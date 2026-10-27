"""
The full run through the Message Batches API: the same research -> structure agent as agent.py,
at 50% of the token price, with 1-hour prompt caching and a checkpoint after every stage.

Every pass in this project (the main run, the follow-up, the verification passes) is the same
shape -- a research batch, then a structure batch fed by the research text -- so that shape lives
in one function, research_then_structure(). With a Checkpoint, each submitted batch id is saved
the moment it exists and each finished stage's results are saved when parsed, so an interrupted
run (credit exhaustion, network drop, manual kill) resumes without paying for any stage twice.
"""

import json
import os
import time

import anthropic
import httpx
import pandas as pd
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from .agent import (
    MODEL, USAGE_KEYS, estimate_cost, total_usage, usage_dict, web_search_tool, system_block,
    research_message, structure_message, message_text, parse_json_message, ownership_fields,
    suspect_truncated,
)
from .preprocess import MIXED_MULTI_BRAND_FLAG, examples_from_row
from .prompts import RESEARCH_SYSTEM_INSTRUCTIONS, STRUCTURE_SYSTEM_INSTRUCTIONS, STRUCTURE_JSON_SCHEMA

OWNERSHIP_COLUMNS = ["unified_brand_name", "parent_company", "parent_country", "confidence"]
OUTPUT_COLUMNS = ["raw_brand_name", "attributed_brand", "row_count", "total_customs_value_usd", *OWNERSHIP_COLUMNS]


class Checkpoint:
    """A JSON dict on disk, rewritten after every change. Without a path it is in-memory only."""

    def __init__(self, path: str = None):
        self.path = path
        self.data = {}
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
            print(f"[checkpoint] resuming from {path} (keys: {sorted(self.data)})\n")

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, key, value):
        self.data[key] = value
        if self.path:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.data, f)


# --------------------------------------------------------------------------
# Batch API plumbing
# --------------------------------------------------------------------------

def research_request(custom_id: str, user_message: str, model: str = MODEL) -> Request:
    return Request(custom_id=custom_id, params=MessageCreateParamsNonStreaming(
        model=model, max_tokens=4096,
        system=system_block(RESEARCH_SYSTEM_INSTRUCTIONS, ttl="1h"),
        tools=[web_search_tool(model)],
        messages=[{"role": "user", "content": user_message}],
    ))


def structure_request(custom_id: str, user_message: str, schema: dict = STRUCTURE_JSON_SCHEMA,
                      max_tokens: int = 512, model: str = MODEL) -> Request:
    return Request(custom_id=custom_id, params=MessageCreateParamsNonStreaming(
        model=model, max_tokens=max_tokens,
        system=system_block(STRUCTURE_SYSTEM_INSTRUCTIONS, ttl="1h"),
        output_config={"format": {"type": "json_schema", "schema": schema}},
        messages=[{"role": "user", "content": user_message}],
    ))


def poll_batch(client: anthropic.Anthropic, batch_id: str, label: str):
    start = time.monotonic()
    while True:
        try:
            batch = client.messages.batches.retrieve(batch_id)
        except anthropic.APIConnectionError as exc:
            print(f"  [{label}] poll failed ({exc.__class__.__name__}), retrying in 15s", flush=True)
            time.sleep(15)
            continue
        if batch.processing_status == "ended":
            break
        print(f"  [{label}] status={batch.processing_status} counts={batch.request_counts} "
              f"(waited {time.monotonic() - start:.0f}s)", flush=True)
        time.sleep(15)
    print(f"[{label}] ended after {time.monotonic() - start:.0f}s. counts={batch.request_counts}")
    return batch


def collect_results(client: anthropic.Anthropic, batch_id: str, max_retries: int = 5) -> dict:
    """{custom_id: Message, or None if that request failed}.

    The batch has already succeeded and been billed by the time results are streamed, so a
    dropped connection mid-stream (seen in practice on a 10k+ request batch) is retried by
    re-fetching from scratch, which costs nothing."""
    for attempt in range(1, max_retries + 1):
        try:
            return {r.custom_id: (r.result.message if r.result.type == "succeeded" else None)
                    for r in client.messages.batches.results(batch_id)}
        except (anthropic.APIConnectionError, anthropic.APITimeoutError, httpx.TransportError) as exc:
            if attempt == max_retries:
                raise
            print(f"  [collect] fetch failed ({exc.__class__.__name__}), retrying ({attempt}/{max_retries}) in 5s", flush=True)
            time.sleep(5)


def run_batch(client: anthropic.Anthropic, requests: list, label: str, ckpt: Checkpoint) -> dict:
    """Submits the batch (or resumes the one already recorded under this label) and returns its
    results."""
    batch_id = ckpt.get(f"{label}_batch_id")
    if batch_id:
        print(f"[{label}] resuming batch {batch_id}")
    else:
        print(f"[{label}] submitting {len(requests)} requests...")
        batch_id = client.messages.batches.create(requests=requests).id
        ckpt.set(f"{label}_batch_id", batch_id)
        print(f"  batch id: {batch_id}")
    poll_batch(client, batch_id, label)
    return collect_results(client, batch_id)


def research_then_structure(client: anthropic.Anthropic, label: str, research_messages: list,
                            structure_message_fn, ckpt: Checkpoint, schema: dict = STRUCTURE_JSON_SCHEMA,
                            structure_max_tokens: int = 512, model: str = MODEL, usage_log: list = None) -> list:
    """Runs one research batch and one structure batch over the same items.

    research_messages: one research user message per item.
    structure_message_fn(i, research_text) -> the structure user message for item i.
    Returns one parsed JSON dict per item ({} where a request failed). Results are cached in the
    checkpoint under the label, so a finished pass is never re-run.
    """
    cached = ckpt.get(f"{label}_results")
    if cached is not None:
        print(f"[{label}] already done, loaded from checkpoint.")
        return cached

    n = len(research_messages)
    research = run_batch(client, [research_request(f"r-{i}", m, model) for i, m in enumerate(research_messages)],
                         f"{label}-research", ckpt)
    texts = [message_text(research[f"r-{i}"]) if research.get(f"r-{i}") else "" for i in range(n)]

    structured = run_batch(client, [structure_request(f"s-{i}", structure_message_fn(i, texts[i]), schema,
                                                      structure_max_tokens, model) for i in range(n)],
                           f"{label}-structure", ckpt)
    parsed = [parse_json_message(structured.get(f"s-{i}")) for i in range(n)]

    if usage_log is not None:
        for stage, results in (("research", research), ("structure", structured)):
            for custom_id, msg in results.items():
                if msg is not None:
                    usage_log.append({"stage": f"{label}-{stage}", "custom_id": custom_id, **usage_dict(msg.usage)})

    ckpt.set(f"{label}_results", parsed)
    return parsed


def write_telemetry(usage_log: list, path: str, model: str = MODEL) -> float:
    """Saves per-request token usage and returns the batch-discounted token cost."""
    df = pd.DataFrame(usage_log, columns=["stage", "custom_id", *USAGE_KEYS])
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return estimate_cost(total_usage(df.to_dict("records")), model, batch=True)


def apply_updates(df: pd.DataFrame, updates: dict) -> pd.DataFrame:
    """updates: {raw_brand_name: {ownership field: value}} -> df with those fields replaced."""
    df = df.copy()
    for col in OWNERSHIP_COLUMNS:
        df[col] = df.apply(lambda r: updates[r["raw_brand_name"]][col] if r["raw_brand_name"] in updates else r[col], axis=1)
    return df


# --------------------------------------------------------------------------
# Full production run
# --------------------------------------------------------------------------

def run_production(client: anthropic.Anthropic, input_path: str, output_path: str, model: str = MODEL) -> pd.DataFrame:
    """input_path: the prepared extract (preprocess.prepare). MIXED_MULTI_BRAND rows are passed
    through as not_attributed; every other brand goes through tier 1, and tier 2 where needed."""
    extract = pd.read_csv(input_path, encoding="utf-8-sig", keep_default_na=False)
    mixed = extract["attributed_brand"] == MIXED_MULTI_BRAND_FLAG
    todo = extract[~mixed].reset_index(drop=True)
    brands = [str(b) for b in todo["raw_brand_name"]]
    n = len(brands)
    print(f"{len(extract)} rows; {n} to research, {int(mixed.sum())} MIXED_MULTI_BRAND passed through.\n")

    ckpt = Checkpoint(output_path.replace(".csv", "_checkpoint.json"))
    usage_log = []

    tier1 = research_then_structure(
        client, "tier1", [research_message(b) for b in brands],
        lambda i, text: structure_message(brands[i], text), ckpt, model=model, usage_log=usage_log,
    )
    final = [ownership_fields(p, brands[i]) for i, p in enumerate(tier1)]

    escalate = [i for i in range(n) if final[i]["confidence"] == "low" or suspect_truncated(brands[i])]
    print(f"\n{len(escalate)}/{n} brands escalate to tier 2.\n")
    if escalate:
        tier2 = research_then_structure(
            client, "tier2", [research_message(brands[i], examples_from_row(todo.loc[i])) for i in escalate],
            lambda j, text: structure_message(brands[escalate[j]], text), ckpt, model=model, usage_log=usage_log,
        )
        for i, p in zip(escalate, tier2):
            if p:  # a failed tier-2 request keeps the tier-1 answer
                final[i] = ownership_fields(p, brands[i])

    keep = OUTPUT_COLUMNS[:4] + (["category_ids"] if "category_ids" in extract.columns else [])
    researched = pd.concat([todo.reindex(columns=keep), pd.DataFrame(final)], axis=1)
    passthrough = extract[mixed].reindex(columns=keep).assign(
        unified_brand_name=MIXED_MULTI_BRAND_FLAG, parent_company="", parent_country="unknown", confidence="not_attributed",
    )
    out = pd.concat([researched, passthrough], ignore_index=True)
    out = out.sort_values("total_customs_value_usd", ascending=False, key=lambda s: pd.to_numeric(s, errors="coerce"))
    out.to_csv(output_path, index=False, encoding="utf-8-sig")

    cost = write_telemetry(usage_log, output_path.replace(".csv", "_telemetry.csv"), model)
    print(f"\n{len(out)} rows -> {output_path}")
    print(f"confidence=low: {int((out['confidence'] == 'low').sum())}")
    print(f"Token cost this invocation (Batch API, 50% off): ${cost:.2f}")
    return out


def run_smoke_test(client: anthropic.Anthropic, brands: list, model: str = MODEL) -> None:
    """Exercises the Batch API path end to end on a few brands, without a checkpoint file, and
    prints cache reads/writes so prompt caching can be confirmed before a large run."""
    print(f"Batch smoke test on {len(brands)} brands: {brands}\n")
    usage_log = []
    parsed = research_then_structure(
        client, "smoke", [research_message(b) for b in brands],
        lambda i, text: structure_message(brands[i], text), Checkpoint(), model=model, usage_log=usage_log,
    )
    print("\nResults:")
    for b, p in zip(brands, parsed):
        print(f"  {b!r} -> {ownership_fields(p, b) if p else 'FAILED'}")
    structure_usage = [u for u in usage_log if u["stage"].endswith("structure")]
    print(f"\nStructure calls: cache_read={sum(u['cache_read_input_tokens'] for u in structure_usage):,} "
          f"cache_write={sum(u['cache_creation_input_tokens'] for u in structure_usage):,}")
    print(f"Token cost (Batch API, 50% off): ${estimate_cost(total_usage(usage_log), model, batch=True):.4f}")
