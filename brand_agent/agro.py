"""
Agro-specificity: for each (brand, customs category) pair, is this brand's product actually what
an agricultural business buys in that category -- not lawn and garden, HoReCa, human pharma or
general industry? Criteria are written per category (prompts.AGRO_CRITERIA).

Closed-book classification on the first goods-description example, no web search, so it is fast
and cheap. Results are appended to the output CSV as they arrive, and a re-run skips pairs that
are already there.
"""

import concurrent.futures
import csv
import os
import threading
import time

import anthropic
import pandas as pd

from .agent import MODEL, system_block, parse_json_message
from .preprocess import examples_for
from .prompts import AGRO_CATEGORIES, AGRO_SYSTEM_PROMPT, AGRO_USER_TEMPLATE, AGRO_JSON_SCHEMA, agro_criteria_for

FIELDNAMES = ["raw_brand_name", "category_id", "category", "agro_specific", "reasoning",
              "unified_brand_name", "parent_company", "parent_country"]


def expand_pairs(final: pd.DataFrame) -> pd.DataFrame:
    """One row per (brand, category) from the ";"-joined category_ids column."""
    rows = []
    for _, r in final[final["confidence"] != "not_attributed"].iterrows():
        for cid in sorted({c.strip() for c in str(r.get("category_ids", "")).split(";") if c.strip()}):
            if cid in AGRO_CATEGORIES:
                rows.append({**r.to_dict(), "category_id": cid, "category": AGRO_CATEGORIES[cid]})
    return pd.DataFrame(rows)


def classify_agro(client: anthropic.Anthropic, final: pd.DataFrame, examples: dict, out_path: str,
                  concurrency: int = 20, limit: int = None, fresh: bool = False, model: str = MODEL) -> None:
    pairs = expand_pairs(final)
    print(f"{len(pairs)} (brand, category) pairs.")
    if limit:
        pairs = pairs.head(limit)

    if fresh and os.path.exists(out_path):
        os.remove(out_path)
    done = set()
    if os.path.exists(out_path):
        prior = pd.read_csv(out_path, encoding="utf-8-sig", keep_default_na=False, dtype=str)
        done = set(zip(prior["raw_brand_name"], prior["category_id"]))
    todo = [r for _, r in pairs.iterrows() if (str(r["raw_brand_name"]), r["category_id"]) not in done]
    print(f"{len(done)} already done, {len(todo)} to do.\n")
    if not todo:
        return

    new_file = not os.path.exists(out_path)
    lock = threading.Lock()
    progress = [0]
    start = time.monotonic()

    with open(out_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if new_file:
            writer.writeheader()

        def worker(row):
            example = (examples_for(examples, row) or ["(немає прикладу)"])[0]
            try:
                msg = client.messages.create(
                    model=model, max_tokens=300,
                    system=system_block(AGRO_SYSTEM_PROMPT),
                    output_config={"format": {"type": "json_schema", "schema": AGRO_JSON_SCHEMA}},
                    messages=[{"role": "user", "content": AGRO_USER_TEMPLATE.format(
                        raw_brand_name=row["raw_brand_name"], unified_brand_name=row["unified_brand_name"],
                        category_id=row["category_id"], category=row["category"],
                        criteria=agro_criteria_for(row["category_id"]), example=example[:600])}],
                )
                parsed = parse_json_message(msg)
            except anthropic.APIError as exc:
                parsed = {}
                print(f"  [error] {row['raw_brand_name']!r}/{row['category_id']}: {exc}")
            with lock:
                writer.writerow({**{k: row[k] for k in FIELDNAMES if k in row},
                                 "agro_specific": parsed.get("agro_specific", ""), "reasoning": parsed.get("reasoning", "")})
                f.flush()
                progress[0] += 1
                if progress[0] % 100 == 0 or progress[0] == len(todo):
                    print(f"  ... {progress[0]}/{len(todo)} ({time.monotonic() - start:.0f}s)", flush=True)

        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            list(executor.map(worker, todo))
    print(f"\nDone in {(time.monotonic() - start) / 60:.1f} min -> {out_path}")
