"""
The ownership-research agent: one brand in, unified brand + ultimate parent + parent country +
confidence out.

Two calls per brand, never one:
  1. RESEARCH: Claude with the server-side web_search tool, free-text findings. No forced schema
     on this call -- combining a search tool with a forced schema risks the model skipping search
     and answering from memory.
  2. STRUCTURE: no tools, the research text as context, output constrained to
     STRUCTURE_JSON_SCHEMA.

Tier 2: if tier 1 comes back with confidence="low", or the brand string looks truncated, the
brand is researched again with up to 3 goods-description snippets from the customs rows (the
full brand name is often spelled out there even when the brand field was cut short).

This module also holds the model config, token/cost accounting and the name-matching helpers
shared by the pipeline and the evals.
"""

import json
import os
import re
import sys
import time

import anthropic
import pandas as pd
from dotenv import load_dotenv

from .prompts import (
    RESEARCH_SYSTEM_INSTRUCTIONS, RESEARCH_USER_MESSAGE_TIER1, RESEARCH_USER_MESSAGE_TIER2,
    STRUCTURE_SYSTEM_INSTRUCTIONS, STRUCTURE_USER_MESSAGE, STRUCTURE_JSON_SCHEMA,
)

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

# Chosen for the production run from the Haiku-vs-Sonnet comparison (evals/compare_models.py).
MODEL = "claude-haiku-4-5"

# Haiku 4.5 only supports the basic web search tool; Sonnet 4.6 supports the dynamic-filtering
# variant. A wrong variant is a 400, not a silent fallback, so it is set per model.
WEB_SEARCH_TOOL_BY_MODEL = {
    "claude-haiku-4-5": {"type": "web_search_20250305", "name": "web_search"},
    "claude-sonnet-4-6": {"type": "web_search_20260209", "name": "web_search"},
}

# $/1M tokens, standard pricing at the time of the runs (August 2026).
PRICING = {
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
}
CACHE_WRITE_MULT = 1.25
CACHE_READ_MULT = 0.10
BATCH_DISCOUNT = 0.50  # Message Batches API: 50% off all token usage

MAX_RETRIES = 3
RETRY_BASE_DELAY = 4.0
MAX_SOURCES_PER_BRAND = 6

USAGE_KEYS = ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")


def make_client(max_retries: int = 8) -> anthropic.Anthropic:
    load_dotenv()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("ANTHROPIC_API_KEY not found in environment/.env")
    return anthropic.Anthropic(api_key=api_key, timeout=60.0, max_retries=max_retries)


def web_search_tool(model: str) -> dict:
    return WEB_SEARCH_TOOL_BY_MODEL.get(model, WEB_SEARCH_TOOL_BY_MODEL[MODEL])


def system_block(text: str, ttl: str = None) -> list:
    """System prompt marked for prompt caching. The Batch API path uses ttl="1h", since a large
    batch can take longer than the default 5-minute cache lifetime to work through."""
    cache_control = {"type": "ephemeral", "ttl": ttl} if ttl else {"type": "ephemeral"}
    return [{"type": "text", "text": text, "cache_control": cache_control}]


# --------------------------------------------------------------------------
# Token usage and cost
# --------------------------------------------------------------------------

def usage_dict(usage) -> dict:
    return {
        "input_tokens": usage.input_tokens or 0,
        "output_tokens": usage.output_tokens or 0,
        "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", 0) or 0,
        "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0) or 0,
    }


def empty_usage() -> dict:
    return dict.fromkeys(USAGE_KEYS, 0)


def add_usage(a: dict, b: dict) -> dict:
    return {k: a.get(k, 0) + b.get(k, 0) for k in USAGE_KEYS}


def total_usage(rows) -> dict:
    """Sums usage over an iterable of dicts (or DataFrame rows) carrying the USAGE_KEYS columns."""
    total = empty_usage()
    for r in rows:
        total = add_usage(total, {k: int(r[k]) for k in USAGE_KEYS})
    return total


def estimate_cost(usage: dict, model: str = MODEL, batch: bool = False) -> float:
    """Token cost in USD. Web search fees ($10 per 1,000 searches) are not included."""
    price = PRICING[model]
    cost = (
        usage["input_tokens"] * price["input"]
        + usage["output_tokens"] * price["output"]
        + usage["cache_creation_input_tokens"] * price["input"] * CACHE_WRITE_MULT
        + usage["cache_read_input_tokens"] * price["input"] * CACHE_READ_MULT
    ) / 1_000_000
    return cost * BATCH_DISCOUNT if batch else cost


# --------------------------------------------------------------------------
# Name matching (truncation heuristic, company/country normalisation)
# --------------------------------------------------------------------------

_VOWELS = set("aeiouAEIOUyY")


def suspect_truncated(name: str) -> bool:
    name = name.strip()
    if not name:
        return False
    if len(name) <= 6:
        return True
    alpha = [c for c in name if c.isalpha()]
    if alpha and not any(c in _VOWELS for c in alpha):
        return True
    last = name[-1]
    if last.isalpha() and last not in _VOWELS:
        if len(name) >= 2 and name[-2].isalpha() and name[-2] not in _VOWELS:
            return True
    return False


_LEGAL_SUFFIX_PATTERNS = [
    r"\bS\.?A\.?\b", r"\bN\.?V\.?\b", r"\bB\.?V\.?\b", r"\bGmbH\b", r"\bmbH\b", r"\bKGaA\b",
    r"\bSE\b", r"\bAG\b", r"\bA/S\b", r"\bASA\b", r"\bOY\b", r"\bAB\b", r"\bAD\b",
    r"\bInc\.?\b", r"\bIncorporated\b", r"\bCorp\.?\b", r"\bCorporation\b",
    r"\bLtd\.?\b", r"\bLimited\b", r"\bLLC\b", r"\bplc\b", r"\bS\.?p\.?A\.?\b", r"\bS\.?r\.?l\.?\b",
    r"\bSrl\b", r"&\s*Co\.?\s*KG", r"\bKG\b", r"\bGroup\b", r"\bHolding[s]?\b", r"\bCo\.?\b",
    r"Sp\.\s*z\s*o\.o\.", r"\bZPH\b",
]


def normalize_company_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    s = name.strip()
    s = re.sub(r"\(.*?\)", "", s)  # drop parenthetical clarifications
    s = re.sub(r"[-/]", " ", s)  # treat hyphens/slashes as word separators, not part of a token
    for pat in _LEGAL_SUFFIX_PATTERNS:
        s = re.sub(pat, "", s, flags=re.IGNORECASE)
    s = re.sub(r"[.,&]", "", s)
    s = re.sub(r"\s+", " ", s).strip().upper()
    return s


def company_names_match(a: str, b: str) -> bool:
    na, nb = normalize_company_name(a), normalize_company_name(b)
    if not na or not nb:
        return na == nb
    if na == nb:
        return True
    if na in nb or nb in na:  # e.g. "KRONE" in "BERNARD KRONE HOLDING"
        return True
    return False


_COUNTRY_UNKNOWN_MARKERS = {"-", "н/д", "", "nan", "none", "unknown"}


def normalize_country(value) -> str:
    if pd.isna(value):
        return "unknown"
    s = str(value).strip()
    if s.lower() in _COUNTRY_UNKNOWN_MARKERS:
        return "unknown"
    s = s.upper()
    if s == "UK":
        s = "GB"
    return s


# --------------------------------------------------------------------------
# Prompt formatting and response parsing (shared with the Batch API path)
# --------------------------------------------------------------------------

def research_message(brand: str, examples: list = None) -> str:
    """Tier-1 message when examples is None, tier-2 message (with goods-description snippets)
    otherwise -- an empty list still selects tier 2, with "(none)" in place of each snippet."""
    if examples is None:
        return RESEARCH_USER_MESSAGE_TIER1.format(attributed_brand=brand)
    e1, e2, e3 = (list(examples) + ["", "", ""])[:3]
    return RESEARCH_USER_MESSAGE_TIER2.format(
        attributed_brand=brand, example_1=e1 or "(none)", example_2=e2 or "(none)", example_3=e3 or "(none)",
    )


def structure_message(brand: str, research_text: str) -> str:
    return STRUCTURE_USER_MESSAGE.format(attributed_brand=brand, research_text=research_text or "(no research available)")


def message_text(msg) -> str:
    return "\n".join(b.text for b in msg.content if b.type == "text")


def message_sources(msg) -> list:
    """Titles and URLs from the web_search results the model actually retrieved -- never from
    the model's own text, so a citation can't be invented."""
    sources = []
    for block in msg.content:
        if block.type == "web_search_tool_result" and isinstance(block.content, list):
            for item in block.content:
                url = getattr(item, "url", None)
                title = getattr(item, "title", None)
                if url:
                    sources.append(f"{title} ({url})" if title else url)
    return sources[:MAX_SOURCES_PER_BRAND]


def n_search_queries(msg) -> int:
    return sum(1 for b in msg.content if b.type == "server_tool_use" and b.name == "web_search")


def parse_json_message(msg) -> dict:
    """The JSON object from a structured-output response, or {} if missing or malformed."""
    if msg is None:
        return {}
    text_block = next((b for b in msg.content if b.type == "text"), None)
    try:
        return json.loads(text_block.text) if text_block else {}
    except json.JSONDecodeError:
        return {}


def ownership_fields(parsed: dict, brand: str, fallback: dict = None) -> dict:
    """The four ownership fields from a parsed structure response. Missing fields fall back to
    `fallback` (an existing answer) or, with no fallback, to a self-referential low-confidence
    answer."""
    fallback = fallback or {"unified_brand_name": brand, "parent_company": brand, "parent_country": "unknown", "confidence": "low"}
    return {k: parsed.get(k, fallback[k]) for k in ("unified_brand_name", "parent_company", "parent_country", "confidence")}


# --------------------------------------------------------------------------
# Synchronous agent (used by the benchmark and the model comparison; the full run goes through
# the Batch API in batch.py with the same prompts and schema)
# --------------------------------------------------------------------------

def _with_retries(label: str, call, retry_on=()):
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return call()
        except (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError, *retry_on) as exc:
            last_error = exc
            if attempt < MAX_RETRIES:
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                print(f"    [{label}] retry {attempt}/{MAX_RETRIES} after {exc!r} (sleeping {delay:.0f}s)")
                time.sleep(delay)
        except anthropic.APIStatusError as exc:
            last_error = exc
            print(f"    [{label}] API error (not retrying): {exc.status_code} {exc.message}")
            break
    print(f"    [{label}] giving up after retries: {last_error}")
    return None


def research_brand(client: anthropic.Anthropic, brand: str, examples: list = None, model: str = MODEL) -> dict:
    def call():
        msg = client.messages.create(
            model=model, max_tokens=4096,
            system=system_block(RESEARCH_SYSTEM_INSTRUCTIONS),
            tools=[web_search_tool(model)],
            messages=[{"role": "user", "content": research_message(brand, examples)}],
        )
        return {"text": message_text(msg), "sources": message_sources(msg),
                "n_search_queries": n_search_queries(msg), "usage": usage_dict(msg.usage)}

    result = _with_retries(f"research/{model}", call)
    return result or {"text": "", "sources": [], "n_search_queries": 0, "usage": empty_usage()}


def structure_brand(client: anthropic.Anthropic, brand: str, research_text: str, model: str = MODEL) -> dict:
    def call():
        msg = client.messages.create(
            model=model, max_tokens=512,
            system=system_block(STRUCTURE_SYSTEM_INSTRUCTIONS),
            output_config={"format": {"type": "json_schema", "schema": STRUCTURE_JSON_SCHEMA}},
            messages=[{"role": "user", "content": structure_message(brand, research_text)}],
        )
        if not any(b.type == "text" for b in msg.content):
            raise ValueError(f"No text block in response (stop_reason={msg.stop_reason})")
        return {**ownership_fields(parse_json_message(msg), brand), "usage": usage_dict(msg.usage)}

    result = _with_retries(f"structure/{model}", call, retry_on=(ValueError,))
    return result or {**ownership_fields({}, brand.strip()), "usage": empty_usage()}


def process_brand(client: anthropic.Anthropic, brand: str, examples: list = None, model: str = MODEL) -> dict:
    """Tier 1, then tier 2 if the answer is low-confidence or the name looks truncated."""
    research = research_brand(client, brand, examples=None, model=model)
    final = structure_brand(client, brand, research["text"], model=model)
    usage = add_usage(research["usage"], final["usage"])
    n_queries = research["n_search_queries"]
    sources = research["sources"]
    tier_used = 1

    if final["confidence"] == "low" or suspect_truncated(brand):
        research2 = research_brand(client, brand, examples=examples or [], model=model)
        final = structure_brand(client, brand, research2["text"], model=model)
        usage = add_usage(usage, add_usage(research2["usage"], final["usage"]))
        n_queries += research2["n_search_queries"]
        sources = research2["sources"] or sources
        tier_used = 2

    return {
        **ownership_fields(final, brand),
        "sources": sources,
        "tier_used": tier_used,
        "n_search_queries": n_queries,
        "usage": usage,
    }
