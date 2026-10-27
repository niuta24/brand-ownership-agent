"""
Evals from the first stage of the project: a Gemini-based classifier that decides whether a
foreign supplier company is US-affiliated, and the two checks run on it.

  classify            the classifier itself (Gemini + Google Search grounding)
  consistency-sample  fixed random sample of unique companies, to classify twice
  compare-runs        run-to-run agreement between two classifications of that sample
  review-sample       stratified sample for blind human review (disagreements, multi-entity rows,
                      rare verdicts, then seeded random fill)
  score-review        model verdicts vs the human labels

  python -m evals.affiliation classify --input companies.csv --output run1.csv
  python -m evals.affiliation consistency-sample --input companies.csv --n 400 --output sample.csv
  python -m evals.affiliation compare-runs --run1 run1.csv --run2 run2.csv --output comparison.csv
  python -m evals.affiliation review-sample --run1 run1.csv --comparison comparison.csv --n 80
  python -m evals.affiliation score-review --review review_labelled.csv

The source company data is not in this repository.

CLASSIFIER

Takes company_name + country from an extraction output (see parse_companies.py)
and asks Gemini, per row, whether the company is "US-affiliated" per four
criteria: US parent incorporation, US HQ, substantial US ownership stake, or
US-originated core technology/license.

Two Gemini calls per row, not one:
  1. RESEARCH call — Google Search grounding tool enabled, no forced JSON
     schema. Empirically, when response_schema is set on the same call,
     gemini-3.1-flash-lite frequently skips the search tool entirely and
     free-form fabricates plausible-looking source URLs instead of citing
     real ones (verified directly: grounding_metadata was None on 3/5 test
     calls that combined tools+schema, yet the model still filled in a
     "sources" list — sometimes with real-looking vertexaisearch.cloud.google.com
     links it had never actually retrieved). Dropping response_schema for
     this call reliably triggers real grounding (5/5 in testing).
  2. STRUCTURE call — no tools, response_schema enforced, given the research
     text as context and asked to produce the final JSON fields. The
     "sources" field in that JSON is discarded and replaced with the real
     URLs pulled from step 1's grounding_metadata, so citations are never
     model-invented.

FIX 1 — Multiple entities in one row: a row like "DELL PRODUCT C O SYNCREON
TECHNOLOGY" merges a producer (Dell) with a logistics/forwarding agent
(Syncreon); extraction/classification can silently anchor on the wrong one.
detect_multiple_entities() runs a regex heuristic over raw_original for each
row — "C/O"-style handoffs, and 2+ non-adjacent legal-entity suffixes (adjacent
ones like "GmbH & Co. KG" are a single compound legal form, not two entities)
— and records multiple_entities_detected + entity_candidates so these rows
can be reviewed separately instead of trusting whichever entity the model
happened to pick.

FIX 2 — Consistency for duplicate company names: previously, the exact same
real-world company (e.g. CNH Industrial subsidiaries) could get different
verdicts across rows because each row triggered its own independent research
call. Now each unique normalized company_name (case-insensitive, trimmed) is
classified exactly ONCE, and that single result is applied to every row
sharing that name — both cheaper (fewer API calls on a file with duplicates)
and structurally immune to the same-name inconsistency.

Requires GEMINI_API_KEY in a .env file (or the environment), or another variable named via
--api-key-env.
"""

import argparse
import json
import os
import re
import sys
import time

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel
from typing import List, Literal, Union

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

DEFAULT_INPUT_CSV = "output/companies.csv"
MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")

MAX_RETRIES = 3
RETRY_BASE_DELAY = 5.0
REQUEST_TIMEOUT_MS = 120_000    # hard cap per HTTP call; see client construction below
REQUEST_DELAY_SECONDS = 1.0     # between rows (each row = 2 calls)
CHECKPOINT_EVERY = 50
MAX_SOURCES_PER_ROW = 6

OUTPUT_DIR = "output"

# Pricing as published at ai.google.dev/gemini-api/docs/pricing (checked 2026-07-13),
# standard (non-batch) tier, USD. Used only to report actual spend after the run —
# prices can change, so treat this as an estimate, not a bill.
PRICE_INPUT_PER_1M = 0.25
PRICE_OUTPUT_PER_1M = 1.50
GROUNDING_FREE_PROMPTS_PER_MONTH = 5000  # shared across all Gemini 3 usage on the project
PRICE_PER_1000_GROUNDED_PROMPTS = 14.0


# --------------------------------------------------------------------------
# FIX 1: multiple-entities-in-one-row detection (regex heuristic, no API call)
# --------------------------------------------------------------------------

LEGAL_SUFFIX_PATTERN = re.compile(
    r"\b("
    r"GMBH|LLC|LTD|LIMITED|INC|PLC|LLP|AG|KG|"
    r"S\.?R\.?O\.?|S\.?R\.?L\.?|S\.?P\.?A\.?|S\.?A\.?S?\.?|S\.?L\.?U?\.?|"
    r"SP\.?\s*Z\.?\s*O\.?\s*O\.?|SP\.?\s*K\.?|"
    r"A/S|N\.?V\.?|B\.?V\.?|KFT|ZRT|D\.?O\.?O\.?|OY|AB|JSC|OOO|"
    r"CO\.?\s*KG|CO\.?\s*LTD|"
    # Swiss/Liechtenstein forms not covered above (verified against real data
    # in this dataset — no false-positive collisions found for any of these):
    r"SAGL|SARL|ANSTALT|"
    # Baltic/other common forms also missing and present in this dataset:
    r"UAB|SIA|APS|E?OOD"
    r")\b",
    re.IGNORECASE,
)
# Requires a real separator (space or slash) between the two letters, and both
# letters isolated as their own token — this is what distinguishes an actual
# "c/o" handoff from "Co." (Company) or a compound suffix like "Co. KG"/"Co.,
# Ltd", where C and O are either fused into one word or immediately adjacent.
CARE_OF_PATTERN = re.compile(r"(?<!\w)C[\s/\\]+O(?!\w)", re.IGNORECASE)
# Standalone "BY" as a handoff marker ("...BY ORDER OF X", "OWNED BY X",
# "VALVOLE HOFMANN BY BONINO ENGINEERING SRL"). Checked against this dataset:
# 19/5,694 rows match, the large majority genuine two-entity handoffs. Known
# false-positive source: place names using "by" as a locative word (English
# "Goring by Sea"; Danish "by" = "town", e.g. "Ans By, Denmark") — accepted,
# since this heuristic only flags rows for human review, it doesn't decide
# anything silently, so an occasional extra row flagged is a low-cost miss.
BY_PATTERN = re.compile(r"(?<!\w)BY(?!\w)", re.IGNORECASE)
SUFFIX_COMPOUND_GAP = 15  # chars; suffixes closer than this are one compound legal form


def detect_multiple_entities(raw_text: str) -> tuple[bool, list[str]]:
    """Heuristic-only (no LLM call): flags rows where raw_original likely merges
    two distinct company names, so they can be reviewed separately rather than
    trusting whichever one the model anchored on. Catches 'c/o'- and 'by'-style
    handoffs, and 2+ non-adjacent legal-entity suffixes. Does not attempt full NER-style
    splitting of two proper-noun sequences that share no suffix at all — that
    would need real NLP, not regex; this covers the concrete failure mode
    (Dell/Syncreon) and similar suffix- or c/o-marked cases."""
    if not raw_text:
        return False, []

    candidates: list[str] = []

    co_match = CARE_OF_PATTERN.search(raw_text)
    if co_match:
        before = raw_text[: co_match.start()].strip(" \"'.,")
        after = raw_text[co_match.end():].strip(" \"'.,")
        candidates.extend(w for w in (before, after) if w)

    by_match = BY_PATTERN.search(raw_text)
    if by_match:
        before = raw_text[: by_match.start()].strip(" \"'.,")
        after = raw_text[by_match.end():].strip(" \"'.,")
        candidates.extend(w for w in (before, after) if w)

    suffix_matches = list(LEGAL_SUFFIX_PATTERN.finditer(raw_text))
    distinct_spans = []
    for m in suffix_matches:
        if distinct_spans and m.start() - distinct_spans[-1][1] <= SUFFIX_COMPOUND_GAP:
            distinct_spans[-1] = (distinct_spans[-1][0], m.end())
        else:
            distinct_spans.append((m.start(), m.end()))

    if len(distinct_spans) >= 2:
        prev_end = 0
        for start, end in distinct_spans:
            window = raw_text[prev_end:end].strip(" \"'.,")
            words = window.split()
            candidate = " ".join(words[-6:]) if len(words) > 6 else window
            if candidate:
                candidates.append(candidate)
            prev_end = end

    multi = bool(co_match) or bool(by_match) or len(distinct_spans) >= 2

    seen = set()
    unique_candidates = []
    for c in candidates:
        key = c.lower()
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)
    return multi, unique_candidates


# --------------------------------------------------------------------------
# FIX 2: dedup classification by normalized company_name
# --------------------------------------------------------------------------

def normalize_company_name(name: str) -> str:
    name = str(name).strip().strip("\"'")
    name = re.sub(r"\s+", " ", name)
    return name.lower()


class USAffiliation(BaseModel):
    is_us_affiliated: Union[bool, Literal["unclear"]]
    confidence: Literal["high", "medium", "low"]
    evidence_type: Literal["ownership", "HQ", "incorporation", "technology_license", "none"]
    parent_company_name: str
    reasoning: str
    sources: List[str]


RESEARCH_PROMPT_TEMPLATE = """Research using Google Search whether the following company is \
"US-affiliated" for export-control screening purposes.

Company name: {company_name}
Country recorded in customs data: {country}
City recorded in customs data (may be a specific facility/branch, not necessarily the \
global HQ — supplementary context only, use it to find the right real-world entity when \
the name is generic or common, not as evidence of HQ location): {city}

A company is US-affiliated if AT LEAST ONE of the following is true:
(a) Its ultimate parent company is incorporated in the United States.
(b) Its global headquarters is located in the United States.
(c) A substantial ownership stake (parent company or majority/controlling shareholder) \
is US-based.
(d) The product it manufactures/sells is produced under a US-originated core technology \
or license.

Actually search for current, verifiable information — do not answer from memory alone, \
especially since companies may have merged, been acquired, spun off, or renamed. If the \
company is part of a larger group (subsidiary, brand, joint venture), research the group's \
ultimate parent and controlling shareholder, not just the local entity.

Important: being listed on a US stock exchange, having US customers/subsidiaries/sales \
offices, supplying US government agencies, or maintaining a US regional/divisional office \
does NOT by itself establish US affiliation under these criteria. Criterion (b) means the \
company's GLOBAL/worldwide headquarters — the address in official corporate filings where \
the group is run from — not a regional, national, or divisional office that happens to be \
in the US. If the ultimate parent's global HQ and incorporation are outside the US, say so \
plainly even if the company also has significant US operations, contracts, or customers.

Write a short factual research summary (a few sentences) covering: who the ultimate parent \
or controlling shareholder is, where HQ and incorporation are, and whether there is a \
US-technology-license angle. State plainly if you could not find reliable information."""

STRUCTURE_PROMPT_TEMPLATE = """Based on the research summary below, produce a structured \
US-affiliation classification.

Company: {company_name}
Country: {country}

Research summary:
{research_text}

Return JSON with:
- is_us_affiliated: true, false, or "unclear" — per the four criteria (US parent \
incorporation / US HQ / substantial US ownership stake / US-originated core technology \
license). Use "unclear" if the research summary does not give enough reliable information, \
rather than guessing.
- confidence: "high"/"medium"/"low", based on how clear and well-sourced the research \
summary is.
- evidence_type: "ownership"/"HQ"/"incorporation"/"technology_license"/"none" — whichever \
criterion primarily supports the conclusion; "none" if is_us_affiliated is false or unclear.
- parent_company_name: name of the ultimate parent / controlling shareholder if the research \
identifies one, else "".
- reasoning: a short explanation IN UKRAINIAN summarizing why, based on the research summary.
- sources: leave as an empty list [] — sources are attached separately from search metadata.

Return only the JSON object, no other text."""


# --------------------------------------------------------------------------
# Gemini calls
# --------------------------------------------------------------------------

class BillingQuotaError(Exception):
    """Non-transient API error (billing/quota exhausted, not a rate-limit blip
    or network hiccup) — retrying won't help, so this is raised instead of
    being retried, to stop the whole run cleanly rather than burn through
    retries or silently mark rows as failed."""


# Phrases that mean "this account cannot make another paid call right now,
# retrying won't help" as opposed to a short-lived per-minute throttle (which
# also surfaces as 429 RESOURCE_EXHAUSTED but genuinely is transient — a few
# seconds of backoff fixes it). Verified against a real production failure:
# a "prepayment credits are depleted" 429 that did NOT contain the word
# "quota" at all, which an earlier, narrower version of this check missed
# entirely — it kept retrying and silently continuing for every subsequent
# company instead of stopping. Structured .code/.message fields (below) are
# checked first since they're far more reliable than string-matching the
# full exception repr; this phrase list is the fallback for exception types
# that don't expose those fields (e.g. network-level errors).
_BILLING_PHRASES = (
    "billing", "credits are depleted", "prepayment", "insufficient_quota",
    "exceeded your current quota",
)


def is_billing_or_quota_error(exc: Exception) -> bool:
    if isinstance(exc, genai_errors.APIError):
        code = exc.code
        message = (exc.message or "").lower()
        if code == 403:
            return True
        if code == 429:
            return any(p in message for p in _BILLING_PHRASES)
        return False

    # Fallback for non-APIError exceptions (network timeouts, etc.) — string
    # match on the repr since structured fields aren't available.
    msg = str(exc)
    lowered = msg.lower()
    if "403" in msg and ("permission_denied" in lowered or "permission" in lowered):
        return True
    if ("429" in msg or "resource_exhausted" in lowered) and any(p in lowered for p in _BILLING_PHRASES):
        return True
    return False


def research_company(client: genai.Client, company_name: str, country: str, city: str = "") -> dict:
    """Step 1: grounded free-text research call. Returns text + real source URLs + token usage."""
    prompt = RESEARCH_PROMPT_TEMPLATE.format(
        company_name=company_name, country=country, city=city or "(not available)"
    )
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                ),
            )
            gm = response.candidates[0].grounding_metadata
            sources = []
            if gm and gm.grounding_chunks:
                for chunk in gm.grounding_chunks:
                    if chunk.web and chunk.web.uri:
                        title = chunk.web.title or ""
                        sources.append(f"{title} ({chunk.web.uri})" if title else chunk.web.uri)
            sources = sources[:MAX_SOURCES_PER_ROW]
            # Google bills grounding overage per individual search QUERY, not
            # per prompt/request — a single request can issue several queries
            # internally (verified: one test call issued 2). The free monthly
            # tier is measured in prompts, the paid overage in queries — a
            # unit mismatch that undercounted real cost until this was added.
            n_queries = len(gm.web_search_queries) if gm and gm.web_search_queries else 0

            usage = response.usage_metadata
            return {
                "text": response.text or "",
                "sources": sources,
                "search_used": bool(gm and gm.grounding_chunks),
                "n_search_queries": n_queries,
                "prompt_tokens": usage.prompt_token_count or 0,
                "output_tokens": usage.candidates_token_count or 0,
            }
        except Exception as exc:  # noqa: BLE001
            if is_billing_or_quota_error(exc):
                raise BillingQuotaError(str(exc)) from exc
            last_error = exc
            if attempt < MAX_RETRIES:
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                print(f"    [research] retry {attempt}/{MAX_RETRIES} after error: {exc} (sleeping {delay:.0f}s)")
                time.sleep(delay)

    print(f"    [research] giving up after {MAX_RETRIES} attempts: {last_error}")
    return {"text": "", "sources": [], "search_used": False, "n_search_queries": 0, "prompt_tokens": 0, "output_tokens": 0}


def structure_classification(client: genai.Client, company_name: str, country: str, research_text: str) -> dict:
    """Step 2: no-tools structured call that turns the research text into the final schema."""
    prompt = STRUCTURE_PROMPT_TEMPLATE.format(
        company_name=company_name, country=country, research_text=research_text or "(no research available)"
    )
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=USAffiliation,
                    temperature=0.1,
                ),
            )
            parsed = response.parsed
            if parsed is None:
                parsed = USAffiliation.model_validate(json.loads(response.text))
            usage = response.usage_metadata
            result = parsed.model_dump()
            result["prompt_tokens"] = usage.prompt_token_count or 0
            result["output_tokens"] = usage.candidates_token_count or 0
            return result
        except Exception as exc:  # noqa: BLE001
            if is_billing_or_quota_error(exc):
                raise BillingQuotaError(str(exc)) from exc
            last_error = exc
            if attempt < MAX_RETRIES:
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                print(f"    [structure] retry {attempt}/{MAX_RETRIES} after error: {exc} (sleeping {delay:.0f}s)")
                time.sleep(delay)

    print(f"    [structure] giving up after {MAX_RETRIES} attempts: {last_error}")
    return {
        "is_us_affiliated": "unclear",
        "confidence": "low",
        "evidence_type": "none",
        "parent_company_name": "",
        "reasoning": "Не вдалося отримати структуровану відповідь від моделі.",
        "sources": [],
        "prompt_tokens": 0,
        "output_tokens": 0,
    }


def classify_row(client: genai.Client, company_name: str, country: str, city: str = "") -> dict:
    research = research_company(client, company_name, country, city)
    structured = structure_classification(client, company_name, country, research["text"])

    return {
        "is_us_affiliated": structured["is_us_affiliated"],
        "confidence": structured["confidence"],
        "evidence_type": structured["evidence_type"],
        "parent_company_name": structured["parent_company_name"],
        "reasoning": structured["reasoning"],
        "sources": json.dumps(research["sources"], ensure_ascii=False),
        "search_used": research["search_used"],
        "n_search_queries": research["n_search_queries"],
        "model_used": MODEL_NAME,
        "research_summary": research["text"],
        "_prompt_tokens": research["prompt_tokens"] + structured["prompt_tokens"],
        "_output_tokens": research["output_tokens"] + structured["output_tokens"],
        "_grounded_prompt": 1 if research["search_used"] else 0,
        "_search_queries": research["n_search_queries"],
    }


# --------------------------------------------------------------------------
# Checkpointing
# --------------------------------------------------------------------------

def load_checkpoint(checkpoint_path: str) -> pd.DataFrame:
    if os.path.exists(checkpoint_path):
        # keep_default_na=False: without it, a company with a blank/empty
        # normalized name (_norm_key == "") gets parsed back as NaN on
        # reload, which then fails to match the freshly-computed "" string
        # key on resume — the row is never recognized as "already done" and
        # gets silently reprocessed on every subsequent resume. Verified
        # directly: this caused 6 duplicate-ish entries for one blank-name
        # company across today's many resume cycles (harmless to the final
        # output since only the final matching entry ever gets merged in,
        # but wasteful — this fixes the root cause).
        try:
            existing = pd.read_csv(checkpoint_path, encoding="utf-8-sig", keep_default_na=False)
        except pd.errors.EmptyDataError:
            # A checkpoint from before the save_checkpoint fix (or any other
            # zero-byte/degenerate file) — treat as no checkpoint rather than crash.
            return pd.DataFrame()
        print(f"Resuming: found {len(existing):,} already-processed rows in {checkpoint_path}")
        return existing
    return pd.DataFrame()


def save_checkpoint(results: list[dict], checkpoint_path: str) -> None:
    # An empty results list produces a zero-column CSV that pd.read_csv can't
    # parse back on the next resume (crashes with EmptyDataError) — verified
    # directly (a run that failed on its very first company before appending
    # anything wrote a broken checkpoint that then crashed the next attempt).
    # Nothing to checkpoint yet, so just don't write.
    if not results:
        return
    pd.DataFrame(results).to_csv(checkpoint_path, index=False, encoding="utf-8-sig")


# --------------------------------------------------------------------------
# Consistency check
# --------------------------------------------------------------------------

def flag_parent_conflicts(df: pd.DataFrame) -> pd.DataFrame:
    """Independent per-row research calls can disagree on the same real-world parent
    (e.g. one subsidiary's call surfaces a tech-license angle another call misses).
    Flag any group of rows sharing a non-empty parent_company_name whose
    is_us_affiliated verdicts disagree, so it can be manually reconciled."""
    df = df.copy()
    df["parent_conflict"] = False
    named = df[df["parent_company_name"].str.strip() != ""]
    for parent, group in named.groupby("parent_company_name"):
        if group["is_us_affiliated"].astype(str).nunique() > 1:
            df.loc[group.index, "parent_conflict"] = True
    n_conflicts = df["parent_conflict"].sum()
    if n_conflicts:
        print(f"\nWARNING: {n_conflicts} rows disagree with another row sharing the same "
              f"parent_company_name — see parent_conflict column for manual review:")
        conflicting = df[df["parent_conflict"]]
        for parent, group in conflicting.groupby("parent_company_name"):
            verdicts = group[["company_name", "is_us_affiliated"]].to_string(index=False)
            print(f"  parent={parent!r}:\n{verdicts}")
    return df


def build_unique_companies(df: pd.DataFrame) -> pd.DataFrame:
    """One representative row per normalized company_name. Warns if the same
    normalized name maps to more than one distinct recorded country, since
    FIX 2 dedupes on name alone (as requested) — the most common country for
    that name is used for the shared classification call. Also carries a
    representative city as supplementary search context only (not warned on
    variance — the same company legitimately has facilities in many cities,
    unlike country, which is actually part of the classification criteria)."""
    has_city = "city" in df.columns
    groups = []
    for key, group in df.groupby("_norm_key", sort=False):
        countries = [c for c in group["country"] if str(c).strip() != ""]
        if countries:
            country = pd.Series(countries).mode().iloc[0]
            distinct = sorted(set(countries))
            if len(distinct) > 1:
                print(
                    f"  NOTE: normalized name {key!r} appears with {len(distinct)} different "
                    f"countries in the data ({distinct}); using most common ({country!r}) "
                    f"for the shared classification."
                )
        else:
            country = ""

        city = ""
        if has_city:
            cities = [c for c in group["city"] if str(c).strip() != ""]
            if cities:
                city = pd.Series(cities).mode().iloc[0]

        groups.append({
            "_norm_key": key, "company_name": group["company_name"].iloc[0],
            "country": country, "city": city,
        })
    return pd.DataFrame(groups)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def classify(args):

    load_dotenv()
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        sys.exit(f"{args.api_key_env} not found. Add it to a .env file as {args.api_key_env}=your_key")
    print(f"Using API key from {args.api_key_env}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    is_test = args.limit is not None
    suffix = "_test" if is_test else ""
    output_path = args.output or os.path.join(OUTPUT_DIR, f"us_affiliation{suffix}.csv")
    checkpoint_path = args.checkpoint or os.path.join(OUTPUT_DIR, f"us_affiliation{suffix}_checkpoint.csv")

    df = pd.read_csv(args.input, encoding="utf-8-sig", keep_default_na=False)
    df["_row_id"] = df.index
    if is_test:
        df = df.iloc[: args.limit].copy()
        print(f"TEST MODE: limiting to first {len(df):,} rows.")
    print(f"Loaded {len(df):,} rows from {args.input}")

    # FIX 1: per-row heuristic flag, independent of dedup below.
    if "raw_original" in df.columns:
        detections = df["raw_original"].apply(detect_multiple_entities)
        df["multiple_entities_detected"] = detections.apply(lambda t: t[0])
        df["entity_candidates"] = detections.apply(lambda t: json.dumps(t[1], ensure_ascii=False))
        n_flagged = df["multiple_entities_detected"].sum()
        print(f"FIX 1: {n_flagged:,} / {len(df):,} rows flagged for possible multiple merged entities.")
    else:
        print("FIX 1: no 'raw_original' column found in input — skipping multi-entity detection.")
        df["multiple_entities_detected"] = False
        df["entity_candidates"] = "[]"

    # FIX 2: classify each unique normalized company_name once.
    df["_norm_key"] = df["company_name"].apply(normalize_company_name)
    unique_companies = build_unique_companies(df)
    print(f"FIX 2: {len(unique_companies):,} unique company names out of {len(df):,} rows "
          f"({len(df) - len(unique_companies):,} rows share a name with another row).")

    checkpoint_df = load_checkpoint(checkpoint_path)
    done_keys = set(checkpoint_df["_norm_key"]) if not checkpoint_df.empty else set()
    results = checkpoint_df.to_dict("records") if not checkpoint_df.empty else []

    # Explicit timeout: without one, a stalled connection (e.g. wifi drop/laptop
    # sleep) can hang a call forever with no exception ever raised, silently
    # bypassing the retry logic below (verified directly — a run hung 1h19m
    # with zero progress and near-zero CPU time before this was added).
    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS))

    remaining = unique_companies[~unique_companies["_norm_key"].isin(done_keys)]
    print(f"{len(remaining):,} unique companies left to classify.")

    total_prompt_tokens = sum(r.get("_prompt_tokens", 0) for r in results)
    total_output_tokens = sum(r.get("_output_tokens", 0) for r in results)
    total_grounded_prompts = sum(r.get("_grounded_prompt", 0) for r in results)
    total_search_queries = sum(r.get("_search_queries", 0) for r in results)

    processed_since_checkpoint = 0
    for i, (_, row) in enumerate(remaining.iterrows(), start=1):
        company_name = str(row["company_name"])
        country = str(row["country"])
        city = str(row["city"])
        print(f"[{i}/{len(remaining)}] {company_name} | {country} | {city}")

        try:
            result = classify_row(client, company_name, country, city)
        except BillingQuotaError as exc:
            save_checkpoint(results, checkpoint_path)
            print(
                f"\nSTOPPED: billing/quota issue, not a transient error.\n"
                f"Details: {exc}\n"
                f"Checkpoint saved with {len(results):,} companies completed so far -> {checkpoint_path}\n"
                f"Fix the billing/quota issue, then rerun the same command to resume from here."
            )
            sys.exit(1)
        total_prompt_tokens += result["_prompt_tokens"]
        total_output_tokens += result["_output_tokens"]
        total_grounded_prompts += result["_grounded_prompt"]
        total_search_queries += result["_search_queries"]
        print(f"    -> is_us_affiliated={result['is_us_affiliated']} confidence={result['confidence']} search_used={result['search_used']} n_queries={result['n_search_queries']}")

        result_row = {"_norm_key": row["_norm_key"], "company_name": company_name, "country": country}
        result_row.update(result)
        results.append(result_row)
        processed_since_checkpoint += 1

        if processed_since_checkpoint >= CHECKPOINT_EVERY:
            save_checkpoint(results, checkpoint_path)
            print(f"  checkpoint saved ({len(results):,} unique companies total) -> {checkpoint_path}")
            processed_since_checkpoint = 0

        time.sleep(args.delay)

    save_checkpoint(results, checkpoint_path)

    classification_cols = [
        "is_us_affiliated", "confidence", "evidence_type", "parent_company_name",
        "reasoning", "sources", "search_used", "n_search_queries", "model_used", "research_summary",
    ]
    classified = pd.DataFrame(results)[["_norm_key"] + classification_cols]
    # extraction's own "confidence" column would otherwise silently collide with
    # classification's "confidence" into pandas' auto _x/_y suffixes on merge.
    if "confidence" in df.columns:
        df = df.rename(columns={"confidence": "extraction_confidence"})
    classified = classified.rename(columns={"confidence": "classification_confidence"})
    final_df = df.drop(columns=["_row_id"]).merge(classified, on="_norm_key", how="left")
    final_df = final_df.drop(columns=["_norm_key"])
    final_df = flag_parent_conflicts(final_df)
    final_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\nDone. Wrote {len(final_df):,} rows to {output_path}")

    input_cost = total_prompt_tokens / 1_000_000 * PRICE_INPUT_PER_1M
    output_cost = total_output_tokens / 1_000_000 * PRICE_OUTPUT_PER_1M
    avg_queries_per_prompt = total_search_queries / total_grounded_prompts if total_grounded_prompts else 0
    # NOTE: the free monthly tier is metered in PROMPTS, the paid overage in
    # QUERIES (different units — see ai.google.dev/gemini-api/docs/pricing
    # footnote: "a request may result in one or more queries to Google
    # Search; you are charged for each individual search query"). This script
    # cannot know how many of the 5,000 free monthly prompts were already
    # used elsewhere before this run started, so the split below is NOT a
    # verified bill — treat query_cost_if_all_billable as the conservative
    # upper bound once the free monthly prompt allowance is exhausted, and
    # reconcile against the real Cloud Billing Transactions page for ground truth.
    query_cost_if_all_billable = total_search_queries / 1000 * PRICE_PER_1000_GROUNDED_PROMPTS
    print(
        f"\nToken usage: {total_prompt_tokens:,} input / {total_output_tokens:,} output tokens "
        f"across {len(unique_companies):,} unique companies\n"
        f"Grounded prompts: {total_grounded_prompts:,} | "
        f"Search queries: {total_search_queries:,} (avg {avg_queries_per_prompt:.2f} queries/prompt)\n"
        f"Estimated cost @ standard-tier pricing (checked 2026-07-13, may be stale; "
        f"NOT reconciled against actual Cloud Billing — verify there before relying on this).\n"
        f"Free monthly tier is {GROUNDING_FREE_PROMPTS_PER_MONTH:,} PROMPTS (not queries), shared across all "
        f"Gemini 3 usage on this project — this run alone may or may not still be within it:\n"
        f"  tokens:                        ${input_cost + output_cost:.4f}\n"
        f"  grounding, if fully billable:  ${query_cost_if_all_billable:.4f} "
        f"({total_search_queries:,} queries @ ${PRICE_PER_1000_GROUNDED_PROMPTS}/1000)\n"
        f"  total (upper bound):           ${input_cost + output_cost + query_cost_if_all_billable:.4f}\n"
        f"  (this is a ceiling, not a bill — check Cloud Billing Transactions for the real figure)"
    )



# --------------------------------------------------------------------------
# Self-consistency and stratified manual review
# --------------------------------------------------------------------------

def consistency_sample(args):
    """Samples at the unique normalized-company_name level, the unit the classifier works on --
    sampling raw rows would just re-test duplicates against themselves."""
    df = pd.read_csv(args.input, encoding="utf-8-sig", keep_default_na=False)
    df["_norm_key"] = df["company_name"].apply(normalize_company_name)
    unique_rows = df.drop_duplicates(subset="_norm_key").drop(columns=["_norm_key"])
    print(f"{len(unique_rows):,} unique company names out of {len(df):,} rows.")
    sample = unique_rows.sample(n=min(args.n, len(unique_rows)), random_state=args.seed).reset_index(drop=True)
    sample.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(sample):,} companies (seed={args.seed}) -> {args.output}")


def compare_runs(args):
    """Joins two classification runs of the same sample and reports verdict agreement."""
    cols = ["_norm_key", "company_name", "country", "is_us_affiliated", "classification_confidence"]
    runs = []
    for path in (args.run1, args.run2):
        df = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
        df["_norm_key"] = df["company_name"].apply(normalize_company_name)
        runs.append(df.drop_duplicates(subset="_norm_key")[cols])
    m = runs[0].merge(runs[1].drop(columns=["company_name", "country"]), on="_norm_key", suffixes=("_run1", "_run2"))
    m = m.rename(columns={"classification_confidence_run1": "confidence_run1", "classification_confidence_run2": "confidence_run2"})
    m["match"] = m["is_us_affiliated_run1"] == m["is_us_affiliated_run2"]
    m.drop(columns=["_norm_key"]).to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Same verdict in both runs: {int(m['match'].sum())}/{len(m)}")
    print(pd.crosstab(m["is_us_affiliated_run1"], m["is_us_affiliated_run2"]))
    for run in ("run1", "run2"):
        print(f"confidence {run}: {m[f'confidence_{run}'].value_counts().to_dict()}")
    print(f"Saved -> {args.output}")


def review_sample(args):
    """Builds the human-review sample, in priority order, capped at --n:
      1. every run1/run2 disagreement from the consistency check;
      2. every row flagged multiple_entities_detected (the model may have anchored on the wrong
         merged entity, whatever its confidence);
      3. every remaining True / unclear verdict -- the rare, higher-stakes classes, which a
         proportional sample would leave almost empty;
      4. a seeded random fill of False verdicts.
    classification_confidence was "high" on every row in both runs, so it carries no signal and
    is not used as a stratification axis. Writes a full file and a blind copy without the
    model's answers for the reviewer."""
    r1 = pd.read_csv(args.run1, encoding="utf-8-sig", keep_default_na=False)
    comparison = pd.read_csv(args.comparison, encoding="utf-8-sig", keep_default_na=False)
    r1["_norm_key"] = r1["company_name"].apply(normalize_company_name)
    comparison["_norm_key"] = comparison["company_name"].apply(normalize_company_name)

    detections = r1["raw_original"].apply(detect_multiple_entities)
    r1["multiple_entities_detected"] = detections.apply(lambda t: t[0])
    r1["entity_candidates"] = detections.apply(lambda t: ", ".join(t[1]) if t[1] else "")

    reason_by_key = {}
    mismatch_keys = set(comparison.loc[~comparison["match"].astype(str).eq("True"), "_norm_key"])
    for reason, mask in (
        ("run1_run2_disagreement", r1["_norm_key"].isin(mismatch_keys)),
        ("multiple_entities_detected", r1["multiple_entities_detected"]),
        ("rare_verdict_(True/unclear)", r1["is_us_affiliated"].isin(["True", "unclear"])),
    ):
        for key in r1.loc[mask, "_norm_key"]:
            if key not in reason_by_key and (reason != "rare_verdict_(True/unclear)" or len(reason_by_key) < args.n):
                reason_by_key[key] = reason

    remaining_false = r1[~r1["_norm_key"].isin(reason_by_key) & (r1["is_us_affiliated"] == "False")].drop_duplicates(subset="_norm_key")
    n_fill = max(0, args.n - len(reason_by_key))
    for key in remaining_false.sample(n=min(n_fill, len(remaining_false)), random_state=args.seed)["_norm_key"]:
        reason_by_key[key] = "random_fill"
    print(f"Composition: {pd.Series(reason_by_key).value_counts().to_dict()}")

    bench = r1[r1["_norm_key"].isin(reason_by_key)].drop_duplicates(subset="_norm_key").copy()
    bench["sample_reason"] = bench["_norm_key"].map(reason_by_key)
    bench["manual_verdict"] = ""
    bench["manual_notes"] = ""
    bench = bench.rename(columns={"is_us_affiliated": "model_is_us_affiliated"})
    cols = ["company_name", "country", "raw_original", "sample_reason", "model_is_us_affiliated",
            "classification_confidence", "evidence_type", "parent_company_name", "reasoning",
            "multiple_entities_detected", "entity_candidates", "manual_verdict", "manual_notes"]
    bench = bench[cols].reset_index(drop=True)
    bench.to_csv(args.output, index=False, encoding="utf-8-sig")
    blind = bench.drop(columns=["sample_reason", "model_is_us_affiliated", "reasoning"])
    blind.to_csv(args.output.replace(".csv", "_blind.csv"), index=False, encoding="utf-8-sig")
    print(f"Wrote {len(bench)} companies -> {args.output} (+ _blind copy for the reviewer)")


def score_review(args):
    """Model verdict columns vs manual_verdict on the labelled review sample."""
    df = pd.read_csv(args.review, encoding="utf-8-sig", keep_default_na=False, dtype=str)
    verdict_cols = [c for c in df.columns if c.endswith("is_us_affiliated")]
    for col in ["manual_verdict", *verdict_cols]:
        df[col] = df[col].str.strip()
    df = df[df["manual_verdict"] != ""]
    for col in verdict_cols:
        agree = int((df[col] == df["manual_verdict"]).sum())
        true_rows = df[df["manual_verdict"] == "True"]
        recall = int((true_rows[col] == "True").sum())
        print(f"\n{col}: agrees with the human label on {agree}/{len(df)}; "
              f"finds {recall}/{len(true_rows)} of the true US affiliations")
        print(pd.crosstab(df[col], df["manual_verdict"], rownames=["model"], colnames=["human"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("classify", help="Classify companies (Gemini API).")
    p.add_argument("--input", type=str, default=DEFAULT_INPUT_CSV, help="CSV with company_name, country, city (and raw_original).")
    p.add_argument("--limit", type=int, default=None, help="Only process the first N rows (for testing).")
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--delay", type=float, default=REQUEST_DELAY_SECONDS, help="Seconds between unique companies.")
    p.add_argument("--api-key-env", type=str, default="GEMINI_API_KEY", help="Name of the variable holding the Gemini API key.")
    p.set_defaults(func=classify)

    p = sub.add_parser("consistency-sample", help="Fixed random sample of unique companies (no API).")
    p.add_argument("--input", required=True)
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="output/consistency_sample.csv")
    p.set_defaults(func=consistency_sample)

    p = sub.add_parser("compare-runs", help="Run-to-run agreement (no API).")
    p.add_argument("--run1", required=True)
    p.add_argument("--run2", required=True)
    p.add_argument("--output", default="output/consistency_comparison.csv")
    p.set_defaults(func=compare_runs)

    p = sub.add_parser("review-sample", help="Stratified sample for blind human review (no API).")
    p.add_argument("--run1", required=True)
    p.add_argument("--comparison", required=True)
    p.add_argument("--n", type=int, default=80)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="output/manual_review.csv")
    p.set_defaults(func=review_sample)

    p = sub.add_parser("score-review", help="Model verdicts vs human labels (no API).")
    p.add_argument("--review", required=True)
    p.set_defaults(func=score_review)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
