"""
Everything before the LLM agent: turning raw customs rows into one row per brand string.

  1. extract     -- one row per unique brand string, with row count, summed customs value and up
                    to 3 readable goods-description examples (pure pandas).
  2. attribution -- declarants sometimes put several brands in one field ("JOHN DEERE, CASE IH",
                    "Кентавр/NOWA"). A rule-based detector flags these as MIXED_MULTI_BRAND: they
                    are kept as their own line, never researched and never folded into any one
                    brand's total.
  3. near-duplicate check (after the agent has run) -- independent per-brand calls can resolve
                    the same brand to several spellings ("JOHN DEERE" / "John Deere",
                    "Sampo Rosenlew" / "Sampo-Rosenlew"). Same-word-set groups are found without
                    an API call, then each group is verified with one grounded Gemini call pair
                    before merging.

raw_brand_name is never modified: it is the exact-match join key back onto the raw rows.
"""

import json
import os
import re
import sys
import time
import unicodedata
from typing import Literal

import pandas as pd
from pydantic import BaseModel

from .prompts import NEAR_DUPLICATE_RESEARCH_PROMPT, NEAR_DUPLICATE_STRUCTURE_PROMPT

# Input columns of the raw customs file.
BRAND_COL = "brand_unified"
DESC_COL = "goods_description"
VALUE_COL = "customs_value_usd"
CATEGORY_COL = "category_id"  # optional; carried through as ";"-joined category_ids

EMPTY_MARKERS = {"", "нема даних", "no data", "nan", "none"}
N_EXAMPLES = 3

# --------------------------------------------------------------------------
# 1. Unique-brand extract
# --------------------------------------------------------------------------


def is_empty_brand(value) -> bool:
    if pd.isna(value):
        return True
    return str(value).strip().lower() in EMPTY_MARKERS


def pick_examples(descriptions: list, brand: str, n: int = N_EXAMPLES) -> list:
    """Up to n example descriptions for a brand, preferring ones that actually contain a legible
    mention of the brand over the longest/most cluttered row (a customs description can run to
    hundreds of characters of part numbers with the brand buried or absent).

    Ranking, best first: contains the brand name (case-insensitive) -> lower digit density ->
    shorter text. Duplicates are collapsed first.
    """
    brand_lower = brand.strip().lower()
    unique_descs = list(dict.fromkeys(str(d).strip() for d in descriptions if not pd.isna(d) and str(d).strip()))

    def score(text: str):
        contains_brand = brand_lower in text.lower() if brand_lower else False
        digit_ratio = sum(c.isdigit() for c in text) / max(len(text), 1)
        return (0 if contains_brand else 1, digit_ratio, len(text))

    return sorted(unique_descs, key=score)[:n]


def load_raw(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    df = pd.read_excel(path) if ext in (".xlsx", ".xls") else pd.read_csv(path, encoding="utf-8-sig")
    missing = [c for c in (BRAND_COL, DESC_COL, VALUE_COL) if c not in df.columns]
    if missing:
        sys.exit(f"Input is missing required column(s): {missing}")
    return df


def build_extract(df: pd.DataFrame) -> pd.DataFrame:
    populated = df[~df[BRAND_COL].apply(is_empty_brand)].copy()
    print(f"{len(populated):,} / {len(df):,} rows have a non-empty {BRAND_COL}.")
    populated[VALUE_COL] = pd.to_numeric(populated[VALUE_COL], errors="coerce").fillna(0.0)

    rows = []
    for brand, group in populated.groupby(BRAND_COL, sort=False):
        examples = pick_examples(group[DESC_COL].tolist(), str(brand))
        examples += [""] * (N_EXAMPLES - len(examples))
        row = {
            "raw_brand_name": brand,
            "row_count": len(group),
            "total_customs_value_usd": group[VALUE_COL].sum(),
            **{f"example_description_{i + 1}": examples[i] for i in range(N_EXAMPLES)},
        }
        if CATEGORY_COL in group.columns:
            row["category_ids"] = ";".join(sorted({str(c).strip() for c in group[CATEGORY_COL].dropna() if str(c).strip()}))
        rows.append(row)

    return pd.DataFrame(rows).sort_values("total_customs_value_usd", ascending=False).reset_index(drop=True)


def examples_from_row(row) -> list:
    return [str(row.get(f"example_description_{i}", "") or "").strip() for i in range(1, N_EXAMPLES + 1)]


def load_examples(path: str) -> dict:
    """{attributed_brand and raw_brand_name: [non-empty example descriptions]} from an extract."""
    df = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
    examples = {}
    for _, row in df.iterrows():
        texts = [t for t in examples_from_row(row) if t]
        for key in (str(row.get("attributed_brand", "")).strip(), str(row["raw_brand_name"]).strip()):
            if key and key not in examples:
                examples[key] = texts
    return examples


def examples_for(examples: dict, row) -> list:
    return examples.get(str(row["attributed_brand"]).strip()) or examples.get(str(row["raw_brand_name"]).strip()) or []


# --------------------------------------------------------------------------
# 2. Multi-brand detection and attribution
#
# Each candidate string (one containing a comma, slash, "&", "and" or "та") is split on those
# separators and each PART is judged as a standalone brand name: too short (the "A"/"I" in
# "A&I") or containing a generic business word (SUPPLIES, MACHINERY, TRADING, GROUP...: a
# fragment of one company's descriptive name) -> implausible.
#   confident_genuine         every part plausible -> MIXED_MULTI_BRAND
#   confident_false_positive  no part plausible ("A&I", "JUMMOS AGRICULTURAL SUPPLIES AND
#                             MACHINERY") -> left as one brand
#   borderline                mixed signal -> left as one brand, listed for manual review
#
# "&" and "and"/"та" alone are weak evidence: real single companies use them (Procter & Gamble,
# Pratt and Whitney). In a 24-file scan essentially every genuine multi-brand list used a comma
# or slash, so a string with only weak separators is borderline unless it is on the curated
# exception list. A "/" between digits and a 1-2 letter suffix ("Via Stelloni 12/B") is a
# building number, not a separator, and is masked out first.
# --------------------------------------------------------------------------

MIXED_MULTI_BRAND_FLAG = "MIXED_MULTI_BRAND"

GENERIC_BUSINESS_WORDS = {
    "SUPPLIES", "MACHINERY", "TRADING", "INDUSTRIES", "INDUSTRY", "GROUP",
    "COMPANY", "CORPORATION", "ENTERPRISE", "ENTERPRISES", "SOLUTIONS",
    "SERVICES", "EQUIPMENT", "PRODUCTS", "MANUFACTURING", "INTERNATIONAL",
    "SYSTEMS", "IMPORT", "EXPORT", "LOGISTICS", "DISTRIBUTION", "HOLDING",
    "HOLDINGS",
    # added after the 24-file scan
    "TECHNOLOGY", "SCIENCE", "IMP", "EXP", "MACHINE", "ELECTRICAL",
    "ELECTRIC", "MECHANICAL", "SERVICE", "INSTITUTE", "RESEARCH",
    "STANDARDS", "CONTROL", "CARE", "NUTRITION", "HEALTH", "HEALTHCARE",
    # Chinese-trading-company / institutional patterns from the same scan, and bare legal-entity
    # fragments that show up as their own "part" after a split
    # ("Weifang Frd Machinery Co.,Ltd" -> [..., "Co.", "Ltd"])
    "INSTRUMENT", "INSTRUMENTS", "CHEMICAL", "CHEMICALS", "PHARMACEUTIC",
    "PHARMACEUTICAL", "BIOSCIENCES", "BIOLOGICALS", "MINING",
    "AGROCHEMICALS", "AUTOMATION", "CONSTRUCTION", "CO", "LTD", "CORP",
    "INC",
}
MIN_PART_LEN = 2

# Evidence-based only: every entry was found wrongly flagged as confident_genuine in the scan.
# Add a name only on the same basis -- a verified real single entity, not a guess.
KNOWN_SINGLE_ENTITY_EXCEPTIONS = {
    "PRATT AND WHITNEY",
    "ENDRESS AND HAUSER",
}

SEPARATOR = re.compile(r"\s*(?:,|/|&|(?<![\w])and(?![\w])|(?<![\w])та(?![\w]))\s*", re.IGNORECASE)
STRONG_SEPARATOR_PRESENT = re.compile(r",|/")
ADDRESS_UNIT_SLASH = re.compile(r"(?<=\d)\s*/\s*(?=[A-Za-z]{1,2}(?![A-Za-z]))")
_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЇїІіЄєҐґ]+")


def part_looks_plausible(part: str) -> bool:
    stripped = part.strip()
    if len(stripped) < MIN_PART_LEN:
        return False
    words = {w.upper() for w in _WORD_RE.findall(stripped)}
    return not (words & GENERIC_BUSINESS_WORDS)


def classify_candidate(raw_name: str):
    """Returns (category, parts, note), category in {"not_candidate", "confident_genuine",
    "confident_false_positive", "borderline"}."""
    masked = ADDRESS_UNIT_SLASH.sub(" ", raw_name)
    parts = [p.strip() for p in SEPARATOR.split(masked) if p.strip()]
    if len(parts) < 2:
        return "not_candidate", parts, ""

    if not STRONG_SEPARATOR_PRESENT.search(masked):
        normalized = re.sub(r"\s+", " ", raw_name.strip()).upper()
        if normalized in KNOWN_SINGLE_ENTITY_EXCEPTIONS:
            return "confident_false_positive", parts, "known single-entity name (curated exception list)"
        return "borderline", parts, (
            "only weak separator(s) ('&'/'and'/'та', no comma or slash) — could be one company's/"
            "institution's own name (e.g. 'Pratt and Whitney') — not auto-decided"
        )

    plausible = [part_looks_plausible(p) for p in parts]
    if not any(plausible):
        return "confident_false_positive", parts, "no part looks like a standalone brand name"
    if all(plausible):
        return "confident_genuine", parts, ""
    implausible_parts = [p for p, ok in zip(parts, plausible) if not ok]
    return "borderline", parts, f"mixed plausibility — implausible part(s): {implausible_parts}"


def scan_multi_brand(values) -> pd.DataFrame:
    """One row per value that contains a separator at all: category, parts, note."""
    rows = []
    for v in values:
        v = str(v).strip()
        if not v:
            continue
        category, parts, note = classify_candidate(v)
        if category != "not_candidate":
            rows.append({"raw_value": v, "category": category, "n_parts": len(parts), "parts": parts, "note": note})
    return pd.DataFrame(rows, columns=["raw_value", "category", "n_parts", "parts", "note"])


def add_attributed_brand(extract: pd.DataFrame) -> pd.DataFrame:
    categories = extract["raw_brand_name"].apply(lambda name: classify_candidate(str(name))[0])
    extract = extract.copy()
    extract.insert(1, "attributed_brand", extract["raw_brand_name"].where(categories != "confident_genuine", MIXED_MULTI_BRAND_FLAG))

    n_mixed = int((categories == "confident_genuine").sum())
    print(f"{n_mixed} row(s) flagged {MIXED_MULTI_BRAND_FLAG} (kept as their own line, not researched).")
    borderline = extract.loc[categories == "borderline", "raw_brand_name"].tolist()
    if borderline:
        print(f"{len(borderline)} BORDERLINE row(s) left as single brands -- review manually:")
        for name in borderline:
            print(f"  {name!r}")
    return extract


def prepare(input_path: str, output_path: str) -> pd.DataFrame:
    """Raw customs rows -> unique-brand extract with attributed_brand."""
    df = load_raw(input_path)
    print(f"Loaded {len(df):,} rows from {input_path}")
    extract = add_attributed_brand(build_extract(df))
    extract.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n{len(extract):,} unique brand strings -> {output_path}")
    return extract


# --------------------------------------------------------------------------
# 3. Near-duplicate spellings (run on the agent's output)
#
# Detection is deliberately narrow: same token set after case/diacritic folding and splitting on
# whitespace, hyphen, period, comma and slash. "ROSENLEW" does NOT match "SAMPO ROSENLEW" --
# a strict same-words rule, not fuzzy matching, to avoid over-merging. Case-only groups merge
# without a call; everything else gets one research + structure call pair per group. Only
# confirmed (high/medium) groups are merged; the rest are listed for manual review.
# --------------------------------------------------------------------------

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")
GEMINI_MAX_RETRIES = 3
GEMINI_RETRY_BASE_DELAY = 5.0
GEMINI_TIMEOUT_MS = 120_000
_BILLING_PHRASES = ("billing", "credits are depleted", "prepayment", "insufficient_quota", "exceeded your current quota")


def token_set(name: str) -> frozenset:
    folded = "".join(c for c in unicodedata.normalize("NFKD", name) if not unicodedata.combining(c)).upper()
    return frozenset(t for t in re.split(r"[\s\-.,/]+", folded) if t)


def is_case_only_duplicate(spellings: list) -> bool:
    """"SCHUMACHER" and "Schumacher" cannot be different entities, so there is nothing to verify.
    (The verification prompt's anti-merge caution was wrongly rejecting exactly these.)"""
    return len({s.upper() for s in spellings}) == 1


def detect_candidate_groups(df: pd.DataFrame) -> list:
    """df: one row per attributed brand with unified_brand_name, row_count,
    total_customs_value_usd. Returns [{spellings, row_counts, values}] per candidate group."""
    work = df[df["unified_brand_name"].str.strip() != ""].copy()
    work["_token_key"] = work["unified_brand_name"].apply(token_set)
    groups = []
    for _, sub in work.groupby("_token_key"):
        spellings = sorted(sub["unified_brand_name"].unique())
        if len(spellings) < 2:
            continue
        agg = sub.groupby("unified_brand_name").agg(
            row_count=("row_count", "sum"), total_value=("total_customs_value_usd", "sum")
        )
        groups.append({"spellings": spellings, "row_counts": agg["row_count"].to_dict(), "values": agg["total_value"].to_dict()})
    return groups


class NearDuplicateVerification(BaseModel):
    same_entity: bool
    canonical_spelling: str
    confidence: Literal["high", "medium", "low"]


class BillingQuotaError(Exception):
    pass


def _is_billing_or_quota_error(exc: Exception) -> bool:
    from google.genai import errors as genai_errors
    if isinstance(exc, genai_errors.APIError):
        message = (exc.message or "").lower()
        return exc.code == 403 or (exc.code == 429 and any(p in message for p in _BILLING_PHRASES))
    msg = str(exc).lower()
    return ("429" in msg or "resource_exhausted" in msg) and any(p in msg for p in _BILLING_PHRASES)


def _gemini_with_retry(fn, label):
    last_error = None
    for attempt in range(1, GEMINI_MAX_RETRIES + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            if _is_billing_or_quota_error(exc):
                raise BillingQuotaError(str(exc)) from exc
            last_error = exc
            if attempt < GEMINI_MAX_RETRIES:
                delay = GEMINI_RETRY_BASE_DELAY * (2 ** (attempt - 1))
                print(f"    [{label}] retry {attempt}/{GEMINI_MAX_RETRIES} after error: {exc} (sleeping {delay:.0f}s)")
                time.sleep(delay)
    print(f"    [{label}] giving up after {GEMINI_MAX_RETRIES} attempts: {last_error}")
    return None


def verify_group(client, spellings: list) -> dict:
    from google.genai import types
    spellings_list = ", ".join(f'"{s}"' for s in spellings)

    def do_research():
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=NEAR_DUPLICATE_RESEARCH_PROMPT.format(spellings_list=spellings_list),
            config=types.GenerateContentConfig(tools=[types.Tool(google_search=types.GoogleSearch())]),
        )
        return response.text or ""

    research_text = _gemini_with_retry(do_research, "research") or ""

    def do_structure():
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=NEAR_DUPLICATE_STRUCTURE_PROMPT.format(spellings_list=spellings_list, research_text=research_text or "(no research available)"),
            config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=NearDuplicateVerification, temperature=0.1),
        )
        parsed = response.parsed or NearDuplicateVerification.model_validate(json.loads(response.text))
        return parsed.model_dump()

    return _gemini_with_retry(do_structure, "structure") or {"same_entity": False, "canonical_spelling": spellings[0], "confidence": "low"}


def consolidate_near_duplicates(lookup_path: str, extract_path: str = None) -> dict:
    """Merges confirmed spelling variants in place. Returns {old spelling: canonical}."""
    from dotenv import load_dotenv
    from google import genai
    from google.genai import types

    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit("GEMINI_API_KEY not found in environment/.env")
    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=GEMINI_TIMEOUT_MS))

    lookup = pd.read_csv(lookup_path, encoding="utf-8-sig", keep_default_na=False)
    for col in ("unified_brand_name", "row_count", "total_customs_value_usd"):
        if col not in lookup.columns:
            sys.exit(f"{lookup_path} is missing required column: {col}")

    groups = detect_candidate_groups(lookup)
    print(f"Detected {len(groups)} near-duplicate candidate group(s).")

    rename_map, review = {}, []
    for i, g in enumerate(groups, start=1):
        if is_case_only_duplicate(g["spellings"]):
            canonical = max(g["values"], key=g["values"].get)  # the spelling carrying more value
            print(f"[{i}/{len(groups)}] {g['spellings']} -> case-only, merged as {canonical!r}")
            rename_map.update(dict.fromkeys(g["spellings"], canonical))
            continue
        print(f"[{i}/{len(groups)}] verifying {g['spellings']} ...")
        try:
            result = verify_group(client, g["spellings"])
        except BillingQuotaError as exc:
            print(f"\nSTOPPED by billing/quota error: {exc}\nApplying whatever was verified so far.")
            break
        if result["same_entity"] and result["confidence"] in ("high", "medium"):
            corrected = "" if result["canonical_spelling"] in g["spellings"] else " (model-corrected)"
            print(f"    -> confirmed, canonical spelling{corrected}: {result['canonical_spelling']!r}")
            rename_map.update(dict.fromkeys(g["spellings"], result["canonical_spelling"]))
        else:
            print(f"    -> NOT confirmed (same_entity={result['same_entity']}, confidence={result['confidence']}), left separate")
            review.append(g["spellings"])
        time.sleep(0.5)

    if review:
        print(f"\n{len(review)} group(s) left separate for manual review: {review}")
    if not rename_map:
        print("No confirmed merges -- nothing written.")
        return rename_map

    lookup["unified_brand_name"] = lookup["unified_brand_name"].map(lambda n: rename_map.get(n, n))
    sum_cols = {c: "sum" for c in ("row_count", "total_customs_value_usd")}
    first_cols = {c: "first" for c in lookup.columns if c not in sum_cols and c != "unified_brand_name"}
    lookup = lookup.groupby("unified_brand_name", as_index=False).agg({**sum_cols, **first_cols})
    lookup.to_csv(lookup_path, index=False, encoding="utf-8-sig")
    print(f"{len(rename_map)} spelling(s) remapped -> {lookup_path}")

    if extract_path:
        extract = pd.read_csv(extract_path, encoding="utf-8-sig", keep_default_na=False)
        if "unified_brand_name" in extract.columns:
            extract["unified_brand_name"] = extract["unified_brand_name"].map(lambda n: rename_map.get(n, n))
            extract.to_csv(extract_path, index=False, encoding="utf-8-sig")
            print(f"Updated -> {extract_path}")
    return rename_map
