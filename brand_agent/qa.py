"""
Quality passes over a finished run. Each writes a backup before touching the results file, and
the full re-verification only writes a report for review.

  find_suspects    no API. Flags the "wrong-company name confusion" error found by hand in
                   GSY, MEGAN, LAMONS and TROUW NUTRITION: the agent matches an obscure brand to
                   a bigger, similarly spelled but unrelated company, although the customs text
                   itself names the real manufacturer (whose own name contains the brand).
  verify_suspects  re-researches medium/low-confidence suspects with the named manufacturer as a
                   hint -- asking whether that manufacturer is plausibly the trademark owner,
                   without showing the current (possibly wrong) answer. Can confirm as well as
                   correct.
  reverify_all     every researched brand: the existing answer + up to 3 customs snippets ->
                   an explicit KEEP / CHANGE verdict, plus collision_detected for brand strings
                   that merge several unrelated companies (forced to an honest low-confidence
                   "unknown" rather than a confident wrong answer). Report only.
  followup         re-researches confidence=low (and medium/high with an unknown country) rows
                   with the goods-description snippets as context, in place.
"""

import os
import re

import anthropic
import pandas as pd

from .agent import MODEL, normalize_company_name, ownership_fields, research_message, structure_message
from .batch import Checkpoint, research_then_structure, write_telemetry, apply_updates
from .preprocess import examples_for
from .prompts import (
    RESEARCH_USER_MESSAGE_MANUFACTURER_HINT, RESEARCH_USER_MESSAGE_REVERIFY, STRUCTURE_USER_MESSAGE_REVERIFY,
    REVERIFY_JSON_SCHEMA,
)

# Manufacturer name in Ukrainian customs text, ended by whichever field label comes next.
MFR_RE = re.compile(
    r"ВИРОБНИК\s+(.+?)(?=\s+(?:ТОРГОВЕЛЬНА\s+МАРКА|КРАЇНА\s+ВИРОБНИЦТВА|АРТИКУЛ|МОДЕЛ[ЬІ]|КОД\b)|$)"
)

STOPWORDS = {
    "AND", "OR", "THE", "OF", "FOR", "DE", "DA", "DO", "DOS", "DAS", "LA", "LE", "LES",
    "CO", "LTD", "GROUP", "INC", "CORP", "CORPORATION", "COMPANY", "GMBH", "AG", "SA",
    "SE", "NV", "BV", "SRL", "SPA", "PLC", "LLC", "KG", "TRADING", "TRADE", "IMPORT",
    "EXPORT", "INTERNATIONAL", "HOLDING", "HOLDINGS", "INDUSTRIAL", "INDUSTRIES",
    "SP", "Z", "O", "CYWILNA", "SPOLKA",
}


def _tokens(s: str) -> set:
    return {t for t in normalize_company_name(s).split() if len(t) >= 3 and t not in STOPWORDS}


def _researched(final: pd.DataFrame) -> pd.DataFrame:
    return final[final["confidence"] != "not_attributed"].reset_index(drop=True)


def _backup(path: str, suffix: str) -> str:
    backup = path.replace(".csv", f"_pre_{suffix}_backup.csv")
    if not os.path.exists(backup):
        pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False).to_csv(backup, index=False, encoding="utf-8-sig")
        print(f"Backed up -> {backup}")
    return backup


def find_suspects(final: pd.DataFrame, examples: dict) -> pd.DataFrame:
    """Signature (shared by all 4 confirmed cases):
      1. the customs text names an explicit "ВИРОБНИК <manufacturer>";
      2. the normalised brand is contained in the normalised manufacturer name (so the brand is
         very likely self-owned);
      3. the researched parent shares no token with either the manufacturer or the brand.
    Contract manufacturing under an unrelated name ("Whiskas" made by "Mars Petcare Poland") does
    not meet condition 2 and is not flagged."""
    suspects = []
    for _, row in _researched(final).iterrows():
        raw = row["raw_brand_name"].strip()
        raw_norm = normalize_company_name(raw)
        if not raw_norm:
            continue
        for text in examples_for(examples, row):
            m = MFR_RE.search(text)
            if not m:
                continue
            mfr_name = m.group(1).strip()
            if raw_norm not in normalize_company_name(mfr_name):
                continue
            parent_toks = _tokens(row["parent_company"])
            if parent_toks and not (parent_toks & _tokens(mfr_name)) and not (parent_toks & _tokens(raw)):
                suspects.append({**row[["raw_brand_name", "attributed_brand", "unified_brand_name", "parent_company",
                                        "parent_country", "confidence", "total_customs_value_usd"]].to_dict(),
                                 "manufacturer_in_text": mfr_name, "raw_text": text})
            break  # one example is enough signal per brand

    df = pd.DataFrame(suspects)
    if len(df):
        df = df.sort_values("total_customs_value_usd", ascending=False, key=lambda s: pd.to_numeric(s, errors="coerce"))
    print(f"{len(df)} brand(s) match the wrong-company-confusion signature.")
    return df


def verify_suspects(client: anthropic.Anthropic, suspects_path: str, final_path: str, model: str = MODEL) -> pd.DataFrame:
    suspects = pd.read_csv(suspects_path, encoding="utf-8-sig", keep_default_na=False)
    target = suspects[suspects["confidence"].isin(["medium", "low"])].reset_index(drop=True)
    brands = list(target["raw_brand_name"])
    print(f"Verification target: {len(target)} brands.\n")

    usage_log = []
    parsed = research_then_structure(
        client, "verify",
        [RESEARCH_USER_MESSAGE_MANUFACTURER_HINT.format(attributed_brand=b, manufacturer_name=m)
         for b, m in zip(brands, target["manufacturer_in_text"])],
        lambda i, text: structure_message(brands[i], text),
        Checkpoint(final_path.replace(".csv", "_verify_checkpoint.json")), model=model, usage_log=usage_log,
    )
    updates = {b: ownership_fields(p, b, fallback=target.loc[i].to_dict()) for i, (b, p) in enumerate(zip(brands, parsed)) if p}

    changes = pd.DataFrame([
        {"raw_brand_name": b, "old_parent_company": target.loc[i, "parent_company"], "old_parent_country": target.loc[i, "parent_country"],
         "new_parent_company": updates[b]["parent_company"], "new_parent_country": updates[b]["parent_country"],
         "new_confidence": updates[b]["confidence"]}
        for i, b in enumerate(brands)
        if b in updates and normalize_company_name(target.loc[i, "parent_company"]) != normalize_company_name(updates[b]["parent_company"])
    ])

    _backup(final_path, "verification")
    final = pd.read_csv(final_path, encoding="utf-8-sig", keep_default_na=False)
    apply_updates(final, updates).to_csv(final_path, index=False, encoding="utf-8-sig")
    changes.to_csv(final_path.replace(".csv", "_verification_changes.csv"), index=False, encoding="utf-8-sig")

    cost = write_telemetry(usage_log, final_path.replace(".csv", "_verification_telemetry.csv"), model)
    print(f"\n{len(updates)}/{len(target)} answered: {len(changes)} CHANGED, {len(updates) - len(changes)} CONFIRMED. "
          f"Token cost: ${cost:.2f}")
    for _, c in changes.iterrows():
        print(f"  {c['raw_brand_name']!r}: {c['old_parent_company']!r}/{c['old_parent_country']} -> "
              f"{c['new_parent_company']!r}/{c['new_parent_country']} [{c['new_confidence']}]")
    return changes


def reverify_all(client: anthropic.Anthropic, final_path: str, examples: dict, report_path: str, model: str = MODEL) -> pd.DataFrame:
    target = _researched(pd.read_csv(final_path, encoding="utf-8-sig", keep_default_na=False))
    print(f"Full re-verification target: {len(target)} brands.\n")

    def examples_block(row):
        texts = examples_for(examples, row)
        if not texts:
            return "(no example goods_description text available for this brand)"
        return "\n\n".join(f"Example {i + 1}: {t[:800]}" for i, t in enumerate(texts[:3]))

    def answer(row):
        return {k: row[k] for k in ("raw_brand_name", "unified_brand_name", "parent_company", "parent_country")}

    usage_log = []
    parsed = research_then_structure(
        client, "reverify",
        [RESEARCH_USER_MESSAGE_REVERIFY.format(**answer(row), examples_block=examples_block(row)) for _, row in target.iterrows()],
        lambda i, text: STRUCTURE_USER_MESSAGE_REVERIFY.format(**answer(target.loc[i]), research_text=text or "(no research available)"),
        Checkpoint(report_path.replace(".csv", "_checkpoint.json")), schema=REVERIFY_JSON_SCHEMA, structure_max_tokens=768,
        model=model, usage_log=usage_log,
    )

    rows = []
    for i, p in enumerate(parsed):
        old = target.loc[i]
        new = {k: old[k] for k in ("unified_brand_name", "parent_company", "parent_country", "confidence")}
        verdict = p.get("verdict", "KEEP") if p else "NO_ANSWER"
        collision = bool(p.get("collision_detected", False))
        if collision:
            new.update(parent_company=old["raw_brand_name"], parent_country="unknown", confidence="low")
        elif verdict == "CHANGE":
            new = ownership_fields(p, old["raw_brand_name"], fallback=new)
        rows.append({
            "raw_brand_name": old["raw_brand_name"], "verdict": verdict, "collision_detected": collision,
            "reasoning": p.get("reasoning", ""),
            **{f"old_{k}": old[k] for k in new}, **{f"new_{k}": v for k, v in new.items()},
            "total_customs_value_usd": old["total_customs_value_usd"],
        })

    report = pd.DataFrame(rows).sort_values("total_customs_value_usd", ascending=False, key=lambda s: pd.to_numeric(s, errors="coerce"))
    report.to_csv(report_path, index=False, encoding="utf-8-sig")
    cost = write_telemetry(usage_log, report_path.replace(".csv", "_telemetry.csv"), model)
    counts = report["verdict"].value_counts().to_dict()
    print(f"\nREPORT (not applied): {counts.get('CHANGE', 0)} CHANGE, {counts.get('KEEP', 0)} KEEP, "
          f"{counts.get('NO_ANSWER', 0)} no answer; {int(report['collision_detected'].sum())} collision(s). "
          f"Token cost: ${cost:.2f}\nSaved -> {report_path}")
    return report


def followup(client: anthropic.Anthropic, final_path: str, examples: dict, model: str = MODEL) -> pd.DataFrame:
    final = pd.read_csv(final_path, encoding="utf-8-sig", keep_default_na=False)
    mask = (final["confidence"] == "low") | ((final["parent_country"] == "unknown") & final["confidence"].isin(["medium", "high"]))
    target = final[mask].reset_index(drop=True)
    brands = list(target["raw_brand_name"])
    print(f"Follow-up target: {len(target)} brands (confidence=low, or country unknown despite medium/high).\n")

    usage_log = []
    parsed = research_then_structure(
        client, "followup", [research_message(b, examples_for(examples, target.loc[i])) for i, b in enumerate(brands)],
        lambda i, text: structure_message(brands[i], text),
        Checkpoint(final_path.replace(".csv", "_followup_checkpoint.json")), model=model, usage_log=usage_log,
    )
    updates = {b: ownership_fields(p, b, fallback=target.loc[i].to_dict()) for i, (b, p) in enumerate(zip(brands, parsed)) if p}

    country_resolved = sum(1 for i, b in enumerate(brands) if b in updates and target.loc[i, "parent_country"] == "unknown"
                           and updates[b]["parent_country"] != "unknown")
    confidence_up = sum(1 for i, b in enumerate(brands) if b in updates and target.loc[i, "confidence"] == "low"
                        and updates[b]["confidence"] != "low")

    _backup(final_path, "followup")
    final = apply_updates(final, updates)
    final.to_csv(final_path, index=False, encoding="utf-8-sig")
    cost = write_telemetry(usage_log, final_path.replace(".csv", "_followup_telemetry.csv"), model)
    print(f"\n{len(updates)}/{len(target)} answered; country resolved: {country_resolved}; low -> higher confidence: {confidence_up}")
    print(f"confidence=low now: {int((final['confidence'] == 'low').sum())}. Token cost: ${cost:.2f}")
    return final
