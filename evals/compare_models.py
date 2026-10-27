"""
Model comparison: the same agent (prompts, schema, two-tier logic) on the same brands with
different Claude models, compared on token cost, confidence, tier-2 escalation and agreement.

  python -m evals.compare_models --report data/evals/model_comparison_claude-haiku-4-5.csv \
                                          data/evals/model_comparison_claude-sonnet-4-6.csv   # no API
  python -m evals.compare_models --brands data/benchmark_brands.csv --column verified_unified_name \
                                 --n 20 --models claude-haiku-4-5,claude-sonnet-4-6          # Claude API

The saved results come from the August 2026 run on 50 brands: 6 well-known brands with an
earlier Gemini answer as a baseline, 5 new high-value brands and 39 brands sampled across the
customs-value distribution. They were produced with an earlier version of the prompts, before
the ultimate-parent and self-reference rules were added.
"""

import argparse
import concurrent.futures
import json
import os
from itertools import combinations

import pandas as pd

from brand_agent.agent import PRICING, USAGE_KEYS, estimate_cost, make_client, normalize_country, process_brand, total_usage


def run(brands: list, model: str, concurrency: int) -> pd.DataFrame:
    client = make_client()
    print(f"=== {model} on {len(brands)} brands ===")

    def worker(brand):
        r = process_brand(client, brand, examples=[], model=model)
        print(f"  [{model}] {brand!r} -> {r['parent_company']!r}/{r['parent_country']} [{r['confidence']}]")
        return {"attributed_brand": brand, **{k: r[k] for k in ("unified_brand_name", "parent_company", "parent_country", "confidence")},
                "sources": json.dumps(r["sources"], ensure_ascii=False), "tier_used": r["tier_used"],
                "n_search_queries": r["n_search_queries"], **r["usage"]}

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        return pd.DataFrame(list(executor.map(worker, brands)))


def model_name(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0].replace("model_comparison_", "")


def report(results: dict) -> None:
    for model, df in results.items():
        cost = estimate_cost(total_usage(df[list(USAGE_KEYS)].to_dict("records")), model)
        print(f"\n{model} ({len(df)} brands)")
        print(f"  token cost: ${cost:.2f} total, ${cost / len(df) * 1000:.2f} per 1,000 brands (web search fees extra)")
        print(f"  confidence: {df['confidence'].value_counts().to_dict()}")
        print(f"  tier 2 needed: {int((df['tier_used'].astype(int) == 2).sum())}/{len(df)}")

    for a, b in combinations(results, 2):
        m = results[a].merge(results[b], on="attributed_brand", suffixes=("_a", "_b"))
        agree = sum(normalize_country(x) == normalize_country(y) for x, y in zip(m["parent_country_a"], m["parent_country_b"]))
        print(f"\nSame parent country, {a} vs {b}: {agree}/{len(m)}")
        diff = m[[normalize_country(x) != normalize_country(y) for x, y in zip(m["parent_country_a"], m["parent_country_b"])]]
        if len(diff):
            with pd.option_context("display.width", 200, "display.max_colwidth", 40):
                cols = {"attributed_brand": "brand", "parent_company_a": f"parent ({a})", "parent_country_a": f"country ({a})",
                        "parent_company_b": f"parent ({b})", "parent_country_b": f"country ({b})"}
                print(diff[list(cols)].rename(columns=cols).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", nargs="+", help="Saved per-model result CSVs (model_comparison_<model>.csv).")
    parser.add_argument("--brands", help="CSV with the brands to run.")
    parser.add_argument("--column", default="verified_unified_name")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--models", default="claude-haiku-4-5,claude-sonnet-4-6")
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--output-dir", default="output")
    args = parser.parse_args()

    if args.report:
        results = {model_name(p): pd.read_csv(p, encoding="utf-8-sig", keep_default_na=False) for p in args.report}
    else:
        if not args.brands:
            parser.error("pass --report <csv...> or --brands <csv>")
        brands = pd.read_csv(args.brands, encoding="utf-8-sig")[args.column].dropna().astype(str).str.strip().head(args.n).tolist()
        os.makedirs(args.output_dir, exist_ok=True)
        results = {}
        for model in args.models.split(","):
            if model not in PRICING:
                parser.error(f"unknown model {model!r}, choose from {sorted(PRICING)}")
            results[model] = run(brands, model, args.concurrency)
            results[model].to_csv(os.path.join(args.output_dir, f"model_comparison_{model}.csv"), index=False, encoding="utf-8-sig")
    report(results)


if __name__ == "__main__":
    main()
