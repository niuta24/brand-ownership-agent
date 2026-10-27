"""
Hand-verified benchmark: data/benchmark_brands.csv (77 brands with a verified unified name,
ultimate parent and parent country).

  python -m evals.benchmark                      # run the agent on every brand (Claude API)
  python -m evals.benchmark --rescore data/evals/benchmark_results_2026-08.csv   # no API

Scoring: parent company matches after legal-suffix normalisation with substring tolerance
("KRONE" ~ "Bernard Krone Holding SE & Co. KG"); parent country matches as ISO alpha-2 with
"-" / "н/д" read as unknown. High-confidence mismatches are reported separately: an answer the
model is sure of and gets wrong is the error that slips past review.
"""

import argparse
import concurrent.futures
import os
import time

import pandas as pd

from brand_agent.agent import MODEL, PRICING, company_names_match, estimate_cost, add_usage, empty_usage, make_client, normalize_country, process_brand

BENCHMARK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "benchmark_brands.csv")


def load_benchmark() -> pd.DataFrame:
    bench = pd.read_csv(BENCHMARK_PATH, encoding="utf-8")
    return bench[bench["verified_unified_name"].notna()].reset_index(drop=True)


def run(model: str, concurrency: int) -> pd.DataFrame:
    """Runs the synchronous agent (same prompts and schema as the batch path) on every brand."""
    client = make_client()
    bench = load_benchmark()
    print(f"Benchmark: {len(bench)} brands, model {model}\n")
    start = time.monotonic()

    def worker(row):
        brand = str(row["verified_unified_name"]).strip()
        r = process_brand(client, brand, examples=[], model=model)
        print(f"  {brand!r} -> parent={r['parent_company']!r} country={r['parent_country']} "
              f"confidence={r['confidence']} tier={r['tier_used']}")
        return r

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        results = list(executor.map(worker, [row for _, row in bench.iterrows()]))
    print(f"\nDone in {(time.monotonic() - start) / 60:.1f} min")

    usage = empty_usage()
    for r in results:
        usage = add_usage(usage, r["usage"])
    print(f"Token cost (synchronous, not batch): ${estimate_cost(usage, model):.4f} for {len(bench)} brands")

    return pd.DataFrame({
        "raw_brand_name": bench["raw_brand_name"],
        "input_unified_name": bench["verified_unified_name"],
        "model_unified_brand_name": [r["unified_brand_name"] for r in results],
        "model_parent_company": [r["parent_company"] for r in results],
        "model_parent_country": [r["parent_country"] for r in results],
        "model_confidence": [r["confidence"] for r in results],
        "benchmark_parent_company": bench["parent_company"],
        "benchmark_parent_country": bench["parent_country"],
        "tier_used": [r["tier_used"] for r in results],
    })


def score(results: pd.DataFrame) -> dict:
    df = results.copy()
    df["country_match"] = [normalize_country(m) == normalize_country(b)
                           for m, b in zip(df["model_parent_country"], df["benchmark_parent_country"])]
    df["parent_match"] = [company_names_match(m, b)
                          for m, b in zip(df["model_parent_company"], df["benchmark_parent_company"])]
    mismatches = df[~(df["country_match"] & df["parent_match"])]
    return {
        "n": len(df),
        "country_match": int(df["country_match"].sum()),
        "parent_match": int(df["parent_match"].sum()),
        "both_match": int((df["country_match"] & df["parent_match"]).sum()),
        "mismatches": mismatches,
        "high_confidence_mismatches": mismatches[mismatches["model_confidence"] == "high"],
    }


def report(s: dict) -> None:
    n = s["n"]
    print(f"Parent country match: {s['country_match']}/{n} ({s['country_match'] / n:.0%})")
    print(f"Parent company match: {s['parent_match']}/{n} ({s['parent_match'] / n:.0%})")
    print(f"Both fields match:    {s['both_match']}/{n} ({s['both_match'] / n:.0%})")
    cols = ["input_unified_name", "model_parent_company", "benchmark_parent_company",
            "model_parent_country", "benchmark_parent_country", "model_confidence"]
    with pd.option_context("display.max_colwidth", 40, "display.width", 220):
        print(f"\n{len(s['mismatches'])} mismatch(es), {len(s['high_confidence_mismatches'])} of them high-confidence:\n")
        print(s["mismatches"][cols].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rescore", help="Score saved results instead of calling the API.")
    parser.add_argument("--model", default=MODEL, choices=sorted(PRICING))
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--output", default="output/benchmark_results.csv")
    args = parser.parse_args()

    if args.rescore:
        results = pd.read_csv(args.rescore, encoding="utf-8-sig", keep_default_na=False)
    else:
        results = run(args.model, args.concurrency)
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        results.to_csv(args.output, index=False, encoding="utf-8-sig")
        print(f"Saved -> {args.output}\n")
    report(score(results))


if __name__ == "__main__":
    main()
