"""
One entry point for the whole pipeline. Run from the repository root:

  python -m brand_agent.cli prepare     --input data/demo/customs_demo.csv --output output/extract.csv
  python -m brand_agent.cli scan-multi-brand --input output/extract.csv
  python -m brand_agent.cli smoke-test  --n 5
  python -m brand_agent.cli run         --input output/extract.csv --output output/brands.csv
  python -m brand_agent.cli followup    --final output/brands.csv --extract output/extract.csv
  python -m brand_agent.cli dedupe      --lookup output/brands.csv
  python -m brand_agent.cli find-suspects --final output/brands.csv --extract output/extract.csv
  python -m brand_agent.cli verify      --suspects output/suspects.csv --final output/brands.csv
  python -m brand_agent.cli reverify    --final output/brands.csv --extract output/extract.csv
  python -m brand_agent.cli agro        --final output/brands.csv --extract output/extract.csv

Commands that call an API say so in their help text. The evals live in evals/.
"""

import argparse
import os

import pandas as pd

from . import agent, agro, batch, preprocess, qa
from .agent import MODEL

BENCHMARK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "benchmark_brands.csv")


def _read(path):
    return pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)


def _ensure_parent(path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def main():
    parser = argparse.ArgumentParser(prog="brand_agent", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=MODEL, choices=sorted(agent.PRICING), help=f"Claude model (default {MODEL}).")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="Raw customs rows -> unique-brand extract with attributed_brand (no API).")
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="output/extract.csv")

    p = sub.add_parser("scan-multi-brand", help="List multi-brand candidates in a brand column (no API).")
    p.add_argument("--input", required=True)
    p.add_argument("--column", default="raw_brand_name")

    p = sub.add_parser("smoke-test", help="Batch API path on the first N benchmark brands (Claude API).")
    p.add_argument("--n", type=int, default=5)

    p = sub.add_parser("run", help="Full run via the Batch API, resumable (Claude API).")
    p.add_argument("--input", required=True, help="Output of `prepare`.")
    p.add_argument("--output", default="output/brands.csv")

    p = sub.add_parser("followup", help="Re-research low-confidence / unknown-country rows in place (Claude API).")
    p.add_argument("--final", required=True)
    p.add_argument("--extract", required=True)

    p = sub.add_parser("dedupe", help="Merge verified spelling variants of unified_brand_name (Gemini API).")
    p.add_argument("--lookup", required=True)
    p.add_argument("--extract", default=None)

    p = sub.add_parser("find-suspects", help="Flag likely wrong-company parents (no API).")
    p.add_argument("--final", required=True)
    p.add_argument("--extract", required=True)
    p.add_argument("--output", default="output/suspects.csv")

    p = sub.add_parser("verify", help="Re-research suspects with the manufacturer hint, apply with backup (Claude API).")
    p.add_argument("--suspects", required=True)
    p.add_argument("--final", required=True)

    p = sub.add_parser("reverify", help="KEEP/CHANGE verdict on every answer, report only (Claude API).")
    p.add_argument("--final", required=True)
    p.add_argument("--extract", required=True)
    p.add_argument("--report", default="output/reverify_report.csv")

    p = sub.add_parser("agro", help="Agro-specificity per (brand, category), resumable (Claude API).")
    p.add_argument("--final", required=True)
    p.add_argument("--extract", required=True)
    p.add_argument("--output", default="output/agro_specificity.csv")
    p.add_argument("--concurrency", type=int, default=20)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--fresh", action="store_true")

    args = parser.parse_args()
    for attr in ("output", "report"):
        if getattr(args, attr, None):
            _ensure_parent(getattr(args, attr))

    if args.command == "prepare":
        preprocess.prepare(args.input, args.output)
    elif args.command == "scan-multi-brand":
        result = preprocess.scan_multi_brand(_read(args.input)[args.column].unique())
        with pd.option_context("display.max_colwidth", 80, "display.width", 200):
            for category in ("confident_genuine", "borderline", "confident_false_positive"):
                subset = result[result["category"] == category]
                print(f"\n=== {category} ({len(subset)}) ===")
                if len(subset):
                    print(subset[["raw_value", "parts", "note"]].to_string(index=False))
    elif args.command == "find-suspects":
        suspects = qa.find_suspects(_read(args.final), preprocess.load_examples(args.extract))
        suspects.to_csv(args.output, index=False, encoding="utf-8-sig")
        print(f"Saved -> {args.output}")
    elif args.command == "dedupe":
        preprocess.consolidate_near_duplicates(args.lookup, args.extract)
    else:
        client = agent.make_client(max_retries=6 if args.command == "agro" else 8)
        if args.command == "smoke-test":
            bench = pd.read_csv(BENCHMARK_PATH).dropna(subset=["verified_unified_name"])
            batch.run_smoke_test(client, [str(b).strip() for b in bench["verified_unified_name"].head(args.n)], args.model)
        elif args.command == "run":
            batch.run_production(client, args.input, args.output, args.model)
        elif args.command == "followup":
            qa.followup(client, args.final, preprocess.load_examples(args.extract), args.model)
        elif args.command == "verify":
            qa.verify_suspects(client, args.suspects, args.final, args.model)
        elif args.command == "reverify":
            qa.reverify_all(client, args.final, preprocess.load_examples(args.extract), args.report, args.model)
        elif args.command == "agro":
            agro.classify_agro(client, _read(args.final), preprocess.load_examples(args.extract), args.output,
                               args.concurrency, args.limit, args.fresh, args.model)


if __name__ == "__main__":
    main()
