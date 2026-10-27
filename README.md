# Brand Ownership Agent

An LLM agent that takes messy brand names from customs declarations and works out, for each one,
the unified brand, its **ultimate parent company** and the **parent's country**. Built on the
Anthropic SDK with the server-side web search tool, structured outputs, the Message Batches API
and prompt caching, and developed against evals: a hand-verified benchmark, a model comparison on
cost and accuracy, a self-consistency check and a stratified blind review.

The production run resolved ~25,000 unique brand strings.

> **Status.** This repository is a cleaned-up restructuring of the scripts used in production
> (October 2026). The refactored code passes an offline end-to-end test against a mocked API
> client, and the saved eval results rescore to exactly the original numbers, but it has **not yet
> been run against the live Anthropic API** in this form. Run `python -m brand_agent.cli smoke-test
> --n 3` first; if a request format turns out to be off, that is where it will show.

## The problem

Customs data names brands the way declarants type them: `JOHN DEERE`, `John Deere`,
`DEERE & COMPANY`, several brands jammed into one field, truncated names, distributors in place
of manufacturers. To aggregate imports by country of capital, each string has to be resolved to
a real brand and traced to the top of its ownership chain (New Holland → CNH Industrial → IT,
Massey Ferguson → AGCO → US), with an honest confidence level.

## How it works

```
raw customs rows
  │ prepare        one row per brand string, value totals, 3 readable description snippets;
  │                rule-based detection of multi-brand fields → MIXED_MULTI_BRAND (no API)
  ▼
  │ run            per brand, two calls (Batch API, 50% off, 1-hour prompt cache):
  │                  research  — Claude + web_search, free-text findings with real sources
  │                  structure — findings → JSON schema (unified brand, parent, country, confidence)
  │                tier 2: low-confidence or truncated-looking names are re-researched with the
  │                customs description snippets as context
  │                checkpoint after every stage: an interrupted run resumes without paying twice
  ▼
  │ followup       re-research low confidence / unknown country
  │ dedupe         merge verified spelling variants (Gemini + Google Search grounding)
  │ find-suspects  flag a known error signature: the customs text names the manufacturer, but
  │                the answer is an unrelated similarly-spelled company (no API)
  │ verify         re-research suspects with the manufacturer as a hint
  │ reverify       KEEP / CHANGE verdict on every answer, with collision detection for brand
  │                strings that merge several unrelated companies (report only)
  ▼
  │ agro           closed-book classification: is this brand's product really agricultural
                   in its customs category?
```

Two design choices carry most of the quality:

- **Research and structuring are separate calls.** Forcing a JSON schema on the call that has the
  search tool lets the model skip searching and answer from memory, and its sources then can't be
  trusted. Sources are taken from the actual `web_search_tool_result` blocks, never from model text.
- **Prompt rules come from eval failures.** The ultimate-parent rule ("Kverneland Group is real
  but not the answer — Kubota owns it") and the self-reference rule (CLAAS → CLAAS rather than an
  empty field or an invented holding) were added after the benchmark and the model comparison
  showed exactly those errors.

## Repository layout

```
brand_agent/
  prompts.py      every prompt and output schema, with worked examples of real failure modes
  agent.py        the agent (research → structure → tier 2), model config, cost accounting,
                  name and country normalisation
  batch.py        Batch API runs with checkpoints; one research-then-structure helper for every pass
  preprocess.py   extract, multi-brand detection, near-duplicate merging
  qa.py           suspects, verification, full re-verification, follow-up
  agro.py         agro-specificity classification
  cli.py          one entry point for all of the above
evals/
  benchmark.py        hand-verified benchmark, live or rescored offline
  compare_models.py   same agent on several models: cost, confidence, agreement
  affiliation.py      self-consistency and stratified blind review (earlier Gemini classifier)
data/
  benchmark_brands.csv              77 hand-verified brands
  evals/                            saved model answers, so the evals rescore without an API key
  sample/brand_ownership_sample.csv 200 brands from the production output
  demo/customs_demo.csv             synthetic customs rows to run the pipeline end to end
```

## Evals

Every number below can be reproduced offline from `data/evals/`.

### 1. Hand-verified benchmark

77 brands with a manually verified unified name, ultimate parent and parent country
(`data/benchmark_brands.csv`). Parent company is matched after legal-suffix normalisation with
substring tolerance, country as ISO alpha-2. High-confidence mismatches are listed separately: a
model that is sure and wrong is the error that slips past review.

```
python -m evals.benchmark --rescore data/evals/benchmark_results_2026-08.csv   # offline
python -m evals.benchmark                                                       # live run
```

First run (August 2026, before the prompt revisions above):

| Metric | Result |
|---|---|
| Parent country match | 66 / 77 (86%) |
| Parent company match | 66 / 77 (86%) |
| Both fields match | 58 / 77 (75%) |
| Mismatches marked `high` confidence | 12 of 19 |

Most errors came with high confidence, so confidence alone could not route rows to human review.
That drove the ownership rules and the separate verification passes. The revised prompts have not
been re-scored yet; `python -m evals.benchmark` does that for a few dollars (tokens plus web
search fees).

### 2. Model comparison

The same agent on 50 brands (6 well-known brands with an earlier Gemini answer as a baseline,
5 new high-value brands, 39 sampled across the customs-value range), with Claude Haiku 4.5 and
Claude Sonnet 4.6.

```
python -m evals.compare_models --report data/evals/model_comparison_claude-haiku-4-5.csv \
                                        data/evals/model_comparison_claude-sonnet-4-6.csv
```

| | Haiku 4.5 | Sonnet 4.6 |
|---|---|---|
| Token cost per 1,000 brands (web search fees extra) | $43.55 | $122.64 |
| `high` confidence answers | 38 / 50 | 39 / 50 |
| Needed tier 2 | 31 / 50 | 32 / 50 |
| Same parent country as the other model | 44 / 50 | |

Haiku runs at about a third of the cost. Of the six country disagreements, three were an empty
parent field (twice from Haiku, once from Sonnet) rather than a wrong company, which is a prompt
problem, not a model one — and it became the self-reference rule. Haiku went to production, and
the Batch API halves its token price again.

### 3. Self-consistency and stratified blind review

From the first stage of the project, a Gemini-based classifier deciding whether a foreign supplier
is US-affiliated (`evals/affiliation.py`). The company data is not published, so these run on your
own data; the numbers are from the original run.

- **Self-consistency**: the same 400 unique companies classified twice — 396 / 400 identical
  verdicts. Confidence was `high` on every row in both runs, so it carried no signal and was
  dropped as a stratification axis.
- **Blind review sample** of 80, in priority order: every run-to-run disagreement (4), every row
  with several merged entities detected (6), the rare True / unclear verdicts (14), then a seeded
  random fill of False (56). The reviewer worked from a copy without the model's answers.
- **Result**: the model matched the human label on 79 / 80. The later production run matched on
  76 / 80 and found 13 of 16 true US affiliations — a recall gap that only showed up because the
  rare class was oversampled.

## Running it

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # ANTHROPIC_API_KEY; GEMINI_API_KEY only for `dedupe` and affiliation

# no API key needed
python -m brand_agent.cli prepare --input data/demo/customs_demo.csv --output output/extract.csv
python -m brand_agent.cli scan-multi-brand --input output/extract.csv

# Claude API: the demo is 23 brands, about a dollar including web search fees
python -m brand_agent.cli smoke-test --n 5
python -m brand_agent.cli run --input output/extract.csv --output output/brands.csv
python -m brand_agent.cli find-suspects --final output/brands.csv --extract output/extract.csv
python -m brand_agent.cli agro --final output/brands.csv --extract output/extract.csv
```

`python -m brand_agent.cli --help` lists every command and says which ones call an API.

## Data

The real customs declaration data is proprietary and is not in this repository. What is here:

- `benchmark_brands.csv` and `data/evals/`: brand names, parent companies, countries and the
  models' answers with their web sources — public company information.
- `sample/brand_ownership_sample.csv`: 200 randomly chosen brands from the production output,
  with no customs values or row counts. Brand strings with Cyrillic, digits or Ukrainian parents
  were excluded so that no sole proprietor appears.
- `demo/customs_demo.csv`: **synthetic** rows. Brand names come from the benchmark; descriptions,
  models and values are generated. It includes spelling variants, multi-brand fields, false
  positives ("A&I"), a truncated name and empty brands, to exercise each step.

## License

MIT, see [LICENSE](LICENSE).
