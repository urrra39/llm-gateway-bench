# llm-gateway-bench

Question in one sentence: on a fixed request workload, how much does a
semantic cache plus a cheap/expensive router actually save, and what does it
cost in answer quality against an always-use-the-expensive-model baseline?

Ceiling in own words: this benchmark cannot weigh production traffic. The
workload is constructed from PAWS pairs with fixed duplicate fractions, both
tiers have been one working model with two operating points since 2026-09-17,
and quality is one judge pair from the same model with no human verification.
Every cost number below is a function of the duplicate fractions, not a
prediction about anyone's traffic. See docs/CEILING.md.

Validity gates: all_gates_passed=true for run `full` (9/9 PASS).

## Headline table

Run `full`: full report-half runs, both duplicate fractions. Threshold 0.79
from replay tuning on the held-out half. Kappa 0.8034 with n=338
(self-consistency of one judge model; no human has verified any label).

Duplicate fractions (input parameters, recorded per workload): low is 10%
exact, 15% paraphrase, 5% trap, 70% novel; high is 30% exact, 30% paraphrase,
5% trap, 35% novel. Each fraction builds 500 requests; the reported half is
about 245 rows for low and 246 rows for high after the tune/report split.
The cache hit rate is a property of this construction.

Low duplicate fraction (report half, n≈237):

| config | cost USD | p50 ms | p95 ms | p99 ms | hit rate | false-hit rate | quality equiv |
|---|---|---|---|---|---|---|---|
| baseline | 0.9226 | 3075.5 | 11657.0 | 25987.8 | 0.0 | null | null (reference) |
| cache | 0.6139 | 2505.1 | 10620.2 | 18942.7 | 0.2899 | 0.0889 (4 rows) | 0.9308 |
| router_cascade | 0.5336 | 1802.9 | 17513.6 | 37085.7 | 0.2815 | 0.1364 (6 rows) | 0.8678 |
| router_heuristic | 0.3282 | 1772.6 | 9637.9 | 10972.3 | 0.2743 | 0.1463 (6 rows) | 0.8348 |

High duplicate fraction (report half, n=246):

| config | cost USD | p50 ms | p95 ms | p99 ms | hit rate | false-hit rate | quality equiv |
|---|---|---|---|---|---|---|---|
| baseline | 0.9187 | 3205.4 | 12331.5 | 25999.2 | 0.0 | null | null (reference) |
| cache | 0.3792 | 7.0 | 7825.5 | 31169.6 | 0.6423 | 0.0941 (8 rows) | 0.9085 |
| router_cascade | 0.2294 | 7.3 | 5751.3 | 24450.2 | 0.6463 | 0.0698 (6 rows) | 0.8619 |
| router_heuristic | 0.2212 | 10.0 | 9084.1 | 10517.9 | 0.6057 | 0.125 (10 rows) | 0.8264 |

Quality equiv is mean judge score divided by 2, where 2 is equivalent to the
baseline, 1 is partial, 0 is wrong. False hits are semantic-cache hits the
judge scored 0; they are counted separately and never folded into hit rate.
Latency percentiles are warm (first request excluded); first-request latency
is about 2.2-3.0 s per config and is reported separately in results.json.

## Duplicate-fraction sensitivity

Cost saving grows with repeats, quality cost does not disappear. Cache-only
saves 33% on low (0.9226 to 0.6139) and 59% on high (0.9187 to 0.3792).
Cache plus heuristic router saves 64% on low and 76% on high, at quality
0.8348 and 0.8264 respectively. A workload with 60% repeats would show larger
savings that mean nothing about production traffic.

## Figures

- docs/figures/cost_by_config.svg: assumed cost by config for each fraction.
- docs/figures/quality_by_config.svg: quality equiv by config for each fraction.
- docs/figures/threshold_sweep.txt: recall and false-hit rate across the tuning
  grid (source table data/runs/threshold_sweep.parquet).

## Reproduction

Requires Python 3.11, uv, and the dev gateway at http://127.0.0.1:8787/v1.
No key is needed locally; GSK_API_KEY is read when set.

```
uv sync --frozen
make all
```

`make pilot` runs a 50-request sanity path on the low fraction. `make all`
rebuilds workloads, tunes the threshold on one half, runs all four configs on
both fractions, judges, assembles results.json, and audits docs. All stages
are resumable with atomic parquet checkpointing; a kill loses at most one row.

## Hardware and runtime

Measured 2026-09-17 on a local mac (darwin, arm64) with CPU only. Embeddings
are local all-MiniLM-L6-v2 (about 90 MB, resident under 1.5 GB); no component
needs a GPU. Full run wall-clock was about 42 minutes for 8 execution runs
plus 6 judging runs at concurrency 4 (see git log 11:49 to 12:31 local).
Latency numbers above are gateway-measured per request.

## Model and price-table cards

Models verified 2026-09-17 by probing every listed model: only
deepseek-v4-flash answers. Cheap and expensive tiers share
deepseek-v4-flash with different recipes: cheap is one sentence with 512
tokens; expensive is up to three sentences with 4096 tokens and
reasoning_effort=low. Judge primary and secondary are both
deepseek-v4-flash at temperature 0 with max_tokens 1200.

Prices as_of 2026-09-17 (assumed rate, not a provider quote): deepseek-v4-flash
input 1.50 and output 6.00 per million tokens. The gateway reports usage but
no price; every dollar figure is this table applied to measured tokens. Both
tiers share the rate, so savings are token savings.

## Label provenance

Requests derive from PAWS labeled pairs (google-research-datasets/paws,
labeled_final/train, 3000-row committed sample in data/workload/paws_sample.csv).
Label 1 pairs give paraphrase should-hit rows; label 0 pairs give trap
should-not-hit rows. The similarity threshold 0.79 was tuned on one half and
reported on the other; the random-threshold control mean F1 is
0.8279916296353538 against tuned F1 0.9007633587786259.

Judge rubric is frozen in config/bench.yaml. Per-row verdicts persist under
data/runs/*/judge.parquet. Second-judge agreement is kappa 0.8034 with n=338
(observed 0.9408, expected 0.6991). Both judges are the same model, so this is
self-consistency, not inter-family agreement. No human has verified any label.
data/human_validation.csv ships the 60 highest-value rows (false hits, judge
disagreements, escalations first) with an empty human_label column; fill it and
run `python -m lgb human-agreement --csv path` to compute judge-versus-human
agreement. Unverified labels are written as unverified.

## What I would ship

Ship cache plus heuristic router at threshold 0.79 on high-duplicate traffic
only, and only with false-hit monitoring. Reasoning: on high duplicates it
saves 76% of assumed cost (0.9187 to 0.2212) at quality 0.8264 with a 12.5%
false-hit rate; on low duplicates the same stack saves 64% at quality 0.8348
with a 14.6% false-hit rate. The cascade is safer on false hits (6.9% on high)
at nearly the same cost (0.2294) and higher quality (0.8619), so ship cascade
where a wrong cached answer costs more than latency. Do not ship either router
where 0.83 quality is unacceptable; the savings do not justify the quality
cost there. Cache-only is the conservative choice: 33-59% saving at quality
0.91-0.93 with false hits under 10%.

## Limitations

- Workload is constructed; hit rate does not predict any real traffic.
- Two operating points of one model are not a market.
- One judge pair from the same model is not ground truth; no human labels exist.
- Prices are assumed; the measured quantity is tokens.
- Gateway exposes only one working model as of 2026-09-17; history with glm-5.3
  is archived and does not reproduce.
- Judge costs are not included in reported spend; total workload spend is about
  4.15 assumed dollars across all eight runs.
