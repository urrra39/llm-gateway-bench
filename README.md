# llm-gateway-bench

One-model constraint, stated first because it changes what every number below
means: only deepseek-v4-flash answers on this gateway (probed 2026-09-17; five
listed models plus glm-5.3, all others quota-blocked or channel-less), so both
tiers are recipes of that one model — short (one sentence, 512 tokens) and
long (three sentences, 4096 tokens). Every "saving" here is measured token
saving from shorter answers and cache hits, not a vendor price comparison.

Question in one sentence: on a fixed request workload, how much does a
semantic cache plus a short/long-recipe router actually save, and what does it
cost in answer quality against an always-use-the-long-recipe baseline?

## Quickstart

Three commands from clone to a served response (needs Python 3.11 and uv;
no API key — the dev gateway answers keyless):

```
git clone https://github.com/urrra39/llm-gateway-bench && cd llm-gateway-bench
uv sync --frozen --extra embeddings
.venv/bin/python -m lgb serve --port 8000
```

Then, in a second terminal:

```
curl -s localhost:8000/health
# {"ok": true, "cache_size": 0}
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "deepseek-v4-flash", "messages": [{"role": "user", "content": "Say hello in one sentence."}]}'
# {"choices": [{"message": {"content": "...", ...}}], "x_gateway": {"mode": "router", "hit": false, ...}}
```

One command reproduces the benchmark (about 45 minutes, resumable): `make all`.

Ceiling in own words: this benchmark cannot weigh production traffic. The
workload is constructed from PAWS pairs with fixed duplicate fractions, both
tiers have been one working model with two operating points since 2026-09-17,
and quality is one judge pair from the same model with no human verification.
Every cost number below is a function of the duplicate fractions, not a
prediction about anyone's traffic. See docs/CEILING.md.

Validity gates: all_gates_passed=true for run `full` (9/9 PASS).

## Findings

1. The cache saves real tokens at a measurable quality cost. Cache-plus-long
   cuts assumed cost 33% on low duplicates (0.9226 to 0.6139) and 59% on high
   (0.9187 to 0.3792), at quality equiv 0.9308 and 0.9085 against the
   long-only baseline.
2. The p99 got worse while the p50 got dramatically better, and that is the
   most operationally important result here. High-duplicate cache p99 is
   31169.6 ms against a 25999.2 ms baseline (+20%); low-duplicate cascade p99
   is 37085.7 ms against 25987.8 ms (+43%). The tail belongs to the upstream
   model, not the cache: tail rows are cache misses emitting thousands of
   tokens with model time nearly equal to total latency, while cache lookup
   overhead is tens of milliseconds. See "The tail belongs upstream" below.
3. The false-hit rate is 6.98% to 14.63% of semantic hits, and no threshold
   fixes it: on the tuning half the false-hit rate never falls below 6.45%
   even at threshold 0.995, where recall has already collapsed to 0.25.
   The semantic cache is therefore not shippable for correctness-sensitive
   traffic on this workload; see the revised shipping recommendation.
4. The heuristic router beats the cascade on cost and loses on quality, so the
   choice is a genuine trade, not a ranking: high duplicates, heuristic 0.2212
   at quality 0.8264 with 12.5% false hits versus cascade 0.2294 at 0.8619
   with 6.98% false hits.
5. This experiment measured short answers, not cheap models. A real
   cheap/expensive pair with a genuine price ratio would show larger cost
   separation and a different quality cost; this benchmark estimates neither,
   and none of the router savings transfer to a vendor-switching decision.
   What transfers: the cache mechanics, the threshold trade curve shape, and
   the tail-latency behavior.

## Headline table

Run `full`: full report-half runs, both duplicate fractions. Threshold 0.79
from replay tuning on the held-out half. Judge self-consistency (κ) 0.8034
with n=338 (same model judging twice at temperature 0; no human has verified
any label).

Duplicate fractions (input parameters, recorded per workload): low is 10%
exact, 15% paraphrase, 5% trap, 70% novel; high is 30% exact, 30% paraphrase,
5% trap, 35% novel. Each fraction builds 500 requests; the report half is 245
attempted rows for low (237–238 successful, the rest gateway errors excluded
from metrics) and 249 attempted for high (246 successful). The cache hit rate
is a property of this construction.

Machine config IDs are kept in parentheses so figures and tables map back to
`results.json` and the committed parquet; display names describe the recipe.

low duplicate fraction (report half: 245 attempted, 237-238 successful):

| config | recipe | cost USD | p50 ms | p95 ms | p99 ms | hit rate | false-hit rate | quality equiv |
|---|---|---|---|---|---|---|---|---|
| baseline | long only | 0.9226 | 3075.5 | 11657.0 | 25987.8 | 0.0 | null | null (reference) |
| cache | cache + long | 0.6139 | 2505.1 | 10620.2 | 18942.7 | 0.2899 | 0.0889 (4 rows) | 0.9308 |
| router_cascade | cache + short, escalate to long | 0.5336 | 1802.9 | 17513.6 | 37085.7 | 0.2815 | 0.1364 (6 rows) | 0.8678 |
| router_heuristic | cache + short/long by heuristic | 0.3282 | 1772.6 | 9637.9 | 10972.3 | 0.2743 | 0.1463 (6 rows) | 0.8348 |

high duplicate fraction (report half: 249 attempted, 246 successful):

| config | recipe | cost USD | p50 ms | p95 ms | p99 ms | hit rate | false-hit rate | quality equiv |
|---|---|---|---|---|---|---|---|---|
| baseline | long only | 0.9187 | 3205.4 | 12331.5 | 25999.2 | 0.0 | null | null (reference) |
| cache | cache + long | 0.3792 | 7.0 | 7825.5 | 31169.6 | 0.6423 | 0.0941 (8 rows) | 0.9085 |
| router_cascade | cache + short, escalate to long | 0.2294 | 7.3 | 5751.3 | 24450.2 | 0.6463 | 0.0698 (6 rows) | 0.8619 |
| router_heuristic | cache + short/long by heuristic | 0.2212 | 10.0 | 9084.1 | 10517.9 | 0.6057 | 0.125 (10 rows) | 0.8264 |

Quality equiv is mean judge score divided by 2, where 2 is equivalent to the
baseline, 1 is partial, 0 is wrong. False hits are semantic-cache hits the
judge scored 0; they are counted separately and never folded into hit rate.
Latency percentiles are warm (first request excluded); first-request latency
is about 2.2-3.0 s per config and is reported separately in results.json.

## The tail belongs upstream

Decomposed from the stored per-request records (`latency_ms`, `embed_ms`,
`lookup_ms`, `model_ms` in `data/runs/*/outcomes.parquet`; no re-run needed):

- Every tail row is a cache miss (or baseline direct call) emitting thousands
  of output tokens — e.g. low-duplicate cascade idx 436: 9216 tokens, 70403 ms
  total; high-duplicate cache idx 215: 8192 tokens, 55488 ms. Model time is
  nearly the whole latency.
- Cache mechanics cost milliseconds: semantic-hit p99 is 12–67 ms across all
  runs; typical miss overhead (embed plus lookup) is 10–60 ms, and miss p50 on
  high duplicates (3033 ms) matches baseline p50 (3205 ms). Two outliers show
  lock contention during full-store re-encode: one 2.4 s lookup on high/cache,
  one 16.5 s lookup on low/heuristic.
- The high-duplicate cache p99 (+20% vs baseline) is sampling noise over a
  heavy-tailed upstream distribution at a different wall-clock time: 87 misses
  versus 245 baseline calls. The cache adds no per-request cost that could
  explain seconds.
- The low-duplicate cascade p99 (+43%) is structural: an escalation pays two
  sequential model calls, short then long, with budgets doubled on empty first
  tries. That stacking is visible in the summed token counts.
- The heuristic p99 improves on both fractions because the short recipe caps
  output length and truncates the right tail.

Operational consequence: this stack halves medians but cannot cap the worst
case. Anyone with a latency SLO gets a different product than the p50
suggests — they need a deadline with fallback, not a median. The cascade
needs a tail-latency budget that skips escalation once the short call has
eaten most of it (recorded in docs/OPEN_DEFECTS.md, not implemented).
`docs/figures/latency_low.svg` and `docs/figures/latency_high.svg` show the
full distributions on a log time axis: hits clustered left, model-bound tail
right.

Measurement caveat: on escalated cascade rows `model_ms` double-counts the
short call (e.g. 80514 ms of model time inside a 70403 ms request), so the
decomposition above uses `latency_ms` for tail rows. Token and cost accounting
sum each call once and are unaffected.

## Duplicate-fraction sensitivity

Cost saving grows with repeats, quality cost does not disappear. Cache-plus-
long saves 33% on low (0.9226 to 0.6139) and 59% on high (0.9187 to 0.3792).
Cache plus heuristic routing saves 64% on low and 76% on high, at quality
0.8348 and 0.8264 respectively. A workload with 60% repeats would show larger
savings that mean nothing about production traffic.

## False hits: the trade curve and what ships

Loss function, stated plainly: a false hit serves a confidently wrong answer
to a user, while a miss only pays the full model price. The tuning objective
(F1, precision and recall weighted equally) does not know this, so threshold
0.79 is tuned for the wrong objective. Under any loss where a false hit costs
an order of magnitude more than a miss, the optimum moves up the curve below —
and the curve shows the cache cannot get there:

| threshold | hit rate (tuning half) | recall | false-hit rate |
|---|---|---|---|
| 0.79 (shipped) | 0.3556 | 1.0000 | 0.1806 |
| 0.85 | 0.3506 | 0.9915 | 0.1761 |
| 0.90 | 0.3432 | 0.9661 | 0.1799 |
| 0.95 | 0.3259 | 0.9237 | 0.1742 |
| 0.97 | 0.2988 | 0.8644 | 0.1570 |
| 0.99 | 0.1728 | 0.5339 | 0.1000 |
| 0.995 | 0.0765 | 0.2458 | 0.0645 |

No threshold reaches 2% false hits; at 0.995 the rate is still 6.45% with
recall destroyed. `docs/figures/threshold_tradeoff.svg` plots all three
curves with the chosen point marked. Revised recommendation: do not ship the
semantic cache for correctness-sensitive traffic on this workload. Ship the
exact-match cache — a verbatim repeat served from store, wrong only if the
original answer was wrong — which contributes roughly 10% of low-duplicate
hits (24 of 69) and 28% of high-duplicate hits (73 of 158) at zero added
error, plus short-recipe routing wherever one-sentence answers are
acceptable. Where approximate answers are tolerable, threshold 0.97 keeps a
0.30 hit rate at 15.7% tuning-half false hits; that is a product decision
with eyes open, not a default.

What fooled the cache, in full. PAWS traps are near-duplicates with flipped
meaning, and embeddings rank them nearest:

1. high/cache idx 296, similarity 0.9786, trap. Request: `Explain this
   statement: "The problem of testing whether a given polynomial is a
   permutation polynomial through a finite field can be resolved in
   polynomial time ."` Cached answer explains brute-force evaluation over
   the finite field ("one can test this in finite time by evaluating f at
   every element"); baseline explains a polynomial-time decision procedure
   via algebraic criteria. Same words, different claim about complexity.
2. high/cache idx 269, similarity 0.9877, trap. Request about two pumping
   stations outside city limits in Toronto/York water service. Cached answer
   reverses the direction ("some of Toronto's water service is supplied by
   York"); baseline keeps Toronto supplying York Region. Direction flipped,
   vocabulary nearly identical.
3. low/cache idx 130, similarity 0.9990, trap. Request: `"Originally from
   Canberra , Holmes attended the Australian Institute of Sport in Sydney ."`
   Cached answer asserts Holmes was "originally from Sydney" and moved to
   Canberra — the exact opposite of the request — because the stored anchor
   had the cities the other way round. One swapped proper noun at 0.9990.

## Figures

- docs/figures/cost_by_config.svg: assumed cost by config for each fraction.
- docs/figures/quality_by_config.svg: quality equiv by config for each fraction.
- docs/figures/threshold_tradeoff.svg: recall, hit rate and false-hit rate
  across the tuning grid with 0.79 marked (source data/runs/threshold_sweep.parquet).
- docs/figures/latency_low.svg and docs/figures/latency_high.svg: warm
  per-request latency distributions per config on a log axis.

## Reproduction

Requires Python 3.11, uv, and the dev gateway at http://127.0.0.1:8787/v1.
No key is needed locally; GSK_API_KEY is read when set and sent as a bearer
token (verified: keyless probes answered on 2026-09-17).

```
uv sync --frozen --extra embeddings
make all
```

`make pilot` runs a 50-request sanity path on the low fraction. `make all`
rebuilds workloads, tunes the threshold on one half, runs all four configs on
both fractions, judges, assembles results.json, and audits docs. All stages
are resumable with atomic parquet checkpointing; a kill loses at most one row.
The embeddings extra (sentence-transformers plus torch CPU) is needed for
tuning and cached runs; serve alone answers with exact-match caching without
downloaded weights.

## Hardware and runtime

Measured 2026-09-17 on a local mac (darwin, arm64) with CPU only. Embeddings
are local all-MiniLM-L6-v2 (about 90 MB, resident under 1.5 GB); no component
needs a GPU. Full run wall-clock was about 42 minutes for 8 execution runs
plus 6 judging runs at concurrency 4 (see git log 11:49 to 12:31 local).
Latency numbers above are gateway-measured per request.

## Model and price-table cards

Gateway models listed 2026-09-17T07:43Z via `GET /v1/models`:
claude-opus-4-8, claude-opus-5, deepseek-v4-flash, gpt-5.6-sol, gpt-6-astra
(glm-5.3, the 2026-09-08 cheap tier, no longer listed and without a channel).
Probe used for each, 2026-09-17: `POST /v1/chat/completions` with model,
system prompt "Answer concisely, in at most three sentences.",
user content `Explain this statement: "The cat sat on the mat."`,
`max_tokens` 4096, `temperature` 0, `reasoning_effort` low. Only
deepseek-v4-flash returned content (219 chars); the other four returned
budget-quota exhaustion and glm-5.3 no available channel. No second model was
found, so nothing was re-run and no price difference was simulated. A full
router re-run would have cost about 42 minutes wall-clock and about 4.15
assumed dollars of workload calls plus judging; it was not spent because
there is no second model to run it on.

Both tiers are deepseek-v4-flash with different recipes: short is one
sentence with 512 tokens; long is up to three sentences with 4096 tokens and
reasoning_effort=low. Judge primary and secondary are both
deepseek-v4-flash at temperature 0 with max_tokens 1200.

Prices as_of 2026-09-17 (assumed rate, not a provider quote): deepseek-v4-flash
input 1.50 and output 6.00 per million tokens. The gateway reports usage but
no price; every dollar figure is this table applied to measured tokens. Both
tiers share the rate, so savings are token savings.

Transfer note: a real cheap/expensive pair with a genuine price ratio would
show larger cost separation and a different quality cost. This benchmark
cannot estimate either and its router savings do not transfer to a
vendor-switching decision. What transfers is the cache mechanics, the shape
of the threshold trade curve, and the tail-latency behavior.

## Label provenance

Requests derive from PAWS labeled pairs (google-research-datasets/paws,
labeled_final/train, 3000-row committed sample in data/workload/paws_sample.csv).
Label 1 pairs give paraphrase should-hit rows; label 0 pairs give trap
should-not-hit rows. The similarity threshold 0.79 was tuned on one half and
reported on the other; the random-threshold control mean F1 is
0.8279916296353538 against tuned F1 0.9007633587786259.

Judge rubric is frozen in config/bench.yaml. Per-row verdicts persist under
data/runs/*/judge.parquet. Judge self-consistency is κ 0.8034 with n=338
(observed 0.9408, expected 0.6991): the same model judging twice, not
inter-family agreement. No human has verified any label.
data/human_validation.csv ships the 60 highest-value rows (false hits, judge
disagreements, escalations first) with an empty human_label column; fill it and
run `python -m lgb human-agreement --csv path` to compute judge-versus-human
agreement. Unverified labels are written as unverified.

## Serving with Docker (unverified here: no Docker on this machine)

```
docker compose up --build
# gateway on localhost:8000 (container binds 0.0.0.0:8000)
curl -s localhost:8000/health
# {"ok": true, "cache_size": 0}
```

Expected: image builds (torch CPU wheel plus ~90 MB embedding weights
download to data/models on first embed, needs network), `health` returns ok,
and the chat endpoint answers as in Quickstart. Without GSK_API_KEY the
container behaves like local runs: requests pass with no Authorization
header, which the dev gateway accepts. Against a keyed upstream, requests
fail with `RuntimeError: chat failed for deepseek-v4-flash: http ...` after
3 attempts (HTTP 500 with that message; uvicorn logs the traceback). The
container reaches the upstream via LGB_GATEWAY_BASE_URL, default
http://host.docker.internal:8787/v1 (Docker Desktop); on Linux set
LGB_GATEWAY_BASE_URL=http://172.17.0.1:8787/v1 or use host networking.

## Limitations

- Workload is constructed; hit rate does not predict any real traffic.
- Two recipes of one model are not a market; router savings are brevity
  savings and do not transfer to vendor choice.
- One judge pair from the same model is not ground truth; no human labels exist.
- Prices are assumed; the measured quantity is tokens.
- Gateway exposes only one working model as of 2026-09-17; history with glm-5.3
  is archived and does not reproduce.
- Judge costs are not included in reported spend; total workload spend is about
  4.15 assumed dollars across all eight runs.
- Tail latencies reflect upstream variance at measurement time; p99 gaps
  between configs are dominated by it (see finding above).
