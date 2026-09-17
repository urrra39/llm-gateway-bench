# Decisions

Index:

1. Embeddings are local all-MiniLM-L6-v2 on CPU, not an API.
2. Model pair is one working model with two operating points since 2026-09-17.
3. Threshold is tuned on one half and reported on the other.
4. Tuning replays the growing cache, not isolated pairs.
5. Cheap tier is a brief prompt with 512 tokens; expensive is full with 4096.
6. Prices are assumed per-token rates with the date recorded.
7. Latency percentiles exclude the first request.
8. False hits are counted separately from misses.
9. Second judge is the same model; the reported number is judge self-consistency.
10. Run outputs under data/runs/ are tracked; model weights and scratch are not.
11. Tables and figures name recipes (short/long); machine config IDs are kept.
12. No second model exists (probed 2026-09-17); no router re-run, no faked price.
13. Threshold 0.79 optimised F1; under a false-hit-weighted loss no point ships.
14. Serve uses benchmark recipes, binds 0.0.0.0, takes an upstream URL override.
15. The p99 gap is upstream variance plus cascade stacking, not cache overhead.

## 1. Embeddings are local all-MiniLM-L6-v2 on CPU, not an API.

Rationale: the environment has no GPU and no embedding API key; the 90 MB
sentence-transformer runs in under 1.5 GB resident.

## 2. Model pair is one working model with two operating points since 2026-09-17.

Rationale: probing every listed model showed only deepseek-v4-flash answering;
glm-5.3 has no channel and the rest are quota-blocked. Two operating points of
one model still measure the cache-plus-router trade honestly.

## 3. Threshold is tuned on one half and reported on the other.

Rationale: tuning on the reported rows would overfit the threshold to the
headline number. Groups never straddle the split.

## 4. Tuning replays the growing cache, not isolated pairs.

Rationale: the live cache takes the maximum similarity over everything stored;
pairwise tuning understates the false-hit rate once the cache fills.

## 5. Cheap tier is a brief prompt with 512 tokens; expensive is full with 4096.

Rationale: with one model, the cost split must come from measured tokens. Brief
answers use roughly half the output tokens.

## 6. Prices are assumed per-token rates with the date recorded.

Rationale: the gateway reports usage but no price. Dollars are the table
applied to measured tokens.

## 7. Latency percentiles exclude the first request.

Rationale: the first request pays cold connection and model-load cost, reported
separately rather than averaged in.

## 8. False hits are counted separately from misses.

Rationale: a cache that returns confidently wrong answers is worse than no
cache; folding them into hit rate would hide the failure.

## 9. Second judge is the same model; the reported number is judge self-consistency.

Rationale: no second family answers, so the denominator and the limitation are
published rather than hidden. The word kappa never appears without the
self-consistency qualifier in prose.

## 10. Run outputs under data/runs/ are tracked; model weights and scratch are not.

Rationale: a previous run lost its dataset to an ignored parquet. Ignore rules
are verified by test_gitignore.py against the actual paths the code writes.

## 11. Tables and figures name recipes (short/long); machine config IDs are kept.

Rationale: renaming served_by values and config keys would change committed
parquet and results.json, forcing a full re-run for identical numbers. Display
names describe what runs; IDs in parentheses map back to the artifacts.

## 12. No second model exists (probed 2026-09-17); no router re-run, no faked price.

Rationale: at 2026-09-17T07:43Z every exposed model was probed with the same
request (POST /v1/chat/completions, max_tokens 4096, temperature 0,
reasoning_effort low): claude-opus-4-8, claude-opus-5, gpt-5.6-sol and
gpt-6-astra are budget-quota-blocked, glm-5.3 has no channel, only
deepseek-v4-flash answers. A real-pair re-run was budgeted at about 42 minutes
wall-clock plus judging and about 4.15 assumed dollars of workload calls; it
was not spent because there is no second model to run it on. The publication
brief was checked against these docs first; no discrepancies were found.

## 13. Threshold 0.79 optimised F1; under a false-hit-weighted loss no point ships.

Rationale: F1 weights precision and recall equally, but a served wrong answer
costs an order of magnitude more than a paid model call. The sweep shows no
threshold under 2% false hits (6.45% at 0.995 with recall 0.25), so the honest
recommendation is exact-match caching plus short recipes, not a tuned point.

## 14. Serve uses benchmark recipes, binds 0.0.0.0, takes an upstream URL override.

Rationale: the live gateway previously called the short tier with the long
prompt, disagreeing with the benchmark; it now passes the cheap recipe.
Binding all interfaces and LGB_GATEWAY_BASE_URL exist so the container can
serve and reach an upstream off container-localhost. The image installs the
embeddings extra and ships the workload sample so `make pilot` works offline
of the weight download.

## 15. The p99 gap is upstream variance plus cascade stacking, not cache overhead.

Rationale: per-request records show tail rows are model-bound misses with
thousands of output tokens, lookup overhead in milliseconds, and small
miss-sample noise across wall-clock-separated runs. The cascade tail is
structural (two sequential calls); its fix is recorded, not implemented.
