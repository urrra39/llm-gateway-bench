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
9. Second judge is the same model; kappa is self-consistency.
10. Run outputs under data/runs/ are tracked; model weights and scratch are not.

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

## 9. Second judge is the same model; kappa is self-consistency.

Rationale: no second family answers, so the denominator and the limitation are
published rather than hidden.

## 10. Run outputs under data/runs/ are tracked; model weights and scratch are not.

Rationale: a previous run lost its dataset to an ignored parquet. Ignore rules
are verified by test_gitignore.py against the actual paths the code writes.
