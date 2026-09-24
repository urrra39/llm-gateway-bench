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
16. Exact-match shares are stated per denominator and machine-checked.
17. Every rate carries a Wilson or bootstrap interval; comparisons carry verdicts.
18. The exact_only recommendation is measured by replay, quality unmeasured.
19. Cost-weighted loss picks the grid edge from r=10 up; recall saturation is measured.
20. Row counts are integers; the tuning/report gap is noise; model_ms stays a defect.
21. Two label-quality gates fail publicly; the gate line reads 9/11, not 7/9.
22. Figures are embedded; Docker is verified by a CI job.
23. No published number moved this round; the description names the finding.
24. The Quickstart states the keyless boundary; upstream failure names its variable.
25. The false-hit gates get a 5% bar and fail, rather than being deleted.
26. The docker job drops buildx and the GHA cache; the cache key was the defect.

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

## 16. Exact-match shares are stated per denominator and machine-checked.

Rationale: the published sentence mixed two denominators (24 of 69 hits is
34.8%; 24 of 245 requests is 9.8%). Every "X of Y" and every "N%" in the four
prose files must now match a recomputed pair or value or the audit fails; a
value-presence check could not catch this class. Proven by doctoring 69 to 70
and watching the audit fail, then reverting.

## 17. Every rate carries a Wilson or bootstrap interval; comparisons carry verdicts.

Rationale: binomial rates use Wilson 95% (correct at small counts and near
zero); means, sums and percentiles use percentile bootstrap (B=10000, recorded
seeds, index-matrix shas in results.json); false-hit differences use Newcombe,
shared-row differences use lockstep paired bootstrap. Most router orderings do
not survive: heuristic-vs-cascade cost separates on low only, quality on high
only, false hits nowhere (about 451 semantic hits per arm needed at
conventional power). The audit holds a registry of comparative claims and
fails unregistered comparative sentences.

## 18. The exact_only recommendation is measured by replay, quality unmeasured.

Rationale: exact matching is order-deterministic, so replaying the baseline's
stored rows reproduces the config without model calls (26 low and 73 high
exact hits). Those 26 are not the cache run's 24 renamed. A semantic hit is
answered from store but never added to it, so when the cache answered requests
438, 464 and 482 semantically at similarity 0.9966 against anchor idx 148,
their own text never became an exact key, while the tier-free replay stores
438 and matches 464 and 482 verbatim: 26 is 24 plus those two rows. Replaying
the exact tier over the cache run's own rows also yields 26, so the error sets
are not the cause. This entry previously read "replay mirrors the baseline
error set, hence 26 vs the cache run's 24 on low", which was wrong; the
explanation is corrected above and no count moved. Quality cannot be judged
without judge calls — temperature-0 repeats still differ, so the replay serves
texts no judge scored — and is reported unmeasured, with the cache run's
exact-hit verdicts as an explicit proxy (1 of 24 scored 0 on low, 0 of 71 on
high with 2 unjudged, which also retires the "zero added error" claim).

## 19. Cost-weighted loss picks the grid edge from r=10 up; recall saturation is measured.

Rationale: L = r·fp + fn over the committed sweep selects 0.745 at r=1, 0.98
at r=3, and the 0.995 grid edge at r=10/30/100 with recall 0.2458 — no
acceptable point exists. Recall is 1.0000 from 0.30 through 0.79 with the
first drop at 0.795, so saturation is measured, not a truncated grid (the
sweep already starts at 0.30).

## 20. Row counts are integers; the tuning/report gap is noise; model_ms stays a defect.

Rationale: successful rows per config are 237/238/238/237 on low and 246 on
high (gateway errors excluded); no count is written as a range. Tuning-half
26 of 144 (95% Wilson 0.1263-0.2514) overlaps report-half 4 of 45 and 8 of 85,
so the factor-of-two gap is sampling noise against small trap counts, not a
split defect. model_ms double-counting cannot be repaired (no per-call times
stored); D5 now carries the exact fix and what it blocks.

## 21. Two label-quality gates fail publicly; the gate line reads 9/11, not 7/9.

Superseded by #25: the gate line now reads 7/11 because two more gates were
given a bound and both fail. The reasoning below still holds for the two
label-quality gates.

Rationale: human_label_coverage reads 0/60 = 0.000 against a 0.50 bar and
judge_independence fails while both judges are deepseek-v4-flash, so
all_gates_passed is false and the front page shows a full 11-row gate table.
The instruction said 7/9; the real counts are 9 pass of 11 total, and the
instruction itself demands real counts, so 9/11 stands and the deviation is
recorded here.

## 22. Figures are embedded; Docker is verified by a CI job.

Rationale: key figures render inline from relative paths; every committed
figure is referenced and none orphaned (audit-enforced, http sources banned).
The container path runs in CI (build, health poll, keyless chat assertion)
because this machine has no Docker; the local caveat stands only for local runs.

## 23. No published number moved this round; the description names the finding.

Rationale: the results.json diff against the previous primary is purely
additive (interval fields, two exact_only entries, two gates, cost-weighted
and comparisons blocks) plus the entailed all_gates_passed flip from true to
false. The GitHub description now leads with the measured finding
(136 characters, set via API) instead of "model routing", which the one-model
constraint contradicts; topics already covered llm, caching, routing,
cost-optimization, benchmark and fastapi, so they stand.

## 24. The Quickstart states the keyless boundary; upstream failure names its variable.

Rationale: the old Quickstart read "no API key — the dev gateway answers
keyless", true only on the author's machine, so a stranger followed three
commands to a connection error. The Quickstart now splits keyless from
endpoint-requiring before the commands (serve starts and /health answers,
plus workload/tune/metrics/audit over committed parquet; a completion does
not), shows LGB_GATEWAY_BASE_URL and GSK_API_KEY in the command block as read
from src/lgb/config.py, and carries the private-gateway disclosure at the
point the reproduction claim is made rather than only under Limitations.
Gateway.chat now raises lgb.chat.UpstreamError carrying the variable name and
distinguishes unreachable (transport), rejected credentials (401/403, no
retry) and answered-without-a-completion; the API returns HTTP 500 with
{"type": "upstream_unavailable", "param": "<variable>"} and logs the same
message through uvicorn. The `chat failed for <model>` prefix is preserved
because the CI probe and the README both quote it; CI now also asserts the
variable name appears in both the body and the container log. `make all`
cannot run without an upstream (run/judge call the model), stated beside the
45-minute claim.

## 25. The false-hit gates get a 5% bar and fail, rather than being deleted.

Rationale: false_hit_rate_reported_low and false_hit_rate_reported_high
asserted only that a false-hit rate existed. No observation could have
failed them, so they inflated the pass count without testing anything. Two
options were available: delete them and move the numbers into the results
table as recorded observations, or give them a bound. The bound was chosen
because it is the stronger option and because it agrees with the no-ship
verdict the README already reaches: a semantic cache that serves one
confidently wrong answer per twenty semantic hits is not shippable for
correctness-sensitive traffic, and a false hit is worse than a miss in that
the user gets no signal. The bar is 5%, defined once as FALSE_HIT_RATE_BAR in
src/lgb/metrics.py and stated in the README as an engineering choice, not a
measurement. Both gates now read FAIL at 0.0889 (4 of 45 semantic hits) and
0.0941 (8 of 85 semantic hits), and the gates were renamed
false_hit_rate_within_bound_{low,high} because the old names described
reporting, not a bound.

Re-auditing the remaining nine: six baseline_costs_more_* gates compare two
measured costs and would fail if a cached or routed run cost at least as much
as the baseline; tuned_threshold_beats_random_control compares tuned F1
against the mean of 64 random thresholds and would fail on a tie;
human_label_coverage compares filled rows against a 0.50 bar and does fail at
0.000; judge_independence compares two model names and does fail. Every gate
in the set now compares a measurement against a bound the data could have
crossed. That property is machine-checked: scripts/audit_docs.py holds
GATE_BOUNDS, one entry per gate naming its comparison, and fails on any gate
absent from it — a gate with nothing to compare has no entry to write. The
same check re-derives both false-hit verdicts from the metrics block and
requires the README table to restate every gate row verbatim, so a failing
gate cannot be dropped from the front page.

No measured number moved. The results.json diff is six fields: two gate
names, two observed strings, two passed booleans. The gate line moved from
9/11 PASS, 2 FAIL to 7/11 PASS, 4 FAIL, and all_gates_passed was already
false.

## 26. The docker job drops buildx and the GHA cache; the cache key was the defect.

Rationale: the job already had layer caching and the caching was the cost.
Measured from the Actions API, not estimated: run 33 (`cache-to` writing a
cold cache) took 16m41s of docker; run 35 (the same cache warm, meant to be
the fast one) took 21m25s, of which the Build step alone was 20m44s. Caching
made the job 4m44s slower.

Run 35's build log says why. `grep -c CACHED` over it returns 2, and the two
restored layers are WORKDIR and the apt-get/pip line. The expensive step,
`#14 RUN uv sync --frozen --no-dev --extra embeddings` at 72.1s, was never
once a cache hit, because `COPY pyproject.toml uv.lock README.md Makefile ./`
sat directly above it. This repository's product is documents; README.md
changes on nearly every commit, so the dependency layer was invalidated on
nearly every commit. Meanwhile `cache-to: type=gha,mode=max` spent 650.7s
uploading layers to save that 72.1s step, and buildx's `docker-container`
driver cannot write into the daemon's image store, so `load: true` exported
the 6.34 GB image to a tarball and re-imported it: 511.4s in `exporting to
docker image format` plus 150.1s in `importing to docker`. Roughly 1300s of
the 1285s Build step was transfer, not build.

So the fix is subtraction plus one reordering, and it is two changes, not one.
The Dockerfile now copies `pyproject.toml` and `uv.lock` alone, runs
`uv sync --no-install-project` against them, and only then copies README.md,
Makefile, config/, src/ and the workload sample before a second `uv sync`
that installs the project itself. The dependency layer is keyed on the lock
file, which is the thing that actually determines the dependencies.
(`readme = "README.md"` in pyproject.toml is why README was in the first COPY
at all; `--no-install-project` does not read it.) The workflow then drops
docker/setup-buildx-action, docker/build-push-action, `cache-from`,
`cache-to` and `load: true` for a plain `docker build` on the default daemon
driver, which writes layers where the next `docker run` can already see them.

What this does and does not buy. On any host that keeps an image store
between builds, the dependency layer is now reused across every prose commit.
A GitHub-hosted runner is ephemeral and keeps nothing, so it rebuilds from
scratch every time — but rebuilding is the 72.1s `uv sync` plus the apt and
wheel work, not the ~1300s of export and upload that the caching machinery
added to reach the same place. The brief asked for warm builds in single-digit
minutes; the honest statement is that a fresh GitHub runner has no warm build
to have, and the way to a single-digit job here was to stop paying for a cache
that never hit. The next run measures the result, and until it does the
before/after in the README is one number, not two.

Kept on every push rather than gated on paths. A `paths:` filter plus a
scheduled full run would leave the ci badge green on a commit where docker
never executed, and the badge cannot say which of those two things it means.
Freshness that cannot be guaranteed is worse than a minute of runner time,
so the job runs on every push to main and on every pull request.

`timeout-minutes` stays at 35. The worst cold build actually observed is run
35's 20m44s Build step, under the export path now removed, and a ceiling is
there to make a hung build fail as a build rather than as a mystery. It gets
tightened when the new configuration has a measurement of its own, not
before.

No published measurement of the benchmark changed; the numbers in this entry
are CI durations read from the Actions API and from run 35's build log.
