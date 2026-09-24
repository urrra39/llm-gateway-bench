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
27. The authorship guard checks six phrases, and one commit hash is allowlisted.
28. The human-validation file ships the rows that could change a conclusion.

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
scratch every time — and the measurement is that rebuilding from scratch is
cheap. Runs 36 and 37, the first two under this configuration, both report
zero cached layers and docker jobs of **1m32s and 2m59s**. Run 36: 5.1s of
apt and pip, 33.9s installing 71 packages including the torch CPU wheel, 1.1s
installing the project, 29.1s exporting layers, 1m18s of Build inside a 1m32s
job. Run 37 did identical work on a slower runner: 5.8s, 76.1s, 1.1s, 75.1s,
2m45s of Build inside a 2m59s job. The spread is runner speed, not caching —
neither run restored a layer. Against 21m25s in run 35, the caching machinery
was 86 to 93% of the job it was meant to shorten. The brief asked for warm
builds in single-digit minutes; a fresh GitHub runner has no warm build to
have, and what it has instead is a cold build of one and a half to three
minutes, every time.

Kept on every push rather than gated on paths. A `paths:` filter plus a
scheduled full run would leave the ci badge green on a commit where docker
never executed, and the badge cannot say which of those two things it means.
Freshness that cannot be guaranteed is worse than a minute of runner time,
so the job runs on every push to main and on every pull request. At under
three minutes this is no longer a trade worth revisiting.

`timeout-minutes` moved 35 to 15 once runs 36 and 37 measured the new cold
build. Fifteen minutes is five times the slower of the two Build steps
(2m45s); the old ceiling was sized for the export path that no longer
exists, whose worst observed Build was 20m44s. A ceiling exists to make a
hung build fail as a build rather than as a mystery, and five times the
measured worst case still does that.

No published measurement of the benchmark changed; the numbers in this entry
are CI durations read from the Actions API and from the runs' build logs.

## 27. The authorship guard checks six phrases, and one commit hash is allowlisted.

Rationale: the repository had two authorship guards and they had drifted. The
file-contents guard checked six phrases; the commit-message guard checked
four. Two of the six were therefore unguarded in commit messages for the
whole history. Widening the message guard to the same six is the fix, and the
widening was proved against history before it shipped: the six-phrase pattern
matched nothing in any existing message, exit 1.

It then failed on the very next run, and the offender was the commit that did
the widening. Its message explains what the guard now catches and quotes two
of the six phrases as search terms to do so. This is recorded rather than
tidied away because the repository does not rewrite history and does not
force-push, so the message is permanent and the guard has to be told about
it. Commit 04589de8bbf806912ad97b369d9caedce419bcbc is allowlisted by full
SHA, one hash, no pattern and no prefix match. Every other commit is checked,
including every future one, and the guard reports the offending SHA and
subject rather than a bare grep line. This mirrors the file-contents guard,
which already excludes .github/workflows/ci.yml because that file carries the
search pattern itself, and data/workload/paws_sample.csv because a PAWS
source sentence contains one of the phrases.

The guard can fail, which is the property that matters, and it was
demonstrated twice: once in CI, in run 36, where it failed on a real commit
and blocked the run; and once against a throwaway repository carrying a
deliberate attribution trailer, where the new per-commit form reported the
offending hash and exited 1. Both jobs check out with fetch-depth: 0, so the
guard sees the whole history rather than a shallow slice of it.

No published number changed.

## 28. The human-validation file ships the rows that could change a conclusion.

Rationale: `data/human_validation.csv` existed, had the right column, and was
useless. It held 60 cache_exact rows from the high-duplicate cache run, every
one scored 2 or 1 by the judge — the least informative 60 rows the store can
produce — while the README said it held the highest-value rows ordered false
hits first, then judge disagreements, then escalations. Two defects in the
same function made that happen. The sort key tested
`isinstance(score, int)` while the exporter wrote scores through a
stringifier, so the false-hit and disagreement tests were false for every
row and the sort fell through to alphabetical `dup_type`, which begins with
"exact". And the false-hit test needed to know whether a row was a semantic
hit, which judge.parquet does not record: `kind` lives in outcomes.parquet
and has to be joined on (config, frac, idx), which the exporter never did.

The exporter now merges the outcome columns, assigns a priority tier, writes
the tier and its reason into the file, and sorts on (tier, config, frac, idx)
so the output is a pure function of the run store and regenerates
byte-identically. Recomputed over all 1449 judged rows: 34 semantic hits the
judge scored 0, 19 rows where the two judge passes disagree, 26 cascade
escalations, 1370 others. The 60-row file takes all of tier 1, all of tier 2
and 7 of the 26 tier-3 escalations. Tier 1 and tier 2 overlap by one row,
counted in tier 1. Three exact hits scored 0 exist and are tier 4 by this
ordering; they are reachable with a larger `--limit`.

Two further defects in the same module were fixed because they would have
silently corrupted the statistic the file exists to produce. `_norm` mapped
"2"/"1"/"0" but not the "2.0"/"1.0"/"0.0" spellings parquet actually writes,
so a judge score could never have equalled a human "correct"; and "nan"
became a fourth category instead of a missing value, deflating the marginals.
The agreement report now also states expected agreement, which kappa cannot
be read without: where one category dominates, a high observed agreement and
a near-zero kappa are the same measurement and only the chance term says so.
`human-agreement` prints n, observed, expected and kappa per judge pass, says
so explicitly when kappa is undefined because chance agreement is 1, and
distinguishes labelled from unlabelled rows. Verified end to end against a
committed synthetic fixture of seven rows whose arithmetic is recomputed by
hand in the test.

docs/HUMAN_LABELING.md is new and states the convention: the 2/1/0 scale
defined operationally, ten deciding rules applied in order, a column-by-column
description, and the rule that the judge's own columns are not to be read
before deciding. Two of its rules exist because they are the ones two
labellers would otherwise split on: judge the answer against the request and
never against the baseline text, and a fluent accurate answer to a different
question is 0.

Nothing writes a human label. The file ships with all 60 cells empty, the
exporter writes the empty string unconditionally, and a test fails if
data/human_validation.csv ever contains a label. Estimated effort for the 60
rows is about 2 to 3 hours, stated as an estimate in the same sentence in
both README and protocol, with its basis given as 7,406 words of request and
answer text; no row has been labelled, so there is no observed rate and none
is implied.

No published number moved. human_label_coverage was 0/60 before the change
and is 0/60 after it, because the file was never labelled and still is not.
judge_independence is untouched and cannot close under the one-model
constraint; the two conditions that would close it are named in the README
block and at the end of docs/HUMAN_LABELING.md.
