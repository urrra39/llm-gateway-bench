# Human labelling protocol

This file defines the one human step this benchmark has left. It exists so
that two people labelling the same row independently write the same value. If
you find a row where this document does not decide the answer, the document is
the defect; record the case at the bottom rather than guessing.

Nothing in this repository writes a human label. Not the exporter, not the
judge, not the audit. `data/human_validation.csv` ships with `human_label`
empty on every row and the only thing that may fill it is a person reading the
row.

## What to do

```
.venv/bin/python -m lgb human-agreement --csv data/human_validation.csv
```

Open `data/human_validation.csv` in a spreadsheet or an editor, fill the
`human_label` column, save as CSV, and run the command above. It works on a
partially filled file: rows with an empty `human_label` are dropped from the
denominator, not counted as agreement. You can stop after ten rows and get a
kappa over ten rows.

Estimated effort for all 60 rows: **about 2 to 3 hours**. This is an estimate,
not a measurement — no row has been labelled, so there is no observed rate to
report. The basis: the file holds 7,406 words of request and answer text
across 60 rows, roughly 123 words per row, and a comparative verdict on two
answers takes longer than reading them once.

## The scale

Label the **answer** in `answer_text` as a response to the **request** in
`request`. Three values, written as the digit or the word; both spellings are
accepted and mean the same thing.

| value | word | means |
| --- | --- | --- |
| 2 | correct | Answers the request, and every factual claim it makes about the request is right. |
| 1 | partial | Addresses the request but is incomplete, hedged into uselessness, or mixes a right answer with a wrong claim. |
| 0 | incorrect | Does not answer the request, answers a different request, or makes a false claim central to the answer. |

Leave the cell **empty** if you cannot decide. An empty cell is a real
outcome, is reported as unlabelled, and is better than a coin flip.

## Deciding rules, in order

Apply these in order and stop at the first one that settles the row. They are
ordered because the later rules would otherwise contradict the earlier ones.

1. **Judge the answer against the request, not against `baseline_text`.**
   `baseline_text` is what the uncached run produced for the same request. It
   is context, and it can itself be wrong. Never mark an answer 0 only because
   it differs from the baseline, and never mark it 2 only because it matches.

2. **A wrong topic is 0, however good the prose.** The workload contains
   near-duplicate request pairs whose meanings are flipped. If the answer is a
   fluent, accurate answer to a question that is not the one in `request`,
   that is 0. This is the case the whole exercise is about: `kind` =
   `cache_semantic` rows were served an answer stored for a different request.

3. **Style is not correctness.** Length, tone, formatting, markdown, a missing
   greeting, a refusal to use a full sentence: none of these move the label.
   The short recipe is meant to be short.

4. **A hedge that still answers is 2; a hedge instead of an answer is 1.**
   "Probably X, though sources differ" where X is right is 2. "This is
   difficult to determine" with no X is 1. "I cannot help with that" for an
   answerable request is 0.

5. **One wrong central claim is 0. One wrong incidental claim is 1.** Central
   means the request is not answered without it. If the request asks who
   dedicated a work to whom and the answer names the wrong person, that is 0
   even if the surrounding paragraph is accurate.

6. **Partially correct lists: 1.** If the request asks for several things and
   the answer gets some right and some wrong or omits some, that is 1, not an
   average of 2 and 0.

7. **Truncation mid-sentence is 1 if what exists is right, 0 if what exists is
   already wrong.** Do not credit an answer for what it was about to say.

8. **An empty or whitespace-only `answer_text` is 0.**

9. **Do not look at `judge1_score` or `judge2_score` before deciding.** They
   are in the file because the agreement statistic needs them, not because
   they are evidence. If you find yourself agreeing with the judge on every
   row, hide those two columns and redo a sample.

10. **Do not consult a language model for the verdict.** The number this file
    produces is the only thing in the repository that is not a model
    judging a model. A model-assisted label makes it worthless and there is no
    way to detect it afterwards.

## What each column is

| column | what it is |
| --- | --- |
| `priority` | Tier, 1 is highest value. See below. |
| `priority_reason` | The tier in words. |
| `config` | Which run produced the row: `cache`, `router_cascade`, `router_heuristic`. |
| `frac` | Duplicate fraction of the workload, `low` or `high`. |
| `idx` | Request index within that run. `config` + `frac` + `idx` identifies the row uniquely. |
| `kind` | `cache_semantic` (served from store on a similarity match), `cache_exact` (served from store on identical text), `miss` (a real model call). |
| `served_by` | `cheap` or `expensive` recipe. |
| `similarity` | Cosine similarity to the stored answer, for `cache_semantic` rows only. |
| `hit_source_idx` | The `idx` whose stored answer was served. For a semantic hit this is the request the answer was actually written for. |
| `dup_type` | How the workload generated the request: `novel`, `exact`, `paraphrase`, `trap`. |
| `request` | The user request. Judge against this. |
| `answer_text` | What the run returned. Label this. |
| `baseline_text` | What the uncached run returned for the same request. Context only. |
| `judge1_score`, `judge2_score` | The two judge passes, empty where a row was not sampled for the second. Do not read before deciding. |
| `human_label` | Yours. Empty on every shipped row. |

## Why these 60 rows

The exporter takes every judged row in the run store, assigns a tier, and
writes the highest tiers first. Within a tier the order is `config`, `frac`,
`idx`, so the file is a pure function of the store and regenerating it with
`python -m lgb export-human` produces byte-identical output.

| tier | rows in the store | rows in the file | why it is worth a human minute |
| --- | --- | --- | --- |
| 1 — semantic hit the judge scored 0 | 34 | 34 | These are the false hits. The false-hit rate is the number the no-ship verdict rests on; a human disagreeing here moves the headline. |
| 2 — the two judge passes disagree | 19 | 19 | Judge self-consistency is the only label-quality figure this repository has, and these rows are where it is weakest. |
| 3 — cascade escalation | 26 | 7 | Two sequential model calls produced the answer; the escalation rule is a published design choice. |
| 4 — every other judged row | 1370 | 0 | Present so the tail can be sampled, not prioritised. |

The tier-1 and tier-2 sets overlap by one row, which is counted in tier 1.
Tier 3 is truncated by the 60-row limit; `python -m lgb export-human --limit
200` writes more.

The file shipped before 2026-09-24 was sixty `cache_exact` rows from a single
run, all scored 2 or 1, which is the least informative set the store can
produce. The sort key tested `isinstance(score, int)` while the exporter wrote
the scores as strings, so no row ever matched the false-hit or disagreement
test and the sort fell through to alphabetical `dup_type`. That is fixed and
the tiers above are recomputed from the store.

## What the numbers mean when you run it

The command prints, for each judge pass:

- `n (rows with both)` — rows carrying both a human label and that judge's
  score. This is the denominator for everything below it.
- `observed agreement` — the share of those rows where the human and the judge
  wrote the same category.
- `expected agreement` — chance agreement, the sum over the three categories
  of the human's marginal share times the judge's. Reported because kappa
  alone is unreadable: where one category dominates, high observed agreement
  and near-zero kappa are the same measurement and only this term says so.
- `Cohen's kappa` — `(observed - expected) / (1 - expected)`. Undefined, and
  reported as such, when both raters used one category and expected agreement
  is 1.

With no labels the command reports `n = 0` and no statistics. That is the
current state and it is the honest one: agreement is unmeasured, not zero.

## What this closes and what it does not

Filling this file closes the `human_label_coverage` gate at 30 labelled rows,
which is the 0.50 bar over 60.

It does not by itself close `judge_independence`. That gate compares the
primary and secondary judge model names, they are both
`deepseek-v4-flash`, and the reported kappa of 0.8034 over 338 rows is
therefore judge self-consistency and not judge accuracy. Exactly one of two
things closes it: a second model family becoming reachable from the gateway,
so that two independent judges can be compared; or a human-labelled subset
large enough to score the judge against, at which point the human labels are
the reference and the judge is the thing being measured. All 60 rows labelled
is a start on the second and is not on its own a large enough subset to
replace a second judge.

## Cases this document does not decide

None recorded. Add them here with `config`, `frac`, `idx` and the question the
rules above failed to answer.
