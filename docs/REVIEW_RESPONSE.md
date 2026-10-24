# Response to reviewer comments

Status of every point raised. Three dispositions are used:

- **Fixed** — corrected in `docs/SymptoGraph-LLM_Paper.docx` in this revision.
- **Disclosed** — cannot be fixed by editing; the claim has been withdrawn or
  the limitation stated explicitly in §5.3.
- **Blocked** — needs a run that has not happened. Listed in §5.2 and below.

Rebuild with `python3 scripts/build_paper.py`; the four checkers in `scripts/`
enforce the invariants this revision introduces.

---

## A1. Baseline numbers do not match notebook 05 — **BLOCKED, and it is the
## most serious point**

This one is unresolved and the authors must settle it before submission.

The reviewer reports notebook 05 giving MentalRoBERTa symptom F1 **0.766**,
AUROC 0.882, risk QWK 0.470, composite 0.568. The file in this repository,
`comparison_vs_baselines/leaderboard_test.csv`, gives **0.696**, 0.859, 0.428,
0.534. Notebook 05 in the repository has **no saved outputs**, so the
discrepancy cannot be adjudicated from what is committed here.

This matters because it decides whether the paper has a result:

| | repo CSV | reviewer's notebook-05 figures |
|---|---|---|
| MentalRoBERTa symptom F1 | 0.696 | 0.766 |
| Proposed symptom F1 | 0.775 | 0.775 |
| **Margin** | **+0.079** | **+0.009** |

At +0.009 against a seed dispersion of 0.012 the headline claim evaporates.
Nothing was changed in either direction on the strength of the reviewer's
numbers, because picking whichever set favours the model is not a defensible
act. What was done instead:

1. §4.1 and §5.3 now state that the reference comparison is **not controlled**
   (see A4), which is true under either set of numbers, and
2. the architectural argument has been moved onto the protocol-matched ladder
   in §4.5, so the paper no longer rests on Table 10.

**Required before submission:** re-run notebook 05 with outputs retained, and
regenerate `comparison_vs_baselines/`. If 0.766 is correct, Table 10 shows
approximate parity and the abstract must say so.

## A2. Gains are within the paper's own noise — **Disclosed**

Correct, and the internal inconsistency was real: §4.5 dismissed L4→L5
(+0.024 F1) as noise while the abstract advertised L0→L5 (+0.023) as a result.

§5.3 now opens by stating that **no architectural effect in this study is
established in the statistical sense**, that differences of the size separating
adjacent rungs are comparable to the seed dispersion, and that the ladder
should be read as a description of six configurations rather than a
demonstration that any component works. The abstract and §5.1 no longer claim
otherwise.

## A3. No significance tests anywhere — **Disclosed**

Correct. §4.2 promised paired bootstrap tests, per-metric confidence intervals
and a "significant on every seed" rule, and delivered none of it; the
"interval attainable on 74 instances" was promised in §3.1, deferred twice and
never computed.

Those promises are removed. §4.2 now defines the paired test, states plainly
that **it is not reported**, and gives the reason: per-example predictions were
not retained, so no paired test can be computed after the fact. The three
cross-references that pointed at the missing interval now point at the measured
seed dispersion (0.136 on risk QWK) instead. Table 6 lists significance testing
as "not performed, see Section 5.3".

`±` is now defined on first use in §4.2 as the mean and sample standard
deviation over three seeds, explicitly not a confidence interval.

## A4. Unfair baseline protocol — **Fixed**

Verified against notebook 05: `epochs=8, batch_size=16, max_length=256`, no
early stopping, against `10 / 32 / 192` with patience 4 for the proposed model.
The paper's claim of "identical epoch budget, batch size, maximum sequence
length" was **false**.

§4.1 now states the two budgets side by side and concludes that §4.6 "is a
comparison against reasonable reference configurations and not a controlled
experiment; its margins cannot be attributed to architecture". §5.3 repeats it
and adds the two related asymmetries the reviewer raised: the reference models
never see the weak-labelled posts, and L0 and the MentalRoBERTa reference share
an encoder yet differ by 0.056 symptom macro-F1, a gap the paper states it
cannot explain.

No LLM baseline is included. Disclosed in §5.3 as missing.

## A5. Metric counts are wrong — **Fixed**

Correct on every count. The claims are now **computed from the tables** by
`scripts/results.py` rather than asserted, and comparison is on the
three-decimal printed values, so the number in the prose is the number a reader
can verify:

- Table 9: L5 is best on **8 of 9**, of which **7 outright and 1 a tie**, and
  does not lead on risk QWK.
- Table 10: leads all three reference models on **8 of 10**, losing risk QWK
  and risk MAE.
- The AUROC margin is **0.027**, not 0.028. `margin()` now rounds both operands
  to three decimals before subtracting, which is what removed the off-by-one.

`scripts/check_consistency.py` fails the build if any performance number in the
prose is not present in a table.

## A6. Interpretability claim contradicted by the architecture — **Disclosed**

Correct. §3.3 now states that the heads receive the concatenation of the 22
concept logits with the 768-dimensional pooled vector, that the layer is
therefore obligatory only in the weaker sense that the pooled vector is itself
concept-gated, and that **no concept-intervention or faithfulness test is
reported**, so interpretability is a property of the architecture as described
and is not experimentally established. §5.3 repeats this and records the
missing clinician evaluation and the absent separate ablations of
definition-initialisation, independent scoring and gating.

## A7. The ordinal claim contradicts itself — **Fixed**

Correct, and this was a genuine defect: the model uses plain linear heads, so
the paper could not also demand that severity "follow the scoring rule of the
instrument" or fault [3] and [14] for unconstrained heads. The abstract, the
introduction and the Novelty paragraph no longer make the instrument-ordering
claim; §3.4 and §3.5 state that no ordering constraint is imposed and that the
ordering is measured by QWK and MAE rather than assumed, and §4.2 says the
kappa is the instrument by which it is verified.

## A8. Leakage and split claims contradict the notebooks — **Fixed**

All four sub-points verified against `processed/manifest.json`
(`author_overrides_applied = 0`, `authors_total = 500`).

§3.1 and §3.2 now state that the author audit applies to the unlabelled pool,
that the labelled corpus carries usable author identifiers for only **500**
authors, that **13,386 of 19,762** posts take the split published by their
source corpus with the remainder assigned by author hashing, and that the
repair pass **applied zero overrides** because it had nothing to move. The
sentence claiming it "relocates the posts of 192,893 authors" is gone, as is
the 23.3% figure with its wrong denominator.

## A9. "19,762 gold-labelled posts" is misleading — **Partly fixed**

§3.1 now reports the per-task label coverage, and Table 3's stress row is
corrected to **2,456 / 541** to match the notebook. The paper states that every
symptom metric covers 11 of 22 concepts.

Still **disclosed rather than fixed**: Table 2's split percentages are computed
over all rows including explanation-only IMHI posts that contribute to no
metric. Recomputing Table 2 per task is the right fix and needs the row-level
provenance.

## A10. Weak supervision claims are overstated — **Fixed**

Correct. The title no longer says "calibrated", and §5.3 reports that
temperature scaling moved expected calibration error only from **0.224 to
0.183** with a non-monotone reliability curve, that labeller agreement averages
0.66 F1 and 0.41 kappa falling to 0.12 on low self-esteem, that **6,000 of the
49,982** labelled posts were used, and that the weak-only symptoms were never
validated. §4.5 already reported that L3 costs performance on every measurable
task; §5.3 now states that the branch "cannot be credited by any measurement
here".

Labeller throughput corrected to ~8 posts/s, consistent with the 35-hour
full-pass estimate.

## A11. Evaluation covers half the ontology — **Fixed**

§3.1 now states that only **5 of the 16** PHQ-9 and GAD-7 items are scored,
that **8 of the 11** gold concepts come from DepressionEmo via a heuristic
emotion-to-item correspondence rather than item-level annotation, and that
prevalence is high (77.3% depressed mood, 69.5% hopelessness) so a
threshold-tuned positive-class F1 is a less demanding quantity than the same
number on a balanced set. Per-symptom tables and trivial baselines are
disclosed as missing.

---

## B. Major methodological points

| Point | Disposition |
|---|---|
| N5 shows no benefit; L4 never discussed | **Fixed** — §4.5 now walks every rung including L4 and states the contrastive term is not shown to help |
| Jaccard degenerate for IRF/LoST; definition differs between Contribution 3 and Eq. 12 | **Fixed** — both now say "over symptoms both posts have annotated"; the degeneracy is disclosed in §5.3 |
| Graph edges quoted as conditional p(b\|a) but the values are PPMI | **Fixed** — the three numbers are removed; the text now describes the ordering of co-membership enrichments without quoting values |
| Only 31 of 231 symptom pairs observable; prior support rests on 1–3 pairs | **Fixed** — stated in §3.4 and §5.3 |
| Symptom–symptom block never defined | **Disclosed** in §5.3 |
| Disorder nodes have no supervision or output; comorbidity never evaluated | **Disclosed** in §5.3 |
| No random-graph control, which the paper faults [18] for lacking | **Disclosed** in §5.3, naming the inconsistency |
| "Loneliness"/"suicidality" treated as disorders; prior not clinician-reviewed | **Disclosed** in §3.4 and §5.3 |
| "L2 is the clearest contributor to severity" unsupported | **Fixed** — claim withdrawn |
| Abstention never shown to be calibrated; no softmax/margin comparison | **Disclosed** in §5.3 |
| Abstention score averages 22 log-variances but error is on 11 | **Disclosed** in §5.3 |
| §3.1 "predicts squared error" vs §3.5 attenuation NLL | **Fixed** — both now describe the attenuation form of Eq. 14 |
| Checkpoint selection past val-loss minimum; headline depends on a rule the paper calls wrong | **Disclosed** — §4.3, §4.4 and §5.3; re-run in §5.2 |
| "Composite" undefined and conflated with "selection score" | **Fixed** — §4.2 defines both exactly: composite is the mean of symptom F1, severity QWK and risk QWK and excludes stress; the selection score is the mean of symptom F1, symptom AUROC and severity QWK |
| Table 5 total does not add up; SBERT is 384-d; D symbolic | **Fixed** — SBERT corrected to 384 with a 384→768 projection and the parameter rows corrected |
| L3 AUROC "0.881 ± 0.000" suspicious | **Blocked** — needs the per-seed values, which the aggregated CSV does not carry |
| Table 7 shows no epoch 9–10 although 9 is selected | **Partly fixed** — columns are now 1/2/3/6/8, chosen so every epoch quoted in the prose is a column; seed 1 stopped at epoch 8, so 9–10 would be a two-seed mean and is deliberately not shown |
| Development anecdotes unsupported | **Disclosed** — retained but now explicitly labelled as development observations |
| Dedup: 200-char prefix, no fuzzy matching, merge rules unstated | **Disclosed** in §5.3 |
| 57% of pool off-topic; disorder labels are subreddit proxies | **Disclosed** in §5.3 and §5.4 |
| 69.3% keyword-leakage statistic unsourced | **Fixed** — removed |
| No external/OOD evaluation; N4 absent though numbering skips it | **Disclosed** in §5.3; the N-numbering is now explained where the novelties are introduced |

## C. Claims and positioning

| Point | Disposition |
|---|---|
| "none has done so as this paper does", "not combined before" | **Fixed** — replaced with "the following five works are the closest in intent" |
| "directly usable in a clinical screening pathway" | **Fixed** — §5.4 states it is not a screening instrument and must not be deployed as one |
| §4.6 titled "State-of-the-Art" but compares three self-trained baselines | **Partly fixed** — the heading is kept to mirror the template, but both §4.1 and §4.6 state the comparison set is retrained reference models, not published systems |
| Missing citations for concept bottleneck, SupCon, selective prediction, loss attenuation | **Blocked** — needs new references; the survey is fixed at [1]–[30] by the citation rule in force |
| Datasets and tools uncited | **Blocked** — same constraint |
| Omits eRisk symptom tasks, PRIMATE | **Blocked** — same constraint |
| Table 1 gap column sarcastic ("without conducting one") | **Fixed** |
| Formulaic survey wording | **Not addressed** |
| Title ~30 words and misleading | **Fixed** — now 12 words, and "calibrated" is gone |

## D. Missing, formatting, style

| Point | Disposition |
|---|---|
| No limitations section | **Fixed** — new §5.3, the longest section in the paper |
| No ethics, IRB, licence, misuse or availability statement | **Fixed** — new §5.4, covering licensing, pseudonymity, local-only labelling, intended use, suicide-risk false negatives and availability |
| Wrong cross-references | **Fixed** — the weak branch now points at §3.2/§3.4, and the three pointers to the uncomputed interval are redirected |
| "linear schedule" vs "cosine schedule" | **Fixed** — cosine, matching `config.json` |
| "seven-corpus" vs "eight-corpus" | **Fixed** — seven |
| Weak subsample "proportional" vs "evenly spread" | **Fixed** |
| Table 3 stress counts wrong; "test share" undefined | **Fixed** — counts corrected; the column is defined in the caption |
| Table 8 duplicated all/gold-supervised rows | **Fixed** — the duplicate pair is removed; the two were identical because only the 11 gold concepts are ever scored |
| N-labels in Tables 4–5 never defined | **Fixed** |
| Abstract: dense, unparseable opening, repetition, no limitations | **Partly fixed** — opening rewritten, repetition removed, symptom coverage and the scope of the comparison now stated |
| §3 opens with a ~500-word paragraph | **Not addressed** |
| Informal register, "we" in conclusion | **Fixed** — "load-bearing", "against the grain", "stands out" removed |
| Reference defects: [28] "Mod" typo, [19]/[29] missing volume/pages, [21] year vs volume, DOIs, [8] claim | **Blocked** — needs source verification against the publisher records |
| Notebook hygiene: wrong column in notebook 06, `.bak` mock labels, four repo commits | **Blocked** — notebook-side, not manuscript-side |

---

## What this revision does not fix

Five items decide whether the paper is publishable and none can be settled by
editing prose:

1. **Reconcile the baseline numbers** (A1). Re-run notebook 05 with outputs
   retained.
2. **Retrain the reference models on the proposed model's budget** so §4.6
   becomes controlled.
3. **Retain per-example predictions** and compute paired tests and intervals,
   so §4 can make a claim rather than a description.
4. **Re-run checkpoint selection on validation loss**, which §4.4 shows would
   favour the two graded heads.
5. **Build the anxiety gold set**, without which the weak branch and half the
   ontology remain unmeasurable.

Until 1–3 are done, the defensible contributions of this work are the
leakage-controlled harmonisation, the abstention result (AURC 0.099, symptom
error 0.192 → 0.095 at half coverage), and the negative findings — not a
performance win over the standard approach.
