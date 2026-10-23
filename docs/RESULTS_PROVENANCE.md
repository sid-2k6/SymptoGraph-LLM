# Results provenance

`docs/SymptoGraph-LLM_Paper.docx` contains **no placeholders**. Every
performance number is read at build time from the run outputs by
`scripts/results.py`, which is the single source of truth, so a value quoted in
prose and the same value in a table cannot drift apart.

Regenerate: `python3 scripts/build_paper.py`
Verify: `python3 scripts/verify_paper.py`, `scripts/check_doc_order.py`,
`scripts/check_consistency.py`

## Where the numbers come from

| Destination | Source file |
|---|---|
| Tables 8, 9 (ladder) | `comparison_vs_ablations/leaderboard_test.csv` |
| Table 10 (comparison) | `comparison_vs_baselines/leaderboard_test.csv` |
| Table 7 (progression) | `L5_full/aggregate/history_all_seeds.csv` |
| Selective rows, Figure 5b | `L5_full/seed_*/performance_metrics.json` |
| Table 6 (configuration) | `L5_full/seed_*/config.json` |
| Figures 3a, 3b | `L5_full/aggregate/history_all_seeds.csv` |
| Figures 4, 5a | the two `leaderboard_test.csv` files |
| Figures 1, 2 | drawn by `build_paper.py`, no data |

Every reported quantity is on the **held-out test split** except Table 7 and
Figure 3, which are a per-epoch validation trace and are labelled as such.
The figures are regenerated from the test split rather than reusing the
notebook PNGs, which plot validation and would contradict the tables.

## Reported configuration

- Ladder: L0_flat, L1_bottleneck, L2_graph, L3_weak, L4_contrastive, L5_full
- Comparison: tfidf_mlp, bilstm_attn, mentalroberta
- Seeds 3, epoch budget 10, patience 4,
  cosine schedule
- Selection metric `selection_score`, selected epochs 9, 4, 9
- Splits: train 18911 (12911 gold + 6000
  weak), val 2862, test 3989
- Trainable parameters 125,766,023

## Headline findings

- L5 leads 9 of 11 ablation metrics; the ladder is **not** monotone.
- L5 beats every comparison model on 9 of 11 metrics, losing on
  Risk QWK, Risk MAE.
- Selective prediction: AURC 0.099 ± 0.015, symptom error
  0.192 at full coverage falling to
  0.095 at half coverage.

## Negative and unattributable findings, reported in the paper

1. **4.5** - the weak-label stream at L3 *costs* performance on every
   measurable task. Eleven of 22 symptoms have no gold label anywhere, those
   eleven are what the branch supervises, and every symptom metric is computed
   on the other eleven. Do not present L3 as an improvement.
2. **4.5** - the L4 to L5 gain cannot be causal. The variance head reads
   detached features and emits no gradient into any predictor, so it cannot
   change the symptom, severity, risk or stress outputs. The difference is
   initialisation and gradient-norm noise. N6's contribution is the abstention
   capability, measured separately in 4.6.
3. **4.3 / 4.4** - the selection score picks epochs 9, 4, 9, at which
   validation loss is 2.16x its minimum of 1.182 and
   training risk QWK has reached 1.000. The two graded heads
   are therefore reported past their best. A validation-loss re-run is listed
   in 5.2.
4. **4.3 / 4.6** - the flat encoder and the domain transformer both beat the
   proposal on suicide-risk QWK. Seed sd on that metric alone is
   0.136 on 74 test instances; the column is
   retained so the limitation is visible, not because the ordering means
   anything.
5. **4.1 / 4.6** - the three comparison models are reference models retrained
   under matched conditions, **not** reimplementations of published systems.
   That sentence must stay; without it the comparison overclaims.

## Deviations from the PRISM template

- Equations are matplotlib images, not OMML; `python-docx` cannot author OMML.
- Table 1 is 12x5 against PRISM's 17x5, because the citation rule fixes the
  literature table at `[21]`-`[30]` plus the proposed row.
- Table 8 is 19x3 against PRISM's 15x3, because the four selective-prediction
  rows were added so that the abstention numbers quoted in 4.6 are backed by a
  table. The other eight tables match PRISM exactly.
