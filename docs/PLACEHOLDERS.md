# Placeholder index

`docs/SymptoGraph-LLM_Paper.docx` follows the structure and formatting of
`PRISM - Copy.docx`. Every **performance** number is `X.XXX` or
`X.XXX ± X.XXX`. Search for `X.XXX`.

Dataset counts, corpus overlaps, the graph edge weights and the labeller noise
channel are **measured facts and already real** - do not overwrite them.

Regenerate: `python scripts/build_paper.py`

## Tables

| Table | Status | Fills from |
|---|---|---|
| 1. Literature comparison | final, no numbers | - |
| 2. Dataset partitioning | real | `processed/manifest.json` |
| 3. Task composition | real | `processed/label_coverage.csv` |
| 4. Layer-wise architecture | real | `src/symptograph/proposed.py` |
| 5. Parameter distribution | real | `config.json` field `trainable_parameters` |
| 6. Hyperparameter configuration | real | `ProposedConfig` |
| **7. Training progression** | placeholder | `proposed_v2_e8/*/seed_*/history.csv` |
| **8. Final performance** | placeholder | `proposed_v2_e8/*/aggregate/aggregate_metrics.json` |
| **9. Ablation study** | placeholder | notebook 08 sections 9 and 11 |
| **10. State-of-the-art comparison** | placeholder | `comparison/comparison_table.csv`, notebook 09 |

## Figures

| Figure | Status | Replace with |
|---|---|---|
| 1. Overall workflow | **final** | - |
| 2. Architecture | **final** | - |
| 3. Convergence curves (2 panels) | placeholder | `proposed_v2_e8/M4_denoised/seed_0/plots/` |
| 4. Ablation fidelity | placeholder | `proposed_v2_e8/comparison_ladder/*.png` |
| 5. SOTA comparison + CIs (2 panels) | placeholder | `comparison/comparison_vs_*/`, `bootstrap_cis.csv` |

## Citation placement, as instructed

- `[1]`-`[5]` Introduction only
- `[6]`-`[20]` Literature Survey prose, in order across the 7 paragraphs
- `[21]`-`[30]` Table 1, in order, plus the `Proposed` row
- `[11]`, `[18]`, `[21]` re-appear in sections 4.1 and 4.6 only to name the
  three re-implemented comparison models

## Claims to check rather than fill

1. **4.5 / 4.6** - if the learned item loadings show near-zero dispersion
   around uniform weighting, the head never left its initialisation and the
   interpretability claim must be **withdrawn**, not reported.
2. **4.1** - the three comparison models are re-implementations, not
   reproductions. That sentence must stay.
3. **4.3 / 4.5** - the weak-only symptom subset is marked "not measurable"
   because eleven symptoms have no gold labels anywhere. Keep it.
4. **Future work** - the matched-data caveat stands until the baselines also
   see weak labels.
