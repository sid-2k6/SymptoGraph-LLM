# Placeholder index for SymptoGraph-LLM_Paper.docx

Every performance number in the paper is written as `X.XXX` (or
`X.XXX ± X.XXX`). Search the document for `X.XXX` to find them all. Dataset
counts, corpus-overlap measurements and the labelling-model noise channel are
**measured quantities and are already real** — do not overwrite them.

Regenerate the document after editing the builder:

```
python scripts/build_paper.py
```

## Tables

| Table | Fills from | Produced by |
|---|---|---|
| 1. Literature comparison | no numbers | — |
| 2. Corpus overlap | already real | `src/symptograph/preprocess.py` docstring |
| 3. Dataset partitioning | already real | `processed/manifest.json` |
| 4. Shared configuration | already real | `TrainConfig` / `ProposedConfig` |
| **5. Baselines and comparison models** | `comparison/comparison_table.csv` and `comparison/test_comparison.csv` | notebook 09 §7 |
| **6. Ablation ladder and controls** | `proposed_v2_e8/comparison_ladder/` aggregates | notebook 08 §9, §11 |
| **7. Per-task confidence intervals** | `proposed_v2_e8/comparison_ladder/bootstrap_cis.csv` | notebook 08 §9 |
| **8. Paired bootstrap** | `comparison/paired_vs_proposed.csv` and `proposed_v2_e8/comparison_ladder/paired_bootstrap.csv` | notebook 08 §10, notebook 09 §8 |

## Figures

| Figure | Status | Replace with |
|---|---|---|
| 1. Architecture | **final** | — |
| 2. Against baselines | placeholder | `comparison/comparison_vs_*/comparison.png` |
| 3. Ablation ladder | placeholder | `proposed_v2_e8/comparison_ladder/*.png` |
| 4. Item loadings | placeholder | `proposed_v2_e8/item_loadings_*.png` |
| 5. Training dynamics | placeholder | `proposed_v2_e8/M4_denoised/seed_0/plots/` |

## Prose placeholders

Section 4 paragraphs carry `X.XXX` inline for: the headline symptom
macro-AUROC and its margin over the strongest comparison model (4.1); the
largest and smallest contributing ablation step (4.2); the count of established
differences and the suicide-risk CI width (4.3); the largest and smallest item
loading and the dispersion around uniform weighting (4.4); the count of runs
above the memorisation threshold and the mean loss ratio (4.5). The abstract and
conclusion repeat the headline figures.

## Claims that must be checked before submission, not filled in

1. **Section 4.4** — if the loading dispersion is near zero the head never left
   its initialisation, and the interpretability claim must be **withdrawn**
   rather than reported. The paragraph is written to accommodate either outcome.
2. **Section 4.1** — the comparison models are re-implementations, not
   reproductions. That sentence must stay.
3. **Section 4.6** — the matched-data caveat must stay until a run with weak
   labels enabled for the baselines exists.
