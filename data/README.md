# data/

Raw datasets for SymptoGraph-LLM. **They are not committed to git**: they total about 3.1 GB, DepressionEmo may not be redistributed, and the Reddit posts should not be republished in a public repo.

Re-create this folder from the repo root:

```bash
bash scripts/download_data.sh
```

The script skips anything already downloaded. It checks every Reddit Mental Health Dataset file against its Zenodo MD5 checksum.

| Folder | Dataset | Contents | Role | Source | Licence |
|---|---|---|---|---|---|
| `depressionemo/` (15M) | DepressionEmo | 6,037 posts (4,225 / 906 / 906) | Gold symptom concepts (N1), weak-label calibration (N3) | github.com/abuBakarSiddiqurRahman/DepressionEmo | Research only; All Rights Reserved |
| `reddit_mental_health/` (3.0G) | Reddit Mental Health Dataset (Low et al., 2020) | 108 CSVs, 28 subreddits, 2018 / 2019 / pre- and mid-COVID | Disorder heads (N2), unlabeled pool (N3), temporal test | Zenodo 3941387 | ODC-PDDL |
| `depseverity/` (1.6M) | DepSeverity | 3,553 posts, 4 severity levels | Ordinal severity head | github.com/usmaann/Depression_Severity_Dataset | Research |
| `cssrs_reddit_500/` (3.6M) | CSSRS-Reddit-500 | 500 users, 5 risk levels | Suicide-risk severity, abstention (N6) | github.com/manasgaur/Knowledge-aware-Assessment-of-Severity-of-Suicide-Risk-for-Early-Intervention | Research |
| `imhi/` (119M) | IMHI (MentaLLaMA) | 20 train / 18 test files | Rationale SFT and evaluation (N4) | github.com/SteveKGYang/MentalLLaMA | MIT |
| `dreaddit/` (2.2M) | Dreaddit | train + test parquet | Out-of-domain test | huggingface.co/datasets/andreagasparini/dreaddit | Research |
| `irf/` (2.5M) | IRF | 3,522 posts + spans | Extra concepts, faithfulness spans | github.com/drmuskangarg/Irf | Research |
| `lost/` (5.2M) | LoST | 3,251 posts + spans | Low self-esteem concept | github.com/drmuskangarg/LoST | Research |

Main files:

- `depressionemo/Dataset/{train,val,test}.json`: one JSON object per line, with an `emotions` list
- `depseverity/Reddit_depression_dataset.csv`: columns `text`, `label`
- `cssrs_reddit_500/anonymized_dataset/500_Reddit_users_posts_labels.csv`: columns `User`, `Post`, `Label`
- `reddit_mental_health/<subreddit>_<period>_features_tfidf_256.csv`: 350 columns (`subreddit`, `author`, `date`, `post`, plus precomputed features)
- `imhi/train_data/`, `imhi/test_data/`: instruction data with explanations
- `irf/{train,val,test}_data.csv`, `lost/LoSTdataset.csv`
