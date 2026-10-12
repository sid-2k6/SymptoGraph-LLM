# data/

Raw datasets for SymptoGraph-LLM. **Not committed to git**: the collection is a few GB, DepressionEmo may not be redistributed, and Reddit posts should not be republished in a public repo.

## Getting the data

From the repo root:

```bash
bash scripts/download_data.sh
```

Options:

```bash
DATA_DIR=/path/to/elsewhere bash scripts/download_data.sh   # different target folder
SKIP=imhi,lost bash scripts/download_data.sh                # skip datasets
RMHD_FILTER='depression|anxiety' bash scripts/download_data.sh  # subset of subreddits
JOBS=4 bash scripts/download_data.sh                        # fewer parallel downloads
```

The script is resumable, and every Reddit Mental Health file is checked against its Zenodo checksum.

To confirm a copy is complete (useful after copying to Google Drive, where folder sizes are hidden):

```bash
python scripts/verify_data.py --data data        # add --deep for a full row count
```

To download straight into Google Drive instead, open `notebooks/download_data_to_drive.ipynb` in Colab.

## Shrinking the Reddit Mental Health Dataset

The Zenodo CSVs carry 350 columns, of which 346 are precomputed features this project does not use:

```bash
python scripts/slim_rmhd.py   # 3.14 GB -> 0.41 GB, 87% smaller
```

Add `--delete-src` to remove each raw CSV as it converts.

## Contents

| Folder | Dataset | Contents | Role | Source | Licence |
|---|---|---|---|---|---|
| `depressionemo/` (15M) | DepressionEmo | 6,037 posts (4,225 / 906 / 906), 8 symptom labels | Gold symptom concepts (N1), weak-label calibration (N3) | github.com/abuBakarSiddiqurRahman/DepressionEmo | Research only; All Rights Reserved |
| `reddit_mental_health_slim/` (392M) | Reddit Mental Health Dataset — slim | 108 gzip CSVs, 1,107,302 posts, 28 subreddits | Disorder heads (N2), unlabeled pool (N3), temporal test | Zenodo 3941387 | ODC-PDDL |
| `reddit_mental_health/` (3.0G) | Reddit Mental Health Dataset — raw | Same posts plus 346 precomputed tf-idf/readability columns | Optional; not used by the model | Zenodo 3941387 | ODC-PDDL |
| `depseverity/` (1.6M) | DepSeverity | 3,553 posts, 4 severity levels | Ordinal severity head | github.com/usmaann/Depression_Severity_Dataset | Research |
| `cssrs_reddit_500/` (3.6M) | CSSRS-Reddit-500 | 500 users, 5 risk levels | Suicide-risk severity, abstention (N6) | github.com/manasgaur/Knowledge-aware-...-Suicide-Risk... | Research |
| `imhi/` (119M) | IMHI (MentaLLaMA) | 20 train / 18 test files with explanations | Rationale SFT and evaluation (N4) | github.com/SteveKGYang/MentalLLaMA | MIT |
| `dreaddit/` (2.2M) | Dreaddit | 3,553 segments (train + test parquet) | Out-of-domain test | huggingface.co/datasets/andreagasparini/dreaddit | Research |
| `irf/` (2.5M) | IRF | 3,522 posts + evidence spans | Extra concepts, faithfulness spans | github.com/drmuskangarg/Irf | Research |
| `lost/` (5.2M) | LoST | 3,251 posts + evidence spans | Low self-esteem concept | github.com/drmuskangarg/LoST | Research |

All row counts above were verified after download.

## Main files

- `depressionemo/Dataset/{train,val,test}.json` — one JSON object per line, with an `emotions` list
- `depseverity/Reddit_depression_dataset.csv` — columns `text`, `label` (minimum / mild / moderate / severe)
- `cssrs_reddit_500/anonymized_dataset/500_Reddit_users_posts_labels.csv` — columns `User`, `Post`, `Label`
- `reddit_mental_health_slim/<subreddit>_<period>.csv.gz` — columns `subreddit`, `author`, `date`, `post`; periods are `2018`, `2019`, `pre` and `post` (meaning pre- and mid-COVID)
- `imhi/train_data/`, `imhi/test_data/` — instruction data with human-checked explanations
- `irf/{train,val,test}_data.csv`, `lost/LoSTdataset.csv` — posts with annotated spans

Read a slim file with:

```python
import pandas as pd
df = pd.read_csv('data/reddit_mental_health_slim/depression_pre.csv.gz', compression='gzip')
```

## Terms of use

- Keep this folder private. DepressionEmo is research-only and All Rights Reserved; it forbids use in web apps, mobile apps or commercial tools without written permission.
- Paraphrase any example posts quoted in the paper, and do not attempt to identify users.
- Run LLM weak labelling with a locally hosted open model rather than a third-party API.
- See `docs/Datasets.md` for the full dataset plan and the symptom mapping.
