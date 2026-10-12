# SymptoGraph-LLM: Open-Source Dataset Plan

All datasets below can be downloaded immediately, with no data-use agreement, email approval, or access request. Availability was checked on 12 October 2026.

## 1. What changed

| Original (needs access request) | Replaced with (open) | Why it is a fair substitute |
|---|---|---|
| **PsySym** (symptom labels, 7 disorders) | **DepressionEmo** + **in-house GAD-7 gold set** (Section 4) | DepressionEmo gives expert-designed, multi-label depression symptom labels; the small in-house set covers anxiety symptoms, which no open dataset labels |
| **PRIMATE** (PHQ-9 items; needs a signed form) | **DepressionEmo** (mapped to PHQ-9 items, Section 3) | Its 8 labels map to 5 of 9 PHQ-9 items plus irritability; the remaining items come from calibrated LLM weak labels (N3) |
| **BDI-Sen** (BDI-II sentences) | Dropped | Covered by DepressionEmo + weak labels |
| **SWMH** (disorder labels + unlabeled pool) | **Reddit Mental Health Dataset** (Low et al., 2020) | Larger, public-domain licence, covers depression, anxiety, bipolar, suicide and 11 more conditions, plus 11 control subreddits |
| **DepSeverity** | **Kept**, already open | — |
| **IMHI** | **Kept**, already open (MIT) | — |
| **Dreaddit** | **Kept**, already open | — |
| — | **Added: CSSRS-Reddit-500** | Open 5-level suicide-risk labels, for the severity head and abstention tests |

## 2. Final dataset list

| Dataset | Size | Labels | Role in SymptoGraph-LLM | Licence / terms | Download |
|---|---|---|---|---|---|
| **DepressionEmo** (Rahman et al., 2024, *J. Affective Disorders*) | 6,037 Reddit posts (4,225 / 906 / 906 split) | 8 multi-label: anger, brain dysfunction (forget), emptiness, hopelessness, loneliness, sadness, suicide intent, worthlessness | **Gold symptom concepts** for the bottleneck (N1); calibration set for weak labels (N3) | Free for research; no apps or commercial tools without permission | `github.com/abuBakarSiddiqurRahman/DepressionEmo` (`Dataset/`) |
| **Reddit Mental Health Dataset** (Low et al., 2020, *JMIR*) | ~3.1 GB, 28 subreddits (15 mental health, others control), pre- and mid-COVID periods | Subreddit = disorder label (weak) | **Disorder heads** (N2); **unlabeled pool** for LLM weak labelling (N3); temporal generalisation test | ODC-PDDL (public domain) | Zenodo record 3941387 (DOI 10.17605/OSF.IO/7PEYQ) |
| **DepSeverity** (Naseem et al., 2022, WWW) | 3,553 posts | minimum 2,587 / mild 290 / moderate 394 / severe 282 | **Ordinal severity head** (CORAL) | Public research data | `github.com/usmaann/Depression_Severity_Dataset` |
| **CSSRS-Reddit-500** (Gaur et al., 2019, WWW) | 500 users | Supportive 108 / Indicator 99 / Ideation 171 / Behavior 77 / Attempt 45 | **Suicide-risk severity**; testing abstention on high-risk cases (N6) | Public, anonymised | `github.com/manasgaur/Knowledge-aware-Assessment-of-Severity-of-Suicide-Risk-for-Early-Intervention` |
| **IMHI** (Yang et al., 2024, WWW, MentaLLaMA) | ~105k instruction samples | Label + human-checked explanations across 10 sources | **Rationale LLM SFT and explanation evaluation** (N4) | MIT | `github.com/SteveKGYang/MentalLLaMA` (`train_data/`, `test_data/`) |
| **Dreaddit** (Turcan & McKeown, 2019) | 3,553 segments | Stress yes/no | **Auxiliary stress label on the DepSeverity posts** (see Section 10) | Public research data | Hugging Face `andreagasparini/dreaddit` |
| *Optional:* **IRF** (Garg et al., ACL 2023) | 3,522 posts | Thwarted belongingness, perceived burdensomeness + explanation spans | Extra concepts (suicide-risk factors) + span-level **faithfulness** checks | Free for research | `github.com/drmuskangarg/Irf` |
| *Optional:* **LoST** (Garg et al., 2023) | Reddit posts | Low self-esteem + spans | Extra concept for "worthlessness" | Free for research | `github.com/drmuskangarg/LoST` |

## 3. Unified symptom ontology mapping (open data version)

| Unified symptom | PHQ-9 / GAD-7 item | DepressionEmo label | Source of supervision |
|---|---|---|---|
| Depressed mood | PHQ-2 | sadness, emptiness, hopelessness | **Gold** |
| Worthlessness / guilt | PHQ-6 | worthlessness (+ LoST) | **Gold** |
| Concentration problems | PHQ-7 | brain dysfunction (forget) | **Gold** |
| Suicidal thoughts | PHQ-9 | suicide intent (+ CSSRS-500) | **Gold** |
| Irritability | GAD-6 | anger | **Gold** |
| Loneliness / disconnection | (extra concept) | loneliness (+ IRF) | **Gold** |
| Anhedonia | PHQ-1 | — | Weak (LLM) |
| Sleep disturbance | PHQ-3 | — | Weak (LLM) |
| Fatigue | PHQ-4 | — | Weak (LLM) |
| Appetite change | PHQ-5 | — | Weak (LLM) |
| Psychomotor change | PHQ-8 | — | Weak (LLM) |
| Nervousness, uncontrollable worry, excessive worry, trouble relaxing, restlessness, fear | GAD-1 to 5, 7 | — | Weak (LLM) + **in-house gold** |

This split strengthens the N3 claim: the paper can report weak-label quality separately for symptoms that have gold labels and symptoms that do not.

## 4. In-house anxiety gold set (fills the only real gap)

No open dataset labels GAD-7 symptoms, so build a small one:

- **Sample:** 600 posts from `anxiety`, `healthanxiety` and `socialanxiety` in the Reddit Mental Health Dataset, plus 200 control posts.
- **Labels:** the 7 GAD-7 items plus the 5 weak-only PHQ-9 items, each marked present or absent with an evidence span.
- **Annotators:** two people annotate independently, then resolve disagreements; report Cohen's κ per item.
- **Effort:** about 15–20 hours each.
- **Uses:** (a) calibrating LLM weak labels (temperature scaling, N3), (b) testing symptom F1 on the weak-only symptoms, (c) a releasable resource contribution.

## 5. Splits

- **DepressionEmo, DepSeverity, IMHI:** use the official splits if given; otherwise 70/15/15, stratified.
- **Reddit Mental Health Dataset:** train on pre-COVID posts and test on mid-COVID posts. This gives a temporal-shift test for free.
- **User-disjoint splitting** wherever author IDs exist, so no user appears in both train and test.
- **Out-of-domain tests:** Dreaddit, CSSRS-500, and the IMHI test sets. None are used for training.

## 6. Metric changes

- **Symptom F1:** report separately for gold-labelled symptoms (DepressionEmo) and for weak-only symptoms (in-house set).
- **Severity:** ordinal MAE and weighted F1 on DepSeverity; macro-F1 on CSSRS-500.
- **Explanation faithfulness:** comprehensiveness and sufficiency (ERASER), plus span overlap with IRF and LoST human spans.

## 7. Terms to respect

- **No apps:** DepressionEmo forbids web apps, mobile apps or commercial tools without written permission, so any demo should stay a research prototype.
- **Reddit content:** paraphrase any example posts quoted in the paper, and do not try to identify users.
- **Weak labelling:** run it with a locally hosted open model (e.g. Llama-3 / Qwen via vLLM), not a third-party API.
- **Ethics:** check whether your institution requires IRB review or a formal exemption for secondary analysis of public data.

## 8. Download commands

**Into Google Drive (Colab):** open
[`notebooks/01_download_data_to_drive.ipynb`](https://colab.research.google.com/github/sid-2k6/SymptoGraph-LLM/blob/main/notebooks/01_download_data_to_drive.ipynb)
in Colab. It downloads on Colab's network and writes the result to your Drive,
so nothing transfers through your own machine.

**Locally:**

```bash
bash scripts/download_data.sh          # everything into ./data
python scripts/slim_rmhd.py            # Reddit MH: 3.14 GB -> 0.41 GB
```

Options: `DATA_DIR`, `SKIP`, `RMHD_FILTER`, `JOBS` (see `data/README.md`).

<details>
<summary>Equivalent manual commands</summary>


```bash
mkdir -p data && cd data
git clone --depth 1 https://github.com/abuBakarSiddiqurRahman/DepressionEmo.git
git clone --depth 1 https://github.com/usmaann/Depression_Severity_Dataset.git
git clone --depth 1 https://github.com/manasgaur/Knowledge-aware-Assessment-of-Severity-of-Suicide-Risk-for-Early-Intervention.git cssrs
git clone --depth 1 https://github.com/SteveKGYang/MentalLLaMA.git
git clone --depth 1 https://github.com/drmuskangarg/Irf.git
git clone --depth 1 https://github.com/drmuskangarg/LoST.git
# Reddit Mental Health Dataset (~3.1 GB)
python -c "import json,urllib.request as u;[print(f['links']['self']) for f in json.load(u.urlopen('https://zenodo.org/api/records/3941387'))['files']]" > rmhd_urls.txt
mkdir -p rmhd && (cd rmhd && xargs -n1 -P4 wget -q < ../rmhd_urls.txt)
# Dreaddit
python -c "from datasets import load_dataset; load_dataset('andreagasparini/dreaddit').save_to_disk('dreaddit')"
```

</details>

## 9. Quick-access options (not required)

These can be added later if time allows:

- **SWMH:** on Hugging Face (`AIMH/SWMH`, CC BY-NC 4.0) with automatic approval. Log in and click accept, and access is immediate.
- **ReDSM5** (Bao et al., 2025): 1,484 posts with sentence-level labels for all nine DSM-5 depression symptoms by a psychologist, plus rationales. This is the closest match to PsySym, but it needs an emailed agreement; only a 25-post preview is open.
- **PRIMATE:** needs a signed agreement form, after which the link is sent immediately.

## 10. Corpus overlap and leakage control

Running `notebooks/02_preprocess_and_split.ipynb` over the downloaded data turned
up substantial overlap between these public corpora. Measured, not assumed:

| Finding | Effect on the design |
|---|---|
| **DepSeverity is a re-annotation of Dreaddit** - the same 3,526 posts, severity labels in one and binary stress labels in the other | Dreaddit cannot be the out-of-domain test set. Both label sets now attach to one post pool (`severity_stress`), and stress becomes an auxiliary target |
| **LoST and IRF share 3,089 posts** | Merged into one pool (`concept_spans`) and split once |
| **IMHI re-packages the IRF and Dreaddit posts** behind a `"Post: "` prefix and truncates them | Exact-match deduplication misses this; matching uses a normalised 200-character prefix key |
| **23.3% of Reddit Mental Health authors (192,893 of 826,961) appear in more than one file** | Reddit posts are split author-disjoint, never at random |
| **85,659 Reddit posts are duplicated across the period files** | Removed, so the pre/mid-COVID temporal split stays clean |
| Of 46,542 rows in the labelled corpora, only **19,762 are distinct posts** | Splitting happens once, globally, over unique posts |

**External evaluation.** Because Dreaddit is no longer usable as a held-out
domain, external evaluation uses the IMHI subsets that overlap nothing else in
the collection: `swmh`, `CAMS`, `t-sid`, `loneliness` and `DR`. These are
test-only by construction.

### Processed outputs

`notebooks/02_preprocess_and_split.ipynb` writes to `processed/`:

| File | Contents |
|---|---|
| `labelled.parquet` | 19,762 unique posts with symptom, severity, risk, stress, span and explanation labels |
| `weak_pool.parquet` | 1,017,959 Reddit posts for LLM weak labelling (N3), author-disjoint, 427 MB |
| `label_coverage.csv` | Which of the 22 symptoms have gold labels and which need weak labels |
| `manifest.json` | Counts, settings and the leakage audit |

Symptom columns use `1` present, `0` absent and `-1` not annotated by that
source. Only non-negative entries should contribute to the concept loss, since
treating `-1` as a negative would teach the model that unannotated symptoms are
absent.

**Gold coverage.** 11 of the 22 unified symptoms have gold labels. The other 11,
mostly GAD-7 anxiety items, have no open gold data, which is precisely the gap
LLM weak supervision (N3) and the small in-house anxiety set (Section 4) address.
Reporting symptom F1 separately for the two groups is what makes that claim
measurable.

The run peaks near 1.7 GB of RAM because the Reddit pool is streamed file by
file; a single in-memory pass needs about 16 GB and will not fit a free Colab
runtime.
