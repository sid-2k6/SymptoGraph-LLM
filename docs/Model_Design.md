# SymptoGraph-LLM: model design and change log

The authoritative record of what the model does and why, for the write-up stage.
Every change below is in `src/symptograph/proposed.py` unless stated otherwise.

## 1. Architecture

```
post
 └─ encoder (MentalRoBERTa)  ──►  token states H (B, L, D)
     ├─ mean-pool ─────────────►  pooled (B, D)        [L0_flat uses this directly]
     └─ N1 concept bottleneck
          symptom queries Q (K, D), initialised from the sentence embeddings
          of the clinical definitions
          att   = softmax_L( H · Qᵏ / √D )             (B, L, K)
          Z     = Σ_L att · H                          (B, K, D)   per-symptom evidence
          c_log = Σ_D ( Z ⊙ Wᵏ ) + bᵏ                  (B, K)      ONE vector per symptom
            │
            ├─ N2 graph: A = relu(A₀ + ΔA), symmetric-normalised, 2 residual
            │   LayerNorm layers over [K symptoms ‖ D disorders]
            │   c_log ← c_log + refine(X_sym);  graph_rep = pool(X)
            │
            └─ concept-gated pooling: g = σ(c_log);  summary = Σ_k g_k·Z_k / Σ_k g_k
          feat = [ c_log ‖ graph_rep ‖ summary ]
            └─ heads: severity (4) · risk (5) · stress (1) · N6 log-variance
```

`K = 22` symptoms, `D = 14` disorders.

## 2. The ablation ladder

| Rung | Adds | Novelty |
|---|---|---|
| `L0_flat` | encoder + flat heads from `pooled` | — |
| `L1_bottleneck` | concept bottleneck | N1 |
| `L2_graph` | symptom–disorder graph | N2 |
| `L3_weak` | confidence-weighted weak labels | N3 |
| `L4_contrastive` | symptom-profile contrastive loss | N5 |
| `L5_full` | learned uncertainty + abstention | N6 |

`L5_full` is the proposed model. N4 (rationale generation) is out of scope here.

## 3. Change log

Each entry records what changed, why, and what was measured. Entries marked
**plan change** alter a claim in the original proposal and must be reflected in
the write-up.

### 3.1 Graph adjacency: ReLU before normalising — `ARCH_VERSION 2`

`A = A₀ + ΔA` with `ΔA` learnable and unconstrained. Once `ΔA` went negative the
row sums crossed zero, and `deg.clamp(min=1e-9) ** -0.5` evaluated to 31,623,
scaling the adjacency by ~10⁹.

**Measured:** `ΔA = -0.05` already gave `|A_norm| = 9.5 × 10⁸`; training losses
reached 1.6 × 10¹³ in 7 of 18 runs, all of them graph rungs.

**Fix:** `relu` the adjacency so edges stay non-negative and `deg ≥ 1` after
adding the identity; degree clamped at 1.0; LayerNorm after each residual GNN
step. `|A_norm|` is now bounded by 1.0 for `ΔA ∈ [-5, +10]`.

**Measured after:** max loss 2.92 over 6 epochs × 3 seeds, zero explosions.

### 3.2 Concept bottleneck: one scoring vector per symptom — `ARCH_VERSION 3`

**plan change (correction, not a redesign).** The bottleneck scored every symptom
with a single shared `Linear(D, 1)`:

```
c_log[k] = (w · z_k + b) × scale_k + bias_k
```

so all 22 symptoms shared one weight vector and each had only a scalar rescale
of one shared direction. The proposal specified a per-symptom `w_k`; the
implementation did not match it.

**Measured:** every bottleneck rung sat at symptom F1 = 0.000 while `L0_flat`,
which has a proper `Linear(D, K)`, reached 0.239.

**Fix:** `sym_w` of shape `(K, D)` — 2,816 parameters instead of 129.

**Measured after** (BERT-mini, 2,500 rows, 4 epochs, validation):

| rung | symptom F1 | AUROC | severity QWK | composite |
|---|---|---|---|---|
| `L0_flat` | 0.473 | 0.707 | 0.148 | 0.207 |
| `L1_bottleneck` | 0.447 | 0.607 | 0.204 | 0.253 |
| `L2_graph` | 0.438 | 0.640 | 0.323 | 0.254 |

### 3.3 Class imbalance and decision thresholds

Symptom prevalence ranges from 19% (`concentration_problems`) to 77%
(`depressed_mood`). Unweighted BCE plus a fixed 0.5 threshold let a well-ranked
model score F1 ≈ 0 because every probability sat below the cut.

**Fix:** per-symptom BCE positive weight `n_neg / n_pos`, capped at 10
(`metrics.positive_weights`), and per-symptom thresholds tuned on **validation
only** and applied unchanged to test (`metrics.tune_thresholds`). Applied
identically to every rung **and to the baselines**, so the comparison is unaffected.

**Measured** (`L1_bottleneck`, BERT-mini, 900 rows, 3 epochs, validation):

| setting | symptom F1 | AUROC | selection score |
|---|---|---|---|
| no pos-weight, threshold 0.5 | 0.336 | 0.573 | 0.303 |
| pos-weight + tuned thresholds | **0.603** | 0.581 | **0.395** |

AUROC barely moves, which is the expected signature: thresholds do not change
ranking quality, they change where the decision is taken. The F1 gain is real
but it is a *calibration* fix, not a representation improvement, and should be
described that way.

### 3.4 N6 uncertainty trained on detached features — `ARCH_VERSION 4`

**plan change.** The original design attenuated the primary loss with the learned
variance (`exp(-s)·BCE + s/2`). That lets the model reduce its loss by declaring
uncertainty instead of fitting, so the abstention rung could not beat the rung
below it by construction.

**Fix:** the variance head now predicts its own squared error on **detached**
features. It cannot trade accuracy for confidence, and abstention is still
available from the same quantity.

**Measured** (L4 vs L5, BERT-mini, 900 rows, 3 epochs, validation selection
score):

| variant | selection score | Δ vs L4 |
|---|---|---|
| `L4` (no uncertainty) | 0.3905 | — |
| `L5` attenuating (old) | 0.3879 | −0.0026 |
| `L5` detached (new) | 0.3877 | −0.0028 |

**The hypothesis was wrong.** Attenuation was not costing accuracy at this scale;
both variants sit within noise of L4. The detached form is kept because it is
sound by construction, not because it was measured to help. Report it that way.

### 3.5 Concept-gated pooling — `ARCH_VERSION 4`

**plan change, and it weakens an interpretability claim.** The original design
was a *strict* bottleneck: severity, risk and stress were predicted from the 22
concept logits alone. That starves those heads — `L0_flat` gives them 768 dims,
`L1` gave them 22.

The heads now also receive `Σ_k σ(c_k)·Z_k / Σ_k σ(c_k)`, the concept-gated
average of the per-symptom evidence vectors.

- **Still bottleneck-like:** every path to the heads is gated by a concept
  activation, and the evidence `Z_k` is what the symptom query attended to.
- **No longer strictly bottlenecked:** `Z_k` is a `D`-dimensional encoder
  representation, so information can reach the heads without being fully
  summarised by the scalar `c_k`.

The paper must call this a **concept-gated** bottleneck, not a strict one. The
strict variant is still available (`concept_gated_pooling=False`) and should be
reported as an ablation so the interpretability/accuracy trade-off is explicit.

### 3.6 Training schedule

`epochs` 8 → 10, cosine schedule with warmup, early stopping with patience 4 on
the validation selection score. Identical across rungs.

### 3.7 Model selection metric

`composite` mixed symptom macro-F1 with both severity QWK **and** suicide-risk
QWK. Risk has 340 training and 74 test rows and swings by ±0.15 across seeds, so
selecting checkpoints on it partly selects on noise.

**Fix:** a new `selection_score` = mean(symptom macro-F1, symptom macro-AUROC,
depression-severity QWK) drives checkpoint selection. `composite` is still
computed and reported unchanged, and risk QWK is still reported in full.

### 3.8 Mock weak labels refused

`weak_label.MockLabeller` is a keyword heuristic for testing the pipeline with no
GPU. Its output was being written to `weak_labels.parquet` and consumed by N3.
Training N3 on it is training on noise, which flattens every rung from `L3_weak`
upward.

**Fix:** `load_frames` raises if the parquet's `labeller` column contains `mock`,
unless `allow_mock_weak_labels=True`.

### 3.9 Ordinal class weights — the single largest measured gain

Depression severity is 72% `minimum` (1767 / 205 / 288 / 196) and risk is
similarly skewed. Unweighted cross-entropy made both heads predict the majority
class, so QWK sat at exactly 0.000 in nearly every run — which also dragged
`composite` down, since QWK is one of its three terms.

**Fix:** inverse-frequency class weights, capped at 10, for both ordinal heads
(`proposed.class_weights`). Severity weights become `[0.35, 3.00, 2.13, 3.13]`.

**Measured** (`L1_bottleneck`, BERT-mini, 2,000 rows, 3 epochs, validation):

| setting | severity QWK | severity MAE | severity macro-F1 | selection score |
|---|---|---|---|---|
| unweighted | +0.005 | 0.428 | 0.219 | 0.426 |
| class-weighted | **+0.509** | 0.386 | 0.294 | **0.587** |

This is a real modelling fix rather than a calibration trick: the head was
degenerate before and is now discriminative. Risk QWK stayed at 0.000 in this
test because the subset held only ~50 risk rows; at full scale it has 340.

### 3.10 Weak-label resume was not labeller-aware

`run_labelling` resumed on `post_id` alone while already recording a `labeller`
field per row. Re-running notebook 04 with `MOCK = False` therefore found every
post "already done" and silently kept the mock labels. The user hit exactly this:
the mock guard still fired after a real re-run.

**Fix:** resume is keyed on `(post_id, labeller)`. A log written entirely by a
different labeller is archived to `<name>.<labeller>.bak` and labelling starts
fresh; a mixed log re-labels the foreign rows and reports how many. The chosen
labeller is recorded in `run_stats`.

**Verified:** mock → mock resumes correctly (55 done, 0 to go); mock → real
archives the old log and relabels all 55, after which the mock guard passes.

### 3.11 Weak labelling was resumable only within a session

Resume reads the JSONL log, and notebook 04 wrote that log to
`/content/weak_labels` with a single `rsync` to Drive in the *final* cell. Colab
wipes `/content` when a runtime is recycled, so a disconnect at hour 4 of a
5-hour labelling run destroyed the log and the next session restarted from zero.
The in-notebook claim "if the session drops, just re-run this cell" was true only
for an interrupted cell on a live runtime, not for a lost runtime.

**Fix:** `run_labelling` takes `mirror_to` and snapshots the log to Drive every
`mirror_every` batches (20, i.e. ≤640 posts ≈ 7 GPU-minutes at risk). The
snapshot is atomic (`copy` to `.partial`, then `os.replace`) so a death
mid-upload leaves the previous complete snapshot, and it is taken in a `finally`
so a `KeyboardInterrupt` also flushes. On start, a mirror with more rows than the
local log is copied back before resume is computed. `sink.flush()` is now
followed by `os.fsync`.

The log stays a local write target rather than appending straight to Drive:
Drive's FUSE mount re-uploads the whole file on each flush, which costs tens of
MB per batch once the log is large.

`LABELLER_MODEL` is also now pinned in the notebook instead of being chosen from
available GPU memory. Because resume is keyed on the labeller (§3.10), a session
that landed on a T4 would have picked the AWQ model, treated the existing log as
foreign, archived it, and relabelled everything. The notebook now raises if the
GPU cannot hold the pinned model.

**Verified:** simulated runtime loss — interrupt after 40/200 rows leaves 40 rows
on the mirror, `/content` is then deleted entirely, and a fresh call recovers the
40 rows from Drive and reports "40 already labelled, 160 to go", finishing at 200
with no `.partial` residue. A third call reports 0 to go.

### 3.12 Weak rows were concatenated into the training frame — the reason the ladder inverted at L3

`load_frames` appended the weak rows to the training frame. For L3–L5 that
changed the training protocol itself, not just the data:

| | L0–L2 | L3–L5 (concatenated) |
|---|---|---|
| train rows | 12,911 | 18,911 |
| optimiser steps / epoch | 404 | 591 |
| cosine schedule spans | 4,040 steps | 5,910 steps |
| rows per batch with a risk label | 0.84 | 0.58 |
| `pos_weight` computed over | gold only | gold + weak |

So the upper rungs ran a different learning-rate schedule and a thinner gold
batch than the rungs they were being compared against. The observed damage
tracks label sparsity, which is the signature of per-batch dilution rather than
of weak-label noise:

| task | train rows | L2 → L3 | delta |
|---|---|---|---|
| symptoms | 6,227 | 0.896 → 0.891 | −0.005 |
| stress | ~3,526 | 0.907 → 0.905 | −0.002 |
| severity (depression) | 2,456 | 0.480 → 0.443 | −0.037 |
| risk | 340 | 0.438 → 0.249 | **−0.189** |

**Fix:** `weak_as_stream=True` makes the weak rows a separate loader, drawn once
per optimiser step and contributing only the symptom terms (`weak_pass=True` in
`compute_loss` skips severity/risk/stress, the graph L1 — a *parameter* penalty
that would otherwise be counted twice — and the contrastive loss, whose
symptom-profile positives are not meaningful on a row where only 11 of 22
symptoms are observed). Steps per epoch, the schedule, `pos_weight` and the gold
batch composition all return to their L0–L2 values. The legacy path is kept so
the earlier run can be reproduced.

Two consequences worth recording:

- Stream cost is set by `weak_batch_size`, not by the size of the weak pool, so
  **all ~50,000 real Qwen labels cost the same as the 6,000 used before**.
- `ARCH_VERSION` stays at 5 and `weak_as_stream` enters the fingerprint **only
  when `use_weak` is true**. A rung with `use_weak=False` executes identical
  code either way, so the existing L0–L2 checkpoints and results remain valid
  and directly comparable. This is what lets notebook 07 retrain 9 runs instead
  of 18.

**Verified** (CPU, synthetic data): stream mode leaves train rows (300), steps
per epoch (19), `pos_weight` mean (1.08592) and risk rows per batch (0.640)
bit-identical to a `use_weak=False` run, while legacy mode inflates steps
19 → 75 and thins risk density 0.640 → 0.160. The weak stream runs on all 19
steps per epoch and `weak_every=2` halves that to 10. Notebook 07 additionally
asserts at runtime that the per-epoch encoder LR matches `L2_graph`'s to within
1e-9, which is only possible if steps per epoch and total steps agree.

### 3.13 The metric set cannot credit N3 directly

`symptom_macro_auroc[all]` equals `symptom_macro_auroc[gold_supervised]` in
every run, because the 11 weak-only symptoms have **no gold labels in any
split**. Weak supervision can therefore only show up as transfer to the 11 gold
symptoms; its direct effect on the symptoms it actually supervises is
unmeasurable with current data. This is a measurement gap, not a model
weakness, and it caps how large an N3 effect any experiment here can detect.
Closing it requires the 800-post in-house anxiety gold set (notebook 04,
section 8), not more training.

## 4. Infrastructure fixes (no effect on model behaviour)

| Issue | Cause | Fix |
|---|---|---|
| Encoder silently substituted | `mental/mental-roberta-base` is gated; a 403 fell back to `roberta-base` while still labelling the run MentalRoBERTa | `strict_encoder` raises; `check_encoder_access` pre-flights; fallback recorded in `config.json` |
| GPU resume crashed | `map_location="cuda"` moved the RNG state to GPU; `set_rng_state` needs CPU `uint8` | `rng.py` coerces back, and saves/restores CUDA generator states too |
| `JSONDecodeError: Extra data` | Drive's FUSE mount does not truncate on overwrite, so a shorter rewrite left a stale tail | `jsonio.write_json` writes a temp file and renames; `read_json` recovers damaged files; `repair_json_tree` fixes a tree |
| Quick test hid real bugs | An untrained random encoder scores at chance whether the model is correct or not | Quick test uses pretrained BERT-mini plus explicit sanity thresholds |

## 5. Honest status of the ladder

Latest full run: 10 epochs, MentalRoBERTa, real Qwen2.5-7B weak labels, with
§3.3–§3.10 applied. Mean ± std across 3 seeds (validation):

| rung | composite | selection_score | sev QWK | risk QWK | sym AUROC | stress AUROC |
|---|---|---|---|---|---|---|
| `L0_flat` | 0.531 ± 0.009 | 0.717 ± 0.004 | 0.486 | 0.324 | 0.883 | 0.908 |
| `L1_bottleneck` | 0.549 ± 0.027 | 0.720 ± 0.004 | 0.471 | 0.384 | 0.896 | 0.906 |
| `L2_graph` | **0.571 ± 0.015** | **0.724 ± 0.005** | 0.480 | **0.438** | 0.896 | 0.907 |
| `L3_weak` | 0.494 ± 0.003 | 0.708 ± 0.004 | 0.443 | 0.249 | 0.891 | 0.905 |
| `L4_contrastive` | 0.544 ± 0.025 | 0.712 ± 0.007 | 0.475 | 0.380 | 0.883 | 0.910 |
| `L5_full` | 0.526 ± 0.027 | 0.714 ± 0.006 | 0.471 | 0.323 | 0.887 | 0.909 |

Two separate readings:

**Absolute performance is good where the data supports it.** Symptom macro AUROC
~0.89 and stress AUROC ~0.91 across all rungs, with seed std ≤ 0.015. These are
the defensible headline numbers for a detection paper.

**The ladder inverts at L3**, and §3.12 identifies why: weak rows were
concatenated into the training frame, so L3–L5 ran a different LR schedule and a
diluted gold batch. That is a protocol confound, not evidence about N3. Notebook
07 retrains L3–L5 only, under a protocol asserted to match L0–L2, which is the
cheapest valid way to re-test it (9 runs, not 18).

`composite` and `selection_score` both average suicide-risk QWK — **74 test
rows, 340 train rows** — in with everything else, so they are dominated by the
noisiest task in the suite. Per-task reporting (notebook 07 §9) is the honest
presentation; the composite should be a secondary number at most.

### What must not happen

Tuning hyperparameters per rung until `L5_full` wins. Every setting in §3 is
shared across rungs, which is the only thing that makes the ladder
interpretable. A ladder where only the top rung received attention measures the
tuning, not the novelties.

Equally: targeting a fixed headline figure. "Above 0.85 on all metrics" is not
reachable on these corpora and would not be believed if reported —
`severity_risk_qwk` is bounded by 340 training rows and by human agreement on
ordinal suicide-risk scales, and the MAE metrics are lower-is-better, so pushing
them "above 0.85" would mean making the model worse. Two metrics
(`stress_auroc`, `symptom_macro_auroc`) already clear 0.85 honestly; those are
the ones to lead with.

If `L5_full` still does not beat the lower rungs after §3.12, the correct action
is to report that. A clean negative result on contrastive and uncertainty is
more credible than five components that all conveniently help, and §3.13 already
explains why the current metric set cannot credit N3 directly.

## 6. v2: what was removed, what was added, and why

Revision in response to external review. The review's ten comments map to fixes
as follows: **seven are measurement, protocol or data issues that no
architecture can address** (baseline encoder fallback, protocol mismatch,
best-epoch selection, labeller bias, the N3 measurement gap, the post-hoc
retrain, and the absence of significance tests). Only label-noise propagation
and the graph's justification are architectural. None of the review's six action
items asks for a new model.

### 6.1 The constraint that drove the design

Training risk QWK reaches **1.000** and training severity QWK **1.000** by epoch
6, against validation ~0.40 and ~0.47, with validation loss rising from epoch 2.
A model that fits its training set perfectly is not short of representational
power. Every v2 change therefore either **reduces effective capacity** or
**corrects supervision**; none adds a module.

### 6.2 Removed

| component | evidence |
|---|---|
| N5 contrastive | 0.544 → 0.537 over two full runs; within seed noise, and it costs a projection head plus a loss term |
| N6 uncertainty | measured −0.003; hypothesis already recorded as wrong in §3.4 |
| N2 graph (demoted) | L1→L2 is 0.549→0.571 with sds 0.027 and 0.015, so it does not clear seed noise; retained as control `C_graph` |

### 6.3 N-A — instrument-grounded monotone ordinal head

PHQ-9 severity is defined as a threshold function of the sum of its nine item
scores, and 9 of the 22 symptoms are those items (7 more are GAD-7, confirmed in
`configs/symptom_ontology.json`). The free `Linear(head_in, 4)` severity head
discarded that structure. v2 uses one latent trait with a cumulative-link model
per ordinal task:

```
z       = Σ_k softplus(v_k) · σ(c_k)   [+ small residual]
P(y ≥ j) = σ(z − τ_j),  τ_1 < … < τ_{J−1}
```

Risk now fits **4 cutpoints** on its 340 rows instead of a whole head, and `z`
is trained by the 2,456 severity rows as well. Non-negative `v_k` makes severity
provably monotone in symptom presence. Class log-probabilities are returned
rather than logits, which is correct for `cross_entropy` because `log_softmax`
is a no-op on normalised log-probabilities — so the existing loss and metric
code needed no special case.

Built from established components (cumulative-link / CORAL ordinal regression,
monotone networks); the contribution is the composition and the instrument
grounding. **A literature check is still required before claiming novelty.**

**Verified:** item weights non-negative (min 0.0486); cutpoints strictly
increasing for both tasks; expected severity monotone in `z` over a sweep from
−3 to 6; class probabilities sum to 1 to 1e-5.

### 6.4 N-B — per-symptom labeller noise channel

A single confidence temperature rescales every label equally and cannot fix
systematic bias: the labeller calls `low_self_esteem` present 80% of the time
where gold prevalence is 24%. `scripts/fit_noise_channel.py` estimates
`tpr_k = P(LLM positive | gold positive)` and `fpr_k = P(LLM positive | gold
negative)` from `validation_labels.jsonl`, with Laplace smoothing and a minimum
support of 20 per cell. Training then applies **forward correction** — the
network predicts the clean probability `p`, and the loss scores
`p·tpr_k + (1−p)·fpr_k` against the noisy label.

**Limitation, stated rather than buried:** the 11 weak-only symptoms have no
gold labels, so their channel is **not identifiable**. It is pooled within
instrument and every such symptom is marked `"source": "pooled_by_instrument"`.
This is the same measurement gap as §3.13 and it means the weak-only half of N-B
cannot be validated until the in-house gold set exists.

**Verified:** on synthetic data with three deliberately over-calling symptoms,
the fitter recovered exactly those three (fpr 0.70 / 0.66 / 0.66 against 0.08–0.17
for the rest).

### 6.5 N-C — author-consistency, deliberately not implemented

Author-level consistency would add supervision from the ~968k unlabelled posts,
which targets the real bottleneck. But `weak_pool.parquet` averages **1.34 posts
per author** and the distribution has not been measured. Notebook 08 §2b
measures it and prints a verdict. Implementing it before that would be shipping
on an assumption.

### 6.6 Overfitting controls, applied identically to every rung

| control | evidence for it |
|---|---|
| `epochs = 4` | validation loss bottoms out at epoch 1–2 in every seed |
| `selection_metric = 'neg_val_loss'` | `selection_score` is 2/3 symptom metrics; it selected epoch 10 at 1.92× and 2.02× the minimum validation loss on two L4 seeds |
| `early_stopping_patience = 2` | matched to where the minimum is |
| `label_smoothing = 0.05` | ordinal heads only |
| `freeze_encoder_layers` | available; verified to cut trainable parameters 6.82M → 0.19M on the test encoder |

### 6.7 Significance testing

`src/symptograph/significance.py` adds paired bootstrap tests and per-task
confidence intervals, working from `predictions_{split}.npz` written by
`train_one`, so no retraining is needed. QWK, AUROC (including ties) and masked
macro F1 were checked to agree with scikit-learn **exactly** (< 1e-9).

The headline finding from it: on the 74-row suicide-risk test split the 95%
bootstrap CI for QWK is roughly **0.4 wide**. No architectural change of any
plausible size is measurable on that split, and this should be reported as a
property of the data rather than papered over.

A difference is called significant only when it clears p < 0.05 on every seed
**and** has the same sign on every seed; a per-seed p-value alone ignores
training variance.

### 6.8 First v2 result: overfitting fixed, N-A's main prediction falsified

Four rungs, 3 seeds, 4 epochs, MentalRoBERTa, selection on validation loss.
Mean ± std over seeds, validation at the selected epoch:

| rung | best epochs | symptom F1 | symptom AUROC | sev QWK | risk QWK | composite |
|---|---|---|---|---|---|---|
| `M0_flat` | 2,2,2 | 0.761 ± 0.004 | 0.877 ± 0.002 | 0.445 ± 0.072 | 0.383 ± 0.078 | 0.529 ± 0.032 |
| `M1_bottleneck` | 2,2,2 | 0.759 ± 0.003 | 0.874 ± 0.004 | **0.493 ± 0.018** | **0.398 ± 0.030** | **0.550 ± 0.017** |
| `M2_monotone` | 3,2,2 | 0.775 ± 0.004 | **0.892 ± 0.005** | 0.438 ± 0.010 | 0.294 ± 0.057 | 0.502 ± 0.020 |
| `M3_weak` | 2,3,3 | 0.774 ± 0.010 | 0.892 ± 0.006 | 0.410 ± 0.010 | 0.304 ± 0.007 | 0.496 ± 0.008 |

**The overfitting controls worked.** Every seed now selects epoch 2–3 instead of
8–10, and training risk QWK at the final epoch is 0.62 under the monotone head
against 0.99–1.000 in v1. Seed standard deviations also collapsed: severity QWK
0.072 → 0.010 and risk QWK 0.078 → 0.007 between `M0_flat` and `M3_weak`. The
monotone constraint is doing real regularisation work.

**N-A's central prediction is falsified.** §6.3 predicted risk QWK would gain
most, because risk drops to 4 cutpoints on 340 rows. Instead:

* symptom AUROC **+0.019** and F1 **+0.016** (consistent, sds ≤ 0.005)
* severity QWK **−0.055**
* risk QWK **−0.105**

The mechanism is the shared trait, not the monotonicity. Forcing depression
severity and suicide risk through **one scalar** `z` makes them the same ordering
of patients, so the model cannot represent high depression severity with low
suicide risk — and risk, as the task whose ordering differs most from severity,
pays the larger price. The regularisation and interpretability came through; the
single-axis assumption did not.

`monotone_shared_trait=False` gives each ordinal task its own monotone trait
over the same concepts, which keeps the instrument grounding, the capacity
reduction and the item-loading readout while dropping the assumption that fails.
It is recorded as a correction with a stated mechanism, not a tuning sweep, and
the flag is in `ARCH_FIELDS` so the two variants cannot be silently mixed.

**N3 remains flat** (`M2 → M3`: composite −0.007, AUROC +0.000), consistent with
§3.13 — the 11 weak-only symptoms have no gold labels, so the metric set cannot
credit it.

**Best configuration so far is `M1_bottleneck`.** That has to be reported as it
stands if the per-task variant does not beat it.

### 6.9 Bug: forward correction crashed under autocast

`F.binary_cross_entropy` raises under `torch.autocast` ("unsafe to autocast"),
so `M4_denoised` died on its first weak batch. The CPU test suite missed it
because autocast resolves to `None` without CUDA. The correction now converts
the channel output back to a logit and uses
`binary_cross_entropy_with_logits`, in float32.

**Verified:** identical to the original formulation to 2.4e-7 with identical
gradient norms, and a positive weak label on a symptom with tpr 0.9 / fpr 0.7
produces a gradient of −0.0625 against −0.5 uncorrected, i.e. the correction
discounts an over-called symptom to 12.5% of its uncorrected push.

The fitted channel on the real labeller is itself a review deliverable:
**tpr spans 0.25–0.94 and fpr spans 0.06–0.74 across the 11 measurable
symptoms.** A symptom with tpr 0.25 means the labeller finds one true positive
in four; fpr 0.74 means it claims presence in three of four genuine negatives.
That is the quantitative form of review comment 5, and it is not something a
single temperature can repair.
