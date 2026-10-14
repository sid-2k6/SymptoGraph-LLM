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

## 4. Infrastructure fixes (no effect on model behaviour)

| Issue | Cause | Fix |
|---|---|---|
| Encoder silently substituted | `mental/mental-roberta-base` is gated; a 403 fell back to `roberta-base` while still labelling the run MentalRoBERTa | `strict_encoder` raises; `check_encoder_access` pre-flights; fallback recorded in `config.json` |
| GPU resume crashed | `map_location="cuda"` moved the RNG state to GPU; `set_rng_state` needs CPU `uint8` | `rng.py` coerces back, and saves/restores CUDA generator states too |
| `JSONDecodeError: Extra data` | Drive's FUSE mount does not truncate on overwrite, so a shorter rewrite left a stale tail | `jsonio.write_json` writes a temp file and renames; `read_json` recovers damaged files; `repair_json_tree` fixes a tree |
| Quick test hid real bugs | An untrained random encoder scores at chance whether the model is correct or not | Quick test uses pretrained BERT-mini plus explicit sanity thresholds |

## 5. Honest status of the ladder

As of the last full run (6 epochs, MentalRoBERTa, before §3.3–§3.7), best
validation composite per seed:

| rung | seeds | mean |
|---|---|---|
| `L0_flat` | 0.552 / 0.532 / 0.545 | 0.543 |
| `L1_bottleneck` | 0.526 / 0.537 / 0.522 | 0.528 |
| `L2_graph` | 0.521 / 0.577 / 0.525 | 0.541 |
| `L3_weak` | 0.531 / 0.519 / 0.535 | 0.528 |

The ladder is **flat within seed noise**. No component has yet demonstrated a
gain, and `L3_weak` was trained on mock labels so its result carries no
information about N3 at all.

Two things could legitimately change this, in order of expected effect:

1. **Real weak labels** (§3.8). N3, and the contrastive objective that consumes
   symptom profiles, were both running on noise.
2. **Ordinal class weights** (§3.9), which moved severity QWK from 0.005 to
   0.509 in a small-scale test. Since severity QWK is one of three terms in both
   `composite` and `selection_score`, this lifts every rung and makes the
   severity task actually informative for the first time.
3. **The calibration and capacity fixes** (§3.3, §3.5), measured to help
   substantially in small-scale tests but not yet run at full scale.

What must **not** happen: tuning hyperparameters per rung until `L5_full` wins.
Every setting in §3 is shared by all six rungs, which is the only thing that
makes the ladder interpretable. If `L5_full` still does not beat the lower rungs
after the fixes above, the correct action is to report that — a clean negative
result on one or two components is far more credible than six components that all
conveniently help.
