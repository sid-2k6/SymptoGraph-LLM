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
2. **The calibration and capacity fixes** (§3.3, §3.5), which are measured to
   help substantially in small-scale tests but have not been run at full scale.

What must **not** happen: tuning hyperparameters per rung until `L5_full` wins.
Every setting in §3 is shared by all six rungs, which is the only thing that
makes the ladder interpretable. If `L5_full` still does not beat the lower rungs
after the fixes above, the correct action is to report that — a clean negative
result on one or two components is far more credible than six components that all
conveniently help.
