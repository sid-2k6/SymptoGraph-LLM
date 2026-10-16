"""SymptoGraph-LLM: the proposed model, implemented as an ablation ladder.

Each rung adds exactly one novelty on top of the previous one, so every reported
gain is attributable to a single component.

| rung | adds | novelty |
|---|---|---|
| ``L0_flat`` | encoder with flat heads (no bottleneck) | - |
| ``L1_bottleneck`` | clinical symptom concept bottleneck | N1 |
| ``L2_graph`` | symptom-disorder comorbidity graph | N2 |
| ``L3_weak`` | confidence-weighted LLM weak labels | N3 |
| ``L4_contrastive`` | symptom-profile contrastive objective | N5 |
| ``L5_full`` | learned uncertainty and abstention | N6 |

``L5_full`` is SymptoGraph-LLM. ``L0_flat`` is intentionally the same model family
as the ``mentalroberta`` baseline, which anchors the ladder to notebook 05.

N4, the symptom-grounded rationale generator, is a generative component and is
not part of this notebook; it needs its own training and human evaluation.

Metrics come from ``metrics.compute_metrics``, the identical function the
baselines use, so the numbers are directly comparable.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .graph import SymptomGraph
from .metrics import (aggregate_seeds, compute_metrics, flatten_metrics,
                      headline_table, plot_history_pub, plot_per_symptom_pub,
                      plot_risk_coverage, plot_seed_aggregate, positive_weights,
                      selective_metrics, set_publication_style, tune_thresholds)
from .jsonio import read_json, write_json
from .preprocess import Ontology
from .rng import capture_rng_state, restore_rng_state

N_SEVERITY, N_RISK = 4, 5

# The ladder, in order. Each entry switches on exactly one more component.
LADDER: list[tuple[str, dict]] = [
    ("L0_flat", dict(use_bottleneck=False, use_graph=False, use_weak=False,
                     use_contrastive=False, use_uncertainty=False)),
    ("L1_bottleneck", dict(use_bottleneck=True, use_graph=False, use_weak=False,
                           use_contrastive=False, use_uncertainty=False)),
    ("L2_graph", dict(use_bottleneck=True, use_graph=True, use_weak=False,
                      use_contrastive=False, use_uncertainty=False)),
    ("L3_weak", dict(use_bottleneck=True, use_graph=True, use_weak=True,
                     use_contrastive=False, use_uncertainty=False)),
    ("L4_contrastive", dict(use_bottleneck=True, use_graph=True, use_weak=True,
                            use_contrastive=True, use_uncertainty=False)),
    ("L5_full", dict(use_bottleneck=True, use_graph=True, use_weak=True,
                     use_contrastive=True, use_uncertainty=True)),
]
LADDER_ORDER = [name for name, _ in LADDER]
FULL_MODEL = "L5_full"

# --------------------------------------------------------------------------- #
# v2 ladder
# --------------------------------------------------------------------------- #
# Contrastive (old N5) and uncertainty (old N6) are dropped: both measured at or
# below seed noise over two full runs, and uncertainty's hypothesis was already
# recorded as wrong in Model_Design.md 3.4. The graph (old N2) is demoted from a
# headline novelty to a controlled ablation, because L1 -> L2 (0.549 -> 0.571,
# sds 0.027 and 0.015) does not clear seed noise and the prior is hand-authored.
#
# In their place: a monotone instrument-grounded ordinal head (N-A) and a
# per-symptom noise channel on the weak labels (N-B). Both REDUCE effective
# capacity or correct supervision rather than adding parameters, which is the
# only kind of change that can help a model whose training risk QWK already
# reaches 1.000 while validation sits near 0.40.
LADDER_V2: list[tuple[str, dict]] = [
    ("M0_flat", dict(use_bottleneck=False, use_graph=False, use_weak=False,
                     use_monotone_ordinal=False, use_noise_correction=False)),
    ("M1_bottleneck", dict(use_bottleneck=True, use_graph=False, use_weak=False,
                           use_monotone_ordinal=False, use_noise_correction=False)),
    ("M2_monotone", dict(use_bottleneck=True, use_graph=False, use_weak=False,
                         use_monotone_ordinal=True, use_noise_correction=False)),
    ("M3_weak", dict(use_bottleneck=True, use_graph=False, use_weak=True,
                     use_monotone_ordinal=True, use_noise_correction=False)),
    ("M4_denoised", dict(use_bottleneck=True, use_graph=False, use_weak=True,
                         use_monotone_ordinal=True, use_noise_correction=True)),
]
LADDER_V2_ORDER = [name for name, _ in LADDER_V2]
FULL_MODEL_V2 = "M4_denoised"

# Controls, run separately from the ladder. Each isolates one claim.
CONTROLS_V2: list[tuple[str, dict]] = [
    # does the graph beat no graph, on top of the full model?
    ("C_graph", dict(use_bottleneck=True, use_graph=True, use_weak=True,
                     use_monotone_ordinal=True, use_noise_correction=True)),
    # is the monotone head winning because of structure, or just fewer params?
    # this keeps the shared latent trait but drops the non-negativity constraint
    ("C_monotone_unconstrained",
     dict(use_bottleneck=True, use_graph=False, use_weak=True,
          use_monotone_ordinal=True, monotone_nonneg=False,
          use_noise_correction=True)),
    # does denoising beat simply down-weighting the weak labels uniformly?
    ("C_weak_downweight",
     dict(use_bottleneck=True, use_graph=False, use_weak=True,
          use_monotone_ordinal=True, use_noise_correction=False,
          weak_stream_weight=0.5)),
]


@dataclass
class ProposedConfig:
    """Shared protocol. Identical to the baseline protocol wherever it overlaps."""
    epochs: int = 10
    batch_size: int = 16
    eval_batch_size: int = 64
    max_length: int = 256
    lr: float = 2e-5
    head_lr: float = 1e-3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    grad_clip: float = 1.0
    dropout: float = 0.1
    seeds: tuple = (0, 1, 2)
    encoder_name: str = "mental/mental-roberta-base"
    encoder_fallback: str = "roberta-base"
    # A silent encoder substitution would mean reporting RoBERTa numbers as
    # MentalRoBERTa. Set True for any run whose results you intend to publish.
    strict_encoder: bool = False
    # Build a small randomly initialised RoBERTa instead of downloading one.
    # Used by the quick test so it never depends on a third-party tiny model.
    tiny_random_encoder: bool = False

    # component switches, set per ladder rung
    use_bottleneck: bool = True
    use_graph: bool = True
    use_weak: bool = True
    use_contrastive: bool = True
    use_uncertainty: bool = True

    # component hyperparameters
    graph_dim: int = 128
    graph_layers: int = 2
    graph_l1: float = 1e-4
    contrastive_weight: float = 0.1
    contrastive_temperature: float = 0.1
    contrastive_jaccard: float = 0.5
    uncertainty_weight: float = 0.5
    # N6 trained on detached features. Attenuating the primary loss with a
    # learned variance lets the model cut its loss by declaring uncertainty
    # instead of fitting, which made the abstention rung strictly worse than the
    # rung below it. Detached, it adds abstention without costing accuracy.
    uncertainty_detached: bool = True
    # N1: pass the concept-gated evidence summary to the heads, not only the 22
    # concept logits. A bare 22-dim bottleneck starves the severity head, which
    # saw 768 dims in the flat rung. Set False for the strict-bottleneck ablation.
    concept_gated_pooling: bool = True
    # ---- N-A: instrument-grounded monotone ordinal head -------------------- #
    # PHQ-9 severity is defined as a threshold function of the sum of its nine
    # item scores, and 9 of the 22 symptoms ARE those items (7 more are GAD-7).
    # The free Linear(head_in, 4) severity head threw that away. Instead a single
    # latent distress trait z = sum_k softplus(v_k) * sigma(c_k) feeds a
    # cumulative-link model per ordinal task, so each task learns only its own
    # ordered cutpoints. Suicide risk then fits 4 numbers on 340 rows instead of
    # a whole head, and borrows strength from the 2,456 severity rows through z.
    use_monotone_ordinal: bool = True
    # non-negative item weights make severity provably monotone in symptom
    # presence. False keeps the shared trait but drops the constraint - the
    # control that separates "structure helps" from "fewer parameters help".
    monotone_nonneg: bool = True
    # a small unconstrained term added to z, so the head is not strictly additive
    monotone_residual: bool = True

    # ---- N-B: per-symptom noise channel on the weak labels ----------------- #
    # A single confidence temperature cannot fix systematic bias: the labeller
    # calls low_self_esteem at 0.80 where gold prevalence is 0.24. Instead model
    # the labeller as a per-symptom channel P(noisy | clean) and train the clean
    # probability through it (forward correction).
    use_noise_correction: bool = False
    # written by scripts/fit_noise_channel.py from weak_labels/validation_labels.jsonl
    noise_channel_path: str | None = None

    # ---- overfitting controls ---------------------------------------------- #
    # Validation loss bottoms out at epoch 1-2 in every seed while train risk QWK
    # reaches 1.000 by epoch 6. Freezing the lower encoder layers is the cheapest
    # real regulariser at 12,911 training rows.
    freeze_encoder_layers: int = 0
    # applied to the ordinal cross-entropies only
    label_smoothing: float = 0.0
    # class imbalance; 0 disables
    pos_weight_cap: float = 10.0
    # Depression severity is 72% "minimum" (1767/205/288/196) and risk is
    # similarly skewed. Unweighted cross-entropy collapses both heads onto the
    # majority class, which is why QWK sat at exactly 0.000.
    ordinal_class_weights: bool = True
    # per-symptom decision thresholds tuned on validation, applied to test
    tune_thresholds: bool = True
    scheduler: str = "cosine"
    early_stopping_patience: int = 4

    task_weights: dict = field(default_factory=lambda: {
        "symptoms": 1.0, "severity": 1.0, "risk": 1.0, "stress": 0.5})
    weak_label_path: str | None = None
    weak_max_rows: int | None = 20_000
    # Weak rows used to be concatenated into the training frame. That inflated
    # steps per epoch by 46%, so L3-L5 ran a different LR schedule from L0-L2,
    # and it diluted the sparse tasks: rows per batch carrying a risk label fell
    # from 0.84 to 0.58, which cost risk QWK 0.19. As a stream the weak rows are
    # a per-step auxiliary symptom loss instead, so steps per epoch, the
    # schedule and the gold batch composition all stay identical to L0-L2.
    # Cost is fixed by weak_batch_size, NOT by the size of the weak pool, so
    # using all 50k labels costs the same as using 6k.
    weak_as_stream: bool = True
    weak_batch_size: int | None = None      # None -> batch_size
    weak_every: int = 1                     # draw a weak batch every N steps
    weak_stream_weight: float = 1.0
    # Refuse labels produced by weak_label.MockLabeller. Those are keyword
    # heuristics meant only for pipeline testing; training N3 on them is
    # training on noise, and it silently flattens the whole upper ladder.
    allow_mock_weak_labels: bool = False
    limit_train: int | None = None
    limit_eval: int | None = None
    amp: bool = True
    # "auto" uses bfloat16 where the GPU supports it (L4, A100), which is more
    # numerically stable than fp16 and needs no gradient scaler.
    amp_dtype: str = "auto"
    selection_metric: str = "selection_score"
    # Full fp32 state including optimiser, so a resumed run is numerically faithful.
    save_optimizer_state: bool = True
    # Writing straight to Drive costs real time for a ~1.5 GB checkpoint. Raise this
    # to 2 to halve that I/O, at the cost of coarser resume granularity.
    checkpoint_every: int = 1

    def with_flags(self, flags: dict) -> "ProposedConfig":
        return ProposedConfig(**{**asdict(self), **flags})


# Bump when the module layout changes, so stale checkpoints are refused rather
# than silently loaded into a different network.
ARCH_VERSION = 5

ARCH_FIELDS = ("max_length", "encoder_name", "encoder_fallback", "graph_dim",
               "graph_layers", "use_bottleneck", "use_graph", "use_uncertainty",
               "use_contrastive", "dropout", "concept_gated_pooling",
               "uncertainty_detached", "use_monotone_ordinal",
               "monotone_nonneg", "monotone_residual", "use_noise_correction",
               "freeze_encoder_layers")


def architecture_fingerprint(rung: str, cfg: ProposedConfig, onto: Ontology) -> tuple:
    fields = {"rung": rung, "n_symptoms": len(onto.symptoms),
              "arch_version": ARCH_VERSION,
              **{f: getattr(cfg, f) for f in ARCH_FIELDS}}
    # ``weak_as_stream`` only enters the fingerprint for rungs that actually use
    # weak data. A rung with use_weak=False runs identical code either way, so
    # including it unconditionally would invalidate the existing L0-L2
    # checkpoints for a change that provably cannot affect them.
    if cfg.use_weak:
        fields["weak_as_stream"] = cfg.weak_as_stream
    blob = json.dumps(fields, sort_keys=True, default=str).encode()
    return hashlib.md5(blob).hexdigest()[:12], fields


# Encoders that are domain-pretrained for mental health. "gated: auto" means the
# repo needs a Hugging Face login plus a one-click licence acceptance; it is not
# a manual review queue.
ENCODER_OPTIONS = {
    "mental/mental-roberta-base": "MentalRoBERTa - the paper's choice. Gated (auto-approve).",
    "mental/mental-bert-base-uncased": "MentalBERT. Gated (auto-approve).",
    "emilyalsentzer/Bio_ClinicalBERT": "Clinical notes, domain-adjacent. Ungated.",
    "roberta-base": "General domain. Ungated. Rename the run honestly if you use it.",
}


def check_encoder_access(name: str) -> dict:
    """Probe whether an encoder can actually be downloaded, before training starts.

    Loading only the config costs a few hundred bytes, which turns a failure six
    hours into a run into a five-second pre-flight check.
    """
    from transformers import AutoConfig
    try:
        AutoConfig.from_pretrained(name)
        return {"ok": True, "encoder": name, "reason": None}
    except Exception as exc:
        text = str(exc)
        if "gated repo" in text or "403" in text:
            reason = ("gated: accept the licence while logged in. Visit "
                      f"https://huggingface.co/{name} , click 'Agree and access "
                      "repository', then authenticate in this notebook.")
        else:
            reason = f"{type(exc).__name__}: {text.splitlines()[0]}"
        return {"ok": False, "encoder": name, "reason": reason}


def resolve_amp(cfg: "ProposedConfig", device: str):
    """Return ``(autocast_dtype, scaler)``; bfloat16 needs no scaler."""
    import torch
    if not cfg.amp or device != "cuda":
        return None, None
    want = cfg.amp_dtype
    if want == "auto":
        want = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if want == "bf16":
        return torch.bfloat16, None
    return torch.float16, torch.amp.GradScaler("cuda")


def set_seed(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_frames(processed_dir: str, onto: Ontology, cfg: ProposedConfig) -> dict:
    """Splits, with per-symptom loss weights.

    Weights are 1.0 for a gold label, the calibrated confidence for a weak label
    and 0.0 where nothing is annotated. Weak rows are only ever added to train.
    """
    df = pd.read_parquet(os.path.join(processed_dir, "labelled.parquet"))
    sym_cols = onto.columns
    frames = {}
    for split in ("train", "val", "test"):
        part = df[df["split"] == split].reset_index(drop=True)
        limit = cfg.limit_train if split == "train" else cfg.limit_eval
        if limit and limit < len(part):
            part = part.sample(n=limit, random_state=0).reset_index(drop=True)
        labels = part[sym_cols].to_numpy(dtype=np.float32)
        weights = (labels >= 0).astype(np.float32)
        frames[split] = {
            "text": part["text"].fillna("").tolist(),
            "symptoms": labels, "sym_weight": weights,
            "severity": part.get("severity_depression",
                                 pd.Series(np.nan, index=part.index)
                                 ).to_numpy(dtype=np.float32),
            "risk": part.get("severity_risk", pd.Series(np.nan, index=part.index)
                             ).to_numpy(dtype=np.float32),
            "stress": part.get("stress", pd.Series(np.nan, index=part.index)
                               ).to_numpy(dtype=np.float32),
            "n_weak_rows": 0,
        }

    if cfg.use_weak and cfg.weak_label_path and os.path.exists(cfg.weak_label_path):
        weak = pd.read_parquet(cfg.weak_label_path)
        if "labeller" in weak.columns:
            kinds = {str(k).lower() for k in weak["labeller"].dropna().unique()}
            if any("mock" in k for k in kinds) and not cfg.allow_mock_weak_labels:
                raise RuntimeError(
                    f"{cfg.weak_label_path} was produced by the MOCK labeller "
                    f"({sorted(kinds)}).\n\n"
                    "Those labels are keyword heuristics for testing the pipeline, "
                    "not LLM labels. Training N3 on them is training on noise, and "
                    "it drags down every rung from L3_weak upward.\n\n"
                    "Re-run notebook 04 with MOCK = False on a GPU runtime, then "
                    "retry. To proceed anyway (pipeline testing only), set "
                    "allow_mock_weak_labels=True.")
        if cfg.weak_max_rows and len(weak) > cfg.weak_max_rows:
            weak = weak.sample(n=cfg.weak_max_rows, random_state=0)
        n = len(weak)
        labels = np.full((n, len(onto.symptoms)), -1.0, dtype=np.float32)
        weights = np.zeros((n, len(onto.symptoms)), dtype=np.float32)
        for i, s in enumerate(onto.symptoms):
            lcol, ccol = f"wl_{s}", f"conf_{s}"
            if lcol in weak:
                lab = weak[lcol].to_numpy(dtype=np.float32)
                conf = (weak[ccol].to_numpy(dtype=np.float32) if ccol in weak
                        else np.full(n, 0.5, dtype=np.float32))
                ok = lab >= 0
                labels[ok, i] = lab[ok]
                weights[ok, i] = conf[ok]          # confidence weighting (N3)
        tr = frames["train"]
        nan = np.full(n, np.nan, dtype=np.float32)
        weak_frame = {
            "text": weak["text"].fillna("").tolist(),
            "symptoms": labels, "sym_weight": weights,
            "severity": nan, "risk": nan, "stress": nan, "n_weak_rows": n,
        }
        if cfg.weak_as_stream:
            # Separate stream: train keeps exactly the gold rows, so steps per
            # epoch, the LR schedule, pos_weight and the per-batch density of
            # severity/risk/stress labels are all unchanged from L0-L2.
            tr["n_weak_rows"] = n
            frames["weak"] = weak_frame
        else:
            # Legacy concatenation, kept so the earlier run can be reproduced.
            frames["train"] = {
                "text": tr["text"] + weak_frame["text"],
                "symptoms": np.concatenate([tr["symptoms"], labels]),
                "sym_weight": np.concatenate([tr["sym_weight"], weights]),
                "severity": np.concatenate([tr["severity"], nan]),
                "risk": np.concatenate([tr["risk"], nan]),
                "stress": np.concatenate([tr["stress"], nan]),
                "n_weak_rows": n,
            }
    return frames


class ProposedDataset:
    def __init__(self, frame: dict):
        self.f = frame

    def __len__(self):
        return len(self.f["text"])

    def __getitem__(self, i):
        return {"text": self.f["text"][i],
                "symptoms": self.f["symptoms"][i],
                "sym_weight": self.f["sym_weight"][i],
                "severity": self.f["severity"][i],
                "risk": self.f["risk"][i],
                "stress": self.f["stress"][i]}


def make_collate(tokenizer, max_length: int):
    import torch

    def collate(batch):
        enc = tokenizer([b["text"] for b in batch], padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt")
        return {
            "input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
            "symptoms": torch.tensor(np.stack([b["symptoms"] for b in batch])),
            "sym_weight": torch.tensor(np.stack([b["sym_weight"] for b in batch])),
            "severity": torch.tensor(np.array([b["severity"] for b in batch])),
            "risk": torch.tensor(np.array([b["risk"] for b in batch])),
            "stress": torch.tensor(np.array([b["stress"] for b in batch])),
        }
    return collate


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
def normalise_adjacency(A):
    """Symmetric normalisation of ``relu(A) + I``.

    The ReLU is load-bearing, not cosmetic. ``A = A0 + dA`` has a learnable
    ``dA``, and once it goes negative the row sums can cross zero. Normalising by
    ``deg.clamp(min=1e-9) ** -0.5`` then multiplies the adjacency by up to 3e4,
    which drove training losses to 1e13 in the graph rungs. Keeping edges
    non-negative guarantees ``deg >= 1`` after adding the identity, so the
    normaliser is bounded by 1.
    """
    import torch
    A = torch.relu(A)
    A = A + torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
    deg = A.sum(-1).clamp(min=1.0)
    d = deg.pow(-0.5)
    return d.unsqueeze(1) * A * d.unsqueeze(0)


def load_noise_channel(path: str | None, onto: Ontology, device, verbose=True):
    """Per-symptom labeller channel ``(tpr, fpr)`` as two K-vectors.

    ``tpr[k] = P(LLM says present | symptom truly present)`` and
    ``fpr[k] = P(LLM says present | truly absent)``, estimated by
    ``scripts/fit_noise_channel.py`` from the gold-labelled symptoms.

    The 11 weak-only symptoms have no gold labels, so their channel is **not
    identifiable**. The fitting script pools those by instrument and records
    that it did; this loader warns about it rather than letting it pass
    silently, because it is an assumption the paper has to state.
    """
    import torch
    K = len(onto.symptoms)
    if not path or not os.path.exists(path):
        raise FileNotFoundError(
            f"use_noise_correction=True needs a fitted channel, got {path!r}.\n"
            "Run: python scripts/fit_noise_channel.py "
            "--validation <weak_labels>/validation_labels.jsonl "
            "--labelled <processed>/labelled.parquet --out <weak_labels>/noise_channel.json")
    ch = read_json(path)
    per = ch["per_symptom"]
    tpr = torch.tensor([per[s]["tpr"] for s in onto.symptoms], dtype=torch.float32)
    fpr = torch.tensor([per[s]["fpr"] for s in onto.symptoms], dtype=torch.float32)
    if verbose:
        measured = [s for s in onto.symptoms if per[s].get("source") == "measured"]
        pooled = [s for s in onto.symptoms if per[s].get("source") != "measured"]
        print(f"  noise channel: {len(measured)} symptoms measured against gold, "
              f"{len(pooled)} pooled/assumed")
        print(f"    tpr {tpr.min():.2f}-{tpr.max():.2f} | "
              f"fpr {fpr.min():.2f}-{fpr.max():.2f}")
        if pooled:
            print(f"    ASSUMED (no gold labels exist): {', '.join(pooled[:6])}"
                  f"{' ...' if len(pooled) > 6 else ''}")
    # A channel with tpr <= fpr carries no signal about that symptom; clamp so
    # the correction cannot invert the label.
    tpr = torch.maximum(tpr, fpr + 0.05).clamp(max=0.99)
    return tpr.to(device), fpr.clamp(min=0.01).to(device)


def freeze_lower_layers(encoder, n: int, verbose: bool = True) -> int:
    """Freeze the embeddings and the lowest ``n`` transformer blocks.

    The cheapest real regulariser at 12,911 training rows: validation loss
    currently bottoms out at epoch 1-2 while training risk QWK reaches 1.000.
    Returns the number of frozen parameters.
    """
    if n <= 0:
        return 0
    frozen = 0
    emb = getattr(encoder, "embeddings", None)
    if emb is not None:
        for prm in emb.parameters():
            prm.requires_grad = False
            frozen += prm.numel()
    layers = getattr(getattr(encoder, "encoder", None), "layer", None)
    if layers is not None:
        for block in layers[:n]:
            for prm in block.parameters():
                prm.requires_grad = False
                frozen += prm.numel()
    if verbose:
        print(f"  froze embeddings + lowest {n} encoder blocks "
              f"({frozen / 1e6:.1f}M parameters)")
    return frozen


def build_model(cfg: ProposedConfig, onto: Ontology, graph: SymptomGraph | None):
    import torch
    import torch.nn as nn
    from transformers import AutoModel, AutoTokenizer

    K = len(onto.symptoms)

    fallback_used = False
    if cfg.tiny_random_encoder:
        from transformers import AutoConfig
        tokenizer = AutoTokenizer.from_pretrained(cfg.encoder_fallback)
        conf = AutoConfig.from_pretrained(cfg.encoder_fallback)
        conf.num_hidden_layers, conf.hidden_size = 2, 128
        conf.num_attention_heads, conf.intermediate_size = 2, 256
        encoder = AutoModel.from_config(conf)
        name = "tiny-random (untrained)"
    else:
        name = cfg.encoder_name
        try:
            tokenizer = AutoTokenizer.from_pretrained(name)
            encoder = AutoModel.from_pretrained(name)
        except Exception as exc:
            if cfg.strict_encoder:
                raise RuntimeError(
                    f"Could not load the requested encoder {name!r}: "
                    f"{type(exc).__name__}: {exc}\n\n"
                    f"strict_encoder=True, so this will not silently fall back to "
                    f"{cfg.encoder_fallback!r}. Reporting one encoder's results "
                    f"under another model's name would invalidate the comparison.\n"
                    f"Fix the model id or network, or set strict_encoder=False."
                ) from exc
            print("  " + "!" * 68)
            print(f"  WARNING: could not load {name} ({type(exc).__name__}).")
            print(f"  FALLING BACK to {cfg.encoder_fallback}. Results will NOT be "
                  f"MentalRoBERTa results.")
            print("  " + "!" * 68, flush=True)
            fallback_used = True
            name = cfg.encoder_fallback
            tokenizer = AutoTokenizer.from_pretrained(name)
            encoder = AutoModel.from_pretrained(name)

    class SymptoGraphLLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.cfg = cfg
            self.encoder = encoder
            self.loaded_name = name
            self.encoder_requested = cfg.encoder_name
            self.encoder_fallback_used = fallback_used
            D = encoder.config.hidden_size
            self.hidden = D
            self.drop = nn.Dropout(cfg.dropout)

            # ---- N1: clinical concept bottleneck
            self.sym_query = nn.Parameter(torch.randn(K, D) * 0.02)
            # One scoring vector PER SYMPTOM. A single shared Linear(D, 1) here
            # would leave each symptom only a scalar rescale of one shared
            # direction, which is not a concept bottleneck at all - it collapsed
            # every bottleneck rung to F1 = 0 while the flat baseline, which has
            # K independent vectors, still learned.
            self.sym_w = nn.Parameter(torch.randn(K, D) / math.sqrt(D))
            self.sym_bias = nn.Parameter(torch.zeros(K))

            # initialise the queries from the sentence embeddings of the symptom
            # definitions, so attention starts from clinical meaning
            if graph is not None and graph.node_features is not None:
                feats = torch.tensor(graph.node_features[:K], dtype=torch.float32)
                self.register_buffer("sym_init", feats)
                self.sym_init_proj = nn.Linear(feats.shape[1], D)
            else:
                self.sym_init = None
                self.sym_init_proj = None

            # ---- N2: symptom-disorder graph
            self.use_graph = cfg.use_graph and graph is not None
            self.gated = cfg.concept_gated_pooling
            if self.use_graph:
                Dn = len(graph.disorders)
                self.n_nodes = K + Dn
                A0 = np.zeros((self.n_nodes, self.n_nodes), dtype=np.float32)
                A0[:K, :K] = graph.A_ss
                A0[:K, K:] = graph.A_sd
                A0[K:, :K] = graph.A_sd.T
                A0[K:, K:] = graph.A_dd
                self.register_buffer("A0", torch.tensor(A0))
                self.delta = nn.Parameter(torch.zeros(self.n_nodes, self.n_nodes))
                gd = cfg.graph_dim
                self.sym_node = nn.Parameter(torch.randn(K, gd) * 0.02)
                self.dis_node = nn.Parameter(torch.randn(Dn, gd) * 0.02)
                self.gnn = nn.ModuleList(
                    [nn.Linear(gd, gd) for _ in range(cfg.graph_layers)])
                # normalise after each residual step so repeated propagation
                # cannot amplify activations layer over layer
                self.gnn_norm = nn.ModuleList(
                    [nn.LayerNorm(gd) for _ in range(cfg.graph_layers)])
                self.graph_refine = nn.Linear(gd, 1)
                self.graph_pool = nn.Linear(gd * 2, gd)
                head_dim = K + gd + (D if self.gated else 0)
            else:
                head_dim = K + (D if self.gated else 0)

            # ---- heads
            if cfg.use_bottleneck:
                self.head_in = head_dim            # strictly downstream of concepts
            else:
                self.head_in = D                   # flat: straight from the encoder
                self.flat_symptoms = nn.Linear(D, K)
            # ---- N-A: instrument-grounded monotone ordinal head
            # Needs the concept vector, so it is only available with the
            # bottleneck; the flat rung keeps free heads as the control.
            self.monotone = cfg.use_monotone_ordinal and cfg.use_bottleneck
            if self.monotone:
                # one item weight per symptom; softplus keeps it non-negative so
                # z cannot fall when a symptom turns on
                # init so that z lands near the initial cutpoints instead of
                # saturating the top class: softplus(-3) ~ 0.049, and about half
                # the 22 concepts fire at init, giving z ~ 0.5 against
                # tau ~ [0, 0.69, 1.39]
                self.item_w = nn.Parameter(torch.full(
                    (K,), -3.0 if cfg.monotone_nonneg else 1.0 / K))
                # ordered cutpoints per task, parameterised as a first cut plus
                # positive increments, so tau_1 < tau_2 < ... by construction
                self.cut0 = nn.ParameterDict({
                    "severity": nn.Parameter(torch.zeros(1)),
                    "risk": nn.Parameter(torch.zeros(1))})
                self.cut_inc = nn.ParameterDict({
                    "severity": nn.Parameter(torch.zeros(N_SEVERITY - 2)),
                    "risk": nn.Parameter(torch.zeros(N_RISK - 2))})
                self.z_scale = nn.Parameter(torch.ones(1))
                if cfg.monotone_residual:
                    self.z_residual = nn.Linear(self.head_in, 1)
            else:
                self.severity = nn.Linear(self.head_in, N_SEVERITY)
                self.risk = nn.Linear(self.head_in, N_RISK)
            self.stress = nn.Linear(self.head_in, 1)

            # ---- N5: contrastive projection
            if cfg.use_contrastive:
                self.projector = nn.Sequential(
                    nn.Linear(D, D), nn.ReLU(), nn.Linear(D, 128))

            # ---- N6: learned per-symptom uncertainty
            if cfg.use_uncertainty:
                self.log_var = nn.Linear(self.head_in, K)

        def adjacency(self):
            return normalise_adjacency(self.A0 + self.delta)

        def cutpoints(self, key: str):
            """Strictly increasing thresholds for one ordinal task."""
            inc = torch.nn.functional.softplus(self.cut_inc[key]) + 1e-3
            return torch.cat([self.cut0[key], self.cut0[key] + inc.cumsum(0)])

        def _ordinal_logits(self, z, key: str, n_cls: int):
            """Cumulative-link class log-probabilities from the latent trait.

            ``P(y >= j) = sigmoid(z - tau_j)``, differenced into class
            probabilities. Returns ``log p`` rather than probabilities, because
            ``cross_entropy`` applies ``log_softmax`` and ``log_softmax(log p)``
            leaves ``log p`` unchanged when ``p`` sums to one - so the existing
            loss and metric code needs no special case.
            """
            tau = self.cutpoints(key)                      # n_cls-1
            ge = torch.sigmoid(z.unsqueeze(-1) - tau.unsqueeze(0))   # B,n_cls-1
            ones = torch.ones_like(ge[:, :1])
            zeros = torch.zeros_like(ge[:, :1])
            upper = torch.cat([ones, ge], dim=-1)          # P(y >= j),   j=0..n-1
            lower = torch.cat([ge, zeros], dim=-1)         # P(y >= j+1), j=0..n-1
            return (upper - lower).clamp(min=1e-7).log()

        def forward(self, batch):
            mask = batch["attention_mask"]
            H = self.encoder(input_ids=batch["input_ids"],
                             attention_mask=mask).last_hidden_state
            m = mask.unsqueeze(-1).float()
            pooled = (H * m).sum(1) / m.sum(1).clamp(min=1e-9)
            out = {"pooled": pooled}

            if not self.cfg.use_bottleneck:
                feat = self.drop(pooled)
                out.update(symptoms=self.flat_symptoms(feat), concepts=None,
                           attention=None)
            else:
                q = self.sym_query
                if self.sym_init is not None:
                    q = q + self.sym_init_proj(self.sym_init)
                scores = torch.einsum("bld,kd->blk", H, q) / math.sqrt(self.hidden)
                scores = scores.masked_fill(~mask.bool().unsqueeze(-1), float("-inf"))
                att = torch.softmax(scores, dim=1)                 # B,L,K
                Z = torch.einsum("blk,bld->bkd", att, H)           # B,K,D
                c_logit = ((self.drop(Z) * self.sym_w.unsqueeze(0)).sum(-1)
                           + self.sym_bias)                          # B,K
                out["attention"] = att

                if self.use_graph:
                    c = torch.sigmoid(c_logit)
                    B = c.shape[0]
                    Xs = c.unsqueeze(-1) * self.sym_node.unsqueeze(0)
                    Xd = self.dis_node.unsqueeze(0).expand(B, -1, -1)
                    X = torch.cat([Xs, Xd], dim=1)
                    A = self.adjacency()
                    for layer, norm in zip(self.gnn, self.gnn_norm):
                        X = norm(torch.relu(
                            layer(torch.einsum("ij,bjd->bid", A, X))) + X)
                    k = self.sym_node.shape[0]
                    c_logit = c_logit + self.graph_refine(X[:, :k]).squeeze(-1)
                    graph_rep = self.graph_pool(
                        torch.cat([X[:, :k].mean(1), X[:, k:].mean(1)], dim=-1))
                    parts = [c_logit, graph_rep]
                else:
                    parts = [c_logit]
                if self.gated:
                    # concept-weighted average of the per-symptom evidence
                    # vectors: downstream heads see only what the concepts
                    # attended to, gated by how strongly each concept fired
                    gate = torch.sigmoid(c_logit).unsqueeze(-1)
                    parts.append((gate * Z).sum(1) / gate.sum(1).clamp(min=1e-6))
                feat = torch.cat(parts, dim=-1)
                out.update(symptoms=c_logit, concepts=torch.sigmoid(c_logit))
                feat = self.drop(feat)

            if self.monotone:
                w = (torch.nn.functional.softplus(self.item_w)
                     if self.cfg.monotone_nonneg else self.item_w)
                z = (out["concepts"] * w.unsqueeze(0)).sum(-1) * self.z_scale
                if self.cfg.monotone_residual:
                    z = z + self.z_residual(feat).squeeze(-1)
                out["latent_trait"] = z
                out["item_weights"] = w
                for key, n_cls in (("severity", N_SEVERITY), ("risk", N_RISK)):
                    out[key] = self._ordinal_logits(z, key, n_cls)
            else:
                out["severity"] = self.severity(feat)
                out["risk"] = self.risk(feat)
            out["stress"] = self.stress(feat).squeeze(-1)
            if self.cfg.use_contrastive:
                out["projection"] = torch.nn.functional.normalize(
                    self.projector(pooled), dim=-1)
            if self.cfg.use_uncertainty:
                head_in = feat.detach() if self.cfg.uncertainty_detached else feat
                out["log_var"] = self.log_var(head_in).clamp(-6, 6)
            return out

    return SymptoGraphLLM(), tokenizer


# --------------------------------------------------------------------------- #
# loss
# --------------------------------------------------------------------------- #
def contrastive_loss(projection, symptoms, weights, cfg: ProposedConfig):
    """Pull together posts with overlapping symptom profiles (novelty N5)."""
    import torch
    known = (weights > 0).float()
    pos = ((symptoms == 1).float() * known)
    have = known.sum(-1) > 0
    if have.sum() < 2:
        return projection.sum() * 0.0
    p, pos, known = projection[have], pos[have], known[have]
    inter = pos @ pos.t()
    union = pos.sum(-1, keepdim=True) + pos.sum(-1) - inter
    jacc = inter / union.clamp(min=1e-6)
    target = (jacc >= cfg.contrastive_jaccard).float()
    target.fill_diagonal_(0)
    if target.sum() == 0:
        return projection.sum() * 0.0
    sim = (p @ p.t()) / cfg.contrastive_temperature
    sim.fill_diagonal_(-1e4)
    log_prob = sim - torch.logsumexp(sim, dim=-1, keepdim=True)
    per_row = -(target * log_prob).sum(-1) / target.sum(-1).clamp(min=1e-6)
    return per_row[target.sum(-1) > 0].mean()


def class_weights(values: np.ndarray, n_classes: int, cap: float = 10.0):
    """Inverse-frequency weights for an ordinal target, capped."""
    v = values[~np.isnan(values)].astype(int)
    if v.size == 0:
        return None
    counts = np.bincount(v, minlength=n_classes).astype(np.float32)
    w = np.where(counts > 0, counts.sum() / np.maximum(counts, 1) / n_classes, 1.0)
    return np.clip(w, 1.0 / cap, cap).astype(np.float32)


def compute_loss(out: dict, batch: dict, model, cfg: ProposedConfig,
                 pos_weight=None, ordinal_weights=None, weak_pass: bool = False,
                 noise_channel=None):
    """Multi-task loss. ``weak_pass`` restricts it to the symptom terms.

    On a weak batch severity, risk and stress are all NaN and would mask out
    anyway, but two terms must be skipped explicitly:

    * the graph L1 penalty is a *parameter* penalty, so adding it on both passes
      would double the regularisation on L3-L5 relative to L2;
    * the contrastive loss defines positives by symptom-profile overlap, and a
      weak row only observes 11 of 22 symptoms, so the overlap is computed on a
      partially observed profile and is not comparable to a gold row's.
    """
    import torch
    import torch.nn.functional as F

    device = out["symptoms"].device
    total = torch.zeros((), device=device)
    parts = {}

    y, w = batch["symptoms"], batch["sym_weight"]
    valid = (w > 0) & (y >= 0)
    if valid.any():
        logits, target = out["symptoms"][valid], y[valid].clamp(min=0)
        weight = w[valid]
        if pos_weight is not None:
            pw = pos_weight.unsqueeze(0).expand_as(y)[valid]
            weight = weight * torch.where(target > 0.5, pw, torch.ones_like(pw))
        if weak_pass and cfg.use_noise_correction and noise_channel is not None:
            # N-B forward correction. The network predicts the CLEAN probability
            # p; we push it through the labeller's per-symptom channel and score
            # that against the noisy label. A symptom the labeller over-calls
            # (high false-positive rate) therefore stops dragging p upward.
            tpr, fpr = noise_channel
            p = torch.sigmoid(logits)
            t = tpr.unsqueeze(0).expand_as(y)[valid]
            f = fpr.unsqueeze(0).expand_as(y)[valid]
            p_obs = (p * t + (1.0 - p) * f).clamp(1e-6, 1.0 - 1e-6)
            bce = F.binary_cross_entropy(p_obs, target, reduction="none")
        else:
            bce = F.binary_cross_entropy_with_logits(logits, target,
                                                     reduction="none")
        l = (bce * weight).sum() / weight.sum().clamp(min=1e-6)
        total = total + cfg.task_weights["symptoms"] * l
        parts["symptoms"] = float(l.detach())

        if cfg.use_uncertainty and "log_var" in out:
            # Train the variance head to predict its own error, on DETACHED
            # features, so it cannot trade accuracy for confidence.
            s = out["log_var"][valid]
            with torch.no_grad():
                err = (torch.sigmoid(logits) - target).pow(2)
            unc = (torch.exp(-s) * err + 0.5 * s)
            lu = (unc * w[valid]).sum() / w[valid].sum().clamp(min=1e-6)
            total = total + cfg.uncertainty_weight * lu
            parts["uncertainty"] = float(lu.detach())
            parts["mean_log_var"] = float(s.mean().detach())

    if weak_pass:
        return total, parts

    for key in ("severity", "risk"):
        t = batch[key]
        m = ~torch.isnan(t)
        if m.any():
            cw = (ordinal_weights or {}).get(key)
            # out[key] holds log-probabilities under the monotone head and raw
            # logits otherwise; log_softmax is a no-op on normalised log-probs,
            # so cross_entropy is correct either way.
            l = F.cross_entropy(out[key][m], t[m].long(), weight=cw,
                                label_smoothing=cfg.label_smoothing)
            total = total + cfg.task_weights[key] * l
            parts[key] = float(l.detach())

    t = batch["stress"]
    m = ~torch.isnan(t)
    if m.any():
        l = F.binary_cross_entropy_with_logits(out["stress"][m], t[m])
        total = total + cfg.task_weights["stress"] * l
        parts["stress"] = float(l.detach())

    if cfg.use_contrastive and "projection" in out:
        l = contrastive_loss(out["projection"], y, w, cfg)
        total = total + cfg.contrastive_weight * l
        parts["contrastive"] = float(l.detach())

    if cfg.use_graph and hasattr(model, "delta"):
        l = model.delta.abs().mean()               # sparse correction to the prior
        total = total + cfg.graph_l1 * l
        parts["graph_l1"] = float(l.detach())

    return total, parts


# --------------------------------------------------------------------------- #
# epoch
# --------------------------------------------------------------------------- #
def cycle_loader(loader):
    """Endless iterator over a loader, reshuffled each time it wraps around."""
    while True:
        for b in loader:
            yield b


def _add_weak_term(loss, model, cfg, pos_weight, ordinal_weights, weak_iter,
                   device, train, step, noise_channel=None):
    """Add the auxiliary weak-label symptom loss to ``loss`` for one step.

    Returns ``(loss, weak_loss_value_or_None)``. Runs inside the caller's
    autocast context so the weak forward uses the same precision, and inside the
    same graph so the step stays a single optimiser update: the number of
    optimiser steps per epoch must not depend on whether weak labels are used.
    """
    if not (train and weak_iter is not None):
        return loss, None
    if cfg.weak_every > 1 and step % cfg.weak_every != 0:
        return loss, None
    wb = next(weak_iter)
    wb = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in wb.items()}
    wout = model(wb)
    wloss, _ = compute_loss(wout, wb, model, cfg, pos_weight, ordinal_weights,
                            weak_pass=True, noise_channel=noise_channel)
    return loss + cfg.weak_stream_weight * wloss, float(wloss.detach())


def run_epoch(model, loader, cfg, onto, optimizer=None, scheduler=None, scaler=None,
              device="cpu", train: bool = True, amp_dtype=None,
              pos_weight=None, thresholds=None, ordinal_weights=None,
              weak_iter=None, noise_channel=None):
    import torch

    model.train() if train else model.eval()
    keys = ("symptoms", "severity", "risk", "stress")
    buf = {k: [] for k in keys}
    tgt = {k: [] for k in keys}
    unc, total_loss, n = [], 0.0, 0

    step = 0
    weak_loss_sum, weak_steps = 0.0, 0
    for batch in loader:
        batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
        with torch.set_grad_enabled(train):
            if amp_dtype is not None:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    out = model(batch)
                    loss, _ = compute_loss(out, batch, model, cfg, pos_weight,
                                           ordinal_weights)
                    loss, wl = _add_weak_term(loss, model, cfg, pos_weight,
                                              ordinal_weights, weak_iter, device,
                                              train, step, noise_channel)
            else:
                out = model(batch)
                loss, _ = compute_loss(out, batch, model, cfg, pos_weight,
                                       ordinal_weights)
                loss, wl = _add_weak_term(loss, model, cfg, pos_weight,
                                          ordinal_weights, weak_iter, device,
                                          train, step, noise_channel)
        if wl is not None:
            weak_loss_sum += wl
            weak_steps += 1
        step += 1

        if train:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
            if scheduler is not None:
                scheduler.step()

        total_loss += float(loss.detach())
        n += 1
        for k in keys:
            buf[k].append(out[k].detach().float().cpu().numpy())
            tgt[k].append(batch[k].detach().float().cpu().numpy())
        if "log_var" in out:
            unc.append(out["log_var"].detach().float().cpu().numpy())

    preds = {k: np.concatenate(v) if v else np.zeros((0,)) for k, v in buf.items()}
    targets = {k: np.concatenate(v) if v else np.zeros((0,)) for k, v in tgt.items()}
    bundle = compute_metrics({
        "symptom_prob": 1 / (1 + np.exp(-preds["symptoms"])),
        "severity_logits": preds["severity"], "risk_logits": preds["risk"],
        "stress_prob": 1 / (1 + np.exp(-preds["stress"]))}, targets, onto,
        thresholds=thresholds)

    selective = None
    if unc:
        u = np.exp(np.concatenate(unc))            # predictive variance
        prob = 1 / (1 + np.exp(-preds["symptoms"]))
        selective = selective_metrics(prob.ravel(), targets["symptoms"].ravel(),
                                      u.ravel())
    return total_loss / max(n, 1), bundle, selective, {
        "symptom_prob": 1 / (1 + np.exp(-preds["symptoms"])),
        "symptoms": targets["symptoms"],
        "severity_pred": preds["severity"], "severity_true": targets["severity"],
        "risk_pred": preds["risk"], "risk_true": targets["risk"],
        "stress_prob": 1 / (1 + np.exp(-preds["stress"])),
        "stress_true": targets["stress"],
        # Logged so the weak stream can be confirmed to have actually run, and
        # its loss tracked separately from the gold objective.
        "weak_loss": (weak_loss_sum / weak_steps) if weak_steps else None,
        "weak_steps": weak_steps}


def _f(v) -> str:
    return " n/a " if v is None else f"{v:.3f}"


def _print_epoch(epoch, epochs, tr_loss, tr, va_loss, va, secs, best: bool,
                 remaining: float | None = None):
    """Both train and validation, every epoch."""
    def line(tag, loss, b):
        return (f"  {tag:<6} loss {loss:7.4f}  symF1 {_f(b['groups']['all']['macro_f1'])}"
                f"  symAUROC {_f(b['groups']['all']['macro_auroc'])}"
                f"  sevQWK {_f(b['tasks']['severity_depression']['qwk'])}"
                f"  riskQWK {_f(b['tasks']['severity_risk']['qwk'])}"
                f"  comp {_f(b['composite'])}")
    eta = f"  eta {remaining / 60:.0f}m" if remaining else ""
    print(f"epoch {epoch:>2}/{epochs}  [{secs:.0f}s]{eta}"
          f"{'  *best' if best else ''}")
    print(line("train", tr_loss, tr))
    print(line("val", va_loss, va), flush=True)


# --------------------------------------------------------------------------- #
# one run
# --------------------------------------------------------------------------- #
def train_one(rung: str, seed: int, cfg: ProposedConfig, processed_dir: str,
              graph_dir: str | None, out_dir: str, onto: Ontology | None = None,
              resume: bool = True, verbose: bool = True) -> dict:
    """Train one ladder rung at one seed, writing everything to ``out_dir``."""
    import torch
    from torch.utils.data import DataLoader
    from transformers import get_linear_schedule_with_warmup

    onto = onto or Ontology.load()
    os.makedirs(out_dir, exist_ok=True)
    plots_dir = os.path.join(out_dir, "plots")
    history_path = os.path.join(out_dir, "history.csv")
    last_ckpt = os.path.join(out_dir, "checkpoint_last.pt")
    best_ckpt = os.path.join(out_dir, "checkpoint_best.pt")

    set_seed(seed)
    graph = None
    if cfg.use_graph or True:          # features also initialise the concept queries
        try:
            graph = SymptomGraph.load(graph_dir) if graph_dir else None
        except Exception as exc:
            print(f"  graph unavailable ({type(exc).__name__}); continuing without it")
            graph = None
    if cfg.use_graph and graph is None:
        raise RuntimeError(
            f"rung {rung} needs the graph but it could not be loaded from {graph_dir!r}. "
            f"Run notebook 03 first.")

    frames = load_frames(processed_dir, onto, cfg)
    model, tokenizer = build_model(cfg, onto, graph)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    # must precede the optimiser, which skips parameters with requires_grad=False
    if cfg.freeze_encoder_layers:
        freeze_lower_layers(model.encoder, cfg.freeze_encoder_layers, verbose)

    collate = make_collate(tokenizer, cfg.max_length)
    loaders = {
        split: DataLoader(ProposedDataset(frames[split]),
                          batch_size=cfg.batch_size if split == "train"
                          else cfg.eval_batch_size,
                          shuffle=(split == "train"), collate_fn=collate)
        for split in ("train", "val", "test")}

    # Auxiliary weak stream. Its length does not affect steps per epoch, so the
    # size of the weak pool is free: 50k labels cost the same as 6k.
    weak_loader = None
    if frames.get("weak"):
        weak_loader = DataLoader(
            ProposedDataset(frames["weak"]),
            batch_size=cfg.weak_batch_size or cfg.batch_size,
            shuffle=True, drop_last=True, collate_fn=collate)
        if verbose:
            print(f"  weak stream: {len(frames['weak']['text']):,} rows, "
                  f"batch {cfg.weak_batch_size or cfg.batch_size}, "
                  f"every {cfg.weak_every} step(s), "
                  f"weight {cfg.weak_stream_weight} "
                  f"-> steps/epoch unchanged at {len(loaders['train'])}")

    enc_params, head_params = [], []
    for pname, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (enc_params if pname.startswith("encoder.") else head_params).append(p)
    optimizer = torch.optim.AdamW(
        [{"params": enc_params, "lr": cfg.lr},
         {"params": head_params, "lr": cfg.head_lr}],
        weight_decay=cfg.weight_decay)
    steps = max(len(loaders["train"]), 1) * cfg.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(cfg.warmup_ratio * steps), steps)
    pos_weight = None
    if cfg.pos_weight_cap and cfg.pos_weight_cap > 1:
        pw = positive_weights(frames["train"]["symptoms"], cfg.pos_weight_cap)
        pos_weight = torch.tensor(pw, device=device)
        if verbose:
            print(f"  pos_weight: min {pw.min():.2f} max {pw.max():.2f} "
                  f"mean {pw.mean():.2f}")

    ordinal_weights = None
    if cfg.ordinal_class_weights:
        ordinal_weights = {}
        for key, n_cls in (("severity", N_SEVERITY), ("risk", N_RISK)):
            cw = class_weights(frames["train"][key], n_cls)
            if cw is not None:
                ordinal_weights[key] = torch.tensor(cw, device=device)
        if verbose and ordinal_weights:
            for k, v in ordinal_weights.items():
                print(f"  {k} class weights: "
                      f"{[round(float(x), 2) for x in v.tolist()]}")

    noise_channel = None
    if cfg.use_noise_correction:
        noise_channel = load_noise_channel(cfg.noise_channel_path, onto, device,
                                           verbose=verbose)

    amp_dtype, scaler = resolve_amp(cfg, device)
    if verbose and amp_dtype is not None:
        print(f"  autocast: {str(amp_dtype).replace('torch.', '')}"
              f"{' with grad scaler' if scaler else ''}")

    fingerprint, arch_fields = architecture_fingerprint(rung, cfg, onto)
    start_epoch, best_score, best_epoch = 1, -math.inf, None
    if resume and os.path.isfile(last_ckpt):
        ck = torch.load(last_ckpt, map_location=device, weights_only=False)
        saved = ck.get("fingerprint")
        if saved and saved != fingerprint:
            old = ck.get("fingerprint_fields", {})
            changed = {k: (old.get(k), v) for k, v in arch_fields.items()
                       if old.get(k) != v}
            raise RuntimeError(
                f"Refusing to resume: {last_ckpt} came from a different architecture.\n"
                + "\n".join(f"    {k}: checkpoint={o!r} now={n!r}"
                            for k, (o, n) in changed.items())
                + f"\n\nDelete the stale run and start fresh:\n    !rm -rf {out_dir}")
        model.load_state_dict(ck["model"])
        if ck.get("optimizer"):
            optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        if scaler is not None and ck.get("scaler"):
            scaler.load_state_dict(ck["scaler"])
        start_epoch = ck["epoch"] + 1
        best_score, best_epoch = ck.get("best_score", -math.inf), ck.get("best_epoch")
        restore_rng_state(ck.get("rng"))
        if verbose:
            print(f"  resumed after epoch {ck['epoch']} "
                  f"(best {cfg.selection_metric}={best_score:.4f} @ epoch {best_epoch})")

    write_json(os.path.join(out_dir, "config.json"),
               {"rung": rung, "seed": seed, "device": device,
                   "encoder": model.loaded_name,
                   "encoder_requested": model.encoder_requested,
                   "encoder_fallback_used": model.encoder_fallback_used,
                   "architecture_fingerprint": fingerprint,
                   "n_weak_rows": frames["train"]["n_weak_rows"],
                "trainable_parameters": sum(p.numel() for p in model.parameters()
                                            if p.requires_grad),
                **asdict(cfg)})

    if verbose and start_epoch > cfg.epochs:
        print(f"  all {cfg.epochs} epochs already done; re-evaluating best checkpoint")

    best_train_bundle = None
    epoch_times = []
    since_improved = 0
    for epoch in range(start_epoch, cfg.epochs + 1):
        t0 = time.time()
        tr_loss, tr_bundle, _, tr_extra = run_epoch(
            model, loaders["train"], cfg, onto, optimizer, scheduler, scaler,
            device, True, amp_dtype=amp_dtype, pos_weight=pos_weight,
            ordinal_weights=ordinal_weights, noise_channel=noise_channel,
            weak_iter=cycle_loader(weak_loader) if weak_loader else None)
        va_loss, va_bundle, va_sel, _ = run_epoch(
            model, loaders["val"], cfg, onto, device=device, train=False,
            amp_dtype=amp_dtype, pos_weight=pos_weight,
            ordinal_weights=ordinal_weights)
        # "neg_val_loss" selects the epoch with the LOWEST validation loss.
        # selection_score is mean(symptom F1, symptom AUROC, severity QWK): two
        # of three terms are symptom metrics that keep climbing while the
        # ordinal heads memorise, so it picked epoch 10 at ~2x the minimum
        # validation loss on two seeds. Validation loss weights the tasks the
        # way training does and cannot be gamed by one saturating task.
        score = (-va_loss if cfg.selection_metric == "neg_val_loss"
                 else va_bundle.get(cfg.selection_metric))
        score = -math.inf if score is None else score
        improved = score > best_score
        secs = time.time() - t0

        row = {"rung": rung, "seed": seed, "epoch": epoch,
               "train_loss": tr_loss, "val_loss": va_loss,
               "lr_encoder": optimizer.param_groups[0]["lr"],
               "lr_head": optimizer.param_groups[1]["lr"],
               "epoch_seconds": round(secs, 2),
               "timestamp": pd.Timestamp.utcnow().isoformat()}
        row["train_weak_loss"] = tr_extra.get("weak_loss")
        row["train_weak_steps"] = tr_extra.get("weak_steps")
        row.update(flatten_metrics(tr_bundle, "train"))
        row.update(flatten_metrics(va_bundle, "val"))
        if va_sel:
            row["val_aurc"] = va_sel["aurc"]
            row["val_risk_at_80_coverage"] = va_sel["risk_at_80_coverage"]
        pd.DataFrame([row]).to_csv(history_path, mode="a", index=False,
                                   header=not os.path.exists(history_path))

        if improved:
            best_score, best_epoch = score, epoch
            # keep this epoch's train metrics so no extra train pass is needed later
            best_train_bundle = tr_bundle
            write_json(os.path.join(out_dir, "best_train_metrics.json"),
                       tr_bundle, indent=None)
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "best_score": best_score, "best_epoch": best_epoch,
                        "rung": rung, "seed": seed, "fingerprint": fingerprint,
                        "fingerprint_fields": arch_fields}, best_ckpt)
        if epoch % max(cfg.checkpoint_every, 1) == 0 or epoch == cfg.epochs:
            torch.save({"model": model.state_dict(),
                        "optimizer": optimizer.state_dict()
                        if cfg.save_optimizer_state else None,
                        "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict() if scaler else None,
                        "epoch": epoch, "best_score": best_score,
                        "best_epoch": best_epoch, "fingerprint": fingerprint,
                        "fingerprint_fields": arch_fields,
                        "rng": capture_rng_state()}, last_ckpt)

        since_improved = 0 if improved else since_improved + 1
        epoch_times.append(secs)
        if verbose:
            remaining = (cfg.epochs - epoch) * (sum(epoch_times) / len(epoch_times))
            _print_epoch(epoch, cfg.epochs, tr_loss, tr_bundle, va_loss, va_bundle,
                         secs, improved, remaining)
        if cfg.early_stopping_patience and since_improved >= cfg.early_stopping_patience:
            if verbose:
                print(f"  early stop: no improvement for "
                      f"{since_improved} epochs (best epoch {best_epoch})")
            break

    history = pd.read_csv(history_path)
    if os.path.isfile(best_ckpt):
        ck = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])

    # Only val and test need a fresh pass from the best checkpoint. Train metrics
    # are reused from the best epoch, which saves a full pass over the training
    # set per seed - the single largest saving on a time-limited GPU.
    thresholds = None
    if cfg.tune_thresholds:
        _, _, _, raw = run_epoch(model, loaders["val"], cfg, onto, device=device,
                                 train=False, amp_dtype=amp_dtype)
        thresholds = tune_thresholds(raw["symptom_prob"], raw["symptoms"])
        if verbose:
            print(f"  tuned thresholds on val: min {thresholds.min():.2f} "
                  f"max {thresholds.max():.2f}")

    finals, selectives = {}, {}
    for split in ("val", "test"):
        _, bundle, sel, extra = run_epoch(model, loaders[split], cfg, onto,
                                          device=device, train=False,
                                          amp_dtype=amp_dtype, thresholds=thresholds)
        finals[split] = bundle
        if sel:
            selectives[split] = sel
        # Per-example predictions, so a PAIRED bootstrap across seeds is possible
        # later without retraining. Aggregate metrics alone cannot support a
        # significance test, which the review asked for.
        np.savez_compressed(
            os.path.join(out_dir, f"predictions_{split}.npz"),
            **{k: v for k, v in extra.items() if isinstance(v, np.ndarray)})

    if best_train_bundle is None:
        cached = os.path.join(out_dir, "best_train_metrics.json")
        if os.path.isfile(cached):
            try:
                best_train_bundle = read_json(cached)
            except (ValueError, OSError):
                best_train_bundle = None     # unreadable: recompute below
    if best_train_bundle is None:
        _, best_train_bundle, _, _ = run_epoch(model, loaders["train"], cfg, onto,
                                               device=device, train=False,
                                               amp_dtype=amp_dtype,
                                               thresholds=thresholds)
    finals["train"] = best_train_bundle

    payload = {
        "rung": rung, "seed": seed, "encoder": model.loaded_name,
        "encoder_requested": model.encoder_requested,
        "encoder_fallback_used": model.encoder_fallback_used,
        "selection_metric": cfg.selection_metric,
        "best_epoch": best_epoch,
        "best_val_score": None if best_score == -math.inf else best_score,
        "epochs_completed": int(history["epoch"].max()),
        "device": device, "config": asdict(cfg),
        "components": {k: getattr(cfg, k) for k in
                       ("use_bottleneck", "use_graph", "use_weak",
                        "use_contrastive", "use_uncertainty",
                        "use_monotone_ordinal", "use_noise_correction")},
        "splits": {k: len(frames[k]["text"]) for k in frames},
        "n_weak_rows": frames["train"]["n_weak_rows"],
        "train": finals["train"], "val": finals["val"], "test": finals["test"],
        "selective": selectives,
        "thresholds": None if thresholds is None else
                      {s: float(t) for s, t in zip(onto.symptoms, thresholds)},
    }
    write_json(os.path.join(out_dir, "performance_metrics.json"), payload)

    set_publication_style()
    plot_history_pub(history, plots_dir, title=f"{rung} (seed {seed})")
    plot_per_symptom_pub(finals["train"], finals["val"], plots_dir,
                         title=f"{rung} (seed {seed}): per-symptom F1")
    if "val" in selectives:
        plot_risk_coverage(selectives["val"], plots_dir,
                           title=f"{rung} (seed {seed}): selective prediction")
    return payload


# --------------------------------------------------------------------------- #
# ladder
# --------------------------------------------------------------------------- #
def run_rung(rung: str, flags: dict, cfg: ProposedConfig, processed_dir: str,
             graph_dir: str | None, root: str, onto: Ontology | None = None,
             resume: bool = True, verbose: bool = True) -> dict:
    onto = onto or Ontology.load()
    rung_cfg = cfg.with_flags(flags)
    base = os.path.join(root, rung)
    runs = []
    for seed in rung_cfg.seeds:
        if verbose:
            print(f"\n----- {rung} | seed {seed} -----")
        runs.append(train_one(rung, seed, rung_cfg, processed_dir, graph_dir,
                              os.path.join(base, f"seed_{seed}"), onto,
                              resume=resume, verbose=verbose))

    agg_dir = os.path.join(base, "aggregate")
    os.makedirs(agg_dir, exist_ok=True)
    aggregates = {s: aggregate_seeds([r[s] for r in runs], onto)
                  for s in ("train", "val", "test")}
    payload = {"rung": rung, "components": runs[0]["components"],
               "seeds": list(rung_cfg.seeds), "n_seeds": len(runs),
               "encoder": runs[0]["encoder"],
               "encoder_fallback_used": any(r.get("encoder_fallback_used")
                                            for r in runs),
               "per_seed_best_epoch": {str(r["seed"]): r["best_epoch"] for r in runs},
               "aggregate": aggregates["val"], "aggregate_by_split": aggregates,
               "config": asdict(rung_cfg)}
    write_json(os.path.join(agg_dir, "aggregate_metrics.json"), payload)

    headline_table(aggregates["val"], label=rung).to_csv(
        os.path.join(agg_dir, "summary.csv"), index=False)
    headline_table(aggregates["test"], label=rung).to_csv(
        os.path.join(agg_dir, "summary_test.csv"), index=False)
    set_publication_style()
    plot_seed_aggregate(aggregates["val"], os.path.join(agg_dir, "plots"), title=rung)

    hist = [pd.read_csv(os.path.join(base, f"seed_{s}", "history.csv"))
            for s in rung_cfg.seeds
            if os.path.isfile(os.path.join(base, f"seed_{s}", "history.csv"))]
    if hist:
        pd.concat(hist, ignore_index=True).to_csv(
            os.path.join(agg_dir, "history_all_seeds.csv"), index=False)
    return payload


def run_ladder(cfg: ProposedConfig, processed_dir: str, graph_dir: str | None,
               root: str, rungs=None, onto: Ontology | None = None,
               resume: bool = True, verbose: bool = True, ladder=None) -> dict:
    """Train a ladder, or the subset named in ``rungs``.

    ``ladder`` defaults to the v1 ``LADDER``; pass ``LADDER_V2`` for the revised
    model. Keeping both callable from one function means the two generations
    share every piece of the training, checkpointing and logging path.
    """
    onto = onto or Ontology.load()
    os.makedirs(root, exist_ok=True)
    ladder = LADDER if ladder is None else ladder
    selected = [(n, f) for n, f in ladder if rungs is None or n in rungs]
    results = {}
    for name, flags in selected:
        if verbose:
            print(f"\n{'=' * 72}\n{name}  {flags}\n{'=' * 72}")
        results[name] = run_rung(name, flags, cfg, processed_dir, graph_dir, root,
                                 onto, resume=resume, verbose=verbose)
    return results
