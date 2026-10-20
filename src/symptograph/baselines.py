"""Three strong baselines for SymptoGraph-LLM, trained under one shared protocol.

The baselines are deliberately *flat*: they predict the targets directly, with no
concept bottleneck (N1), no comorbidity graph (N2) and no weak supervision (N3)
unless explicitly enabled. They are the reference the proposed model has to beat.

| key | encoder | why it is here |
|---|---|---|
| ``tfidf_mlp`` | TF-IDF (fit on train only) into an MLP | classical bag-of-words reference, still competitive on short clinical text |
| ``bilstm_attn`` | trainable embeddings, BiLSTM, additive attention | the standard deep baseline in this literature, no pretraining |
| ``mentalroberta`` | MentalRoBERTa fine-tuned | domain-pretrained transformer, the strongest published family on these corpora |

Fairness
--------
All three share the same multi-task head module, loss, masking, optimiser family,
schedule, epoch budget, batch size, seeds, metrics and model-selection rule. Only
the text encoder differs, so a difference in score is attributable to the encoder.

The one exception is the learning rate, which is set per architecture. Forcing a
single value would either diverge the transformer or barely move the MLP, and a
crippled baseline is worse than no baseline. Set ``shared_lr`` to override.

Checkpointing
-------------
``checkpoint_last.pt`` is written at the end of every epoch with model, optimiser,
scheduler and RNG state. Resuming continues from the next epoch and **appends** to
``history.csv``; it never truncates or rewrites it.
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

from .metrics import (aggregate_seeds, compute_metrics, flatten_metrics,
                      headline_table, plot_history, plot_per_symptom_bars,
                      plot_seed_aggregate, positive_weights, class_weights, tune_thresholds)
from .jsonio import write_json
from .preprocess import Ontology
from .rng import capture_rng_state, restore_rng_state

BASELINES = ("tfidf_mlp", "bilstm_attn", "mentalroberta")

# name -> builder(cfg, onto, train_texts, SharedHeads, load_transformer)
# Populated by symptograph.sota so comparison models share this whole harness:
# the same splits, loss, class weights, threshold tuning, selection rule,
# checkpointing, history.csv and saved predictions.
MODEL_REGISTRY: dict = {}


def register_model(name: str):
    def deco(fn):
        MODEL_REGISTRY[name] = fn
        return fn
    return deco
N_SEVERITY, N_RISK = 4, 5


@dataclass
class TrainConfig:
    """One configuration shared by every baseline."""
    epochs: int = 8
    batch_size: int = 16
    eval_batch_size: int = 64
    max_length: int = 256
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    grad_clip: float = 1.0
    dropout: float = 0.1
    hidden_dim: int = 256
    seeds: tuple = (0, 1, 2)
    shared_lr: float | None = None
    lr_by_model: dict = field(default_factory=lambda: {
        "tfidf_mlp": 1e-3, "bilstm_attn": 1e-3, "mentalroberta": 2e-5})
    task_weights: dict = field(default_factory=lambda: {
        "symptoms": 1.0, "severity": 1.0, "risk": 1.0, "stress": 0.5})
    # weight on a model's own auxiliary consistency term, if it exposes one
    consistency_weight: float = 1.0
    transformer_name: str = "mental/mental-roberta-base"
    transformer_fallback: str = "roberta-base"
    # mental/mental-roberta-base is a gated repo. Without this, a 403 silently
    # substitutes roberta-base and the run is still labelled "mentalroberta".
    strict_encoder: bool = False
    tfidf_features: int = 30_000
    vocab_size: int = 30_000
    embed_dim: int = 200
    use_weak_labels: bool = False
    weak_label_path: str | None = None
    limit_train: int | None = None
    limit_eval: int | None = None
    amp: bool = True
    # "selection_score", "composite", or "neg_val_loss" (lowest validation loss)
    selection_metric: str = "selection_score"
    pos_weight_cap: float = 10.0
    tune_thresholds: bool = True
    # Parity with proposed.ProposedConfig. Without these the baselines were
    # handicapped on exactly the two tasks the proposed model claims to win:
    # severity is 72% majority class and risk has 340 training rows, and
    # unweighted cross-entropy collapses both heads onto the majority class.
    # Any severity or risk margin measured against a baseline that lacks this
    # is measuring the class weighting, not the architecture.
    ordinal_class_weights: bool = True
    label_smoothing: float = 0.0
    scheduler: str = "linear"          # "linear" or "cosine"
    early_stopping_patience: int = 0   # 0 disables
    strict_encoder_note: str = "see strict_encoder"

    def lr_for(self, model: str) -> float:
        """Learning rate for one model.

        Unknown names fall back to the transformer rate, because every
        comparison model registered in ``MODEL_REGISTRY`` is transformer-backed
        and must use the same encoder rate as ``mentalroberta`` for the
        comparison to isolate architecture. A per-architecture rate is still
        needed for the non-transformer baselines: 2e-5 does not move a
        randomly-initialised MLP, and 1e-3 diverges a pretrained encoder.
        """
        if self.shared_lr is not None:
            return self.shared_lr
        return self.lr_by_model.get(model, self.lr_by_model["mentalroberta"])


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
# Config fields that change the shape of the network. If any of these differs
# from the checkpoint, resuming is impossible and must fail with a clear message
# rather than a bare tensor-shape error.
ARCH_FIELDS = ("hidden_dim", "dropout", "embed_dim", "vocab_size", "tfidf_features",
               "max_length", "transformer_name", "transformer_fallback")


def architecture_fingerprint(name: str, cfg: "TrainConfig", onto: Ontology) -> tuple:
    """Short hash of everything that determines the parameter shapes."""
    import hashlib
    fields = {"baseline": name, "n_symptoms": len(onto.symptoms),
              **{f: getattr(cfg, f) for f in ARCH_FIELDS}}
    blob = json.dumps(fields, sort_keys=True, default=str).encode()
    return hashlib.md5(blob).hexdigest()[:12], fields


def _check_checkpoint_compatible(ck: dict, fingerprint: str, fields: dict,
                                 ckpt_path: str, out_dir: str) -> None:
    """Refuse to resume from a checkpoint built with a different architecture."""
    saved = ck.get("fingerprint")
    if saved is None or saved == fingerprint:
        return
    old = ck.get("fingerprint_fields", {})
    changed = {k: (old.get(k), v) for k, v in fields.items() if old.get(k) != v}
    lines = "\n".join(f"    {k}: checkpoint={o!r} now={n!r}" for k, (o, n) in changed.items())
    raise RuntimeError(
        f"Refusing to resume: {ckpt_path} was written with a different architecture.\n"
        f"{lines}\n\n"
        f"This usually means a quick-test run and a full run shared an output folder.\n"
        f"Delete the stale run and start this one fresh:\n"
        f"    !rm -rf {out_dir}")


def set_seed(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_splits(processed_dir: str, onto: Ontology, cfg: TrainConfig) -> dict:
    """Read labelled.parquet and optionally merge weak labels into the train split."""
    df = pd.read_parquet(os.path.join(processed_dir, "labelled.parquet"))
    sym_cols = onto.columns
    frames = {}
    for split in ("train", "val", "test"):
        part = df[df["split"] == split].reset_index(drop=True)
        limit = cfg.limit_train if split == "train" else cfg.limit_eval
        if limit and limit < len(part):
            # sample rather than take the head, so a smoke test still exercises
            # the severity and risk heads instead of only the symptom head
            part = part.sample(n=limit, random_state=0).reset_index(drop=True)
        frames[split] = part

    if cfg.use_weak_labels and cfg.weak_label_path and os.path.exists(cfg.weak_label_path):
        weak = pd.read_parquet(cfg.weak_label_path)
        wl_cols = [c for c in weak.columns if c.startswith("wl_")]
        extra = weak[["post_id", "text"] + wl_cols].copy()
        for c in wl_cols:
            name = c[3:]
            if onto.col(name) in sym_cols:
                extra[onto.col(name)] = extra[c].astype("int8")
        for c in sym_cols:
            if c not in extra:
                extra[c] = np.int8(-1)
        extra["split"] = "train"
        extra["author_hash"] = None
        keep = ["post_id", "text", "split", "author_hash"] + sym_cols
        frames["train"] = pd.concat([frames["train"][keep], extra[keep]],
                                    ignore_index=True)
    return frames


def targets_from_frame(part: pd.DataFrame, onto: Ontology) -> dict:
    sym = part[onto.columns].to_numpy(dtype=np.int8) if len(part) else \
        np.zeros((0, len(onto.symptoms)), dtype=np.int8)
    def col(name):
        return (part[name].to_numpy(dtype=np.float32) if name in part
                else np.full(len(part), np.nan, dtype=np.float32))
    return {"symptoms": sym, "severity": col("severity_depression"),
            "risk": col("severity_risk"), "stress": col("stress")}


class TextDataset:
    """Rows of text plus masked multi-task targets."""

    def __init__(self, part: pd.DataFrame, onto: Ontology):
        self.texts = part["text"].fillna("").tolist()
        # Stable integer per author, for models that add same-author edges.
        # -1 where the corpus has no author (non-Reddit rows and weak rows).
        if "author_hash" in part:
            self.group = np.array(
                [(-1 if not a else (int(hashlib.md5(str(a).encode()).hexdigest()[:8], 16)))
                 for a in part["author_hash"].tolist()], dtype=np.int64)
        else:
            self.group = np.full(len(self.texts), -1, dtype=np.int64)
        t = targets_from_frame(part, onto)
        self.symptoms = t["symptoms"].astype(np.float32)
        self.severity, self.risk, self.stress = t["severity"], t["risk"], t["stress"]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return {"text": self.texts[i], "symptoms": self.symptoms[i],
                "severity": self.severity[i], "risk": self.risk[i],
                "stress": self.stress[i], "group": self.group[i]}


def make_collate(encoder_kind: str, helper):
    """Build the collate function for the encoder in use."""
    import torch

    def collate(batch):
        out = {
            "symptoms": torch.tensor(np.stack([b["symptoms"] for b in batch])),
            "severity": torch.tensor(np.array([b["severity"] for b in batch])),
            "risk": torch.tensor(np.array([b["risk"] for b in batch])),
            "stress": torch.tensor(np.array([b["stress"] for b in batch])),
            "group": torch.tensor(np.array([b["group"] for b in batch])),
        }
        texts = [b["text"] for b in batch]
        if encoder_kind == "tfidf":
            out["features"] = torch.tensor(
                helper.transform(texts).toarray().astype(np.float32))
        elif encoder_kind == "tokens":
            ids, lengths = helper(texts)
            out["input_ids"] = torch.tensor(ids)
            out["lengths"] = torch.tensor(lengths)
        else:
            # a helper may return extra tensors, e.g. one tokenised view per
            # prompt perspective; merge whatever it produces
            out.update(helper(texts))
        return out
    return collate


class SimpleVocab:
    """Deterministic word vocabulary built from the training split only."""

    def __init__(self, texts, max_size: int, max_length: int):
        from collections import Counter
        counts = Counter(w for t in texts for w in str(t).lower().split())
        self.itos = ["<pad>", "<unk>"] + [w for w, _ in counts.most_common(max_size - 2)]
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.max_length = max_length

    def __len__(self):
        return len(self.itos)

    def __call__(self, texts):
        ids, lengths = [], []
        for t in texts:
            toks = [self.stoi.get(w, 1) for w in str(t).lower().split()][:self.max_length]
            lengths.append(max(len(toks), 1))
            ids.append(toks + [0] * (self.max_length - len(toks)))
        return np.array(ids, dtype=np.int64), np.array(lengths, dtype=np.int64)


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
def load_transformer(cfg: "TrainConfig", model_name: str):
    """``(encoder, tokenizer, loaded_name, fell_back)``, honouring strict_encoder.

    Extracted so the comparison models in ``symptograph.sota`` cannot acquire a
    different encoder-loading policy from the baselines. A silent fallback here
    once produced a run labelled "mentalroberta" that was actually RoBERTa for
    two of three seeds.
    """
    from transformers import AutoModel, AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(model_name)
        enc = AutoModel.from_pretrained(model_name)
        return enc, tok, model_name, False
    except Exception as exc:
        if cfg.strict_encoder:
            raise RuntimeError(
                f"Could not load the requested encoder {model_name!r}: "
                f"{type(exc).__name__}: {exc}\n\n"
                f"strict_encoder=True, so this will not silently fall back to "
                f"{cfg.transformer_fallback!r}.") from exc
        print("  " + "!" * 68)
        print(f"  WARNING: could not load {model_name} ({type(exc).__name__}).")
        print(f"  FALLING BACK to {cfg.transformer_fallback}.")
        print("  " + "!" * 68, flush=True)
        fb = cfg.transformer_fallback
        return (AutoModel.from_pretrained(fb), AutoTokenizer.from_pretrained(fb),
                fb, True)


def build_model(name: str, cfg: TrainConfig, onto: Ontology, train_texts):
    """Return ``(model, encoder_kind, collate_helper)``."""
    import torch
    import torch.nn as nn

    n_sym = len(onto.symptoms)

    class SharedHeads(nn.Module):
        """Identical output heads for every baseline, so only the encoder differs."""

        def __init__(self, dim):
            super().__init__()
            self.drop = nn.Dropout(cfg.dropout)
            self.symptoms = nn.Linear(dim, n_sym)
            self.severity = nn.Linear(dim, N_SEVERITY)
            self.risk = nn.Linear(dim, N_RISK)
            self.stress = nn.Linear(dim, 1)

        def forward(self, h):
            h = self.drop(h)
            return {"symptoms": self.symptoms(h), "severity": self.severity(h),
                    "risk": self.risk(h), "stress": self.stress(h).squeeze(-1)}

    if name in MODEL_REGISTRY:
        return MODEL_REGISTRY[name](cfg, onto, train_texts, SharedHeads,
                                    load_transformer)

    class TfidfMLP(nn.Module):
        def __init__(self, in_dim):
            super().__init__()
            self.body = nn.Sequential(
                nn.Linear(in_dim, 512), nn.ReLU(), nn.Dropout(cfg.dropout),
                nn.Linear(512, cfg.hidden_dim), nn.ReLU())
            self.heads = SharedHeads(cfg.hidden_dim)

        def forward(self, batch):
            return self.heads(self.body(batch["features"]))

    class BiLSTMAttn(nn.Module):
        def __init__(self, vocab):
            super().__init__()
            self.emb = nn.Embedding(vocab, cfg.embed_dim, padding_idx=0)
            self.lstm = nn.LSTM(cfg.embed_dim, cfg.hidden_dim, batch_first=True,
                                bidirectional=True)
            self.attn = nn.Linear(2 * cfg.hidden_dim, 1)
            self.heads = SharedHeads(2 * cfg.hidden_dim)

        def forward(self, batch):
            x = self.emb(batch["input_ids"])
            h, _ = self.lstm(x)
            pad = (batch["input_ids"] == 0)
            scores = self.attn(h).squeeze(-1).masked_fill(pad, float("-inf"))
            w = torch.softmax(scores, dim=1).unsqueeze(-1)
            return self.heads((h * w).sum(dim=1))

    class TransformerEnc(nn.Module):
        def __init__(self, model_name):
            super().__init__()
            from transformers import AutoModel
            self.enc = AutoModel.from_pretrained(model_name)
            self.heads = SharedHeads(self.enc.config.hidden_size)

        def forward(self, batch):
            out = self.enc(input_ids=batch["input_ids"],
                           attention_mask=batch["attention_mask"])
            mask = batch["attention_mask"].unsqueeze(-1).float()
            pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            return self.heads(pooled)

    if name == "tfidf_mlp":
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(max_features=cfg.tfidf_features, ngram_range=(1, 2),
                              min_df=2, sublinear_tf=True)
        vec.fit(train_texts)                      # train split only, no leakage
        dim = len(vec.vocabulary_)
        return TfidfMLP(dim), "tfidf", vec

    if name == "bilstm_attn":
        vocab = SimpleVocab(train_texts, cfg.vocab_size, cfg.max_length)
        return BiLSTMAttn(len(vocab)), "tokens", vocab

    if name == "mentalroberta":
        from transformers import AutoTokenizer
        model_name = cfg.transformer_name
        fallback_used = False
        try:
            tok = AutoTokenizer.from_pretrained(model_name)
            model = TransformerEnc(model_name)
        except Exception as exc:                  # offline or gated
            if cfg.strict_encoder:
                raise RuntimeError(
                    f"Could not load the requested encoder {model_name!r}: "
                    f"{type(exc).__name__}: {exc}\n\n"
                    f"strict_encoder=True, so this will not silently fall back to "
                    f"{cfg.transformer_fallback!r}. A baseline named "
                    f"'mentalroberta' that actually ran RoBERTa would misreport "
                    f"the comparison.") from exc
            print("  " + "!" * 68)
            print(f"  WARNING: could not load {model_name} ({type(exc).__name__}).")
            print(f"  FALLING BACK to {cfg.transformer_fallback}. This baseline is "
                  f"NOT MentalRoBERTa.")
            print("  " + "!" * 68, flush=True)
            fallback_used = True
            model_name = cfg.transformer_fallback
            tok = AutoTokenizer.from_pretrained(model_name)
            model = TransformerEnc(model_name)

        def helper(texts):
            return tok(list(texts), padding=True, truncation=True,
                       max_length=cfg.max_length, return_tensors="pt")
        model.loaded_name = model_name
        model.encoder_fallback_used = fallback_used
        return model, "hf", helper

    raise ValueError(f"unknown baseline {name!r}")


# --------------------------------------------------------------------------- #
# loss
# --------------------------------------------------------------------------- #
def multitask_loss(out: dict, batch: dict, cfg: TrainConfig,
                   pos_weight=None, ordinal_weights=None) -> tuple:
    """Masked multi-task loss. Rows missing a target contribute nothing to it."""
    import torch
    import torch.nn.functional as F

    device = out["symptoms"].device
    total = torch.zeros((), device=device)
    parts = {}

    y = batch["symptoms"]
    mask = y >= 0
    if mask.any():
        target = y[mask].clamp(min=0)
        bce = F.binary_cross_entropy_with_logits(
            out["symptoms"][mask], target, reduction="none")
        if pos_weight is not None:
            pw = pos_weight.unsqueeze(0).expand_as(y)[mask]
            wts = torch.where(target > 0.5, pw, torch.ones_like(pw))
            l = (bce * wts).sum() / wts.sum().clamp(min=1e-6)
        else:
            l = bce.mean()
        total = total + cfg.task_weights["symptoms"] * l
        parts["symptoms"] = float(l.detach())

    for key, logits_key, weight_key in (("severity", "severity", "severity"),
                                        ("risk", "risk", "risk")):
        t = batch[key]
        m = ~torch.isnan(t)
        if m.any():
            cw = (ordinal_weights or {}).get(key)
            l = F.cross_entropy(out[logits_key][m], t[m].long(), weight=cw,
                                label_smoothing=cfg.label_smoothing)
            total = total + cfg.task_weights[weight_key] * l
            parts[key] = float(l.detach())

    t = batch["stress"]
    m = ~torch.isnan(t)
    if m.any():
        l = F.binary_cross_entropy_with_logits(out["stress"][m], t[m])
        total = total + cfg.task_weights["stress"] * l
        parts["stress"] = float(l.detach())

    if "consistency" in out:
        total = total + cfg.consistency_weight * out["consistency"]
        parts["consistency"] = float(out["consistency"].detach())

    return total, parts


# --------------------------------------------------------------------------- #
# epoch loops
# --------------------------------------------------------------------------- #
def _collect(preds: dict) -> dict:
    return {"symptom_prob": 1 / (1 + np.exp(-preds["symptoms"])),
            "severity_logits": preds["severity"],
            "risk_logits": preds["risk"],
            "stress_prob": 1 / (1 + np.exp(-preds["stress"]))}


def _run_epoch(model, loader, cfg, onto, optimizer=None, scheduler=None,
               scaler=None, device="cpu", train: bool = True,
               pos_weight=None, thresholds=None, ordinal_weights=None):
    """One pass. Training predictions are gathered from the same forward passes,
    so train metrics cost nothing extra."""
    import torch

    model.train() if train else model.eval()
    buf = {k: [] for k in ("symptoms", "severity", "risk", "stress")}
    tgt = {k: [] for k in ("symptoms", "severity", "risk", "stress")}
    total_loss, n_batches = 0.0, 0

    for batch in loader:
        batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
        with torch.set_grad_enabled(train):
            if scaler is not None and train:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out = model(batch)
                    loss, _ = multitask_loss(out, batch, cfg, pos_weight,
                                             ordinal_weights)
            else:
                out = model(batch)
                loss, _ = multitask_loss(out, batch, cfg, pos_weight,
                                         ordinal_weights)

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
        n_batches += 1
        for k in buf:
            buf[k].append(out[k].detach().float().cpu().numpy())
            tgt[k].append(batch[k].detach().float().cpu().numpy())

    preds = {k: np.concatenate(v) if v else np.zeros((0,)) for k, v in buf.items()}
    targets = {k: np.concatenate(v) if v else np.zeros((0,)) for k, v in tgt.items()}
    bundle = compute_metrics(_collect(preds), targets, onto, thresholds=thresholds)
    col = _collect(preds)
    return total_loss / max(n_batches, 1), bundle, {
        "symptom_prob": col["symptom_prob"], "symptoms": targets["symptoms"],
        # significance.compare_models reads these; without them no paired test
        # against a baseline is possible at all.
        "severity_pred": preds["severity"], "severity_true": targets["severity"],
        "risk_pred": preds["risk"], "risk_true": targets["risk"],
        "stress_prob": col["stress_prob"], "stress_true": targets["stress"]}


# --------------------------------------------------------------------------- #
# single run with checkpoint and resume
# --------------------------------------------------------------------------- #
def train_one(name: str, seed: int, cfg: TrainConfig, processed_dir: str,
              out_dir: str, onto: Ontology | None = None,
              resume: bool = True, verbose: bool = True) -> dict:
    """Train one baseline at one seed. Resumable, and appends to history.csv."""
    import torch
    from torch.utils.data import DataLoader

    onto = onto or Ontology.load()
    os.makedirs(out_dir, exist_ok=True)
    plots_dir = os.path.join(out_dir, "plots")
    history_path = os.path.join(out_dir, "history.csv")
    last_ckpt = os.path.join(out_dir, "checkpoint_last.pt")
    best_ckpt = os.path.join(out_dir, "checkpoint_best.pt")

    set_seed(seed)
    frames = load_splits(processed_dir, onto, cfg)
    model, kind, helper = build_model(name, cfg, onto, frames["train"]["text"].tolist())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    ordinal_weights = None
    if cfg.ordinal_class_weights:
        ordinal_weights = {}
        for key, col, n_cls in (("severity", "severity_depression", N_SEVERITY),
                                ("risk", "severity_risk", N_RISK)):
            vals = (frames["train"][col].to_numpy(dtype=np.float32)
                    if col in frames["train"] else np.zeros(0, dtype=np.float32))
            cw = class_weights(vals, n_cls)
            if cw is not None:
                ordinal_weights[key] = torch.tensor(cw, device=device)
        if verbose and ordinal_weights:
            for k, v in ordinal_weights.items():
                print(f"  {k} class weights: "
                      f"{[round(float(x), 2) for x in v.tolist()]}")

    collate = make_collate(kind, helper)
    loaders = {}
    for split in ("train", "val", "test"):
        loaders[split] = DataLoader(
            TextDataset(frames[split], onto),
            batch_size=cfg.batch_size if split == "train" else cfg.eval_batch_size,
            shuffle=(split == "train"), collate_fn=collate,
            num_workers=0, drop_last=False)

    decay = [p for n, p in model.named_parameters()
             if p.requires_grad and not any(x in n for x in ("bias", "LayerNorm"))]
    no_decay = [p for n, p in model.named_parameters()
                if p.requires_grad and any(x in n for x in ("bias", "LayerNorm"))]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}], lr=cfg.lr_for(name))
    steps = max(len(loaders["train"]), 1) * cfg.epochs
    from transformers import get_linear_schedule_with_warmup
    _mk = get_linear_schedule_with_warmup
    if cfg.scheduler == "cosine":
        from transformers import get_cosine_schedule_with_warmup as _mk
    elif cfg.scheduler != "linear":
        raise ValueError(f"scheduler must be 'linear' or 'cosine', got {cfg.scheduler!r}")
    scheduler = _mk(
        optimizer, int(cfg.warmup_ratio * steps), steps)
    scaler = torch.amp.GradScaler("cuda") if (cfg.amp and device == "cuda") else None

    pos_weight = None
    if cfg.pos_weight_cap and cfg.pos_weight_cap > 1:
        sym = targets_from_frame(frames["train"], onto)["symptoms"].astype(np.float32)
        pos_weight = torch.tensor(positive_weights(sym, cfg.pos_weight_cap),
                                  device=device)

    fingerprint, arch_fields = architecture_fingerprint(name, cfg, onto)

    start_epoch, best_score, best_epoch = 1, -math.inf, None
    if resume and os.path.isfile(last_ckpt):
        ck = torch.load(last_ckpt, map_location=device, weights_only=False)
        _check_checkpoint_compatible(ck, fingerprint, arch_fields, last_ckpt, out_dir)
        try:
            model.load_state_dict(ck["model"])
        except RuntimeError as exc:
            raise RuntimeError(
                f"checkpoint in {out_dir} does not match the current configuration.\n"
                f"It was almost certainly written by a different run (a quick test, "
                f"for example).\nDelete that folder and start again:\n"
                f"    !rm -rf {out_dir}\n\noriginal error: {exc}") from exc
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        if scaler is not None and ck.get("scaler"):
            scaler.load_state_dict(ck["scaler"])
        start_epoch = ck["epoch"] + 1
        best_score, best_epoch = ck.get("best_score", -math.inf), ck.get("best_epoch")
        restore_rng_state(ck.get("rng"))
        if verbose:
            print(f"  resumed from epoch {ck['epoch']} "
                  f"(best {cfg.selection_metric}={best_score:.4f} @ {best_epoch})")

    write_json(os.path.join(out_dir, "config.json"),
               {"baseline": name, "seed": seed, "device": device,
                   "learning_rate": cfg.lr_for(name),
                   "encoder": getattr(model, "loaded_name", name),
                   "encoder_requested": cfg.transformer_name if name == "mentalroberta" else name,
                   "encoder_fallback_used": getattr(model, "encoder_fallback_used", False),
                "architecture_fingerprint": fingerprint,
                **asdict(cfg)})

    if start_epoch > cfg.epochs and verbose:
        print(f"  already finished {cfg.epochs} epochs; re-evaluating from best checkpoint")

    for epoch in range(start_epoch, cfg.epochs + 1):
        t0 = time.time()
        train_loss, train_bundle, _ = _run_epoch(
            model, loaders["train"], cfg, onto, optimizer, scheduler, scaler,
            device, train=True, pos_weight=pos_weight, ordinal_weights=ordinal_weights)
        val_loss, val_bundle, _ = _run_epoch(
            model, loaders["val"], cfg, onto, device=device, train=False,
            pos_weight=pos_weight, ordinal_weights=ordinal_weights)
        # neg_val_loss selects the epoch with the LOWEST validation loss.
        # selection_score is 2/3 symptom metrics, which keep climbing while the
        # ordinal heads memorise their small training sets.
        score = (-val_loss if cfg.selection_metric == "neg_val_loss"
                 else val_bundle.get(cfg.selection_metric))
        score = -math.inf if score is None else score

        row = {"baseline": name, "seed": seed, "epoch": epoch,
               "train_loss": train_loss, "val_loss": val_loss,
               "lr": optimizer.param_groups[0]["lr"],
               "epoch_seconds": round(time.time() - t0, 2),
               "timestamp": pd.Timestamp.utcnow().isoformat()}
        row.update(flatten_metrics(train_bundle, "train"))
        row.update(flatten_metrics(val_bundle, "val"))
        # append; never truncate an existing history
        pd.DataFrame([row]).to_csv(history_path, mode="a", index=False,
                                   header=not os.path.exists(history_path))

        improved = score > best_score
        if improved:
            since_improved = 0
        else:
            since_improved = locals().get("since_improved", 0) + 1
        if improved:
            best_score, best_epoch = score, epoch
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "best_score": best_score, "best_epoch": best_epoch,
                        "baseline": name, "seed": seed,
                        "fingerprint": fingerprint,
                        "fingerprint_fields": arch_fields}, best_ckpt)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict() if scaler else None,
                    "epoch": epoch, "best_score": best_score, "best_epoch": best_epoch,
                    "fingerprint": fingerprint, "fingerprint_fields": arch_fields,
                    "rng": capture_rng_state()}, last_ckpt)

        if verbose:
            print(f"  epoch {epoch:>2}/{cfg.epochs}  loss {train_loss:.4f}/{val_loss:.4f}"
                  f"  val macroF1 {_fmt(val_bundle['groups']['all']['macro_f1'])}"
                  f"  sevQWK {_fmt(val_bundle['tasks']['severity_depression']['qwk'])}"
                  f"  composite {_fmt(score)}{'  *best' if improved else ''}"
                  f"  [{row['epoch_seconds']:.0f}s]", flush=True)

    history = pd.read_csv(history_path)
    if os.path.isfile(best_ckpt):
        ck = torch.load(best_ckpt, map_location=device, weights_only=False)
        _check_checkpoint_compatible(ck, fingerprint, arch_fields, best_ckpt, out_dir)
        model.load_state_dict(ck["model"])
    thresholds = None
    if cfg.tune_thresholds:
        _, _, raw = _run_epoch(model, loaders["val"], cfg, onto,
                               device=device, train=False)
        thresholds = tune_thresholds(raw["symptom_prob"], raw["symptoms"])
    _, final_train, _ = _run_epoch(model, loaders["train"], cfg, onto,
                                   device=device, train=False, thresholds=thresholds)
    _, final_val, _ = _run_epoch(model, loaders["val"], cfg, onto,
                                 device=device, train=False, thresholds=thresholds)
    _, final_test, test_raw = _run_epoch(model, loaders["test"], cfg, onto,
                                         device=device, train=False,
                                         thresholds=thresholds)
    _, _, val_raw = _run_epoch(model, loaders["val"], cfg, onto, device=device,
                               train=False, thresholds=thresholds)
    # Per-example predictions, so significance.compare_models can run a PAIRED
    # bootstrap of any comparison model against the proposed model without
    # retraining either of them.
    for split, raw in (("val", val_raw), ("test", test_raw)):
        np.savez_compressed(
            os.path.join(out_dir, f"predictions_{split}.npz"),
            **{k: v for k, v in raw.items() if isinstance(v, np.ndarray)})

    payload = {
        "baseline": name, "seed": seed,
        "encoder": getattr(model, "loaded_name", name),
        "encoder_fallback_used": getattr(model, "encoder_fallback_used", False),
        "learning_rate": cfg.lr_for(name),
        "selection_metric": cfg.selection_metric,
        "best_epoch": best_epoch, "best_val_score": None if best_score == -math.inf else best_score,
        "epochs_completed": int(history["epoch"].max()),
        "device": device,
        "config": asdict(cfg),
        "splits": {k: int(len(v)) for k, v in frames.items()},
        "thresholds": None if thresholds is None else
                      {s: float(t) for s, t in zip(onto.symptoms, thresholds)},
        "train": final_train, "val": final_val, "test": final_test,
    }
    write_json(os.path.join(out_dir, "performance_metrics.json"), payload)

    plot_history(history, plots_dir, title=f"{name} seed {seed}")
    plot_per_symptom_bars(final_train, final_val, plots_dir, title=f"{name} seed {seed}")
    return payload


def _fmt(v) -> str:
    return "  n/a" if v is None else f"{v:.3f}"


# --------------------------------------------------------------------------- #
# all seeds, all baselines
# --------------------------------------------------------------------------- #
def run_baseline(name: str, cfg: TrainConfig, processed_dir: str, root: str,
                 onto: Ontology | None = None, resume: bool = True,
                 verbose: bool = True) -> dict:
    """Train one baseline across every seed and aggregate the results."""
    onto = onto or Ontology.load()
    base = os.path.join(root, name)
    runs = []
    for seed in cfg.seeds:
        if verbose:
            print(f"\n[{name}] seed {seed}")
        runs.append(train_one(name, seed, cfg, processed_dir,
                              os.path.join(base, f"seed_{seed}"), onto,
                              resume=resume, verbose=verbose))

    agg_dir = os.path.join(base, "aggregate")
    os.makedirs(agg_dir, exist_ok=True)
    aggregates = {split: aggregate_seeds([r[split] for r in runs], onto)
                  for split in ("train", "val", "test")}
    payload = {"baseline": name, "seeds": list(cfg.seeds),
               "n_seeds": len(runs), "config": asdict(cfg),
               "encoder": runs[0]["encoder"] if runs else None,
               "per_seed_best_epoch": {str(r["seed"]): r["best_epoch"] for r in runs},
               "aggregate": aggregates["val"],
               "aggregate_by_split": aggregates}
    write_json(os.path.join(agg_dir, "aggregate_metrics.json"), payload)

    table = headline_table(aggregates["val"], label=name)
    table.to_csv(os.path.join(agg_dir, "summary.csv"), index=False)
    test_table = headline_table(aggregates["test"], label=name)
    test_table.to_csv(os.path.join(agg_dir, "summary_test.csv"), index=False)
    plot_seed_aggregate(aggregates["val"], os.path.join(agg_dir, "plots"), title=name)

    histories = []
    for seed in cfg.seeds:
        h = os.path.join(base, f"seed_{seed}", "history.csv")
        if os.path.isfile(h):
            histories.append(pd.read_csv(h))
    if histories:
        pd.concat(histories, ignore_index=True).to_csv(
            os.path.join(agg_dir, "history_all_seeds.csv"), index=False)
    return payload


def run_all(cfg: TrainConfig, processed_dir: str, root: str,
            baselines=BASELINES, onto: Ontology | None = None,
            resume: bool = True, verbose: bool = True) -> dict:
    onto = onto or Ontology.load()
    os.makedirs(root, exist_ok=True)
    results = {}
    for name in baselines:
        results[name] = run_baseline(name, cfg, processed_dir, root, onto,
                                     resume=resume, verbose=verbose)
    return results
