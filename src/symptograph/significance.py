"""Paired bootstrap significance tests and confidence intervals.

The review was explicit: "no significance test or paired bootstrap across
seeds. Three seeds is the minimum." Mean +/- std over three seeds cannot tell
you whether a 0.02 gap is real, and suicide-risk QWK is measured on **74 test
rows**, where a handful of examples moves it by more than any architectural
change does.

This module works from the ``predictions_{split}.npz`` files written by
``proposed.train_one``, so no retraining is needed.

Two separate sources of variance, deliberately kept apart:

* **Example variance** - resampling test examples with replacement, holding the
  model fixed. This is what a confidence interval on a single run means, and it
  is the one that matters for the tiny risk split.
* **Seed variance** - the spread across training runs. Reported as the seed std,
  not folded into the bootstrap, because 3 seeds is far too few to bootstrap.

The paired test resamples the *same* example indices for both models, which is
what makes it paired and what gives it power on small splits.
"""
from __future__ import annotations

import os

import numpy as np

QWK_TASKS = {"severity": 4, "risk": 5}


# --------------------------------------------------------------------------- #
# metrics, recomputed from raw predictions so the bootstrap can resample
# --------------------------------------------------------------------------- #
def _qwk(true: np.ndarray, pred: np.ndarray, n_cls: int) -> float:
    """Quadratic weighted kappa. Returns nan if either side is constant."""
    if len(true) == 0:
        return float("nan")
    o = np.zeros((n_cls, n_cls))
    for t, p in zip(true.astype(int), pred.astype(int)):
        o[t, p] += 1
    w = np.array([[(i - j) ** 2 for j in range(n_cls)] for i in range(n_cls)],
                 dtype=float) / (n_cls - 1) ** 2
    ht = np.bincount(true.astype(int), minlength=n_cls).astype(float)
    hp = np.bincount(pred.astype(int), minlength=n_cls).astype(float)
    e = np.outer(ht, hp) / max(len(true), 1)
    den = (w * e).sum()
    if den <= 0:
        return float("nan")
    return float(1.0 - (w * o).sum() / den)


def _macro_f1(true: np.ndarray, prob: np.ndarray, thr: np.ndarray | float = 0.5):
    """Macro F1 over symptom columns, ignoring masked (-1) entries."""
    pred = (prob >= thr).astype(int)
    f1s = []
    for k in range(true.shape[1]):
        m = true[:, k] >= 0
        if m.sum() == 0:
            continue
        t, p = true[m, k], pred[m, k]
        tp = float(((t == 1) & (p == 1)).sum())
        fp = float(((t == 0) & (p == 1)).sum())
        fn = float(((t == 1) & (p == 0)).sum())
        if tp + fp + fn == 0:
            continue
        f1s.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s)) if f1s else float("nan")


def _auroc(true: np.ndarray, score: np.ndarray) -> float:
    """Rank-based AUROC for one binary column."""
    m = true >= 0
    t, s = true[m], score[m]
    pos, neg = (t == 1).sum(), (t == 0).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks over ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / cnt)[inv]
    return float((ranks[t == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def _macro_auroc(true: np.ndarray, prob: np.ndarray) -> float:
    vals = [_auroc(true[:, k], prob[:, k]) for k in range(true.shape[1])]
    vals = [v for v in vals if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def metric_fns(thresholds=None) -> dict:
    """Metric name -> callable(predictions dict, index array) -> float."""
    thr = 0.5 if thresholds is None else np.asarray(thresholds)

    def sym_f1(d, idx):
        return _macro_f1(d["symptoms"][idx], d["symptom_prob"][idx], thr)

    def sym_auroc(d, idx):
        return _macro_auroc(d["symptoms"][idx], d["symptom_prob"][idx])

    def ordinal(task, n_cls):
        def f(d, idx):
            true, pred = d[f"{task}_true"][idx], d[f"{task}_pred"][idx]
            m = ~np.isnan(true)
            if m.sum() == 0:
                return float("nan")
            return _qwk(true[m], pred[m].argmax(-1), n_cls)
        return f

    def stress_auroc(d, idx):
        true = d["stress_true"][idx]
        m = ~np.isnan(true)
        if m.sum() == 0:
            return float("nan")
        return _auroc(true[m].astype(int), d["stress_prob"][idx][m])

    return {"symptom_macro_f1": sym_f1, "symptom_macro_auroc": sym_auroc,
            "severity_qwk": ordinal("severity", 4),
            "risk_qwk": ordinal("risk", 5),
            "stress_auroc": stress_auroc}


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_predictions(run_dir: str, seed: int, split: str = "test") -> dict | None:
    p = os.path.join(run_dir, f"seed_{seed}", f"predictions_{split}.npz")
    if not os.path.isfile(p):
        return None
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


# --------------------------------------------------------------------------- #
# bootstrap
# --------------------------------------------------------------------------- #
def bootstrap_ci(pred: dict, metric, n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> dict:
    """Percentile CI over resampled test examples, model held fixed."""
    rng = np.random.default_rng(seed)
    n = len(pred["symptoms"])
    point = metric(pred, np.arange(n))
    draws = np.empty(n_boot)
    for b in range(n_boot):
        draws[b] = metric(pred, rng.integers(0, n, n))
    draws = draws[~np.isnan(draws)]
    if len(draws) == 0:
        return {"point": point, "lo": float("nan"), "hi": float("nan"), "n": n}
    return {"point": point,
            "lo": float(np.percentile(draws, 100 * alpha / 2)),
            "hi": float(np.percentile(draws, 100 * (1 - alpha / 2))),
            "n": n, "n_valid_draws": int(len(draws))}


def paired_bootstrap(pred_a: dict, pred_b: dict, metric, n_boot: int = 2000,
                     seed: int = 0) -> dict:
    """Two-sided paired bootstrap of ``metric(b) - metric(a)``.

    The same resampled example indices are used for both models, so the test
    only sees differences in prediction, not differences in which examples were
    drawn. ``p`` is the fraction of resamples in which the sign of the
    difference flips relative to the observed difference.
    """
    n = len(pred_a["symptoms"])
    if len(pred_b["symptoms"]) != n:
        raise ValueError("paired test needs the same evaluation set on both sides")
    rng = np.random.default_rng(seed)
    full = np.arange(n)
    obs = metric(pred_b, full) - metric(pred_a, full)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = metric(pred_b, idx) - metric(pred_a, idx)
    diffs = diffs[~np.isnan(diffs)]
    if len(diffs) == 0:
        return {"observed": obs, "p": float("nan"), "lo": float("nan"),
                "hi": float("nan"), "n_boot": 0}
    p = float(2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
    return {"observed": float(obs), "p": min(p, 1.0),
            "lo": float(np.percentile(diffs, 2.5)),
            "hi": float(np.percentile(diffs, 97.5)),
            "n_boot": int(len(diffs))}


def compare_models(run_a: str, run_b: str, seeds=(0, 1, 2), split: str = "test",
                   n_boot: int = 2000, thresholds=None, label_a="A", label_b="B"):
    """Paired bootstrap per seed, plus the across-seed spread, per metric.

    Returns a list of dicts ready for a DataFrame. A difference is reported as
    significant only if the paired test clears 0.05 **and** the sign is
    consistent across all seeds - a per-seed p-value alone ignores training
    variance entirely.
    """
    import itertools
    fns = metric_fns(thresholds)
    rows = []
    for name, fn in fns.items():
        per_seed = []
        for s in seeds:
            a, b = load_predictions(run_a, s, split), load_predictions(run_b, s, split)
            if a is None or b is None:
                continue
            r = paired_bootstrap(a, b, fn, n_boot=n_boot, seed=s)
            r["seed"] = s
            per_seed.append(r)
        if not per_seed:
            continue
        obs = [r["observed"] for r in per_seed]
        ps = [r["p"] for r in per_seed]
        signs = {np.sign(round(o, 6)) for o in obs if round(o, 6) != 0}
        rows.append({
            "metric": name,
            f"{label_a}_vs_{label_b}_mean_delta": round(float(np.mean(obs)), 4),
            "delta_per_seed": [round(o, 4) for o in obs],
            "max_p": round(float(np.max(ps)), 4),
            "consistent_sign": len(signs) <= 1,
            "significant": bool(float(np.max(ps)) < 0.05 and len(signs) <= 1),
        })
    return rows
