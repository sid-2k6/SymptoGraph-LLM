"""Metrics, plots and seed aggregation shared by the baselines and the proposed model.

Every model writes the same ``performance_metrics.json`` schema, so the proposed
SymptoGraph-LLM run can be dropped into the same comparison without touching the
reporting code.

Masking
-------
Symptom targets use 1 present, 0 absent and -1 not annotated. Only non-negative
entries contribute to any metric; a symptom with no annotations reports ``None``
rather than a misleading zero.
"""
from __future__ import annotations

import json
import math
import os
from typing import Sequence

import numpy as np
import pandas as pd

SELECTION_METRIC = "composite"


# --------------------------------------------------------------------------- #
# symptom groups
# --------------------------------------------------------------------------- #
def symptom_groups(onto) -> dict[str, list[str]]:
    """Named groups used for the 'group metrics' reporting."""
    from .weak_label import weak_only_symptoms
    weak_only = set(weak_only_symptoms(onto))
    groups = {
        "all": list(onto.symptoms),
        "gold_supervised": [s for s in onto.symptoms if s not in weak_only],
        "weak_only": [s for s in onto.symptoms if s in weak_only],
    }
    by_instrument: dict[str, list[str]] = {}
    for m in onto.meta:
        by_instrument.setdefault(m["instrument"], []).append(m["name"])
    for inst, names in by_instrument.items():
        groups[f"instrument_{inst.replace('-', '')}"] = names
    return groups


# --------------------------------------------------------------------------- #
# primitive metrics
# --------------------------------------------------------------------------- #
def _safe(fn, *a, **kw):
    try:
        v = fn(*a, **kw)
        return None if v is None or (isinstance(v, float) and not math.isfinite(v)) else float(v)
    except Exception:
        return None


def binary_metrics(y: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict:
    from sklearn.metrics import (average_precision_score, f1_score,
                                 precision_score, recall_score, roc_auc_score)
    if len(y) == 0:
        return {"n": 0, "n_positive": 0, "positive_rate": None, "precision": None,
                "recall": None, "f1": None, "auroc": None, "ap": None, "accuracy": None}
    pred = (prob >= threshold).astype(int)
    both = len(np.unique(y)) > 1
    return {
        "n": int(len(y)),
        "n_positive": int(y.sum()),
        "positive_rate": float(y.mean()),
        "precision": _safe(precision_score, y, pred, zero_division=0),
        "recall": _safe(recall_score, y, pred, zero_division=0),
        "f1": _safe(f1_score, y, pred, zero_division=0),
        "accuracy": float((pred == y).mean()),
        "auroc": _safe(roc_auc_score, y, prob) if both else None,
        "ap": _safe(average_precision_score, y, prob) if both else None,
    }


def ordinal_metrics(y: np.ndarray, logits: np.ndarray, n_classes: int) -> dict:
    """Accuracy, macro-F1, MAE and quadratic weighted kappa for an ordinal target."""
    from sklearn.metrics import cohen_kappa_score, f1_score
    if len(y) == 0:
        return {"n": 0, "accuracy": None, "macro_f1": None, "mae": None, "qwk": None}
    pred = logits.argmax(axis=1)
    return {
        "n": int(len(y)),
        "accuracy": float((pred == y).mean()),
        "macro_f1": _safe(f1_score, y, pred, average="macro", zero_division=0),
        "mae": float(np.abs(pred - y).mean()),
        "qwk": _safe(cohen_kappa_score, y, pred, weights="quadratic",
                     labels=list(range(n_classes))),
    }


def _nanmean(values: Sequence) -> float | None:
    vals = [v for v in values if v is not None and math.isfinite(v)]
    return float(np.mean(vals)) if vals else None


# --------------------------------------------------------------------------- #
# full metric bundle
# --------------------------------------------------------------------------- #
def compute_metrics(preds: dict, targets: dict, onto, threshold: float = 0.5) -> dict:
    """Build the full nested metric bundle for one split.

    ``preds`` holds ``symptom_prob`` (N, K), ``severity_logits`` (N, 4),
    ``risk_logits`` (N, 5) and ``stress_prob`` (N,). ``targets`` holds the
    matching arrays with -1 / NaN marking absent labels.
    """
    out: dict = {"per_symptom": {}, "groups": {}, "tasks": {}}

    # ---- symptoms, individually
    prob, y = preds.get("symptom_prob"), targets.get("symptoms")
    if prob is not None and y is not None and len(y):
        for i, s in enumerate(onto.symptoms):
            m = y[:, i] >= 0
            out["per_symptom"][s] = binary_metrics(y[m, i].astype(int), prob[m, i], threshold)
    else:
        for s in onto.symptoms:
            out["per_symptom"][s] = binary_metrics(np.array([]), np.array([]))

    # ---- symptoms, by group
    for gname, members in symptom_groups(onto).items():
        rows = [out["per_symptom"][s] for s in members if out["per_symptom"][s]["n"] > 0]
        if not rows:
            out["groups"][gname] = {"n_symptoms": 0, "macro_f1": None, "macro_auroc": None,
                                    "macro_ap": None, "micro_f1": None,
                                    "macro_precision": None, "macro_recall": None}
            continue
        idx = [i for i, s in enumerate(onto.symptoms)
               if s in members and out["per_symptom"][s]["n"] > 0]
        micro = None
        if prob is not None and y is not None and len(y) and idx:
            from sklearn.metrics import f1_score
            sub_y, sub_p = y[:, idx], prob[:, idx]
            mask = sub_y >= 0
            if mask.any():
                micro = _safe(f1_score, sub_y[mask].astype(int),
                              (sub_p[mask] >= threshold).astype(int), zero_division=0)
        out["groups"][gname] = {
            "n_symptoms": len(rows),
            "macro_f1": _nanmean([r["f1"] for r in rows]),
            "macro_precision": _nanmean([r["precision"] for r in rows]),
            "macro_recall": _nanmean([r["recall"] for r in rows]),
            "macro_auroc": _nanmean([r["auroc"] for r in rows]),
            "macro_ap": _nanmean([r["ap"] for r in rows]),
            "micro_f1": micro,
        }

    # ---- other tasks
    for key, n_classes, tgt_key, pred_key in [
        ("severity_depression", 4, "severity", "severity_logits"),
        ("severity_risk", 5, "risk", "risk_logits"),
    ]:
        yt, lg = targets.get(tgt_key), preds.get(pred_key)
        if yt is not None and lg is not None and len(yt):
            m = ~np.isnan(yt)
            out["tasks"][key] = ordinal_metrics(yt[m].astype(int), lg[m], n_classes)
        else:
            out["tasks"][key] = ordinal_metrics(np.array([]), None, n_classes)

    yt, pp = targets.get("stress"), preds.get("stress_prob")
    if yt is not None and pp is not None and len(yt):
        m = ~np.isnan(yt)
        out["tasks"]["stress"] = binary_metrics(yt[m].astype(int), pp[m], threshold)
    else:
        out["tasks"]["stress"] = binary_metrics(np.array([]), np.array([]))

    # ---- single number for model selection
    out["composite"] = _nanmean([
        out["groups"]["all"]["macro_f1"],
        out["tasks"]["severity_depression"]["qwk"],
        out["tasks"]["severity_risk"]["qwk"],
    ])
    return out


def flatten_metrics(bundle: dict, prefix: str = "") -> dict:
    """Flatten the bundle for one row of history.csv."""
    flat: dict = {}
    p = f"{prefix}_" if prefix else ""
    flat[f"{p}composite"] = bundle.get("composite")
    for g, vals in bundle.get("groups", {}).items():
        for k, v in vals.items():
            flat[f"{p}group_{g}_{k}"] = v
    for t, vals in bundle.get("tasks", {}).items():
        for k, v in vals.items():
            flat[f"{p}{t}_{k}"] = v
    for s, vals in bundle.get("per_symptom", {}).items():
        for k in ("f1", "auroc", "precision", "recall"):
            flat[f"{p}sym_{s}_{k}"] = vals.get(k)
    return flat


# --------------------------------------------------------------------------- #
# seed aggregation
# --------------------------------------------------------------------------- #
def aggregate_seeds(bundles: list[dict], onto) -> dict:
    """Mean and standard deviation across seeds, preserving the bundle shape."""
    def ms(values):
        vals = [v for v in values if v is not None and isinstance(v, (int, float))
                and math.isfinite(v)]
        if not vals:
            return {"mean": None, "std": None, "n_seeds": 0}
        return {"mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "n_seeds": len(vals)}

    agg: dict = {"per_symptom": {}, "groups": {}, "tasks": {},
                 "composite": ms([b.get("composite") for b in bundles]),
                 "n_seeds": len(bundles)}
    for s in onto.symptoms:
        agg["per_symptom"][s] = {
            k: ms([b["per_symptom"][s].get(k) for b in bundles])
            for k in ("f1", "precision", "recall", "auroc", "ap")}
        agg["per_symptom"][s]["n"] = bundles[0]["per_symptom"][s]["n"] if bundles else 0
    for g in (bundles[0]["groups"] if bundles else {}):
        agg["groups"][g] = {k: ms([b["groups"][g].get(k) for b in bundles])
                            for k in bundles[0]["groups"][g]}
    for t in (bundles[0]["tasks"] if bundles else {}):
        agg["tasks"][t] = {k: ms([b["tasks"][t].get(k) for b in bundles])
                           for k in bundles[0]["tasks"][t]}
    return agg


def headline_table(agg: dict, label: str = "") -> pd.DataFrame:
    """The handful of numbers that go in the paper, as mean +/- std."""
    rows = []

    def add(name, node):
        if isinstance(node, dict) and "mean" in node:
            m, sd = node.get("mean"), node.get("std")
            rows.append({"model": label, "metric": name,
                         "mean": None if m is None else round(m, 4),
                         "std": None if sd is None else round(sd, 4),
                         "formatted": "-" if m is None else f"{m:.3f} +/- {sd or 0:.3f}"})

    add("composite", agg.get("composite", {}))
    for g in ("all", "gold_supervised", "weak_only"):
        if g in agg.get("groups", {}):
            add(f"symptom_macro_f1[{g}]", agg["groups"][g].get("macro_f1", {}))
            add(f"symptom_macro_auroc[{g}]", agg["groups"][g].get("macro_auroc", {}))
    for t, keys in (("severity_depression", ("qwk", "mae", "macro_f1")),
                    ("severity_risk", ("qwk", "mae", "macro_f1")),
                    ("stress", ("f1", "auroc"))):
        for k in keys:
            if t in agg.get("tasks", {}):
                add(f"{t}_{k}", agg["tasks"][t].get(k, {}))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# plots
# --------------------------------------------------------------------------- #
def _setup():
    import warnings
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # harmless when a figure has many tick labels and few columns
    warnings.filterwarnings("ignore", message=".*Tight layout not applied.*")
    return plt


def to_markdown_table(df: pd.DataFrame) -> str:
    """Markdown table without depending on the optional ``tabulate`` package."""
    header = [str(df.index.name or "")] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    for idx, row in df.iterrows():
        cells = [str(idx)] + ["" if pd.isna(v) else str(v) for v in row.tolist()]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plot_history(history: pd.DataFrame, out_dir: str, title: str = "") -> list[str]:
    """Loss and group-level metric curves, train against val, per epoch."""
    plt = _setup()
    os.makedirs(out_dir, exist_ok=True)
    saved = []

    panels = [
        ("loss", "train_loss", "val_loss", "loss"),
        ("composite", "train_composite", "val_composite", "composite score"),
        ("symptom_macro_f1", "train_group_all_macro_f1", "val_group_all_macro_f1",
         "symptom macro F1"),
        ("symptom_macro_auroc", "train_group_all_macro_auroc",
         "val_group_all_macro_auroc", "symptom macro AUROC"),
        ("severity_qwk", "train_severity_depression_qwk", "val_severity_depression_qwk",
         "depression severity QWK"),
        ("risk_qwk", "train_severity_risk_qwk", "val_severity_risk_qwk",
         "suicide risk QWK"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, (_, tr_col, va_col, ylabel) in zip(axes.ravel(), panels):
        for col, lab, style in ((tr_col, "train", "-o"), (va_col, "val", "-s")):
            if col in history:
                ax.plot(history["epoch"], history[col], style, label=lab, markersize=4)
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"{title} - training curves", fontsize=13)
    fig.tight_layout()
    p = os.path.join(out_dir, "curves_group.png")
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)

    # per-symptom F1 over epochs, val
    sym_cols = [c for c in history.columns
                if c.startswith("val_sym_") and c.endswith("_f1")]
    keep = [c for c in sym_cols if history[c].notna().any()]
    if keep:
        fig, ax = plt.subplots(figsize=(11, 6))
        for c in keep:
            ax.plot(history["epoch"], history[c], marker="o", markersize=3,
                    label=c.replace("val_sym_", "").replace("_f1", ""))
        ax.set_xlabel("epoch")
        ax.set_ylabel("val F1")
        ax.set_title(f"{title} - per-symptom validation F1")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=2, loc="best")
        fig.tight_layout()
        p = os.path.join(out_dir, "curves_per_symptom_val.png")
        fig.savefig(p, dpi=140, bbox_inches="tight")
        plt.close(fig)
        saved.append(p)

        fig, ax = plt.subplots(figsize=(min(1.1 * len(history), 14), 0.42 * len(keep) + 2))
        mat = history[keep].to_numpy(dtype=float).T
        im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_yticks(range(len(keep)))
        ax.set_yticklabels([c.replace("val_sym_", "").replace("_f1", "") for c in keep],
                           fontsize=8)
        ax.set_xticks(range(len(history)))
        ax.set_xticklabels(history["epoch"].tolist(), fontsize=8)
        ax.set_xlabel("epoch")
        ax.set_title(f"{title} - per-symptom val F1 by epoch")
        fig.colorbar(im, ax=ax, fraction=0.03)
        fig.tight_layout()
        p = os.path.join(out_dir, "heatmap_per_symptom_val.png")
        fig.savefig(p, dpi=140, bbox_inches="tight")
        plt.close(fig)
        saved.append(p)
    return saved


def plot_per_symptom_bars(train_bundle: dict, val_bundle: dict, out_dir: str,
                          title: str = "") -> str | None:
    """Final per-symptom F1, train against val, for the symptoms that have labels."""
    plt = _setup()
    os.makedirs(out_dir, exist_ok=True)
    names = [s for s, v in val_bundle["per_symptom"].items() if v["n"] > 0]
    if not names:
        return None
    tr = [(train_bundle["per_symptom"].get(s, {}).get("f1") or 0) for s in names]
    va = [(val_bundle["per_symptom"][s]["f1"] or 0) for s in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(7, 0.62 * len(names)), 5))
    ax.bar(x - 0.2, tr, 0.4, label="train")
    ax.bar(x + 0.2, va, 0.4, label="val")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1)
    ax.set_title(f"{title} - per-symptom F1 (final epoch)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    p = os.path.join(out_dir, "bars_per_symptom.png")
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return p


def plot_seed_aggregate(agg: dict, out_dir: str, title: str = "") -> list[str]:
    """Mean +/- std across seeds: group metrics and per-symptom F1."""
    plt = _setup()
    os.makedirs(out_dir, exist_ok=True)
    saved = []

    labels, means, stds = [], [], []
    for g in ("all", "gold_supervised", "instrument_PHQ9", "instrument_GAD7",
              "instrument_extra"):
        node = agg.get("groups", {}).get(g, {}).get("macro_f1")
        if node and node.get("mean") is not None:
            labels.append(f"macro F1\n{g}")
            means.append(node["mean"])
            stds.append(node.get("std") or 0)
    for t in ("severity_depression", "severity_risk"):
        node = agg.get("tasks", {}).get(t, {}).get("qwk")
        if node and node.get("mean") is not None:
            labels.append(f"QWK\n{t.replace('severity_', '')}")
            means.append(node["mean"])
            stds.append(node.get("std") or 0)
    if labels:
        fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(labels)), 4.5))
        ax.bar(range(len(labels)), means, yerr=stds, capsize=4, color="steelblue")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("score")
        ax.set_title(f"{title} - mean +/- std over {agg.get('n_seeds', 0)} seeds")
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        p = os.path.join(out_dir, "aggregate_group_metrics.png")
        fig.savefig(p, dpi=140, bbox_inches="tight")
        plt.close(fig)
        saved.append(p)

    names = [s for s, v in agg["per_symptom"].items()
             if v["f1"]["mean"] is not None and v.get("n", 0) > 0]
    if names:
        m = [agg["per_symptom"][s]["f1"]["mean"] for s in names]
        e = [agg["per_symptom"][s]["f1"]["std"] or 0 for s in names]
        fig, ax = plt.subplots(figsize=(max(7, 0.62 * len(names)), 5))
        ax.bar(range(len(names)), m, yerr=e, capsize=3, color="darkseagreen")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("val F1")
        ax.set_ylim(0, 1)
        ax.set_title(f"{title} - per-symptom val F1, mean +/- std")
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        p = os.path.join(out_dir, "aggregate_per_symptom.png")
        fig.savefig(p, dpi=140, bbox_inches="tight")
        plt.close(fig)
        saved.append(p)
    return saved


def plot_comparison(tables: dict[str, pd.DataFrame], out_dir: str,
                    metrics: Sequence[str] = ("composite",
                                              "symptom_macro_f1[all]",
                                              "severity_depression_qwk",
                                              "severity_risk_qwk")) -> str | None:
    """Grouped bars comparing every model on the headline metrics."""
    plt = _setup()
    os.makedirs(out_dir, exist_ok=True)
    models = list(tables)
    if not models:
        return None
    fig, ax = plt.subplots(figsize=(max(7, 2.1 * len(metrics)), 5))
    width = 0.8 / max(len(models), 1)
    for i, name in enumerate(models):
        t = tables[name].set_index("metric")
        means = [t["mean"].get(m) if m in t.index else None for m in metrics]
        stds = [t["std"].get(m) if m in t.index else None for m in metrics]
        xs = np.arange(len(metrics)) + i * width - 0.4 + width / 2
        ax.bar(xs, [0 if v is None else v for v in means], width,
               yerr=[0 if v is None else v for v in stds], capsize=3, label=name)
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels([m.replace("_", " ") for m in metrics], fontsize=9)
    ax.set_ylabel("score")
    ax.set_title("Model comparison (mean +/- std over seeds)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(out_dir, "comparison_headline.png")
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return p


# --------------------------------------------------------------------------- #
# comparison across runs
# --------------------------------------------------------------------------- #
def collect_runs(root: str) -> dict[str, dict]:
    """Find every ``aggregate/aggregate_metrics.json`` under ``root``."""
    found = {}
    if not os.path.isdir(root):
        return found
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name, "aggregate", "aggregate_metrics.json")
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as fh:
                found[name] = json.load(fh)
    return found


def build_leaderboard(root: str, out_dir: str) -> tuple[pd.DataFrame, dict]:
    """Combine every model under ``root`` into one comparison table."""
    runs = collect_runs(root)
    tables, frames = {}, []
    for name, payload in runs.items():
        agg = payload.get("aggregate", payload)
        t = headline_table(agg, label=name)
        tables[name] = t
        frames.append(t)
    if not frames:
        return pd.DataFrame(), tables
    combined = pd.concat(frames, ignore_index=True)
    os.makedirs(out_dir, exist_ok=True)
    combined.to_csv(os.path.join(out_dir, "all_models.csv"), index=False)

    wide = combined.pivot(index="metric", columns="model", values="formatted")
    wide.to_csv(os.path.join(out_dir, "leaderboard.csv"))
    with open(os.path.join(out_dir, "leaderboard.md"), "w", encoding="utf-8") as fh:
        fh.write("# Model comparison\n\nmean +/- std across seeds.\n\n")
        fh.write(to_markdown_table(wide))
        fh.write("\n")
    plot_comparison(tables, out_dir)
    return combined, tables


# --------------------------------------------------------------------------- #
# publication styling
# --------------------------------------------------------------------------- #
PUB_FONT = 20
PUB_DPI = 300


def set_publication_style(font_size: int = PUB_FONT) -> None:
    """Large fonts and heavy lines, for figures that go straight into the paper."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": font_size,
        "axes.titlesize": font_size + 2,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size - 4,
        "ytick.labelsize": font_size - 4,
        "legend.fontsize": font_size - 4,
        "figure.titlesize": font_size + 4,
        "axes.linewidth": 1.4,
        "lines.linewidth": 2.4,
        "lines.markersize": 8,
        "grid.alpha": 0.3,
        "savefig.dpi": PUB_DPI,
        "figure.dpi": 110,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _save(fig, path: str, dpi: int = PUB_DPI) -> str:
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# selective prediction (novelty N6)
# --------------------------------------------------------------------------- #
def selective_metrics(prob: np.ndarray, y: np.ndarray, uncertainty: np.ndarray,
                      n_points: int = 20) -> dict:
    """Risk-coverage behaviour when the most uncertain predictions are deferred.

    ``risk`` is the error rate on the retained predictions. ``aurc`` is the area
    under the risk-coverage curve, so **lower is better**. A model whose
    uncertainty is informative shows risk falling as coverage drops.
    """
    mask = y >= 0
    if not mask.any():
        return {"aurc": None, "risk_at_full_coverage": None, "curve": []}
    p, t, u = prob[mask], y[mask].astype(int), uncertainty[mask]
    wrong = ((p >= 0.5).astype(int) != t).astype(float)
    order = np.argsort(u)                       # most certain first
    wrong_sorted = wrong[order]

    curve, risks = [], []
    for frac in np.linspace(1.0 / n_points, 1.0, n_points):
        k = max(int(round(frac * len(wrong_sorted))), 1)
        risk = float(wrong_sorted[:k].mean())
        curve.append({"coverage": round(float(k / len(wrong_sorted)), 4),
                      "risk": round(risk, 4), "n_retained": k})
        risks.append(risk)
    return {"aurc": float(np.mean(risks)),
            "risk_at_full_coverage": float(wrong.mean()),
            "risk_at_80_coverage": curve[int(0.8 * n_points) - 1]["risk"],
            "risk_at_50_coverage": curve[int(0.5 * n_points) - 1]["risk"],
            "curve": curve}


def plot_risk_coverage(selective: dict, out_dir: str, title: str = "",
                       font_size: int = PUB_FONT) -> str | None:
    if not selective or not selective.get("curve"):
        return None
    set_publication_style(font_size)
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    cov = [c["coverage"] for c in selective["curve"]]
    risk = [c["risk"] for c in selective["curve"]]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot(cov, risk, "-o")
    ax.axhline(selective["risk_at_full_coverage"], ls="--", color="grey",
               label="no abstention")
    ax.set_xlabel("coverage")
    ax.set_ylabel("risk (error rate)")
    ax.set_title(f"{title}\nAURC = {selective['aurc']:.3f} (lower is better)")
    ax.grid(True)
    ax.legend()
    return _save(fig, os.path.join(out_dir, "risk_coverage.png"))


# --------------------------------------------------------------------------- #
# publication-quality reporting
# --------------------------------------------------------------------------- #
def plot_history_pub(history: pd.DataFrame, out_dir: str, title: str = "",
                     font_size: int = PUB_FONT) -> list[str]:
    """Train-versus-validation curves, one large panel per metric."""
    set_publication_style(font_size)
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    saved = []

    panels = [
        ("loss", "train_loss", "val_loss", "loss"),
        ("composite", "train_composite", "val_composite", "composite"),
        ("symptom_macro_f1", "train_group_all_macro_f1", "val_group_all_macro_f1",
         "symptom macro F1"),
        ("symptom_macro_auroc", "train_group_all_macro_auroc",
         "val_group_all_macro_auroc", "symptom macro AUROC"),
        ("severity_qwk", "train_severity_depression_qwk",
         "val_severity_depression_qwk", "severity QWK"),
        ("risk_qwk", "train_severity_risk_qwk", "val_severity_risk_qwk", "risk QWK"),
    ]
    for key, tr, va, ylabel in panels:
        if tr not in history and va not in history:
            continue
        fig, ax = plt.subplots(figsize=(9, 7))
        if tr in history:
            ax.plot(history["epoch"], history[tr], "-o", label="train")
        if va in history:
            ax.plot(history["epoch"], history[va], "-s", label="validation")
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True)
        ax.legend()
        saved.append(_save(fig, os.path.join(out_dir, f"curve_{key}.png")))

    # combined overview
    fig, axes = plt.subplots(2, 3, figsize=(26, 14))
    for ax, (key, tr, va, ylabel) in zip(axes.ravel(), panels):
        if tr in history:
            ax.plot(history["epoch"], history[tr], "-o", label="train")
        if va in history:
            ax.plot(history["epoch"], history[va], "-s", label="validation")
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    saved.append(_save(fig, os.path.join(out_dir, "curves_overview.png")))
    return saved


def plot_per_symptom_pub(train_bundle: dict, val_bundle: dict, out_dir: str,
                         title: str = "", font_size: int = PUB_FONT) -> str | None:
    """Per-symptom F1, train against validation."""
    set_publication_style(font_size)
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    names = [s for s, v in val_bundle["per_symptom"].items() if v["n"] > 0]
    if not names:
        return None
    tr = [(train_bundle["per_symptom"].get(s, {}).get("f1") or 0) for s in names]
    va = [(val_bundle["per_symptom"][s]["f1"] or 0) for s in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(12, 1.05 * len(names)), 8))
    ax.bar(x - 0.2, tr, 0.4, label="train")
    ax.bar(x + 0.2, va, 0.4, label="validation")
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_", " ") for n in names], rotation=45, ha="right")
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.grid(True, axis="y")
    ax.legend()
    return _save(fig, os.path.join(out_dir, "per_symptom_f1.png"))


def plot_ladder(tables: dict, out_dir: str, order: list[str],
                metrics=("composite", "symptom_macro_f1[all]",
                         "severity_depression_qwk", "severity_risk_qwk"),
                font_size: int = PUB_FONT) -> list[str]:
    """Ablation ladder: each metric against the rungs, in ladder order."""
    set_publication_style(font_size)
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    rungs = [r for r in order if r in tables]
    if not rungs:
        return []
    saved = []
    for metric in metrics:
        means, stds = [], []
        for r in rungs:
            t = tables[r].set_index("metric")
            means.append(t["mean"].get(metric) if metric in t.index else None)
            stds.append(t["std"].get(metric) if metric in t.index else None)
        if all(m is None for m in means):
            continue
        fig, ax = plt.subplots(figsize=(max(11, 2.6 * len(rungs)), 8))
        xs = np.arange(len(rungs))
        ax.errorbar(xs, [0 if m is None else m for m in means],
                    yerr=[0 if s is None else s for s in stds],
                    fmt="-o", capsize=6)
        ax.set_xticks(xs)
        ax.set_xticklabels(rungs, rotation=30, ha="right")
        ax.set_ylabel(metric.replace("_", " "))
        ax.set_title(f"Ablation ladder: {metric.replace('_', ' ')}")
        ax.grid(True)
        saved.append(_save(fig, os.path.join(out_dir, f"ladder_{_slug(metric)}.png")))

    # all metrics on one axis
    fig, ax = plt.subplots(figsize=(max(12, 2.8 * len(rungs)), 8))
    xs = np.arange(len(rungs))
    for metric in metrics:
        means, stds = [], []
        for r in rungs:
            t = tables[r].set_index("metric")
            means.append(t["mean"].get(metric) if metric in t.index else None)
            stds.append(t["std"].get(metric) if metric in t.index else None)
        if all(m is None for m in means):
            continue
        ax.errorbar(xs, [np.nan if m is None else m for m in means],
                    yerr=[0 if s is None else s for s in stds],
                    fmt="-o", capsize=5, label=metric.replace("_", " "))
    ax.set_xticks(xs)
    ax.set_xticklabels(rungs, rotation=30, ha="right")
    ax.set_ylabel("score")
    ax.set_title("Ablation ladder")
    ax.grid(True)
    ax.legend(fontsize=font_size - 6)
    saved.append(_save(fig, os.path.join(out_dir, "ladder_all_metrics.png")))
    return saved


def plot_comparison_pub(tables: dict, out_dir: str, filename: str = "comparison.png",
                        metrics=("composite", "symptom_macro_f1[all]",
                                 "severity_depression_qwk", "severity_risk_qwk"),
                        title: str = "Model comparison",
                        font_size: int = PUB_FONT) -> str | None:
    """Grouped bars with error bars, publication sized."""
    set_publication_style(font_size)
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    models = list(tables)
    if not models:
        return None
    fig, ax = plt.subplots(figsize=(max(13, 3.4 * len(metrics)), 8))
    width = 0.8 / max(len(models), 1)
    for i, name in enumerate(models):
        t = tables[name].set_index("metric")
        means = [t["mean"].get(m) if m in t.index else None for m in metrics]
        stds = [t["std"].get(m) if m in t.index else None for m in metrics]
        xs = np.arange(len(metrics)) + i * width - 0.4 + width / 2
        ax.bar(xs, [0 if v is None else v for v in means], width,
               yerr=[0 if v is None else v for v in stds], capsize=5, label=name)
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels([m.replace("_", " ").replace("[all]", "") for m in metrics],
                       rotation=20, ha="right")
    ax.set_ylabel("score")
    ax.set_title(title)
    ax.grid(True, axis="y")
    ax.legend(fontsize=font_size - 6)
    return _save(fig, os.path.join(out_dir, filename))


def _slug(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")


def compare_runs(run_dirs: dict, out_dir: str, title: str = "Model comparison",
                 ladder_order: list | None = None,
                 font_size: int = PUB_FONT) -> tuple:
    """Compare an explicit set of runs given ``{label: run_directory}``.

    Each directory must contain ``aggregate/aggregate_metrics.json``, which every
    baseline and every ablation rung writes, so comparisons need no extra code.
    """
    os.makedirs(out_dir, exist_ok=True)
    tables, frames, test_frames = {}, [], []
    for label, path in run_dirs.items():
        meta = os.path.join(path, "aggregate", "aggregate_metrics.json")
        if not os.path.isfile(meta):
            continue
        with open(meta, encoding="utf-8") as fh:
            payload = json.load(fh)
        agg = payload.get("aggregate", payload)
        tables[label] = headline_table(agg, label=label)
        frames.append(tables[label])
        by_split = payload.get("aggregate_by_split", {})
        if "test" in by_split:
            test_frames.append(headline_table(by_split["test"], label=label))
    if not frames:
        return pd.DataFrame(), tables

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(os.path.join(out_dir, "all_runs_val.csv"), index=False)
    wide = combined.pivot(index="metric", columns="model", values="formatted")
    wide.to_csv(os.path.join(out_dir, "leaderboard_val.csv"))
    if test_frames:
        tc = pd.concat(test_frames, ignore_index=True)
        tc.to_csv(os.path.join(out_dir, "all_runs_test.csv"), index=False)
        tc.pivot(index="metric", columns="model", values="formatted").to_csv(
            os.path.join(out_dir, "leaderboard_test.csv"))

    with open(os.path.join(out_dir, "leaderboard.md"), "w", encoding="utf-8") as fh:
        fh.write(f"# {title}\n\nmean +/- std across seeds (validation).\n\n")
        fh.write(to_markdown_table(wide))
        fh.write("\n")

    plot_comparison_pub(tables, out_dir, title=title, font_size=font_size)
    if ladder_order:
        plot_ladder(tables, out_dir, ladder_order, font_size=font_size)
    return combined, tables
