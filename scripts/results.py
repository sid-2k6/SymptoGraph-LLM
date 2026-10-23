#!/usr/bin/env python3
"""Single source of truth for every number printed in the manuscript.

`build_paper.py` imports this module and must not hard-code a metric anywhere.
That is the only way to guarantee that a value quoted in prose and the same
value in a table cannot drift apart.

Reads, all produced by the notebooks:

    comparison_vs_ablations/leaderboard_test.csv   L0..L5 on the test split
    comparison_vs_baselines/leaderboard_test.csv   proposed vs the three baselines
    L5_full/aggregate/history_all_seeds.csv        per-epoch validation trace
    L5_full/seed_*/performance_metrics.json        selective prediction, metadata
    L5_full/seed_*/config.json                     the configuration actually run

Everything is reported on the **test** split except Table 7, which is a
per-epoch validation trace and is labelled as such.
"""
from __future__ import annotations

import csv
import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

RUNGS = ["L0_flat", "L1_bottleneck", "L2_graph", "L3_weak", "L4_contrastive",
         "L5_full"]
RUNG_LABEL = {
    "L0_flat": "L0 (flat encoder)",
    "L1_bottleneck": "L1 (+ N1 bottleneck)",
    "L2_graph": "L2 (+ N2 graph)",
    "L3_weak": "L3 (+ N3 weak labels)",
    "L4_contrastive": "L4 (+ N5 contrastive)",
    "L5_full": "L5 (+ N6 uncertainty)",
}
BASELINES = ["tfidf_mlp", "bilstm_attn", "mentalroberta"]
BASELINE_LABEL = {
    "tfidf_mlp": "TF-IDF + MLP",
    "bilstm_attn": "BiLSTM + attention",
    "mentalroberta": "MentalRoBERTa, fine-tuned",
}

# metric key -> (table row label, lower_is_better)
METRICS = {
    "symptom_macro_f1[all]": ("Symptom macro-F1", False),
    "symptom_macro_auroc[all]": ("Symptom macro-AUROC", False),
    "severity_depression_macro_f1": ("Severity macro-F1", False),
    "severity_depression_qwk": ("Severity QWK", False),
    "severity_depression_mae": ("Severity MAE", True),
    "severity_risk_macro_f1": ("Risk macro-F1", False),
    "severity_risk_qwk": ("Risk QWK", False),
    "severity_risk_mae": ("Risk MAE", True),
    "stress_f1": ("Stress F1", False),
    "stress_auroc": ("Stress AUROC", False),
    "composite": ("Composite", False),
}


def _dash(s: str) -> str:
    """Normalise the ASCII +/- the notebooks emit into a real sign."""
    return s.replace("+/-", "\u00b1").strip()


def _load_leaderboard(path: str) -> dict[str, dict[str, str]]:
    """metric -> {model: 'mean ± sd'} from a leaderboard_test.csv."""
    out: dict[str, dict[str, str]] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        models = header[1:]
        for row in reader:
            metric, values = row[0], row[1:]
            out[metric] = {
                m: _dash(v) for m, v in zip(models, values) if v.strip()
            }
    return out


def _load_runs(path: str) -> dict[tuple[str, str], float]:
    """(model, metric) -> mean, from an all_runs_test.csv."""
    out: dict[tuple[str, str], float] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["mean"].strip():
                out[(r["model"], r["metric"])] = float(r["mean"])
    return out


ABL = _load_leaderboard(
    os.path.join(ROOT, "comparison_vs_ablations", "leaderboard_test.csv"))
BASE = _load_leaderboard(
    os.path.join(ROOT, "comparison_vs_baselines", "leaderboard_test.csv"))
ABL_MEAN = _load_runs(
    os.path.join(ROOT, "comparison_vs_ablations", "all_runs_test.csv"))
BASE_MEAN = _load_runs(
    os.path.join(ROOT, "comparison_vs_baselines", "all_runs_test.csv"))


def abl(rung: str, metric: str) -> str:
    """'mean ± sd' for an ablation rung on the test split."""
    return ABL.get(metric, {}).get(rung, "\u2013")


def base(model: str, metric: str) -> str:
    """'mean ± sd' for a baseline or the proposed model on the test split."""
    return BASE.get(metric, {}).get(model, "\u2013")


def prop(metric: str) -> str:
    """'mean ± sd' for the proposed model (L5) on the test split."""
    return BASE.get(metric, {}).get("SymptoGraph-LLM", "\u2013")


def mean(model: str, metric: str) -> float | None:
    """Raw mean, for computing margins. Looks in both leaderboards."""
    if (model, metric) in ABL_MEAN:
        return ABL_MEAN[(model, metric)]
    return BASE_MEAN.get((model, metric))


def m3(model: str, metric: str) -> str:
    """Raw mean formatted to three decimals."""
    v = mean(model, metric)
    return "\u2013" if v is None else f"{v:.3f}"


def sd(model: str, metric: str) -> str:
    """Just the seed standard deviation, three decimals."""
    txt = ABL.get(metric, {}).get(model) or BASE.get(metric, {}).get(model)
    if not txt or "\u00b1" not in txt:
        return "\u2013"
    try:
        return f"{float(txt.split(chr(0xb1))[1]):.3f}"
    except (ValueError, IndexError):
        return "\u2013"


def margin(metric: str, a: str = "L5_full", b: str = "mentalroberta") -> str:
    """Absolute difference between two models on one metric, three decimals."""
    va, vb = mean(a, metric), mean(b, metric)
    if va is None:
        va = mean("SymptoGraph-LLM", metric)
    if va is None or vb is None:
        return "\u2013"
    return f"{abs(va - vb):.3f}"


def best_rung(metric: str) -> str:
    """Which rung leads on a metric, respecting metric direction."""
    lower = METRICS.get(metric, ("", False))[1]
    vals = [(r, mean(r, metric)) for r in RUNGS if mean(r, metric) is not None]
    if not vals:
        return "\u2013"
    pick = min(vals, key=lambda kv: kv[1]) if lower else max(
        vals, key=lambda kv: kv[1])
    return pick[0]


def rungs_led_by_l5() -> tuple[int, int]:
    """(how many of the reported metrics L5 leads, how many were compared)."""
    led = total = 0
    for k in METRICS:
        if mean("L5_full", k) is None:
            continue
        total += 1
        if best_rung(k) == "L5_full":
            led += 1
    return led, total


def beats_all_baselines() -> tuple[int, int, list[str]]:
    """(metrics where the proposal leads every baseline, total, the losses)."""
    led = total = 0
    lost: list[str] = []
    for k, (label, lower) in METRICS.items():
        p = mean("SymptoGraph-LLM", k)
        bs = [mean(b, k) for b in BASELINES]
        bs = [v for v in bs if v is not None]
        if p is None or not bs:
            continue
        total += 1
        ok = all(p < v for v in bs) if lower else all(p > v for v in bs)
        if ok:
            led += 1
        else:
            lost.append(label)
    return led, total, lost


# --------------------------------------------------------------------------- #
# per-epoch validation trace, for the training-progression table
# --------------------------------------------------------------------------- #
HIST_PATH = os.path.join(ROOT, "L5_full", "aggregate", "history_all_seeds.csv")
with open(HIST_PATH, newline="", encoding="utf-8") as fh:
    HIST = list(csv.DictReader(fh))

# epochs every seed reached, so no column is a partial average
_per_seed = {}
for _r in HIST:
    _per_seed.setdefault(_r["seed"], []).append(int(_r["epoch"]))
COMMON_EPOCHS = sorted(set.intersection(*(set(v) for v in _per_seed.values())))
N_SEEDS = len(_per_seed)
EPOCHS_RUN = {s: max(v) for s, v in _per_seed.items()}

# The five checkpoints reported, mirroring the reference template's five.
# Chosen so that every epoch quoted in the prose is also a column: 1 is the
# start, 2 is the validation-loss minimum, 3 is the validation risk-QWK peak,
# 8 is the end of the budget every seed reached.
PROGRESS_EPOCHS = [e for e in (1, 2, 3, 6, 8) if e in COMMON_EPOCHS]

PROGRESS_ROWS = [
    ("Multi-task validation loss", "val_loss"),
    ("Multi-task training loss", "train_loss"),
    ("Symptom macro-F1", "val_group_gold_supervised_macro_f1"),
    ("Symptom macro-AUROC", "val_group_gold_supervised_macro_auroc"),
    ("Symptom macro-AP", "val_group_gold_supervised_macro_ap"),
    ("Stress AUROC", "val_stress_auroc"),
    ("Stress F1", "val_stress_f1"),
    ("Severity QWK", "val_severity_depression_qwk"),
    ("Severity macro-F1", "val_severity_depression_macro_f1"),
    ("Severity MAE", "val_severity_depression_mae"),
    ("Risk QWK", "val_severity_risk_qwk"),
    ("Risk macro-F1", "val_severity_risk_macro_f1"),
    ("Risk MAE", "val_severity_risk_mae"),
    ("Training risk QWK", "train_severity_risk_qwk"),
    ("Selection score", "val_selection_score"),
]


def epoch_mean(epoch: int, col: str) -> float | None:
    vals = [float(r[col]) for r in HIST
            if int(r["epoch"]) == epoch and r.get(col, "") not in ("", "nan")]
    return st.mean(vals) if vals else None


def progress(col: str, epoch: int) -> str:
    v = epoch_mean(epoch, col)
    return "\u2013" if v is None else f"{v:.3f}"


def _argmin_epoch(col: str) -> int:
    cand = [(e, epoch_mean(e, col)) for e in COMMON_EPOCHS]
    cand = [(e, v) for e, v in cand if v is not None]
    return min(cand, key=lambda kv: kv[1])[0]


def _argmax_epoch(col: str) -> int:
    cand = [(e, epoch_mean(e, col)) for e in COMMON_EPOCHS]
    cand = [(e, v) for e, v in cand if v is not None]
    return max(cand, key=lambda kv: kv[1])[0]


VAL_LOSS_MIN_EPOCH = _argmin_epoch("val_loss")
VAL_LOSS_MIN = f"{epoch_mean(VAL_LOSS_MIN_EPOCH, 'val_loss'):.3f}"
VAL_COMPOSITE_PEAK_EPOCH = _argmax_epoch("val_composite")
VAL_COMPOSITE_PEAK = f"{epoch_mean(VAL_COMPOSITE_PEAK_EPOCH, 'val_composite'):.3f}"

_last = max(COMMON_EPOCHS)
VAL_LOSS_LAST = f"{epoch_mean(_last, 'val_loss'):.3f}"
VAL_LOSS_LAST_EPOCH = _last
VAL_LOSS_RATIO = (
    f"{epoch_mean(_last, 'val_loss') / epoch_mean(VAL_LOSS_MIN_EPOCH, 'val_loss'):.2f}")
TRAIN_RISK_QWK_LAST = f"{epoch_mean(_last, 'train_severity_risk_qwk'):.3f}"
VAL_RISK_QWK_PEAK_EPOCH = _argmax_epoch("val_severity_risk_qwk")
VAL_RISK_QWK_PEAK = f"{epoch_mean(VAL_RISK_QWK_PEAK_EPOCH, 'val_severity_risk_qwk'):.3f}"
VAL_RISK_QWK_LAST = f"{epoch_mean(_last, 'val_severity_risk_qwk'):.3f}"

# epoch at which training risk QWK first reaches unity, if it does
_MEMORISE = next(
    (e for e in COMMON_EPOCHS
     if (epoch_mean(e, "train_severity_risk_qwk") or 0) >= 0.999), None)
TRAIN_RISK_MEMORISE_EPOCH = _MEMORISE


# --------------------------------------------------------------------------- #
# selective prediction (novelty N6) and run metadata
# --------------------------------------------------------------------------- #
def _seed_dirs() -> list[str]:
    root = os.path.join(ROOT, "L5_full")
    return sorted(os.path.join(root, d) for d in os.listdir(root)
                  if d.startswith("seed_"))


_SEL: dict[str, list[float]] = {}
_BEST_EPOCH: list[int] = []
CONFIG: dict = {}
SPLITS: dict = {}
COMPONENTS: dict = {}
N_WEAK_ROWS = None

for _d in _seed_dirs():
    with open(os.path.join(_d, "performance_metrics.json"), encoding="utf-8") as fh:
        pm = json.load(fh)
    for k, v in pm["selective"]["test"].items():
        _SEL.setdefault(k, []).append(v)
    _BEST_EPOCH.append(pm["best_epoch"])
    SPLITS = pm["splits"]
    COMPONENTS = pm["components"]
    N_WEAK_ROWS = pm["n_weak_rows"]
    with open(os.path.join(_d, "config.json"), encoding="utf-8") as fh:
        CONFIG = json.load(fh)


def sel(key: str) -> str:
    """'mean ± sd' of a selective-prediction metric on the test split."""
    v = _SEL.get(key)
    if not v:
        return "\u2013"
    if len(v) < 2:
        return f"{v[0]:.3f}"
    return f"{st.mean(v):.3f} \u00b1 {st.stdev(v):.3f}"


def sel_mean(key: str) -> str:
    v = _SEL.get(key)
    return "\u2013" if not v else f"{st.mean(v):.3f}"


BEST_EPOCHS = ", ".join(str(e) for e in _BEST_EPOCH)
PARAMS = CONFIG.get("trainable_parameters")
PARAMS_M = f"{PARAMS / 1e6:.1f}" if PARAMS else "\u2013"
PARAMS_STR = f"{PARAMS:,}" if PARAMS else "\u2013"
EPOCH_BUDGET = CONFIG.get("epochs")
PATIENCE = CONFIG.get("early_stopping_patience")
SCHEDULER = CONFIG.get("scheduler")
SELECTION_METRIC = CONFIG.get("selection_metric")
GRAPH_DIM = CONFIG.get("graph_dim")
GRAPH_LAYERS = CONFIG.get("graph_layers")
GRAPH_L1 = CONFIG.get("graph_l1")
CONTRASTIVE_WEIGHT = CONFIG.get("contrastive_weight")
CONTRASTIVE_TEMP = CONFIG.get("contrastive_temperature")
CONTRASTIVE_JACCARD = CONFIG.get("contrastive_jaccard")
UNCERTAINTY_WEIGHT = CONFIG.get("uncertainty_weight")
UNCERTAINTY_DETACHED = CONFIG.get("uncertainty_detached")
ENCODER = CONFIG.get("encoder")
WEAK_ROWS_STR = f"{N_WEAK_ROWS:,}" if N_WEAK_ROWS else "\u2013"
TRAIN_ROWS = SPLITS.get("train")
VAL_ROWS = SPLITS.get("val")
TEST_ROWS = SPLITS.get("test")
GOLD_TRAIN_ROWS = (TRAIN_ROWS - N_WEAK_ROWS) if (TRAIN_ROWS and N_WEAK_ROWS) else None

# error reduction obtained by abstaining on the least confident half
_full = _SEL.get("risk_at_full_coverage")
_half = _SEL.get("risk_at_50_coverage")
ABSTAIN_REDUCTION = (
    f"{100 * (1 - st.mean(_half) / st.mean(_full)):.0f}"
    if _full and _half else "\u2013")


def summary() -> str:
    led, total = rungs_led_by_l5()
    bled, btotal, blost = beats_all_baselines()
    lines = [
        "RESULTS LOADED",
        f"  rungs            : {', '.join(RUNGS)}",
        f"  baselines        : {', '.join(BASELINES)}",
        f"  seeds            : {N_SEEDS}, epochs run {EPOCHS_RUN}",
        f"  common epochs    : {COMMON_EPOCHS}",
        f"  reported epochs  : {PROGRESS_EPOCHS}",
        f"  selection metric : {SELECTION_METRIC}, best epochs {BEST_EPOCHS}",
        f"  splits           : train {TRAIN_ROWS} "
        f"({GOLD_TRAIN_ROWS} gold + {N_WEAK_ROWS} weak), "
        f"val {VAL_ROWS}, test {TEST_ROWS}",
        f"  params           : {PARAMS_STR} ({PARAMS_M} M)",
        f"  L5 leads         : {led}/{total} ablation metrics",
        f"  beats baselines  : {bled}/{btotal}; loses on {blost or 'nothing'}",
        f"  val loss         : min {VAL_LOSS_MIN} at epoch "
        f"{VAL_LOSS_MIN_EPOCH}, {VAL_LOSS_LAST} at epoch {VAL_LOSS_LAST_EPOCH} "
        f"({VAL_LOSS_RATIO}x)",
        f"  train risk QWK   : {TRAIN_RISK_QWK_LAST} at epoch "
        f"{VAL_LOSS_LAST_EPOCH}, unity from epoch {TRAIN_RISK_MEMORISE_EPOCH}",
        f"  selective (test) : aurc {sel('aurc')}, "
        f"risk@100 {sel('risk_at_full_coverage')}, "
        f"risk@80 {sel('risk_at_80_coverage')}, "
        f"risk@50 {sel('risk_at_50_coverage')}",
        f"  abstain gain     : {ABSTAIN_REDUCTION}% error reduction at 50% coverage",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary())


# --------------------------------------------------------------------------- #
# full risk-coverage curves, averaged across seeds, for the N6 figure
# --------------------------------------------------------------------------- #
def selective_curves(split: str = "test") -> list[tuple[float, float, float]]:
    """[(coverage, mean selective risk, sd across seeds)] sorted by coverage."""
    by_cov: dict[float, list[float]] = {}
    for d in _seed_dirs():
        with open(os.path.join(d, "performance_metrics.json"),
                  encoding="utf-8") as fh:
            pm = json.load(fh)
        for point in pm["selective"][split].get("curve", []):
            by_cov.setdefault(round(point["coverage"], 4), []).append(
                point["risk"])
    out = []
    for cov in sorted(by_cov):
        vals = by_cov[cov]
        sd = st.stdev(vals) if len(vals) > 1 else 0.0
        out.append((cov, st.mean(vals), sd))
    return out
