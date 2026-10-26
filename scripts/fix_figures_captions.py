#!/usr/bin/env python3
"""Three targeted edits to ``docs/SymptoGraph-LLM_Paper_revised.docx``.

1. Re-insert Figures 3 and 5, which lost their images, regenerated at
   font size 20 and 300 dpi.
2. Reduce every over-long caption to a single specific sentence.
3. Replace in-text section cross-references with the name of the section.

Nothing else in the document is touched.

Why the images were missing: Figures 3 and 5 each hold two panels in one
paragraph separated by a two-space run. An earlier whitespace pass assigned to
``run.text`` on that paragraph, and in python-docx assigning ``run.text``
replaces every child of the run, including ``<w:drawing>``. ``set_text`` below
refuses to touch a paragraph containing a drawing, so this cannot recur.
"""
from __future__ import annotations

import csv
import io
import json
import os
import re
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from docx import Document
from docx.shared import Inches, Pt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DOC = os.path.join(ROOT, "docs", "SymptoGraph-LLM_Paper_revised.docx")
FIGS = os.path.join(ROOT, "docs", "paper_figures")
HIST = os.path.join(ROOT, "L5_full", "aggregate", "history_all_seeds.csv")

FONT = "Times New Roman"
SIZE = Pt(12)

# as requested: plot text at 20 pt, rasterised at 300 dpi
PLOT_FONT = 20
PLOT_DPI = 300
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

LOG: list[str] = []


def note(m: str) -> None:
    LOG.append(m)
    print(f"  {m}")


def has_drawing(p) -> bool:
    return bool(p._p.findall(f".//{W}drawing"))


def set_text(p, text: str) -> None:
    """Replace a paragraph's text, keeping the first run's formatting.

    Refuses to act on a paragraph holding a drawing, because assigning to
    ``run.text`` would delete the image.
    """
    if has_drawing(p):
        raise RuntimeError("refusing to set text on a paragraph with a drawing")
    runs = p.runs
    if not runs:
        r = p.add_run(text)
        r.font.name, r.font.size = FONT, SIZE
        return
    runs[0].text = text
    for r in runs[1:]:
        r.text = ""


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
with open(HIST, newline="", encoding="utf-8") as fh:
    ROWS = list(csv.DictReader(fh))
EPOCHS = sorted({int(r["epoch"]) for r in ROWS})
SEEDS_AT = {e: {r["seed"] for r in ROWS if int(r["epoch"]) == e} for e in EPOCHS}
FULL = max(len(v) for v in SEEDS_AT.values())
PARTIAL = [e for e in EPOCHS if len(SEEDS_AT[e]) < FULL]


def mean_at(epoch: int, col: str):
    v = [float(r[col]) for r in ROWS
         if int(r["epoch"]) == epoch and r.get(col, "") not in ("", "nan")]
    return st.mean(v) if v else None


def split_curve(col: str):
    full = [e for e in EPOCHS if e not in PARTIAL]
    part = ([full[-1]] + PARTIAL) if full and PARTIAL else PARTIAL
    return (full, [mean_at(e, col) for e in full],
            part, [mean_at(e, col) for e in part])


def table10(doc):
    """Read Table 10 so Figure 5A matches the table in this document."""
    t = doc.tables[9]
    hdr = [c.text.strip() for c in t.rows[0].cells]
    out = {}
    for row in t.rows[1:]:
        name = row.cells[0].text.strip()
        vals = {}
        for j, h in enumerate(hdr[1:], start=1):
            txt = row.cells[j].text.strip()
            m = re.match(r"([\d.]+)\s*[\u00b1+/-]+\s*([\d.]+)", txt)
            if m:
                vals[h] = (float(m.group(1)), float(m.group(2)))
        out[name] = vals
    return hdr, out


# --------------------------------------------------------------------------- #
# figures, at 20 pt and 300 dpi
# --------------------------------------------------------------------------- #
def _style():
    plt.rcParams.update({
        "font.family": "serif", "font.size": PLOT_FONT,
        "axes.labelsize": PLOT_FONT, "axes.titlesize": PLOT_FONT,
        "xtick.labelsize": PLOT_FONT - 2, "ytick.labelsize": PLOT_FONT - 2,
        "legend.fontsize": PLOT_FONT - 4, "savefig.dpi": PLOT_DPI,
    })


def fig3a(path):
    _style()
    fig, ax = plt.subplots(figsize=(9.2, 6.4))
    for col, colour, label, mk in (
            ("train_loss", "#2b6cb0", "training loss", "o"),
            ("val_loss", "#c05621", "validation loss", "s")):
        fe, fv, pe, pv = split_curve(col)
        ax.plot(fe, fv, marker=mk, lw=2.6, ms=9, color=colour, label=label)
        if len(pe) > 1:
            ax.plot(pe, pv, marker=mk, lw=2.6, ms=9, color=colour,
                    ls=(0, (4, 2)), mfc="white")
    if PARTIAL:
        lo = min(PARTIAL) - 0.5
        ax.axvspan(lo, max(EPOCHS) + 0.4, color="#999", alpha=0.12, lw=0)
        ax.text(lo + 0.1, ax.get_ylim()[1] * 0.96, "2 seeds",
                fontsize=PLOT_FONT - 4, color="#555", va="top")
    vmin = min(EPOCHS, key=lambda e: mean_at(e, "val_loss"))
    ax.axvline(vmin, ls=":", color="#555", lw=2.0)
    ax.annotate(f"min val loss (epoch {vmin})",
                xy=(vmin, mean_at(vmin, "val_loss")),
                xytext=(vmin + 1.1, mean_at(vmin, "val_loss") + 0.95),
                fontsize=PLOT_FONT - 5, color="#333",
                arrowprops=dict(arrowstyle="->", color="#555", lw=1.6))
    ax.set_xlabel("Epoch"), ax.set_ylabel("Multi-task loss")
    ax.set_xticks(EPOCHS), ax.legend(loc="upper left"), ax.grid(alpha=0.3)
    fig.tight_layout(), fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def fig3b(path):
    _style()
    fig, ax = plt.subplots(figsize=(9.2, 6.4))
    for col, colour, label, mk in (
            ("train_severity_risk_qwk", "#2b6cb0", "training risk QWK", "o"),
            ("val_severity_risk_qwk", "#c05621", "validation risk QWK", "s")):
        fe, fv, pe, pv = split_curve(col)
        ax.plot(fe, fv, marker=mk, lw=2.6, ms=9, color=colour, label=label)
        if len(pe) > 1:
            ax.plot(pe, pv, marker=mk, lw=2.6, ms=9, color=colour,
                    ls=(0, (4, 2)), mfc="white")
    if PARTIAL:
        lo = min(PARTIAL) - 0.5
        ax.axvspan(lo, max(EPOCHS) + 0.4, color="#999", alpha=0.12, lw=0)
        ax.text(lo + 0.1, 1.09, "2 seeds", fontsize=PLOT_FONT - 4,
                color="#555")
    ax.axhline(1.0, ls="--", lw=1.6, color="#888")
    ax.text(EPOCHS[0], 1.03, "memorisation ceiling",
            fontsize=PLOT_FONT - 5, color="#666")
    ax.set_xlabel("Epoch"), ax.set_ylabel("Quadratic weighted kappa")
    ax.set_xticks(EPOCHS), ax.set_ylim(0, 1.2)
    ax.legend(loc="center right"), ax.grid(alpha=0.3)
    fig.tight_layout(), fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def fig5a(path, hdr, rows):
    """Grouped bars straight from Table 10 of this document."""
    _style()
    metrics = ["Sym F1 \u2191", "Sym AUROC \u2191", "Sev QWK \u2191",
               "Risk QWK \u2191", "Stress AUROC \u2191", "Composite \u2191"]
    metrics = [m for m in metrics if m in hdr]
    order = [k for k in rows if not k.lower().startswith("proposed")]
    order += [k for k in rows if k.lower().startswith("proposed")]
    short = {"TF-IDF + MLP": "TF-IDF + MLP",
             "BiLSTM + attention": "BiLSTM + attn"}
    # aspect matched to Figure 5B so the two panels align when placed
    # side by side at equal width
    fig, ax = plt.subplots(figsize=(10.4, 6.9))
    n = len(order)
    width = 0.8 / n
    x = np.arange(len(metrics))
    cmap = plt.get_cmap("tab10")
    for i, name in enumerate(order):
        vals = [rows[name].get(m, (0.0, 0.0))[0] for m in metrics]
        errs = [rows[name].get(m, (0.0, 0.0))[1] for m in metrics]
        label = short.get(name, name.split(",")[0].split("(")[0].strip())
        ax.bar(x + i * width - 0.4 + width / 2, vals, width, yerr=errs,
               capsize=3.5, label=label, color=cmap(i), edgecolor="black",
               lw=0.6, error_kw=dict(lw=1.3))
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(" \u2191", "") for m in metrics],
                       rotation=22, ha="right")
    ax.set_ylabel("Score (test split)"), ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.20),
              frameon=False)
    fig.tight_layout(), fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def fig5b(path):
    _style()
    by_cov: dict[float, list[float]] = {}
    root = os.path.join(ROOT, "L5_full")
    for dname in sorted(d for d in os.listdir(root) if d.startswith("seed_")):
        with open(os.path.join(root, dname, "performance_metrics.json"),
                  encoding="utf-8") as fh:
            pm = json.load(fh)
        for pt in pm["selective"]["test"].get("curve", []):
            by_cov.setdefault(round(pt["coverage"], 4), []).append(pt["risk"])
    cov = sorted(by_cov)
    risk = np.array([st.mean(by_cov[c]) for c in cov])
    sd = np.array([st.stdev(by_cov[c]) if len(by_cov[c]) > 1 else 0.0
                   for c in cov])
    fig, ax = plt.subplots(figsize=(9.6, 6.6))
    ax.plot(cov, risk, marker="o", lw=2.6, ms=8, color="#c05621",
            label="selective risk (test)")
    ax.fill_between(cov, risk - sd, risk + sd, alpha=0.18, color="#c05621")
    for c in (0.5, 0.8):
        ax.axvline(c, ls=":", lw=2.0, color="#555")
        ax.text(c + 0.012, max(risk) * 0.22, f"{int(c * 100)}%",
                fontsize=PLOT_FONT - 4, color="#444")
    ax.set_xlabel("Coverage"), ax.set_ylabel("Selective risk")
    ax.legend(loc="upper left"), ax.grid(alpha=0.3)
    fig.tight_layout(), fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 1. re-insert the two figures
# --------------------------------------------------------------------------- #
def insert_two_panels(doc, caption_starts: str, left: str, right: str,
                      each: float = 3.02) -> None:
    idx = next(i for i, p in enumerate(doc.paragraphs)
               if p.text.strip().startswith(caption_starts))
    holder = doc.paragraphs[idx - 1]
    if has_drawing(holder):
        note(f"{caption_starts} already has its images, left alone")
        return
    for r in list(holder.runs):
        r._r.getparent().remove(r._r)
    holder.add_run().add_picture(left, width=Inches(each))
    sep = holder.add_run("  ")
    sep.font.name, sep.font.size = FONT, SIZE
    holder.add_run().add_picture(right, width=Inches(each))
    note(f"{caption_starts} images re-inserted at {each} in per panel")


# --------------------------------------------------------------------------- #
# 2. captions
# --------------------------------------------------------------------------- #
CAPTIONS = {
    "Table 3.":
        "Table 3. Labelled training and test instance counts for each of the "
        "four prediction tasks, with the test share of that task's labelled "
        "total.",
    "Figure 2.":
        "Figure 2. Architecture of SymptoGraph-LLM, in which the symptom head "
        "reads the graph-refined concept logits while the severity, risk and "
        "stress heads read those logits concatenated with the concept-gated "
        "pooled vector.",
    "Table 7.":
        "Table 7. Per-epoch validation metrics of SymptoGraph-LLM, averaged "
        "over three seeds to epoch 8 and over the two seeds that continued at "
        "epochs 9 and 10.",
    "Figure 3.":
        "Figure 3. Multi-task loss (A) and suicide-risk quadratic weighted "
        "kappa (B) on the training and validation splits, the dashed tail "
        "from epoch 9 averaging only the two seeds that ran past early "
        "stopping.",
    "Figure 5.":
        "Figure 5. Test-split scores against the three retrained reference "
        "models (A) and the risk-coverage curve of the abstention signal (B), "
        "both averaged over three seeds.",
}


def fix_captions(doc) -> None:
    changed = 0
    for p in doc.paragraphs:
        t = p.text.strip()
        for key, new in CAPTIONS.items():
            if t.startswith(key) and t != new:
                set_text(p, new)
                changed += 1
                break
    note(f"rewrote {changed} long captions as one specific sentence each")


# --------------------------------------------------------------------------- #
# 3. section cross-references
# --------------------------------------------------------------------------- #
NAMES = {
    "2": "literature survey",
    "3": "proposed methodology",
    "4": "results and discussions",
    "5": "conclusion",
    "3.1": "dataset description",
    "3.2": "data pre-processing",
    "3.3": "feature extraction",
    "3.4": "proposed framework architecture",
    "3.5": "loss function and hyperparameter configuration",
    "4.1": "experimental setup",
    "4.2": "evaluation metrics",
    "4.3": "performance analysis",
    "4.4": "training convergence analysis",
    "4.5": "ablation study",
    "4.6": "comparison with state-of-the-art models",
    "5.1": "conclusion",
    "5.2": "future work",
    "5.3": "ethical considerations",
    "5.4": "data availability",
    "5.5": "code availability",
}

# phrasings that need more than the generic swap to stay readable
SPECIAL = [
    ("as Section 3.5 states",
     "as the loss function and hyperparameter configuration states"),
    ("so Section 4.6 is a comparison against reasonable reference "
     "configurations",
     "so that comparison is against reasonable reference configurations"),
    ("Section 4.6 rather than inferred from this table",
     "the comparison with state-of-the-art models and not inferred from "
     "this table"),
    ("Section 4.6 discusses", "the comparison with state-of-the-art models "
     "discusses"),
    ("Section 4.5 shows, however,", "the ablation study shows, however,"),
    ("Section 4.4 shows would favour", "the training convergence analysis "
     "shows would favour"),
    ("Section 5.2 records", "the future work records"),
    ("Section 5.2 as outstanding", "the future work as outstanding"),
    ("Section 5.2 instead of a finding", "the future work instead of a "
     "finding"),
    ("Section 2 reviews", "The literature survey reviews"),
    ("Section 3 describes", "The proposed methodology describes"),
    ("Section 4 reports", "The results and discussions report"),
    ("Section 5 concludes the paper", "The conclusion closes the paper"),
    ("so Section 4.6 becomes a controlled comparison",
     "so the comparison with state-of-the-art models becomes controlled"),
]


def fix_cross_refs(doc) -> None:
    pat = re.compile(r"\bSections?\s+(\d(?:\.\d)?)")
    touched = before = after = 0
    for p in doc.paragraphs:
        t = p.text
        if "Section" not in t:
            continue
        before += len(pat.findall(t))
        new = t
        for old, repl in SPECIAL:
            new = new.replace(old, repl)

        def sub(m):
            name = NAMES.get(m.group(1))
            return f"the {name}" if name else m.group(0)

        new = pat.sub(sub, new)
        new = (new.replace("in the the ", "in the ")
                  .replace("the the ", "the ")
                  .replace("  ", " "))
        # a replacement at the start of a sentence needs its capital back
        new = re.sub(r"(^|(?<=\. ))the ([a-z])",
                     lambda m: m.group(1) + "The " + m.group(2), new)
        after += len(pat.findall(new))
        if new != t:
            set_text(p, new)
            touched += 1
    note(f"replaced {before - after} section cross-references with section "
         f"names across {touched} paragraphs ({after} left)")


# --------------------------------------------------------------------------- #
def main() -> None:
    doc = Document(DOC)

    print("regenerating figures at font 20, dpi 300")
    hdr, rows = table10(doc)
    fig3a(os.path.join(FIGS, "figure3a_loss.png"))
    fig3b(os.path.join(FIGS, "figure3b_qwk.png"))
    fig5a(os.path.join(FIGS, "figure5a_sota.png"), hdr, rows)
    fig5b(os.path.join(FIGS, "figure5b_ci.png"))
    note("Figure 3 and Figure 5 panels regenerated; Figure 5A read from "
         "Table 10 of this document")

    print("patching document")
    insert_two_panels(doc, "Figure 3.",
                      os.path.join(FIGS, "figure3a_loss.png"),
                      os.path.join(FIGS, "figure3b_qwk.png"))
    insert_two_panels(doc, "Figure 5.",
                      os.path.join(FIGS, "figure5a_sota.png"),
                      os.path.join(FIGS, "figure5b_ci.png"))
    fix_captions(doc)
    fix_cross_refs(doc)

    doc.save(DOC)
    print(f"\nsaved {DOC}")


if __name__ == "__main__":
    main()
