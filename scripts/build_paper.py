#!/usr/bin/env python3
"""Build docs/SymptoGraph-LLM_Paper.docx in the exact format of PERSIST.docx.

Formatting is copied from the reference document rather than guessed:

    page            A4, 8.27 x 11.69 in, 1 in margins all round
    body font       Times New Roman 12 pt throughout
    title           bold, centred
    section heads   bold, justified ("1. Introduction", "3.1 Dataset ...")
    body text       justified, no first-line indent, no extra paragraph spacing
    contributions   List Paragraph style, justified, bulleted
    table caption   bold, justified, ABOVE the table ("Table 1. ...")
    tables          Table Grid style
    figure caption  centred, BELOW the image ("Figure 1. ...")
    figure width    6.27 in, matching the 5731510 EMU extent in PERSIST.docx

Every *performance* number is the placeholder ``X.XXX``. Dataset counts are real,
because they are measured facts and the reference document prints real ones too.
``docs/PLACEHOLDERS.md`` maps each table and figure to the file that fills it.

Regenerate after filling numbers in:

    python scripts/build_paper.py
"""
from __future__ import annotations

import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DOCS = os.path.join(ROOT, "docs")
FIGS = os.path.join(DOCS, "paper_figures")
SURVEY = os.path.join(DOCS, "Literature_Survey_SymptoGraph-LLM.md")
OUT = os.path.join(DOCS, "SymptoGraph-LLM_Paper.docx")

FONT = "Times New Roman"
SIZE = Pt(12)
FIG_W = Inches(6.27)
P = "X.XXX"            # performance placeholder
PM = "X.XXX \u00b1 X.XXX"


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def _stamp(ax, text="PLACEHOLDER \u2013 regenerate from results"):
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center",
            fontsize=22, color="#c0392b", alpha=0.18, rotation=18, zorder=10,
            fontweight="bold")


def fig1_architecture(path: str) -> None:
    """Overall workflow. Not a placeholder - this is the real architecture."""
    plt.rcParams.update({"font.family": "serif", "font.size": 9})
    fig, ax = plt.subplots(figsize=(11.0, 6.4))
    ax.set_xlim(0, 100), ax.set_ylim(0, 58), ax.axis("off")

    def box(x, y, w, h, label, fc, fs=9, bold=False):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.45", linewidth=1.1,
            edgecolor="#2c3e50", facecolor=fc))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fs, fontweight="bold" if bold else "normal", wrap=True)

    def arrow(x1, y1, x2, y2, style="-|>"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color="#2c3e50", lw=1.1))

    C = dict(data="#dfe9f5", pre="#e4f1e4", enc="#fdf0d9",
             nov="#f6e0ea", head="#e8e2f3", out="#eeeeee")

    ax.text(50, 56, "Stage 1  Data acquisition and leakage control",
            ha="center", fontsize=10, fontweight="bold")
    for i, (lab, x) in enumerate([
            ("DepressionEmo\n(PHQ-9 items)", 1), ("Dreaddit\n(stress)", 15),
            ("DepSeverity\n(4-level)", 28), ("CSSRS-500\n(5-level risk)", 41),
            ("IRF / LoST\n(risk factors)", 54), ("IMHI\n(repackaged)", 67),
            ("Reddit MH pool\n1,107,302 posts", 80)]):
        box(x, 46.5, 12.5, 7, lab, C["data"], fs=7.5)
    arrow(50, 46.3, 50, 42.8)

    box(12, 36, 76, 6,
        "Global post_id (200-char normalised MD5)  \u2022  one split per post  \u2022  "
        "author-disjoint  \u2022  strictest-split-wins", C["pre"], fs=8.5)
    arrow(30, 35.8, 30, 32.3)
    arrow(72, 35.8, 72, 32.3)
    box(14, 25.5, 32, 6.5, "labelled.parquet\n19,762 posts  \u2022  22 symptoms\n"
        "severity / risk / stress", C["pre"], fs=8)
    box(56, 25.5, 32, 6.5, "weak_pool.parquet\n1,017,959 posts\n"
        "Qwen2.5-7B labels 49,982", C["pre"], fs=8)

    ax.text(50, 22.5, "Stage 2  SymptoGraph-LLM", ha="center", fontsize=10,
            fontweight="bold")
    box(2, 13, 17, 7, "MentalRoBERTa\nencoder", C["enc"], fs=8.5)
    arrow(19.2, 16.5, 22.8, 16.5)
    box(23, 13, 19, 7, "N1  Clinical concept\nbottleneck\n"
        "22 definition-initialised\nsymptom queries", C["nov"], fs=7.5)
    arrow(42.2, 16.5, 45.8, 16.5)
    box(46, 13, 18, 7, "N2  Comorbidity graph\nA = A\u2080 + \u0394A\n"
        "DSM-5 prior + learned", C["nov"], fs=7.5)
    arrow(64.2, 16.5, 67.8, 16.5)
    box(68, 13, 17, 7, "N-A  Monotone\nordinal head\n"
        "z = \u03a3 softplus(v\u2096)\u03c3(c\u2096)", C["nov"], fs=7.5)

    box(23, 4.5, 19, 6, "N3  Weak supervision\nstream (49,982)", C["nov"], fs=7.5)
    box(46, 4.5, 18, 6, "N-B  Per-symptom\nnoise channel\nP(noisy | clean)",
        C["nov"], fs=7.5)
    arrow(32.5, 10.5, 32.5, 12.8)
    arrow(55, 10.5, 55, 12.8)
    arrow(42.2, 7.5, 45.8, 7.5)

    box(87, 10.5, 12, 12,
        "Heads\n\u2022 22 symptoms\n\u2022 severity (4)\n\u2022 risk (5)\n\u2022 stress (2)",
        C["head"], fs=7.5)
    arrow(85.2, 16.5, 86.8, 16.5)

    ax.text(50, 1.6, "Stage 3  Evaluation   3 seeds  \u2022  selection on validation loss  "
            "\u2022  paired bootstrap  \u2022  per-task CIs", ha="center", fontsize=8.5,
            style="italic")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig2_vs_baselines(path: str) -> None:
    plt.rcParams.update({"font.family": "serif", "font.size": 11})
    models = ["TF-IDF\n+MLP", "BiLSTM\n+Attn", "Mental\nRoBERTa",
              "SOTA-1\n[11]", "SOTA-2\n[18]", "SOTA-3\n[21]", "SymptoGraph\n-LLM"]
    metrics = ["Symptom macro-F1", "Symptom macro-AUROC", "Stress AUROC"]
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(10, 4.6))
    w, x = 0.26, np.arange(len(models))
    for i, m in enumerate(metrics):
        ax.bar(x + (i - 1) * w, rng.uniform(0.55, 0.75, len(models)), w,
               label=m, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x), ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel("Score"), ax.set_ylim(0, 1)
    ax.set_title("Figure 2 placeholder \u2013 proposed framework against baselines "
                 "and comparison models")
    ax.legend(fontsize=9, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.17))
    ax.grid(axis="y", alpha=0.3)
    _stamp(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig3_ablation(path: str) -> None:
    plt.rcParams.update({"font.family": "serif", "font.size": 11})
    rungs = ["M0\nflat", "M1\n+bottleneck", "M2\n+monotone",
             "M3\n+weak", "M4\n+denoised"]
    rng = np.random.default_rng(1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, name in zip(axes, ["Symptom macro-AUROC", "Suicide-risk QWK"]):
        y = np.sort(rng.uniform(0.40, 0.62, len(rungs)))
        e = rng.uniform(0.01, 0.05, len(rungs))
        ax.errorbar(range(len(rungs)), y, yerr=e, marker="o", capsize=5,
                    linewidth=1.8, color="#2b6cb0")
        ax.fill_between(range(len(rungs)), y - e, y + e, alpha=0.15,
                        color="#2b6cb0")
        ax.set_xticks(range(len(rungs))), ax.set_xticklabels(rungs, fontsize=9)
        ax.set_ylabel(name), ax.grid(alpha=0.3)
        ax.set_title(name)
        _stamp(ax, "PLACEHOLDER")
    fig.suptitle("Figure 3 placeholder \u2013 component-wise ablation ladder, "
                 "mean \u00b1 standard deviation over three seeds")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig4_loadings(path: str) -> None:
    plt.rcParams.update({"font.family": "serif", "font.size": 10})
    syms = ["anhedonia", "depressed_mood", "sleep_disturbance", "fatigue",
            "appetite_change", "worthlessness_guilt", "concentration_problems",
            "psychomotor_change", "suicidal_thoughts", "nervousness",
            "uncontrollable_worry", "excessive_worry", "trouble_relaxing",
            "restlessness", "irritability", "fear_apprehension", "hopelessness",
            "emptiness", "loneliness", "low_self_esteem",
            "thwarted_belongingness", "perceived_burdensomeness"]
    inst = ["PHQ-9"] * 9 + ["GAD-7"] * 7 + ["extra"] * 6
    col = {"PHQ-9": "#2b6cb0", "GAD-7": "#c05621", "extra": "#4a5568"}
    rng = np.random.default_rng(2)
    w = rng.uniform(0.6, 1.5, len(syms))
    order = np.argsort(w)
    fig, ax = plt.subplots(figsize=(8.6, 7.2))
    ax.barh([syms[i] for i in order], w[order],
            color=[col[inst[i]] for i in order], edgecolor="black", linewidth=0.4)
    ax.axvline(1.0, ls="--", c="k", lw=1.4)
    ax.set_xlabel("Weight relative to equal (PHQ-9) item weighting")
    ax.set_title("Figure 4 placeholder \u2013 learned item loadings of the "
                 "monotone ordinal head")
    ax.legend(handles=[mpatches.Patch(color=v, label=k) for k, v in col.items()],
              fontsize=9, loc="lower right")
    _stamp(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig5_training(path: str) -> None:
    plt.rcParams.update({"font.family": "serif", "font.size": 11})
    ep = np.arange(1, 9)
    rng = np.random.default_rng(3)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(ep, 3.4 * np.exp(-0.28 * ep), marker="o", label="train")
    axes[0].plot(ep, 1.15 + 0.03 * np.clip(ep - 3, 0, None), marker="s",
                 label="validation")
    axes[0].set_xlabel("Epoch"), axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss"), axes[0].legend(), axes[0].grid(alpha=0.3)
    axes[1].plot(ep, np.minimum(0.95, 0.30 + 0.09 * ep), marker="o",
                 label="train risk QWK")
    axes[1].plot(ep, 0.38 + 0.01 * rng.standard_normal(len(ep)), marker="s",
                 label="validation risk QWK")
    axes[1].set_xlabel("Epoch"), axes[1].set_ylabel("QWK")
    axes[1].set_title("Ordinal task, train against validation")
    axes[1].legend(), axes[1].grid(alpha=0.3)
    for a in axes:
        _stamp(a, "PLACEHOLDER")
    fig.suptitle("Figure 5 placeholder \u2013 training dynamics and the "
                 "overfitting control")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# docx helpers
# --------------------------------------------------------------------------- #
def _style(run):
    run.font.name = FONT
    run.font.size = SIZE


def para(doc, text="", bold=False, align="justify", style=None):
    p = doc.add_paragraph(style=style)
    p.alignment = {"justify": WD_ALIGN_PARAGRAPH.JUSTIFY,
                   "center": WD_ALIGN_PARAGRAPH.CENTER,
                   "left": WD_ALIGN_PARAGRAPH.LEFT}[align]
    r = p.add_run(text)
    r.bold = bold
    _style(r)
    return p


def heading(doc, text):
    return para(doc, text, bold=True)


def bullet(doc, text):
    return para(doc, text, style="List Bullet")


def caption(doc, text):
    """Table caption: bold, justified, above the table."""
    return para(doc, text, bold=True)


def figure(doc, image, number, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(image, width=FIG_W)
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = cap.add_run(f"Figure {number}. {text}")
    _style(r)


def table(doc, rows, widths=None):
    t = doc.add_table(rows=len(rows), cols=len(rows[0]))
    t.style = "Table Grid"
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = t.cell(i, j)
            cell.text = ""
            p = cell.paragraphs[0]
            r = p.add_run(str(val))
            r.bold = (i == 0) or str(val).strip() == "Proposed"
            r.font.name = FONT
            r.font.size = Pt(10)
    if widths:
        for j, w in enumerate(widths):
            for row in t.rows:
                row.cells[j].width = Inches(w)
    doc.add_paragraph()
    return t


def load_references() -> list[str]:
    """Read the 30 Harvard references out of the existing survey document."""
    if not os.path.isfile(SURVEY):
        raise SystemExit(f"missing {SURVEY}")
    text = open(SURVEY, encoding="utf-8").read()
    refs = re.findall(r"^\[(\d+)\]\s+(.*?)$", text, re.M)
    refs = [(int(n), b.strip()) for n, b in refs]
    refs.sort()
    if len(refs) < 30:
        raise SystemExit(f"only found {len(refs)} references in the survey")
    return [f"[{n}] {body}" for n, body in refs]


TITLE = ("SymptoGraph-LLM: A Symptom-Grounded, Comorbidity-Aware Deep Learning "
         "Framework for Interpretable Multi-Task Mental Health Assessment from "
         "Social Media Text Under Calibrated Large Language Model Weak "
         "Supervision")

KEYWORDS = ("Mental Health Detection, Social Media, Concept Bottleneck Models, "
            "Clinical Symptom Identification, Comorbidity Graph, Large Language "
            "Model Weak Supervision, Label Noise Correction, Ordinal "
            "Regression, Interpretable Machine Learning, PHQ-9 and GAD-7.")


def build_document() -> None:
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Inches(8.27), Inches(11.69)
    sec.left_margin = sec.right_margin = Inches(1.0)
    sec.top_margin = sec.bottom_margin = Inches(1.0)
    st = doc.styles["Normal"]
    st.font.name, st.font.size = FONT, SIZE

    para(doc, TITLE, bold=True, align="center")

    # ------------------------------------------------------------------ abstract
    heading(doc, "Abstract")
    para(doc,
        "Detecting mental-health risk from social media text is a necessary step "
        "towards scalable screening, yet the models that currently perform the "
        "task are difficult for a clinician to audit. Although symptom-level "
        "annotation has been shown to improve the trust that health "
        "professionals place in automated decisions, existing approaches suffer "
        "from a number of severe shortcomings. Notably, they predict a single "
        "disorder label rather than the clinical symptoms a practitioner "
        "reasons with, they treat co-occurring conditions as independent "
        "problems despite repeatedly demonstrating that comorbidity is the "
        "norm, they estimate ordinal severity with unconstrained classifiers "
        "that discard the additive scoring rule by which the instruments are "
        "defined, they consume large language model pseudo-labels as if those "
        "labels were certain, and they report accuracy on corpora whose "
        "overlapping provenance leaks training material into test sets. "
        "Therefore, in our research we introduce a novel deep learning "
        "approach, called SymptoGraph-LLM, designed to address the task of "
        "interpretable, comorbidity-aware mental-health assessment from "
        "user-generated text. First, the introduced SymptoGraph-LLM suggests a "
        "clinical concept bottleneck in which twenty-two symptom queries are "
        "initialised from the sentence embeddings of their own clinical "
        "definitions and every downstream prediction is forced through the "
        "resulting concept vector, so that no disorder-level output can bypass "
        "the symptoms that justify it. Then a symptom-disorder comorbidity "
        "graph propagates evidence along a DSM-5 prior corrected by a learned, "
        "sparsity-penalised residual, allowing the conditional dependence "
        "between conditions to be modelled rather than merely diagnosed. "
        "Afterward, an instrument-grounded monotone ordinal head maps the "
        "concept vector onto a bounded latent distress trait through "
        "non-negative item weights and a cumulative-link model with strictly "
        "ordered cutpoints, reproducing the additive structure by which PHQ-9 "
        "severity is defined and reducing the suicide-risk task to four "
        "threshold parameters. Finally, calibrated weak supervision from a "
        "locally executed large language model is corrected by a per-symptom "
        "noise channel estimated against gold labels, so that systematic "
        "labeller bias is inverted instead of being rescaled by a single "
        "confidence temperature, while a global post identifier, "
        "author-disjoint partitioning and a strictest-split-wins rule ensure "
        "that no post and no author crosses a split boundary across seven "
        "overlapping corpora. The framework is validated using 19,762 "
        "gold-labelled posts drawn from seven public corpora, together with a "
        "pool of 1,017,959 unlabelled posts of which 49,982 receive calibrated "
        f"weak labels, and is compared against three baselines and three "
        f"architectures re-implemented from the recent literature under one "
        f"identical protocol, as well as five component-wise ablations and "
        f"three isolating controls. The experimental validation reports {P} "
        f"symptom macro-AUROC, {P} symptom macro-F1, {P} stress AUROC, {P} "
        f"depression-severity quadratic weighted kappa and {P} suicide-risk "
        f"quadratic weighted kappa, exceeding the strongest comparison model by "
        f"{P} absolute symptom macro-AUROC while using {P} times fewer "
        f"parameters. Additionally, a paired bootstrap over resampled test "
        f"examples, reported alongside per-task confidence intervals, "
        f"establishes that {P} of the component-wise differences are "
        f"distinguishable from seed and sampling noise, and identifies the "
        f"concept bottleneck and the monotone ordinal head as the components "
        f"with a measurable effect, whereas the suicide-risk task is shown to "
        f"carry a confidence interval wider than any architectural difference "
        f"it could resolve.")

    heading(doc, "Keywords")
    para(doc, KEYWORDS)
    para(doc, "")

    # -------------------------------------------------------------- introduction
    heading(doc, "1. Introduction")
    para(doc,
        "Mental-health disorders account for a substantial and growing share of "
        "global disease burden, while the clinical capacity available to "
        "identify them remains far below the level required. Social platforms "
        "have become spaces in which individuals describe their psychological "
        "state in unusual detail, and those descriptions contain traces of "
        "clinically meaningful symptoms. The importance of such traces "
        "increases even more when the goal moves from retrospective "
        "description of a population to the production of an assessment that a "
        "practitioner could act upon, because an output that cannot be "
        "inspected cannot be acted upon responsibly in a clinical setting. "
        "Automated screening is therefore constrained less by raw predictive "
        "accuracy than by whether the quantity being predicted is one that a "
        "clinician recognises, and by whether the evidence supporting it can be "
        "traced back to the text.")
    para(doc,
        "Large language models and transformer encoders have recently been "
        "central to this area. Xu et al. [1] evaluate large language models for "
        "mental-health prediction from online text, establishing both that "
        "instruction tuning improves performance on these tasks and that "
        "zero-shot prompting alone remains below supervised alternatives. Shah "
        "et al. [2] advance depression detection on social platforms through "
        "fine-tuned large language models, and Liu et al. [3] propose DepGLM "
        "for recognising depression degree, demonstrating that the ordinal "
        "structure of severity can be learned rather than thresholded. "
        "However, each of these formulations predicts a disorder-level label or "
        "degree directly from text, so the clinical symptoms that would justify "
        "the prediction are never represented explicitly and cannot be "
        "inspected. There is a gap in architectures that make the symptom layer "
        "an obligatory intermediate representation rather than a post-hoc "
        "explanation.")
    para(doc,
        "A second strand addresses suicide risk specifically. Qorich and El "
        "Ouazzani [5] combine deep learning with large language models for "
        "suicide-ideation detection, Bauer et al. [6] use large language models "
        "to analyse suicidality within a social-media-derived taxonomy of "
        "mental-health disorders, and Lee et al. [7] compare large language "
        "models against expert clinicians for crisis prediction, reporting that "
        "model and clinician judgements diverge in instructive ways. McCoy and "
        "Perlis [8] stratify suicide risk from narrative clinical notes, and Xie "
        "et al. [9] introduce retrieval augmentation for depression screening "
        "and suicide-risk stratification. These works establish that graded "
        "risk is learnable from text. Nevertheless, risk is modelled "
        "independently of the depressive and anxious symptomatology that "
        "clinically precedes it, and the ordinal scales are treated as "
        "unordered categories by the loss functions used. There is a gap in "
        "estimators that share statistical strength between a data-rich "
        "severity task and a data-poor risk task through a common latent "
        "construct.")
    para(doc,
        "Interpretability has been pursued through symptom marking and through "
        "graph structure. Bao et al. [10] propose transformer architectures "
        "that both detect and explain depressive symptom markers in "
        "user-generated text, producing natural-language explanations aligned "
        "with validated symptoms. Suksam et al. [11] perform symptom-level "
        "classification with transformer ensembles before aggregating towards "
        "DSM-5-oriented screening, and Garg [12] extracts wellness concepts "
        "from Reddit posts. In parallel, Chen et al. [16] build a heterogeneous "
        "subgraph network with prompt learning for interpretable depression "
        "detection, Jain et al. [17] propose CDME-GAT using multiple "
        "embeddings and graph attention, and Rehman et al. [18] combine deep "
        "learning with graph neural networks over a graph estimated from post "
        "semantics and author relations. Tahir et al. [19] and Gao et al. [20] "
        "construct knowledge graphs, the former ontology-guided and coupled to "
        "a retrieval-augmented language model. However, the graphs in these "
        "works connect posts or users rather than clinical constructs, so "
        "comorbidity between conditions is not an object the model reasons "
        "over. There is a gap in propagating evidence over a graph whose nodes "
        "are symptoms and disorders.")
    para(doc,
        "A final strand concerns supervision itself. Hsu and Roberts [23] show "
        "that large language models can supply knowledge-free weak supervision "
        "for clinical natural language processing, and Hsieh et al. [21] combine "
        "prompt ensembling with auxiliary self-supervision for multi-label "
        "classification. Khowaja et al. [22] use emotion-aware encoders with "
        "contrastive objectives, and Khalil et al. [24] address privacy through "
        "federated learning. Systematic reviews by Guo et al. [25], Omar et al. "
        "[26], Volkmer et al. [27], Levkovich and Omar [28] and Bucher et al. "
        "[30] converge on the same two cautions: that reported performance is "
        "frequently not comparable across studies, and that evaluation rarely "
        "accounts for the provenance of the corpora used. Jin et al. [29] "
        "respond with PsyEval, a benchmark for mental-health evaluation. "
        "Nevertheless, weak labels are typically consumed as if they were "
        "certain, and the systematic per-symptom bias of the labelling model is "
        "not estimated, so the noise it introduces propagates into the trained "
        "classifier. There is a gap in treating the labelling model as a "
        "measurable noise channel whose inverse can be applied during training.")
    para(doc,
        "Despite significant recent advances in text-based mental-health "
        "assessment, several open research questions therefore remain. Research "
        "in this area is fragmented between disorder classification, symptom "
        "marking, graph-based modelling, weak supervision and benchmark "
        "construction, while no work unites these components into a single "
        "framework in which the symptom layer is obligatory, comorbidity is "
        "modelled, ordinal severity respects the additive rule of the "
        "instrument that defines it, and weak supervision is corrected rather "
        "than trusted. While existing works mainly optimise a single prediction "
        "objective, a clinically usable system should expose which symptoms "
        "drove its output and how confident the underlying evidence was. "
        "Furthermore, the corpora used in this field overlap substantially in "
        "provenance, and that overlap is almost never controlled, so it is "
        "unclear what part of a reported score reflects genuine "
        "generalisation. To address these gaps, we propose SymptoGraph-LLM, a "
        "symptom-grounded and comorbidity-aware framework for interpretable "
        "mental-health assessment. The proposed framework transforms disorder "
        "classification into a constrained, symptom-mediated estimation "
        "problem, in which the model learns which clinical symptoms a post "
        "evidences, how those symptoms co-occur, how they accumulate into "
        "graded severity and risk, and how much of the remaining disagreement "
        "is attributable to label noise rather than to the model.")
    para(doc, "The major contributions of this research are summarised as follows:")
    bullet(doc,
        "A Clinical Concept Bottleneck with definition-initialised symptom "
        "attention is introduced, in which each of the twenty-two symptom "
        "queries is initialised from the sentence embedding of its own clinical "
        "definition and carries an independent scoring vector, and in which "
        "every severity, risk and stress prediction is read from a "
        "concept-gated summary of the attended evidence, so that the model "
        "reasons about clinically named symptoms rather than producing a "
        "disorder label and an explanation after the fact.")
    bullet(doc,
        "A Symptom-Disorder Comorbidity Graph is developed in which a DSM-5 "
        "derived prior adjacency is corrected by a learned residual under an "
        "L1 penalty, and in which disorder-disorder edges are estimated as "
        "conditional co-membership probabilities rather than pointwise mutual "
        "information, so that comorbid structure is propagated through the "
        "estimator instead of being reported as a property of the dataset.")
    bullet(doc,
        "An Instrument-Grounded Monotone Ordinal Head is proposed, which maps "
        "the concept vector onto a latent distress trait through non-negative "
        "item weights and predicts each ordinal task by a cumulative-link model "
        "with strictly ordered cutpoints, reproducing the additive scoring rule "
        "by which PHQ-9 severity bands are defined, guaranteeing monotonicity "
        "in symptom presence by construction, and reducing the suicide-risk "
        "task to four threshold parameters so that it can borrow strength from "
        "the far larger severity task.")
    bullet(doc,
        "A Calibrated Weak Supervision scheme with a Per-Symptom Noise Channel "
        "is introduced, in which a locally executed instruction-tuned large "
        "language model labels the symptoms that no public corpus annotates, "
        "its verbalised confidence is temperature-calibrated against gold "
        "agreement, and its per-symptom true-positive and false-positive rates "
        "are estimated and inverted during training through a forward "
        "correction, so that systematic over-calling of a symptom is discounted "
        "rather than learned.")
    bullet(doc,
        "A Leakage-Controlled Multi-Corpus Protocol and Significance-Audited "
        "Evaluation are introduced, combining a global post identifier over a "
        "normalised text prefix, author-disjoint partitioning, a "
        "strictest-split-wins rule across seven corpora with measured mutual "
        "overlap, validation-loss checkpoint selection, three-seed replication, "
        "per-task bootstrap confidence intervals and paired bootstrap tests on "
        "resampled test examples, thereby ensuring that reported differences "
        "are neither leaked nor artefacts of run-to-run variability.")

    # --------------------------------------------------------- literature survey
    heading(doc, "2. Literature Survey")
    para(doc,
        "Current advances in the area of text-based mental-health assessment "
        "can be summarised in the following major trends: disorder-level "
        "classification with pretrained and instruction-tuned language models, "
        "symptom-level and concept-level interpretable detection, graph-based "
        "and knowledge-graph-based modelling, weak and self-supervised "
        "supervision from large language models, and benchmark and review work "
        "concerned with comparability. Among these research streams there have "
        "been several studies on depression detection, suicide-risk "
        "stratification, severity estimation and multi-label mental-health "
        "classification from Reddit and comparable platforms, some focused on "
        "explanation quality and some on architectural fusion. However, these "
        "streams remain largely separate, and the following review considers "
        "each in turn with attention to what it establishes and what it leaves "
        "open.")
    para(doc,
        "Xu et al. [1] conduct a systematic evaluation of large language models "
        "for mental-health prediction from online text, comparing zero-shot, "
        "few-shot and instruction-tuned regimes across several tasks. The study "
        "confirms that instruction tuning on in-domain data recovers a large "
        "part of the gap to supervised encoders and that prompting alone is "
        "insufficient. However, the prediction target throughout is a "
        "disorder-level label, so the clinical symptoms that would support the "
        "decision are not represented, and the evaluation inherits the "
        "provenance of the source corpora without controlling for overlap "
        "between them. There is a gap in symptom-mediated prediction evaluated "
        "under explicit leakage control. Shah et al. [2] fine-tune large "
        "language models for depression detection and report consistent gains "
        "over encoder baselines, confirming that scale helps on this task, but "
        "the model remains a binary classifier whose decision cannot be "
        "attributed to clinical criteria.")
    para(doc,
        "Liu et al. [3] propose DepGLM for depression-degree recognition, "
        "treating severity as the prediction target rather than presence alone. "
        "The work is notable because it acknowledges that severity is graded "
        "and that a binary decision discards clinically relevant information. "
        "Nevertheless, the degrees are predicted by an unconstrained classifier "
        "head, so the ordering between adjacent degrees is not enforced and the "
        "additive rule by which the underlying instrument defines its bands is "
        "discarded. There is a gap in ordinal estimators whose structure "
        "reproduces the scoring rule of the instrument. Abdullah and Negied [4] "
        "combine machine learning, ensemble learning and large language models "
        "to predict future mental disorder from social-media data, "
        "demonstrating that ensembling stabilises predictions, but the ensemble "
        "is over classifiers of the same disorder-level target and does not "
        "introduce an interpretable intermediate layer.")
    para(doc,
        "Qorich and El Ouazzani [5] apply advanced deep learning and large "
        "language models to suicide-ideation detection, and Bauer et al. [6] "
        "analyse suicidality linguistically within a social-media-derived "
        "taxonomy of mental-health disorders, establishing that the linguistic "
        "signature of risk is distinguishable from that of general distress. "
        "Lee et al. [7] compare large language models against expert clinicians "
        "for crisis prediction among telemental-health patients, which is among "
        "the most clinically grounded evaluations available, and report that "
        "agreement is partial. However, these works model risk in isolation "
        "from the depressive and anxious symptomatology that clinically "
        "precedes it, so no statistical strength is shared between a scarce "
        "risk target and the far more abundant symptom annotations. There is a "
        "gap in multi-task estimators that couple risk to symptoms through a "
        "shared latent construct.")
    para(doc,
        "McCoy and Perlis [8] stratify suicide risk from narrative clinical "
        "notes using large language models, demonstrating transfer of the "
        "approach into a clinical-record setting where the text is produced by "
        "professionals rather than patients. Xie et al. [9] introduce "
        "retrieval-augmented large language models for depression screening and "
        "suicide-risk stratification, showing that grounding generation in "
        "retrieved material improves reliability. Nevertheless, in both cases "
        "the output is a stratification without an auditable intermediate "
        "representation, and retrieval grounds the generation rather than "
        "constraining the prediction. There is a gap in constraining the "
        "prediction itself to pass through clinically named concepts.")
    para(doc,
        "Bao et al. [10] introduce transformer architectures that detect and "
        "explain the appearance of depressive symptom markers in "
        "user-generated content, presenting both a separated classify-then-"
        "explain design and a unified model, and evaluating explanation quality "
        "with offline and expert-in-the-loop metrics. This work is directly "
        "relevant because it establishes that symptom markers improve the trust "
        "health professionals place in automated decisions. However, the "
        "symptom layer serves as an explanation accompanying a decision rather "
        "than as a bottleneck the decision must pass through, so nothing "
        "prevents the classifier from relying on evidence the explanation does "
        "not mention. There is a gap in architectures where the symptom "
        "representation is obligatory rather than concomitant.")
    para(doc,
        "Suksam et al. [11] propose transformer-based ensemble learning for "
        "symptom-level classification followed by DSM-5-oriented depression "
        "screening, aggregating predicted symptoms over an observation window. "
        "The two-phase design is valuable because it makes the symptom layer "
        "the primary learning target and derives the screening decision from "
        "it. Nevertheless, the aggregation from symptoms to screening is a "
        "fixed rule applied after classification rather than a learned, bounded "
        "response, and the ensemble members are independent classifiers that "
        "share no clinical structure. There is a gap in learning the "
        "symptom-to-severity mapping as a constrained, monotone function. Garg "
        "[12] extracts and classifies wellness concepts in Reddit posts, "
        "establishing concept extraction as a viable intermediate task, but the "
        "concepts are not tied to a diagnostic instrument.")
    para(doc,
        "Zhang et al. [13] propose a knowledge-aware deep learning approach to "
        "depression detection using digital traces, incorporating external "
        "knowledge into the representation. Qasim et al. [14] detect depression "
        "severity with transformer-based models, and Kerasiotis et al. [15] "
        "combine transformer encoders with auxiliary handcrafted features, "
        "showing that linguistic features remain complementary to learned "
        "representations. However, the external knowledge in these works "
        "conditions the input representation rather than structuring the output "
        "space, so dependence between the predicted conditions is not "
        "represented. There is a gap in modelling dependence among the "
        "predicted clinical constructs themselves.")
    para(doc,
        "Chen et al. [16] construct a heterogeneous subgraph network with "
        "prompt learning for interpretable depression detection, and Jain et "
        "al. [17] propose CDME-GAT, which fuses multiple embeddings with graph "
        "attention over a constructed context graph. Rehman et al. [18] predict "
        "both depression and suicide risk using deep learning and graph neural "
        "networks over a graph estimated from semantic similarity between posts "
        "together with author relations, rather than from social interactions, "
        "and add post-hoc attribution and calibration analysis. These works "
        "establish that graph structure improves both accuracy and "
        "interpretability. Nevertheless, the nodes of these graphs are posts or "
        "users, so the structure encodes who resembles whom rather than which "
        "clinical conditions co-occur, and comorbidity remains outside the "
        "model. There is a gap in graphs whose nodes are symptoms and disorders "
        "and whose edges carry clinical prior information.")
    para(doc,
        "Tahir et al. [19] present an ontology-guided knowledge graph coupled "
        "with a retrieval-augmented large language model framework for "
        "interpretable mental-health assessment, and Gao et al. [20] use large "
        "language models to construct knowledge graphs for mental-health "
        "exploration. Both demonstrate that clinical ontologies can be brought "
        "into the modelling pipeline. However, the knowledge graph is used to "
        "retrieve or to organise evidence for a generative model rather than to "
        "define a differentiable adjacency over which symptom evidence "
        "propagates during training, and neither reports a control establishing "
        "that the ontology-derived structure outperforms a randomised graph of "
        "equal density. There is a gap in differentiable clinical graphs "
        "evaluated against a random-structure control.")
    para(doc,
        "Hsieh et al. [21] address multi-label mental-health classification "
        "with a multi-perspective prompt-design module and a perturbation-based "
        "self-supervised module on a pretrained backbone, which is among the "
        "closest formulations to the multi-label symptom setting considered "
        "here. Khowaja et al. [22] combine emotion-aware encoders with "
        "fuzzy-based contrastive networks. Nevertheless, the label space in "
        "these works consists of disorders rather than instrument items, so the "
        "multi-label structure does not correspond to a clinical scoring "
        "instrument and cannot be aggregated into graded severity. There is a "
        "gap in multi-label formulations whose labels are the items of a "
        "validated instrument.")
    para(doc,
        "Hsu and Roberts [23] demonstrate knowledge-free weak supervision from "
        "large language models for clinical natural language processing, "
        "establishing that model-generated labels can substitute for scarce "
        "expert annotation. This is the most directly relevant precedent for "
        "the weak-supervision component of the present work. However, the weak "
        "labels are consumed as if they were correct, and no per-class estimate "
        "of the labelling model's systematic bias is produced, so a class the "
        "labeller systematically over-calls transfers that bias into the "
        "trained classifier. There is a gap in estimating and inverting the "
        "labelling model's per-class noise channel. Khalil et al. [24] address "
        "a complementary concern, showing that federated training preserves "
        "privacy for multilingual depression detection, but do not address "
        "label reliability.")
    para(doc,
        "Finally, a body of review work converges on the methodological "
        "weaknesses of the field. Guo et al. [25], Omar et al. [26] and Bucher "
        "et al. [30] systematically review large language model applications to "
        "mental health and consistently report that evaluations are not "
        "comparable across studies, that corpus provenance is seldom examined, "
        "and that clinical validity is rarely established. Volkmer et al. [27] "
        "discuss the resulting opportunities and challenges in psychiatry, and "
        "Levkovich and Omar [28] compare encoder-based and generative models for "
        "suicide detection, finding that reported margins are frequently within "
        "the variability of the evaluation itself. Jin et al. [29] respond with "
        "PsyEval, a benchmark intended to standardise evaluation. Nevertheless, "
        "these reviews diagnose the problem without supplying a protocol that "
        "controls corpus overlap and reports significance, which is precisely "
        "what the present work adopts. There is a gap in primary studies that "
        "implement the controls the reviews call for.")
    caption(doc, "Table 1. Comparison of the proposed SymptoGraph-LLM with "
                 "existing mental-health detection and symptom-identification "
                 "methods.")
    table(doc, [
        ["Ref.", "Authors / Year", "Method / Focus", "Major Findings",
         "Limitations / Research Gap"],
        ["[1]", "Xu et al. (2024)",
         "Mental-LLM; zero-shot, few-shot and instruction-tuned LLMs for "
         "mental-health prediction",
         "Instruction tuning recovers much of the gap to supervised encoders; "
         "prompting alone is insufficient",
         "Disorder-level target only; no symptom layer; corpus overlap not "
         "controlled"],
        ["[3]", "Liu et al. (2025)",
         "DepGLM; depression-degree recognition with LLMs",
         "Graded severity is learnable from text and is more informative than "
         "binary labels",
         "Unconstrained ordinal head; ordering not enforced; instrument scoring "
         "rule discarded"],
        ["[7]", "Lee et al. (2024)",
         "LLMs against expert clinicians for crisis prediction",
         "Clinically grounded comparison; model and clinician judgements agree "
         "only partially",
         "Risk modelled in isolation from symptomatology; no shared latent "
         "construct"],
        ["[9]", "Xie et al. (2026)",
         "Retrieval-augmented LLMs for depression screening and suicide-risk "
         "stratification",
         "Grounding generation in retrieved material improves reliability",
         "Retrieval grounds generation, not the prediction; no auditable "
         "intermediate representation"],
        ["[10]", "Bao et al. (2024)",
         "Transformers that detect and explain depressive symptom markers",
         "Symptom markers increase clinician trust; explanations align with "
         "validated symptoms",
         "Symptom layer accompanies the decision rather than constraining it"],
        ["[11]", "Suksam et al. (2026)",
         "Transformer ensemble for symptom-level classification, then DSM-5 "
         "screening",
         "Symptom layer as primary target; screening derived from predicted "
         "symptoms",
         "Fixed post-hoc aggregation rule; ensemble members share no clinical "
         "structure"],
        ["[15]", "Kerasiotis et al. (2024)",
         "Transformer encoders fused with auxiliary handcrafted features",
         "Linguistic features remain complementary to learned representations",
         "Knowledge conditions the input; dependence among predicted constructs "
         "unmodelled"],
        ["[16]", "Chen et al. (2025)",
         "Heterogeneous subgraph network with prompt learning",
         "Graph structure improves both accuracy and interpretability",
         "Nodes are posts or users; comorbidity between conditions not "
         "represented"],
        ["[18]", "Rehman et al. (2026)",
         "Deep learning and GNN over a graph from post semantics and author "
         "relations",
         "Joint depression and risk prediction; adds attribution and "
         "calibration analysis",
         "Graph encodes textual resemblance, not clinical co-occurrence; no "
         "random-graph control"],
        ["[19]", "Tahir et al. (2026)",
         "Ontology-guided knowledge graph with retrieval-augmented LLM",
         "Clinical ontologies can be brought into the assessment pipeline",
         "Ontology retrieves evidence rather than defining a differentiable "
         "adjacency"],
        ["[21]", "Hsieh et al. (2025)",
         "Multi-perspective prompt ensemble with perturbation self-supervision",
         "Prompt diversity and consistency regularisation help multi-label "
         "classification",
         "Labels are disorders, not instrument items; cannot aggregate into "
         "graded severity"],
        ["[23]", "Hsu & Roberts (2025)",
         "Knowledge-free LLM weak supervision for clinical NLP",
         "Model-generated labels can substitute for scarce expert annotation",
         "Weak labels treated as certain; per-class labeller bias neither "
         "estimated nor inverted"],
        ["[25], [26], [30]", "Guo et al. (2024); Omar et al. (2024); Bucher et "
         "al. (2025)",
         "Systematic reviews of LLMs in mental health and psychiatry",
         "Evaluations are not comparable; provenance seldom examined; clinical "
         "validity rarely established",
         "Diagnose the problem without supplying a leakage-controlled, "
         "significance-audited protocol"],
        ["Proposed", "SymptoGraph-LLM (2026)",
         "Symptom-grounded concept bottleneck over a comorbidity graph, with a "
         "monotone instrument-grounded ordinal head and noise-corrected LLM "
         "weak supervision",
         "Symptom layer is obligatory; comorbidity propagated; severity "
         "monotone in symptom presence by construction; labeller bias inverted",
         "Integrates interpretable symptom mediation, comorbid structure and "
         "corrected weak supervision under leakage control and paired "
         "significance testing"],
    ], widths=[0.55, 1.05, 1.55, 1.45, 1.55])
    para(doc,
        "Taking into account the abovementioned literature review, there have "
        "been numerous developments in terms of instruction-tuned language "
        "models for disorder classification, symptom-marker detection and "
        "explanation, graph-based and knowledge-graph-based modelling, weak "
        "supervision from large language models, and benchmark construction for "
        "mental-health natural language processing. However, most of the "
        "existing studies deal only with particular issues, such as improving a "
        "single disorder-level score, generating an explanation alongside a "
        "decision, or fusing additional features, but fail to provide an "
        "integrated method in which the clinical symptom layer is an obligatory "
        "intermediate representation and the disorder-level outputs are derived "
        "from it under clinically motivated constraints. Moreover, most "
        "existing solutions do not incorporate dependence between the predicted "
        "conditions, although comorbidity has repeatedly been shown to be the "
        "norm rather than the exception. Although several studies employ large "
        "language models as annotators, the systematic per-class bias of the "
        "annotating model is not estimated, so a symptom the model "
        "over-recognises transfers that bias into the trained classifier. "
        "Besides, it is rare for existing research to control for the "
        "substantial mutual overlap between the public corpora used in this "
        "field, or to report whether a claimed margin exceeds the variability "
        "of the evaluation itself, so it is difficult to state whether reported "
        "accuracy reflects genuine generalisation or shared provenance between "
        "training and test material.")

    # ------------------------------------------------------------- methodology
    heading(doc, "3. Proposed Methodology")
    para(doc,
        "The proposed SymptoGraph-LLM framework can be applied for "
        "interpretable mental-health assessment from short, noisy, "
        "user-generated text in which clinical labels are scarce, unevenly "
        "distributed across conditions and drawn from corpora of overlapping "
        "provenance. Different from existing frameworks that either classify a "
        "disorder directly, generate an explanation alongside a decision, or "
        "fuse auxiliary features into the input representation, the proposed "
        "framework integrates the following elements into a unified estimation "
        "pipeline: a clinical concept bottleneck with definition-initialised "
        "symptom attention, a symptom-disorder comorbidity graph with a learned "
        "residual, an instrument-grounded monotone ordinal head, calibrated "
        "weak supervision with a per-symptom noise channel, and a "
        "leakage-controlled evaluation protocol with paired significance "
        "testing. First, seven public corpora are harmonised into a single "
        "canonical table over a shared twenty-two symptom space derived from "
        "the PHQ-9 and GAD-7 instruments, with a global post identifier "
        "assigned over a normalised text prefix so that a post appearing in "
        "several corpora collapses to one row. Then the text is encoded by a "
        "domain-pretrained transformer and projected onto the symptom space by "
        "attention queries initialised from the sentence embeddings of the "
        "clinical definitions, producing a concept vector through which every "
        "downstream prediction must pass. The concept vector is refined by "
        "message passing over a graph whose nodes are symptoms and disorders "
        "and whose adjacency combines a DSM-5 derived prior with a learned, "
        "L1-penalised residual. Meanwhile, an instrument-grounded monotone "
        "ordinal head maps the concept vector onto a bounded latent distress "
        "trait through non-negative item weights and predicts each ordinal task "
        "by a cumulative-link model with strictly ordered cutpoints. In "
        "parallel, an instruction-tuned large language model executed locally "
        "labels the eleven symptoms that no public corpus annotates, its "
        "verbalised confidence is temperature-calibrated against gold "
        "agreement, and its per-symptom true-positive and false-positive rates "
        "are estimated and inverted during training. Finally, performance is "
        "evaluated under author-disjoint splits with validation-loss checkpoint "
        "selection, three-seed replication, per-task bootstrap confidence "
        "intervals and paired bootstrap tests on resampled test examples, "
        "against three baselines, three architectures re-implemented from the "
        "recent literature, five component-wise ablations and three isolating "
        "controls. The full pipeline of the proposed SymptoGraph-LLM framework "
        "is illustrated in Figure 1.")
    para(doc, "")
    figure(doc, os.path.join(FIGS, "figure1_architecture.png"), 1,
           "Overall workflow of the proposed SymptoGraph-LLM framework.")

    heading(doc, "3.1 Dataset Description")
    para(doc,
        "A harmonised corpus assembled from seven publicly available "
        "mental-health text collections is selected as the primary dataset for "
        "the development and evaluation of the SymptoGraph-LLM framework, due "
        "to its combination of symptom-level, severity-level and risk-level "
        "annotation over the same canonical label space, which is required for "
        "symptom-mediated estimation by structured deep learning. In contrast "
        "to conventional mental-health datasets that provide a single binary "
        "disorder label or a single severity scale, the assembled corpus "
        "comprises item-level symptom annotations aligned to the PHQ-9 and "
        "GAD-7 instruments, four-level depression severity, five-level "
        "clinician-coded suicide risk, binary perceived stress, author "
        "identifiers, posting dates and subreddit provenance, enabling the "
        "proposed framework to learn symptom-mediated and comorbidity-aware "
        "response behaviour. The gold-labelled partition consists of 19,762 "
        "unique posts over a shared space of 22 symptoms, of which 9 correspond "
        "to PHQ-9 items, 7 to GAD-7 items and 6 to additional constructs drawn "
        "from the interpersonal-risk and loneliness literature. The constituent "
        "sources are DepressionEmo for item-level depressive symptoms, Dreaddit "
        "for perceived stress, DepSeverity for four-level depression severity, "
        "the C-SSRS annotated Reddit collection for five-level suicide risk, "
        "the Interpersonal Risk Factors and LoST collections for thwarted "
        "belongingness, perceived burdensomeness and loneliness, and the IMHI "
        "instruction collection, together with a large unlabelled pool of "
        "1,017,959 posts drawn from the Reddit Mental Health Dataset spanning "
        "108 subreddit files and 822,884 distinct authors. Symptom labels use "
        "three states, denoting presence, absence and absence of annotation by "
        "that source, so that an unannotated symptom is masked out of the loss "
        "rather than being treated as a negative observation.")
    para(doc,
        "Before developing the models, the harmonised corpus is pre-processed "
        "and partitioned so that no post and no author can appear on both sides "
        "of a split. Five distinct routes by which the constituent corpora "
        "overlap were measured and are reported in Table 2; the most severe is "
        "that DepSeverity is a re-annotation of the same 3,526 posts contained "
        "in Dreaddit, so that treating either as an independent evaluation set "
        "for the other would be circular. A global post identifier is therefore "
        "computed as the hash of a normalised two-hundred-character prefix, "
        "each unique post is assigned exactly one split, Reddit posts with a "
        "known author are partitioned author-disjointly, and where corpora "
        "disagree the strictest assignment prevails. The partitioning is "
        "verified by an assertion that raises if any post or any author crosses "
        "a split boundary. De-seasonalised provenance metadata, clear-text "
        "length statistics and observation-validity masks are preserved to "
        "support loss masking and subgroup analysis. The resulting dataset "
        "partition used for the experimental study is illustrated in Table 3.")
    caption(doc, "Table 2. Measured overlap between the constituent corpora, "
                 "and the control applied to each.")
    table(doc, [
        ["Overlap route", "Measured extent", "Control applied"],
        ["DepSeverity re-annotates Dreaddit",
         "3,526 posts identical in both corpora",
         "Single global post identifier; one split per unique post"],
        ["LoST and IRF share posts", "3,089 posts in common",
         "Deduplication before splitting; strictest-split-wins"],
        ["IMHI repackages IRF and Dreaddit",
         "Posts re-emitted behind an instruction prefix",
         "Prefix stripped before identifier computation"],
        ["Authors appear in multiple source files",
         "192,893 of 822,884 authors (23.3%)",
         "Author-disjoint partitioning with a repair pass"],
        ["Duplicate posts within the unlabelled pool", "85,659 duplicates",
         "Removed at identifier assignment"],
    ], widths=[1.9, 1.9, 2.4])
    caption(doc, "Table 3. Dataset Partitioning for SymptoGraph-LLM")
    table(doc, [
        ["Dataset Split", "Posts", "Symptom-annotated", "Severity", "Risk",
         "Split %"],
        ["Training Set", "12,911", "6,227", "2,456", "340", "65.33%"],
        ["Validation Set", "2,862", "1,397", "529", "86", "14.48%"],
        ["Testing Set", "3,989", "1,956", "541", "74", "20.19%"],
        ["Total (gold-labelled)", "19,762", "9,580", "3,526", "500", "100%"],
        ["Unlabelled pool", "1,017,959", "\u2013", "\u2013", "\u2013", "\u2013"],
        ["LLM weak-labelled subset", "49,982", "49,982", "\u2013", "\u2013",
         "\u2013"],
    ], widths=[1.6, 1.0, 1.3, 0.8, 0.6, 0.8])
    para(doc,
        "The training subset is used for the purpose of encoder fine-tuning, "
        "concept-bottleneck and graph-residual optimisation, and ordinal "
        "cutpoint estimation during framework development. The validation "
        "subset is used for monitoring the learning process, selecting the "
        "checkpoint by minimum validation loss, tuning the per-symptom decision "
        "thresholds and applying early stopping, in such a way that the final "
        "test data is never involved in parameter or threshold optimisation. "
        "The independent test subset is used for the evaluation of the "
        "framework's ability to identify clinical symptoms, estimate graded "
        "severity and risk, and maintain calibration under class imbalance. The "
        "same subsets are used both for the proposed framework and its "
        "component-wise ablations, and additionally for all re-trained "
        "baselines and comparison models, for the purpose of making a fair "
        "comparison. It should be noted that the suicide-risk partition is "
        "markedly smaller than the others, comprising 340 training and 74 test "
        "instances across five ordinal levels, and the consequences of this "
        "scarcity for the interpretation of risk metrics are quantified "
        "explicitly in Section 4. In addition to the gold-labelled partition, "
        "the unlabelled pool of 1,017,959 posts supports the weak-supervision "
        "component, of which a stratified subsample of 49,982 posts receives "
        "calibrated large language model labels for the eleven symptoms that no "
        "constituent corpus annotates, with controls capped at fifteen per cent "
        "of the subsample so that rare conditions are not swamped. Consequently, "
        "the harmonised corpus provides item-level symptom annotation, graded "
        "severity, clinician-coded risk, perceived stress and author provenance "
        "required for the SymptoGraph-LLM framework development, while the "
        "unlabelled pool supports calibrated weak supervision and the "
        "measurement of the labelling model's per-symptom noise channel.")

    # ------------------------------------------------- 3.2 preprocessing onwards
    heading(doc, "3.2 Clinical Concept Bottleneck with Definition-Initialised "
                 "Attention")
    para(doc,
        "The first component of the framework projects the encoded text onto "
        "the clinical symptom space and forces every subsequent prediction "
        "through that projection. A domain-pretrained transformer encoder "
        "produces contextual token representations, over which twenty-two "
        "learnable attention queries, one per symptom, compute an independent "
        "attention distribution. The resulting evidence vector for each symptom "
        "is scored by its own scoring vector, yielding a concept logit per "
        "symptom. Two design choices are load-bearing. First, each symptom "
        "query is initialised from the sentence embedding of that symptom's "
        "clinical definition rather than randomly, so attention begins from "
        "clinical meaning and the association between a query and its symptom "
        "is established before any gradient is applied. Second, each symptom "
        "receives an independent scoring vector rather than sharing a single "
        "projection, because a shared projection leaves each symptom only a "
        "scalar rescaling of one common direction, which is not a concept "
        "bottleneck and was observed during development to collapse the "
        "symptom head entirely. The severity, risk and stress heads then read a "
        "concept-gated summary of the attended evidence, in which each "
        "symptom's evidence vector is weighted by that symptom's activation, so "
        "the heads observe only what the concepts attended to while retaining "
        "sufficient representational width to estimate graded outcomes.")

    heading(doc, "3.3 Symptom-Disorder Comorbidity Graph")
    para(doc,
        "The second component propagates evidence between clinical constructs. "
        "A graph is constructed whose nodes comprise the twenty-two symptoms "
        "together with the disorder categories observed in the corpus, and "
        "whose adjacency is the sum of a fixed prior and a learned residual "
        "under an L1 penalty, so that the data may add or suppress edges while "
        "remaining anchored to clinical knowledge. The symptom-disorder block "
        "of the prior is authored from DSM-5 criteria. The disorder-disorder "
        "block is estimated from author co-membership across the unlabelled "
        "pool as a conditional co-membership probability rather than as "
        "pointwise mutual information, because the mean number of distinct "
        "conditions per author in this corpus is 1.125, which bounds the "
        "attainable lift below unity and causes a pointwise mutual information "
        "matrix to vanish identically. The estimated edges are clinically "
        "plausible, with the strongest comorbid associations holding between "
        "bipolar disorder and schizophrenia, between borderline personality "
        "disorder and eating disorders, and between attention-deficit "
        "hyperactivity disorder and autism. Message passing applies a "
        "non-negativity constraint to the summed adjacency before symmetric "
        "normalisation and inserts layer normalisation after each residual "
        "step, because an unconstrained learned residual can drive the degree "
        "of a node towards zero and thereby amplify activations without bound.")

    heading(doc, "3.4 Instrument-Grounded Monotone Ordinal Head")
    para(doc,
        "The third component estimates the graded outcomes. Depression "
        "severity, as operationalised by the PHQ-9, is defined as a threshold "
        "function of the sum of its nine item scores, and nine of the "
        "twenty-two symptoms in the present label space are exactly those "
        "items. An unconstrained linear classifier over the concept vector "
        "discards this structure entirely. Instead, a latent distress trait is "
        "formed as a weighted sum of symptom activations under non-negative "
        "item weights, and each ordinal task is predicted by a cumulative-link "
        "model in which the probability of exceeding each level is a logistic "
        "function of the trait minus a strictly ordered cutpoint. Three "
        "properties follow by construction. The predicted severity is monotone "
        "non-decreasing in the presence of any symptom, which is a hard "
        "constraint rather than a learned tendency. The ordering between "
        "adjacent levels cannot be violated, because the cutpoints are "
        "parameterised as a first threshold plus positive increments. And the "
        "suicide-risk task reduces to four threshold parameters over a trait "
        "that is simultaneously trained by the far more abundant severity "
        "annotations, so strength is shared between a data-rich and a data-poor "
        "task. The cutpoint spacing at initialisation is a consequential "
        "hyperparameter rather than a cosmetic one: in a cumulative-link model "
        "an interior class attains a maximum probability of approximately the "
        "logistic difference across the spacing, so an insufficient initial "
        "spacing renders the interior classes unreachable at any value of the "
        "trait, and the spacing adopted here is chosen so that every class is "
        "attainable.")

    heading(doc, "3.5 Calibrated Weak Supervision with a Per-Symptom Noise "
                 "Channel")
    para(doc,
        "The fourth component supplies supervision for the eleven symptoms, "
        "predominantly the GAD-7 anxiety items, that no constituent corpus "
        "annotates. An instruction-tuned large language model is executed "
        "locally, because transmitting these posts to a third-party service "
        "would breach the data agreements attached to several of the source "
        "corpora, and labels a stratified subsample of 49,982 posts with a "
        "verbalised confidence per symptom. The weak rows enter training as a "
        "separate stream drawn once per optimisation step rather than being "
        "concatenated into the training set, because concatenation inflates the "
        "number of steps per epoch, alters the learning-rate schedule and "
        "reduces the density of the scarce ordinal labels within each batch, "
        "all of which were observed during development to degrade precisely the "
        "tasks with the fewest annotations. The confidence values are "
        "calibrated by a single temperature fitted against observed agreement "
        "on the symptoms that do have gold labels. Crucially, a temperature "
        "rescales all labels by the same factor and therefore cannot correct "
        "systematic bias, which was measured to be severe and strongly "
        "symptom-dependent: across the eleven measurable symptoms the labelling "
        "model's true-positive rate ranges from 0.252 to 0.944 and its "
        "false-positive rate from 0.065 to 0.745, so that one symptom is "
        "recovered in only one case out of four while another is asserted in "
        "three quarters of genuine negatives. A per-symptom noise channel is "
        "therefore estimated from paired gold and model labels under Laplace "
        "smoothing, and training applies a forward correction in which the "
        "network predicts the clean probability and the loss is evaluated after "
        "passing that probability through the estimated channel. For the eleven "
        "symptoms with no gold labels the channel is not identifiable and is "
        "pooled within instrument; this is an assumption rather than a "
        "measurement and is reported as such, since six of the seven GAD-7 "
        "items inherit the channel estimated from a single measurable item.")

    heading(doc, "3.6 Training Protocol and Evaluation")
    para(doc,
        "All models are trained under one shared configuration so that the only "
        "quantity varying between conditions is the representation under test. "
        "Class imbalance is addressed on two axes: capped inverse-frequency "
        "weights are applied to the ordinal cross-entropies, because the "
        "depression-severity distribution is dominated by its lowest band and "
        "unweighted cross-entropy collapses the ordinal heads onto the majority "
        "class, and capped positive weights are applied to the symptom "
        "objective, whose per-symptom prevalence is likewise skewed. "
        "Per-symptom decision thresholds are tuned on validation data and "
        "applied unchanged to the test split. Checkpoint selection uses minimum "
        "validation loss rather than an aggregate score, because an aggregate "
        "dominated by symptom metrics continues to improve while the ordinal "
        "heads memorise their small training partitions, which was observed to "
        "select checkpoints at approximately twice the minimum validation loss. "
        "Every configuration is replicated across three seeds. Reported "
        "differences are assessed by a paired bootstrap over resampled test "
        "examples with the same resampled indices applied to both models, and a "
        "difference is reported as established only if it attains significance "
        "on every seed and preserves its sign on every seed. Per-task bootstrap "
        "confidence intervals are reported alongside every metric, because the "
        "evaluation partitions differ in size by more than an order of "
        "magnitude between tasks. The shared configuration is summarised in "
        "Table 4.")
    caption(doc, "Table 4. Shared training configuration applied to the proposed "
                 "framework, its ablations, the baselines and the comparison "
                 "models.")
    table(doc, [
        ["Setting", "Value", "Rationale"],
        ["Encoder", "MentalRoBERTa (domain-pretrained, 125M)",
         "Strict loading; a silent substitution would misreport the comparison"],
        ["Maximum sequence length", "192 tokens",
         "Covers approximately 80% of posts; median 73 words"],
        ["Batch size", "32", "Identical across all conditions"],
        ["Learning rate", "2e-5 encoder / 1e-3 heads",
         "Pretrained encoder and randomly initialised heads require different "
         "rates"],
        ["Schedule", "Linear warmup and decay, 10% warmup",
         "Identical across all conditions"],
        ["Epochs", "8, early stopping patience 2",
         "Stopping rule is minimum validation loss"],
        ["Checkpoint selection", "Minimum validation loss",
         "An aggregate score dominated by one task selects overfitted epochs"],
        ["Ordinal class weights", "Capped inverse frequency",
         "Severity is dominated by its lowest band; risk is similarly skewed"],
        ["Symptom positive weights", "Capped inverse frequency, cap 10",
         "Per-symptom prevalence is skewed"],
        ["Decision thresholds", "Per symptom, tuned on validation",
         "Applied unchanged to the test split"],
        ["Label smoothing", "0.05 on ordinal heads",
         "Mild regularisation of the small ordinal partitions"],
        ["Seeds", "3", "Minimum for a variance estimate"],
        ["Significance", "Paired bootstrap, 1,000 resamples",
         "Same resampled examples on both sides of every comparison"],
    ], widths=[1.7, 2.1, 2.4])

    # ---------------------------------------------------------------- results
    heading(doc, "4. Results and Discussion")
    para(doc,
        "This section reports the experimental evaluation of the proposed "
        "SymptoGraph-LLM framework. All numerical results in this section are "
        "placeholders of the form X.XXX and are to be replaced by the "
        "corresponding entries produced by the evaluation notebooks; the "
        "mapping from each table and figure to the file that supplies its "
        "values is recorded in the accompanying placeholder index. Dataset "
        "counts, corpus overlap measurements and labelling-model noise-channel "
        "estimates reported in Section 3 are measured quantities and are not "
        "placeholders.")

    heading(doc, "4.1 Comparison with Baselines and Recent Architectures")
    para(doc,
        "The proposed framework is compared against three baselines and three "
        "architectures re-implemented from the recent literature. The baselines "
        "comprise a term-frequency representation with a multilayer perceptron, "
        "a bidirectional recurrent encoder with attention, and a fine-tuned "
        "domain-pretrained transformer. The comparison models are a transformer "
        "stacking ensemble for symptom-level classification following Suksam et "
        "al. [11], a graph neural network over a post graph estimated from "
        "semantic similarity and author relations following Rehman et al. [18], "
        "and a multi-perspective prompt ensemble with perturbation-based "
        "self-supervision following Hsieh et al. [21]. It must be stated "
        "explicitly that these are re-implementations of the architectures "
        "described in those works, adapted to the twenty-two symptom label "
        "space and four-task output of the present study and trained under the "
        "shared protocol of Table 4; they are not reproductions of the "
        "published systems and their results are not comparable with the "
        "figures reported in the original papers. Two components of the cited "
        "designs cannot be exercised on the present data and are reported as "
        "out of scope rather than as failures: the window-based screening stage "
        "of [11] requires several posts per author within a fixed observation "
        "window, whereas the gold-labelled partition here has a median of one "
        f"post per author, and the post-hoc attribution analyses of [18] do not "
        f"affect any metric in this comparison. The results are reported in "
        f"Table 5 and visualised in Figure 2. The proposed framework attains "
        f"{P} symptom macro-AUROC against {P} for the strongest comparison "
        f"model, a difference of {P} absolute, while the stress task saturates "
        f"near {P} for all conditions.")
    caption(doc, "Table 5. Performance of the proposed framework against the "
                 "baselines and the re-implemented comparison models, reported "
                 "as mean \u00b1 standard deviation over three seeds on the "
                 "held-out test split.")
    table(doc, [
        ["Model", "Symptom macro-F1", "Symptom macro-AUROC", "Severity QWK",
         "Risk QWK", "Stress AUROC", "Params (M)"],
        ["TF-IDF + MLP", PM, PM, PM, PM, PM, P],
        ["BiLSTM + Attention", PM, PM, PM, PM, PM, P],
        ["MentalRoBERTa (fine-tuned)", PM, PM, PM, PM, PM, P],
        ["Transformer stacking ensemble, after [11]", PM, PM, PM, PM, PM, P],
        ["Post-graph GNN, after [18]", PM, PM, PM, PM, PM, P],
        ["Prompt ensemble with self-supervision, after [21]", PM, PM, PM, PM,
         PM, P],
        ["Proposed", PM, PM, PM, PM, PM, P],
    ], widths=[1.7, 0.95, 0.95, 0.8, 0.7, 0.8, 0.6])
    figure(doc, os.path.join(FIGS, "figure2_vs_baselines.png"), 2,
           "Placeholder. Performance of the proposed framework against the "
           "baselines and the re-implemented comparison models across the "
           "symptom and stress tasks.")

    heading(doc, "4.2 Component-Wise Ablation Study")
    para(doc,
        "The contribution of each component is isolated by a ladder in which "
        "exactly one element is added at each step, beginning from a flat "
        "encoder with unconstrained heads and ending with the complete "
        "framework. Every rung shares the configuration of Table 4 without "
        "exception, so no rung benefits from a learning rate, schedule, epoch "
        "budget or selection rule that another did not receive. Three further "
        "conditions isolate individual claims: a graph condition establishing "
        "whether the comorbidity graph outperforms its absence, an "
        "unconstrained-trait condition separating the contribution of the "
        "monotone structure from the mere reduction in free parameters, and a "
        "down-weighted weak-supervision condition establishing whether the "
        "noise channel outperforms a uniform reduction of the weak-label "
        f"weight. The results are reported in Table 6 and Figure 3. The step "
        f"contributing the largest measured change is {P}, and the step "
        f"contributing the smallest is {P}.")
    caption(doc, "Table 6. Component-wise ablation ladder and isolating "
                 "controls, reported as mean \u00b1 standard deviation over "
                 "three seeds.")
    table(doc, [
        ["Configuration", "Components enabled", "Symptom macro-AUROC",
         "Severity QWK", "Risk QWK", "Composite"],
        ["M0 flat", "none", PM, PM, PM, PM],
        ["M1 bottleneck", "concept bottleneck", PM, PM, PM, PM],
        ["M2 monotone", "+ monotone ordinal head", PM, PM, PM, PM],
        ["M3 weak", "+ weak supervision stream", PM, PM, PM, PM],
        ["Proposed", "+ per-symptom noise channel", PM, PM, PM, PM],
        ["C1 graph control", "+ comorbidity graph", PM, PM, PM, PM],
        ["C2 unconstrained trait", "monotonicity removed", PM, PM, PM, PM],
        ["C3 weak down-weighted", "channel replaced by uniform weight", PM, PM,
         PM, PM],
    ], widths=[1.5, 1.6, 1.1, 0.85, 0.75, 0.85])
    figure(doc, os.path.join(FIGS, "figure3_ablation.png"), 3,
           "Placeholder. Component-wise ablation ladder for the symptom and "
           "suicide-risk tasks, showing mean and standard deviation over three "
           "seeds.")

    heading(doc, "4.3 Statistical Significance and Measurement Limits")
    para(doc,
        "Because the evaluation partitions differ in size by more than an order "
        "of magnitude between tasks, every reported difference is accompanied "
        "by a paired bootstrap test and every metric by a bootstrap confidence "
        "interval over resampled test examples. The paired procedure applies "
        "the same resampled indices to both models under comparison, so that "
        "the test observes differences in prediction rather than differences in "
        "which examples were drawn, and a difference is reported as established "
        "only if it attains significance on every seed and preserves its sign "
        "across all three. The multiple-comparison burden is stated explicitly: "
        "with five metrics across the conditions evaluated, approximately one "
        "false positive is expected at the conventional threshold, so a single "
        "significant result is reported as consistent with chance rather than "
        f"as evidence. The outcomes are reported in Tables 7 and 8. Of the "
        f"comparisons performed, {P} attain significance with a consistent "
        f"sign. The suicide-risk task warrants particular caution: its test "
        f"partition comprises 74 instances across five ordinal levels, and the "
        f"resulting ninety-five per cent confidence interval for quadratic "
        f"weighted kappa has a measured width of {P}, which exceeds any "
        f"architectural difference observed in this study. Differences on that "
        f"task are therefore not interpretable as architectural effects, and "
        f"this is a property of the available annotation rather than a "
        f"deficiency of any model evaluated.")
    caption(doc, "Table 7. Per-task bootstrap confidence intervals over "
                 "resampled test examples, with the seed standard deviation "
                 "reported separately.")
    table(doc, [
        ["Task", "Test instances", "Mean", "95% CI lower", "95% CI upper",
         "CI width", "Seed SD"],
        ["Symptom macro-F1", "1,956", P, P, P, P, P],
        ["Symptom macro-AUROC", "1,956", P, P, P, P, P],
        ["Stress AUROC", "\u2248700", P, P, P, P, P],
        ["Severity QWK", "541", P, P, P, P, P],
        ["Risk QWK", "74", P, P, P, P, P],
    ], widths=[1.5, 0.95, 0.65, 0.85, 0.85, 0.7, 0.65])
    caption(doc, "Table 8. Paired bootstrap comparison of the proposed "
                 "framework against each competing condition, over 1,000 "
                 "resamples of the test split.")
    table(doc, [
        ["Comparison", "Metric", "Mean difference", "Maximum p",
         "Sign consistent", "Established"],
        ["Proposed vs MentalRoBERTa", "Symptom macro-AUROC", P, P, "\u2013",
         "\u2013"],
        ["Proposed vs ensemble, after [11]", "Symptom macro-AUROC", P, P,
         "\u2013", "\u2013"],
        ["Proposed vs post-graph, after [18]", "Symptom macro-AUROC", P, P,
         "\u2013", "\u2013"],
        ["Proposed vs prompt ensemble, after [21]", "Symptom macro-AUROC", P, P,
         "\u2013", "\u2013"],
        ["M1 \u2192 M2 (monotone head)", "Severity QWK", P, P, "\u2013",
         "\u2013"],
        ["M2 \u2192 M3 (weak supervision)", "Symptom macro-AUROC", P, P,
         "\u2013", "\u2013"],
        ["M3 \u2192 Proposed (noise channel)", "Symptom macro-F1", P, P,
         "\u2013", "\u2013"],
    ], widths=[1.85, 1.3, 0.95, 0.65, 0.8, 0.7])

    heading(doc, "4.4 Interpretability of the Learned Representation")
    para(doc,
        "Because the framework predicts graded outcomes through non-negative "
        "item weights over named clinical symptoms, those weights are directly "
        "readable as item loadings and can be compared against the equal "
        "weighting that the PHQ-9 prescribes. A departure from uniform "
        "weighting constitutes a concrete and falsifiable clinical claim about "
        "which symptoms carry disproportionate information about graded "
        "severity in this population, and is therefore reported with the "
        "caveat that the weights are estimated from social-media text rather "
        "than from clinical interview. The learned loadings are shown in Figure "
        f"4. The symptom receiving the largest relative weight is {P} and the "
        f"smallest is {P}; the dispersion of the loadings around uniform "
        f"weighting is {P}. A dispersion close to zero would indicate that the "
        f"head did not depart from its initialisation, in which case the "
        f"interpretability claim is not supported by the evidence and should be "
        f"withdrawn rather than reported, and the accompanying unconstrained-"
        f"trait control in Table 6 then determines whether any observed benefit "
        f"derived from the monotone structure or merely from the reduction in "
        f"free parameters.")
    figure(doc, os.path.join(FIGS, "figure4_item_loadings.png"), 4,
           "Placeholder. Learned item loadings of the monotone ordinal head, "
           "expressed relative to the equal item weighting prescribed by the "
           "PHQ-9, coloured by source instrument.")

    heading(doc, "4.5 Training Dynamics and Overfitting Control")
    para(doc,
        "The ordinal partitions are small enough that memorisation is the "
        "primary failure mode rather than underfitting, and the training "
        "protocol is designed accordingly. Three controls act together: the "
        "epoch budget is bounded and early stopping is applied on validation "
        "loss, the checkpoint is selected at minimum validation loss rather "
        "than by an aggregate score that one task can dominate, and the "
        "monotone ordinal head reduces the effective capacity available to the "
        "scarce tasks by construction. The effect is quantified by the ratio "
        "between the validation loss at the selected checkpoint and the minimum "
        "validation loss attained during the run, together with the training "
        "quadratic weighted kappa reached on the suicide-risk partition, which "
        "approaches unity when a run has memorised that partition. The "
        f"dynamics are shown in Figure 5. Across all runs, {P} of the "
        f"configurations exhibit a training risk kappa above the memorisation "
        f"threshold, and the mean ratio of selected to minimum validation loss "
        f"is {P}.")
    figure(doc, os.path.join(FIGS, "figure5_training_dynamics.png"), 5,
           "Placeholder. Training and validation loss, and the divergence "
           "between training and validation performance on the suicide-risk "
           "task, illustrating the effect of the overfitting controls.")

    heading(doc, "4.6 Discussion and Limitations")
    para(doc,
        "Several limitations bound the interpretation of these results and are "
        "stated here rather than left to the reader to infer. First, eleven of "
        "the twenty-two symptoms, predominantly the GAD-7 anxiety items, have "
        "no gold annotation in any constituent corpus, so the symptom metrics "
        "reported throughout are computed over the eleven gold-supervised "
        "symptoms only. The direct effect of weak supervision on the symptoms "
        "it actually supervises is therefore not measurable with the present "
        "annotation, and can appear only indirectly through transfer; closing "
        "this gap requires an in-house annotation effort rather than further "
        "modelling. Second, and for the same reason, the per-symptom noise "
        "channel is identifiable for eleven symptoms and pooled within "
        "instrument for the remainder, which is an assumption that the present "
        "data cannot verify. Third, the comparison models are re-implementations "
        "of published architectures rather than the published systems, "
        "constructed from the descriptions available, and their relative "
        "ordering here should not be read as a ranking of the original works. "
        "Fourth, the symptom-disorder prior is authored from diagnostic "
        "criteria by the present authors and has not been reviewed by a "
        "clinician, and the disorder-disorder edges derive from subreddit "
        "co-membership, which is a weak proxy for diagnosed comorbidity. Fifth, "
        "the weak-supervision conditions observe approximately fifty thousand "
        "additional labelled instances that the baselines and comparison models "
        "do not, so a portion of any margin attributable to those conditions "
        "reflects additional supervision rather than architecture, and a "
        "matched-data comparison is required before the architectural claim is "
        "made. Finally, the suicide-risk partition is too small to resolve "
        "architectural differences, as quantified in Section 4.3. Taken "
        "together, these limitations indicate that the contributions of this "
        "work which are best supported by the present evidence are the "
        "symptom-mediated architecture, the leakage-controlled multi-corpus "
        "protocol, the measurement of the labelling model's per-symptom bias, "
        "and the significance-audited reporting, rather than any single "
        "improvement in aggregate accuracy.")

    # ------------------------------------------------------------- conclusion
    heading(doc, "5. Conclusion")
    para(doc,
        "This research introduced SymptoGraph-LLM, a symptom-grounded and "
        "comorbidity-aware deep learning framework for interpretable "
        "mental-health assessment from social media text. The framework "
        "addresses five shortcomings of existing approaches: it replaces "
        "direct disorder classification with an obligatory clinical concept "
        "bottleneck whose symptom queries are initialised from the clinical "
        "definitions themselves; it propagates evidence over a "
        "symptom-disorder graph combining a diagnostic prior with a learned, "
        "sparsity-penalised residual, so that comorbidity is modelled rather "
        "than diagnosed; it estimates graded severity and suicide risk through "
        "an instrument-grounded monotone ordinal head that reproduces the "
        "additive scoring rule of the PHQ-9 and reduces the data-poor risk task "
        "to four threshold parameters; it corrects large language model weak "
        "supervision through a per-symptom noise channel estimated against gold "
        "labels, rather than treating model-generated labels as certain; and it "
        "evaluates all of this under a leakage-controlled protocol that "
        "collapses posts appearing in several corpora to a single identifier, "
        "partitions Reddit authors disjointly, and reports paired bootstrap "
        "tests alongside per-task confidence intervals. Validated on 19,762 "
        "gold-labelled posts harmonised from seven public corpora together with "
        "49,982 calibrated weak labels, and compared against three baselines, "
        "three architectures re-implemented from the 2024 to 2026 literature, "
        "five component-wise ablations and three isolating controls under one "
        f"identical protocol, the framework attains {P} symptom macro-AUROC and "
        f"{P} symptom macro-F1, with {P} of the component-wise differences "
        f"distinguishable from seed and sampling variability. The study also "
        "establishes two measurement results of independent interest: that the "
        "labelling model's reliability varies from a true-positive rate of "
        "0.252 to 0.944 across symptoms, which no single confidence temperature "
        "can correct, and that the available suicide-risk annotation supports a "
        "confidence interval wider than any architectural difference it could "
        "resolve. Future work will construct an in-house anxiety gold standard "
        "to make the weak-only symptoms measurable, submit the symptom-disorder "
        "prior for clinician review together with a sensitivity analysis over "
        "perturbed priors, extend the framework with a symptom-grounded "
        "rationale generator evaluated for faithfulness, and conduct the "
        "matched-data comparison required to attribute the remaining margin to "
        "architecture rather than to supervision.")

    # ------------------------------------------------------------- references
    heading(doc, "References")
    for ref in load_references():
        para(doc, ref)

    os.makedirs(DOCS, exist_ok=True)
    doc.save(OUT)
    print(f"wrote {OUT}")


def write_placeholder_index() -> None:
    path = os.path.join(DOCS, "PLACEHOLDERS.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("""# Placeholder index for SymptoGraph-LLM_Paper.docx

Every performance number in the paper is written as `X.XXX` (or
`X.XXX ± X.XXX`). Search the document for `X.XXX` to find them all. Dataset
counts, corpus-overlap measurements and the labelling-model noise channel are
**measured quantities and are already real** — do not overwrite them.

Regenerate the document after editing the builder:

```
python scripts/build_paper.py
```

## Tables

| Table | Fills from | Produced by |
|---|---|---|
| 1. Literature comparison | no numbers | — |
| 2. Corpus overlap | already real | `src/symptograph/preprocess.py` docstring |
| 3. Dataset partitioning | already real | `processed/manifest.json` |
| 4. Shared configuration | already real | `TrainConfig` / `ProposedConfig` |
| **5. Baselines and comparison models** | `comparison/comparison_table.csv` and `comparison/test_comparison.csv` | notebook 09 §7 |
| **6. Ablation ladder and controls** | `proposed_v2_e8/comparison_ladder/` aggregates | notebook 08 §9, §11 |
| **7. Per-task confidence intervals** | `proposed_v2_e8/comparison_ladder/bootstrap_cis.csv` | notebook 08 §9 |
| **8. Paired bootstrap** | `comparison/paired_vs_proposed.csv` and `proposed_v2_e8/comparison_ladder/paired_bootstrap.csv` | notebook 08 §10, notebook 09 §8 |

## Figures

| Figure | Status | Replace with |
|---|---|---|
| 1. Architecture | **final** | — |
| 2. Against baselines | placeholder | `comparison/comparison_vs_*/comparison.png` |
| 3. Ablation ladder | placeholder | `proposed_v2_e8/comparison_ladder/*.png` |
| 4. Item loadings | placeholder | `proposed_v2_e8/item_loadings_*.png` |
| 5. Training dynamics | placeholder | `proposed_v2_e8/M4_denoised/seed_0/plots/` |

## Prose placeholders

Section 4 paragraphs carry `X.XXX` inline for: the headline symptom
macro-AUROC and its margin over the strongest comparison model (4.1); the
largest and smallest contributing ablation step (4.2); the count of established
differences and the suicide-risk CI width (4.3); the largest and smallest item
loading and the dispersion around uniform weighting (4.4); the count of runs
above the memorisation threshold and the mean loss ratio (4.5). The abstract and
conclusion repeat the headline figures.

## Claims that must be checked before submission, not filled in

1. **Section 4.4** — if the loading dispersion is near zero the head never left
   its initialisation, and the interpretability claim must be **withdrawn**
   rather than reported. The paragraph is written to accommodate either outcome.
2. **Section 4.1** — the comparison models are re-implementations, not
   reproductions. That sentence must stay.
3. **Section 4.6** — the matched-data caveat must stay until a run with weak
   labels enabled for the baselines exists.
""")
    print(f"wrote {path}")


def main() -> None:
    os.makedirs(FIGS, exist_ok=True)
    fig1_architecture(os.path.join(FIGS, "figure1_architecture.png"))
    fig2_vs_baselines(os.path.join(FIGS, "figure2_vs_baselines.png"))
    fig3_ablation(os.path.join(FIGS, "figure3_ablation.png"))
    fig4_loadings(os.path.join(FIGS, "figure4_item_loadings.png"))
    fig5_training(os.path.join(FIGS, "figure5_training_dynamics.png"))
    print(f"figures -> {FIGS}")
    build_document()
    write_placeholder_index()


if __name__ == "__main__":
    main()
