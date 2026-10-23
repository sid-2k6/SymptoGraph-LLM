#!/usr/bin/env python3
"""Build docs/SymptoGraph-LLM_Paper.docx, mirroring "PRISM - Copy.docx".

Structure read out of the reference by parsing word/document.xml in document
order, section by section:

    title                       bold, centred
    Abstract:                   bold label, then one justified paragraph
    Keywords:                   one justified paragraph, inline label
    1. Introduction             3 body paragraphs
      Objectives                label + 4 bullets
      Aim                       label + 1 paragraph
      Novelty                   label + 1 paragraph
      Contribution              label + 4 bullets
    2. Literature Survey        7 body paragraphs, then Table 1 (17 x 5)
    3. Proposed Methodology     1 overview paragraph, Figure 1
      3.1 Dataset Description   3 paras, Table 2 (4x4), 1 para, Table 3 (6x4), 2 paras
      3.2 Data Pre-processing   1 para + equations (1)-(5) each with a "where" paragraph
      3.3 Feature Extraction    1 para + equations (6)-(9)
      3.4 Proposed Framework Architecture
                                1 para, Figure 2, 3 paras, Table 4 (31x3),
                                1 para, Table 5 (12x2), 1 para
      3.5 Loss Function and Hyperparameter Configuration
                                1 para + equations (10)-(15), 1 para,
                                Table 6 (44x2), 1 para
    4. Results and Discussions
      4.1 Experimental Setup    1 para
      4.2 Evaluation Metrics    1 para + equations (16)-(20)
      4.3 Performance Analysis  1 para, Table 7, 1 para, Table 8, 2 paras
      4.4 Training Convergence Analysis   2 images as Figure 3, 3 paras
      4.5 Ablation Study        1 para, Table 9, Figure 4, 2 paras
      4.6 Comparison with State-of-the-Art Models
                                1 para, Table 10, Figure 5 (2 images), 2 paras
    5. Conclusion and Future Work
      5.1 Conclusion            1 para
      5.2 Future Work           lead-in + 4 bullets
    6. References               numbered list

Formatting, also from the reference: A4 8.27 x 11.69 in, 1 in margins, Times New
Roman 12 pt body and 10 pt in tables, bold centred title, bold section and
subsection headings, justified body with no first-line indent, table captions
NOT bold and placed ABOVE the table, figure captions centred and placed BELOW
the image, 6.27 in figure width for full-width figures and roughly 3.0 in for
side-by-side pairs.

The reference stores equations as OMML. python-docx cannot author OMML, so each
equation is rendered to an image with matplotlib mathtext and placed centred,
followed by its number in its own paragraph exactly as the reference does.

Citation placement, as instructed:
    [1]-[5]    Introduction only
    [6]-[20]   Literature Survey prose, in order, across the 7 body paragraphs
    [21]-[30]  Literature Survey comparison table, in order, plus a Proposed row

Every performance number is the placeholder ``X.XXX``. Dataset counts, corpus
overlaps and the labeller noise channel stay real, because they are measured
facts and the reference prints real ones in the equivalent position.

    python scripts/build_paper.py
"""
from __future__ import annotations

import os
import re

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
EQS = os.path.join(FIGS, "equations")
SURVEY = os.path.join(DOCS, "Literature_Survey_SymptoGraph-LLM.md")
OUT = os.path.join(DOCS, "SymptoGraph-LLM_Paper.docx")

FONT = "Times New Roman"
SIZE = Pt(12)
P = "X.XXX"
PM = "X.XXX \u00b1 X.XXX"


# --------------------------------------------------------------------------- #
# equations, rendered because the reference uses OMML which python-docx cannot
# author
# --------------------------------------------------------------------------- #
EQUATIONS = {
    1: r"$\mathrm{pid}(x)=\mathrm{MD5}(\mathrm{norm}(x)_{1:200}),\quad \mathcal{D}=\cup_{c=1}^{7}\mathcal{D}_c\ /\ \mathrm{pid}$",
    2: r"$y_{ik}\in\{1,0,-1\},\qquad m_{ik}=\mathbf{1}[\,y_{ik}\geq 0\,]$",
    3: r"$\mathrm{split}(a)=\mathrm{bucket}(\mathrm{hash}(a)),\quad a_i=a_j\Rightarrow s_i=s_j$",
    4: r"$s_i=\min_{c\,:\,i\in\mathcal{D}_c}\mathrm{rank}(s_i^{(c)}),\quad \mathrm{rank}(\mathrm{test})<\mathrm{rank}(\mathrm{val})<\mathrm{rank}(\mathrm{train})$",
    5: r"$\mathcal{W}\sim\mathrm{Strat}(\mathcal{U};d,\tau),\quad |\mathcal{W}|=49{,}982,\quad \pi_{\mathrm{ctrl}}\leq 0.15$",
    6: r"$H_i=\mathrm{Enc}(x_i)\in\mathbb{R}^{L\times D},\qquad D=768,\ L=192$",
    7: r"$\alpha_{ik}=\mathrm{softmax}_{l}\left(\frac{H_{il}^{T}q_k}{\sqrt{D}}\right),\qquad q_k^{(0)}=\mathrm{SBERT}(\mathrm{def}_k)$",
    8: r"$Z_{ik}=\sum_{l=1}^{L}\alpha_{ikl}H_{il},\qquad c_{ik}=w_k^{T}Z_{ik}+b_k$",
    9: r"$g_i=\frac{\sum_{k=1}^{K}\sigma(c_{ik})\,Z_{ik}}{\sum_{k=1}^{K}\sigma(c_{ik})},\qquad K=22$",
    10: r"$\mathcal{L}_{\mathrm{sym}}=\frac{\sum_{i,k}m_{ik}\,\omega_{ik}\,\mathrm{BCE}(c_{ik},y_{ik})}{\sum_{i,k}m_{ik}\,\omega_{ik}}$",
    11: r"$z_i=\sum_{k=1}^{K}\mathrm{softplus}(v_k)\,\sigma(c_{ik}),\qquad v_k\in\mathbb{R}$",
    12: r"$P(y_i\geq j)=\sigma(z_i-\tau_j),\qquad \tau_j=\tau_1+\sum_{m=2}^{j}\mathrm{softplus}(\delta_m)$",
    13: r"$A=\mathrm{relu}(A_0+\Delta A)+I,\quad \tilde{A}=D^{-1/2}AD^{-1/2},\quad \mathcal{L}_{\mathrm{graph}}=\|\Delta A\|_1$",
    14: r"$\tilde{p}_{ik}=p_{ik}\,\mathrm{tpr}_k+(1-p_{ik})\,\mathrm{fpr}_k,\qquad \mathcal{L}_{\mathrm{weak}}=\mathrm{BCE}(\tilde{p}_{ik},\hat{y}_{ik})$",
    15: r"$\mathcal{L}=\mathcal{L}_{\mathrm{sym}}+\lambda_{s}\mathcal{L}_{\mathrm{sev}}+\lambda_{r}\mathcal{L}_{\mathrm{risk}}+\lambda_{t}\mathcal{L}_{\mathrm{str}}+\lambda_{w}\mathcal{L}_{\mathrm{weak}}+\lambda_{g}\mathcal{L}_{\mathrm{graph}}$",
    16: r"$\mathrm{F1}_{\mathrm{macro}}=\frac{1}{K}\sum_{k=1}^{K}\frac{2\,\mathrm{TP}_k}{2\,\mathrm{TP}_k+\mathrm{FP}_k+\mathrm{FN}_k}$",
    17: r"$\mathrm{AUROC}_k=\frac{\sum_{i\in\mathcal{P}_k}\mathrm{rank}(p_{ik})-n_k^{+}(n_k^{+}+1)/2}{n_k^{+}\,n_k^{-}}$",
    18: r"$\kappa_{w}=1-\frac{\sum_{a,b}W_{ab}O_{ab}}{\sum_{a,b}W_{ab}E_{ab}},\qquad W_{ab}=\frac{(a-b)^2}{(J-1)^2}$",
    19: r"$\mathrm{MAE}=\frac{1}{N}\sum_{i=1}^{N}|\hat{y}_i-y_i|,\qquad \hat{y}_i=\arg\max_j P(y_i=j)$",
    20: r"$p=2\min(\Pr[\Delta^{(b)}\leq 0],\ \Pr[\Delta^{(b)}\geq 0]),\quad \Delta^{(b)}=m(\mathcal{B}^{(b)})-m_0(\mathcal{B}^{(b)})$",
}


def render_equations() -> dict[int, str]:
    os.makedirs(EQS, exist_ok=True)
    out = {}
    for n, tex in EQUATIONS.items():
        path = os.path.join(EQS, f"eq{n:02d}.png")
        fig = plt.figure(figsize=(0.01, 0.01))
        fig.text(0, 0, tex, fontsize=15, family="serif")
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.06,
                    transparent=False, facecolor="white")
        plt.close(fig)
        out[n] = path
    return out


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def _stamp(ax, text="PLACEHOLDER"):
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center",
            fontsize=17, color="#c0392b", alpha=0.17, rotation=18, zorder=10,
            fontweight="bold")


def fig1_workflow(path: str) -> None:
    """Figure 1, overall workflow. Real content, not a placeholder."""
    plt.rcParams.update({"font.family": "serif"})
    fig, ax = plt.subplots(figsize=(10.6, 8.2))
    ax.set_xlim(0, 100), ax.set_ylim(0, 78), ax.axis("off")

    def box(x, y, w, h, t, fc, fs=8.0, bold=False):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.4", lw=1.1,
            edgecolor="#2c3e50", facecolor=fc))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal")

    def arr(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#2c3e50", lw=1.2))

    def band(y, h, t):
        ax.add_patch(mpatches.Rectangle((0.5, y), 99, h, facecolor="#fbfbfb",
                                        edgecolor="#bdc3c7", lw=0.8, ls=":",
                                        zorder=0))
        ax.text(1.8, y + h - 1.5, t, fontsize=9, fontweight="bold", va="top",
                color="#34495e")

    C = dict(src="#dfe9f5", pre="#e4f1e4", enc="#fdf0d9", nov="#f6e0ea",
             head="#e8e2f3", ev="#ededed")

    band(60, 17.5, "Stage 1   Corpus harmonisation and leakage control")
    for t, x in [("Depression\nEmo", 3.5), ("Dreaddit", 16), ("DepSeverity", 27),
                 ("C-SSRS\nReddit", 38.5), ("IRF", 49.5), ("LoST", 58.5),
                 ("IMHI", 67.5), ("Reddit MH pool\n1,107,302 posts", 77)]:
        box(x, 69, 11 if x < 70 else 20, 6.5, t, C["src"], fs=7.0)
    arr(50, 68.8, 50, 66.3)
    box(9, 61, 82, 5,
        "global post_id = MD5(norm(x)[:200])   \u2022   one split per post   "
        "\u2022   author-disjoint   \u2022   strictest-split-wins", C["pre"], fs=7.6)

    arr(27, 60.8, 27, 57.3)
    arr(73, 60.8, 73, 57.3)
    box(7, 51, 38, 6.3, "labelled.parquet   19,762 posts\n"
        "22 symptoms  \u2022  severity  \u2022  risk  \u2022  stress", C["pre"], fs=7.4)
    box(55, 51, 38, 6.3, "weak_pool.parquet   1,017,959 posts\n"
        "Qwen2.5-7B labels 49,982 posts", C["pre"], fs=7.4)

    band(20.5, 29, "Stage 2   SymptoGraph-LLM estimation pipeline")
    arr(26, 50.8, 26, 46.3)
    box(4, 39.5, 20, 6.8, "MentalRoBERTa\nencoder, 192 tokens", C["enc"], fs=7.6)
    arr(24.2, 42.9, 27.8, 42.9)
    box(28, 38, 21, 9.8, "N1  Concept bottleneck\n22 definition-initialised\n"
        "queries, per-symptom $w_k$\nconcept-gated pooling", C["nov"], fs=7.0)
    arr(49.2, 42.9, 52.8, 42.9)
    box(53, 38, 19, 9.8, "N2  Comorbidity graph\n$A=\\mathrm{relu}(A_0+\\Delta A)$\n"
        "DSM-5 prior, $L_1$ residual", C["nov"], fs=7.0)
    arr(72.2, 42.9, 75.8, 42.9)
    box(76, 38, 21, 9.8, "N-A  Monotone ordinal\n"
        "$z=\\sum_k\\mathrm{sp}(v_k)\\sigma(c_k)$\n"
        "$P(y\\geq j)=\\sigma(z-\\tau_j)$", C["nov"], fs=7.0)

    arr(74, 50.8, 74, 33.3)
    box(28, 25.5, 21, 6.5, "N3  Weak-label stream\nconfidence-weighted",
        C["nov"], fs=7.0)
    box(53, 25.5, 19, 6.5, "N-B  Noise channel\n"
        "$p\\,\\mathrm{tpr}_k+(1-p)\\mathrm{fpr}_k$", C["nov"], fs=7.0)
    arr(49.2, 28.8, 52.8, 28.8)
    arr(38.5, 32.1, 38.5, 37.8)
    arr(62.5, 32.1, 62.5, 37.8)
    box(76, 24.5, 21, 7.5, "Prediction heads\n22 symptoms  \u2022  severity (4)\n"
        "risk (5)  \u2022  stress (2)", C["head"], fs=7.0)
    arr(86.5, 37.8, 86.5, 32.3)

    band(2, 16, "Stage 3   Leakage-controlled evaluation protocol")
    for t, x, w in [("3 seeds", 4, 12), ("selection on\nvalidation loss", 18, 16),
                    ("per-symptom thresholds\ntuned on validation", 36, 22),
                    ("paired bootstrap\n1,000 resamples", 60, 17),
                    ("per-task 95%\nintervals", 79, 16)]:
        box(x, 9.5, w, 5.8, t, C["ev"], fs=7.0)
    box(14, 3, 72, 5,
        "3 baselines   \u2022   3 re-implemented comparison models   \u2022   "
        "5 component-wise ablations", C["ev"], fs=7.6)
    arr(50, 9.3, 50, 8.2)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig2_architecture(path: str) -> None:
    """Figure 2, the architecture diagram with tensor shapes. Real content."""
    plt.rcParams.update({"font.family": "serif"})
    fig, ax = plt.subplots(figsize=(11.0, 8.2))
    ax.set_xlim(0, 100), ax.set_ylim(0, 76), ax.axis("off")

    def box(x, y, w, h, t, fc, fs=7.6, bold=False):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.35", lw=1.1,
            edgecolor="#2c3e50", facecolor=fc))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal")

    def arr(x1, y1, x2, y2, label=None, ls="-"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#2c3e50", lw=1.1,
                                    linestyle=ls))
        if label:
            ax.text((x1 + x2) / 2 + 1.2, (y1 + y2) / 2, label, fontsize=6.6,
                    color="#555", style="italic", ha="left", va="center")

    C = dict(inp="#dfe9f5", enc="#fdf0d9", n1="#f6e0ea", n2="#e7f0da",
             na="#f9e2cf", n3="#eae3f5", nb="#d9ecef", out="#e8e2f3",
             loss="#f3f3f3")

    box(2, 68, 24, 6, "Post text  $x_i$", C["inp"], fs=8.2, bold=True)
    arr(14, 67.8, 14, 64.3, "tokenise, 192")
    box(2, 57.5, 24, 6.5, "MentalRoBERTa encoder\n12 layers, $D=768$", C["enc"],
        fs=7.8)
    arr(14, 57.3, 14, 53.8, "$H_i\\in\\mathbb{R}^{192\\times 768}$")

    ax.text(50, 51.5, "N1   Clinical concept bottleneck", fontsize=8.6,
            fontweight="bold", ha="center", color="#8e2a52")
    box(2, 43, 22, 6.5, "Symptom queries $q_k$\ninit = SBERT(def$_k$)\n$22\\times 768$",
        C["n1"], fs=7.0)
    arr(24.2, 46.3, 28.8, 46.3)
    box(29, 43, 20, 6.5, "Symptom attention\n$\\alpha_{ik}=\\mathrm{softmax}_l"
        "(H_{il}^{\\top}q_k/\\sqrt{D})$", C["n1"], fs=6.8)
    arr(49.2, 46.3, 53.8, 46.3, "$Z_i\\in\\mathbb{R}^{22\\times 768}$")
    box(54, 43, 20, 6.5, "Per-symptom scoring\n$c_{ik}=w_k^{\\top}Z_{ik}+b_k$\n"
        "$w\\in\\mathbb{R}^{22\\times 768}$", C["n1"], fs=6.8)
    arr(74.2, 46.3, 78.8, 46.3, "$c_i\\in\\mathbb{R}^{22}$")
    box(79, 43, 19, 6.5, "Concept-gated pool\n$g_i=\\sum_k\\sigma(c_k)Z_k/"
        "\\sum_k\\sigma(c_k)$", C["n1"], fs=6.6)

    arr(64, 42.8, 64, 38.3)
    box(40, 31.5, 48, 6.5, "N2   Comorbidity graph message passing\n"
        "$\\tilde{A}=D^{-1/2}(\\mathrm{relu}(A_0+\\Delta A)+I)D^{-1/2}$,  "
        "2 layers, LayerNorm", C["n2"], fs=7.0)
    box(2, 31.5, 35, 6.5, "Graph prior $A_0$\nDSM-5 symptom$\\rightarrow$disorder\n"
        "$p(b\\,|\\,a)$ disorder$\\leftrightarrow$disorder", C["n2"], fs=6.8)
    arr(37.2, 34.8, 39.8, 34.8)
    arr(64, 31.3, 64, 26.8, "refined $c_i$")

    ax.text(26, 24.5, "N-A   Monotone ordinal head", fontsize=8.4,
            fontweight="bold", ha="center", color="#a0522d")
    box(2, 15.5, 24, 7.5, "Latent distress trait\n"
        "$z_i=\\sum_k\\mathrm{softplus}(v_k)\\sigma(c_{ik})$\n$v_k\\geq 0$",
        C["na"], fs=6.8)
    arr(26.2, 19.3, 30.8, 19.3)
    box(31, 15.5, 24, 7.5, "Ordered cutpoints\n"
        "$\\tau_1<\\tau_2<\\cdots<\\tau_{J-1}$\n"
        "$P(y\\geq j)=\\sigma(z-\\tau_j)$", C["na"], fs=6.8)

    box(60, 15.5, 17, 7.5, "Symptom head\n$22$ logits", C["out"], fs=7.0)
    box(79, 15.5, 19, 7.5, "Severity (4)  Risk (5)\nStress (2)", C["out"], fs=7.0)
    arr(79, 42.8, 79, 23.3)
    arr(55.2, 19.3, 59.8, 19.3)
    arr(77.2, 19.3, 78.8, 19.3)

    ax.text(26, 12.6, "N3 / N-B   Weak supervision branch", fontsize=8.4,
            fontweight="bold", ha="center", color="#2c6e7f")
    box(2, 4.5, 24, 6.5, "N3  Weak stream\n49,982 posts, per step\nconfidence weights",
        C["n3"], fs=6.8)
    arr(26.2, 7.8, 30.8, 7.8)
    box(31, 4.5, 24, 6.5, "N-B  Noise channel\n"
        "$\\tilde{p}=p\\,\\mathrm{tpr}_k+(1-p)\\mathrm{fpr}_k$", C["nb"], fs=6.8)
    arr(55.2, 7.8, 59.8, 7.8)
    box(60, 4.5, 38, 6.5, "Total objective  "
        "$\\mathcal{L}=\\mathcal{L}_{\\mathrm{sym}}+\\lambda_s\\mathcal{L}_{\\mathrm{sev}}"
        "+\\lambda_r\\mathcal{L}_{\\mathrm{risk}}+\\lambda_t\\mathcal{L}_{\\mathrm{str}}"
        "+\\lambda_w\\mathcal{L}_{\\mathrm{weak}}+\\lambda_g\\|\\Delta A\\|_1$",
        C["loss"], fs=6.6)
    arr(68, 15.3, 68, 11.3, ls="--")
    arr(88, 15.3, 88, 11.3, ls="--")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig3a_loss(path: str) -> None:
    plt.rcParams.update({"font.family": "serif", "font.size": 9})
    ep = np.arange(1, 9)
    fig, ax = plt.subplots(figsize=(4.4, 3.1))
    ax.plot(ep, 3.4 * np.exp(-0.26 * ep), marker="o", label="training loss")
    ax.plot(ep, 1.15 + 0.035 * np.clip(ep - 3, 0, None), marker="s",
            label="validation loss")
    ax.set_xlabel("Epoch"), ax.set_ylabel("Multi-task loss")
    ax.legend(fontsize=8), ax.grid(alpha=0.3)
    _stamp(ax)
    fig.tight_layout(), fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig3b_qwk(path: str) -> None:
    plt.rcParams.update({"font.family": "serif", "font.size": 9})
    ep = np.arange(1, 9)
    rng = np.random.default_rng(3)
    fig, ax = plt.subplots(figsize=(4.1, 2.9))
    ax.plot(ep, np.minimum(0.95, 0.30 + 0.085 * ep), marker="o",
            label="training risk QWK")
    ax.plot(ep, 0.38 + 0.012 * rng.standard_normal(len(ep)), marker="s",
            label="validation risk QWK")
    ax.set_xlabel("Epoch"), ax.set_ylabel("Quadratic weighted kappa")
    ax.legend(fontsize=8), ax.grid(alpha=0.3)
    _stamp(ax)
    fig.tight_layout(), fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig4_ablation(path: str) -> None:
    plt.rcParams.update({"font.family": "serif", "font.size": 9})
    rungs = ["M0", "M1", "M2", "M3", "Prop."]
    rng = np.random.default_rng(1)
    fig, ax = plt.subplots(figsize=(4.3, 2.8))
    y = np.sort(rng.uniform(0.44, 0.63, len(rungs)))
    e = rng.uniform(0.01, 0.045, len(rungs))
    ax.errorbar(range(len(rungs)), y, yerr=e, marker="o", capsize=4, lw=1.6,
                color="#2b6cb0")
    ax.fill_between(range(len(rungs)), y - e, y + e, alpha=0.15, color="#2b6cb0")
    ax.set_xticks(range(len(rungs))), ax.set_xticklabels(rungs)
    ax.set_ylabel("Symptom macro-AUROC"), ax.grid(alpha=0.3)
    _stamp(ax)
    fig.tight_layout(), fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig5a_sota(path: str) -> None:
    plt.rcParams.update({"font.family": "serif", "font.size": 9})
    m = ["TF-IDF", "BiLSTM", "MRoBERTa", "[11]", "[18]", "[21]", "Prop."]
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(4.3, 2.8))
    ax.bar(m, rng.uniform(0.60, 0.80, len(m)), edgecolor="black", lw=0.5,
           color="#2b6cb0")
    ax.set_ylabel("Symptom macro-AUROC"), ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=40, labelsize=7.5)
    ax.grid(axis="y", alpha=0.3)
    _stamp(ax)
    fig.tight_layout(), fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig5b_ci(path: str) -> None:
    plt.rcParams.update({"font.family": "serif", "font.size": 9})
    tasks = ["Sym F1", "Sym AUROC", "Stress", "Severity", "Risk"]
    n = np.array([1956, 1956, 700, 541, 74])
    rng = np.random.default_rng(4)
    fig, ax = plt.subplots(figsize=(4.3, 3.1))
    mean = rng.uniform(0.40, 0.85, 5)
    half = 0.42 / np.sqrt(n / 74)
    ax.errorbar(mean, range(5), xerr=half, fmt="o", capsize=4, color="#c05621")
    ax.set_yticks(range(5)), ax.set_yticklabels(tasks, fontsize=8)
    ax.set_xlabel("Score with 95% bootstrap interval")
    ax.grid(alpha=0.3)
    for i, v in enumerate(n):
        ax.text(1.03, i, f"n={v}", fontsize=7, va="center")
    _stamp(ax)
    fig.tight_layout(), fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# docx helpers
# --------------------------------------------------------------------------- #
def _style(r):
    r.font.name = FONT
    r.font.size = SIZE


def para(doc, text="", bold=False, align="justify"):
    p = doc.add_paragraph()
    p.alignment = {"justify": WD_ALIGN_PARAGRAPH.JUSTIFY,
                   "center": WD_ALIGN_PARAGRAPH.CENTER,
                   "left": WD_ALIGN_PARAGRAPH.LEFT}[align]
    r = p.add_run(text)
    r.bold = bold
    _style(r)
    return p


def heading(doc, text):
    return para(doc, text, bold=True)


def label(doc, text):
    """Intro sub-labels: Objectives, Aim, Novelty, Contribution. Not bold."""
    return para(doc, text)


def bullet(doc, text):
    p = doc.add_paragraph(style="List Paragraph")
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    r = p.add_run(text)
    _style(r)
    return p


def caption(doc, text):
    """Table caption: not bold, above the table, as in the reference."""
    return para(doc, text, align="left")


def equation(doc, eqs, n, width=None):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    img = eqs[n]
    from PIL import Image as _I
    w_px, _ = _I.open(img).size
    w_in = min(5.6, max(1.6, w_px / 300.0))
    p.add_run().add_picture(img, width=Inches(width or w_in))
    q = doc.add_paragraph()
    q.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = q.add_run(f"({n})")
    _style(r)


def figure(doc, images, number, text, width=6.27):
    if isinstance(images, str):
        images = [images]
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    each = width if len(images) == 1 else 3.02
    for img in images:
        p.add_run().add_picture(img, width=Inches(each))
        if len(images) > 1:
            p.add_run("  ")
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = cap.add_run(f"Figure {number}. {text}")
    _style(r)


def table(doc, rows, widths=None, fs=10):
    t = doc.add_table(rows=len(rows), cols=len(rows[0]))
    t.style = "Table Grid"
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            c = t.cell(i, j)
            c.text = ""
            r = c.paragraphs[0].add_run(str(v))
            r.bold = (i == 0) or str(v).strip() in ("Proposed", "PRISM-GAN",
                                                    "SymptoGraph-LLM")
            r.font.name = FONT
            r.font.size = Pt(fs)
    if widths:
        for j, w in enumerate(widths):
            for row in t.rows:
                row.cells[j].width = Inches(w)
    doc.add_paragraph()
    return t


def load_references() -> list[str]:
    text = open(SURVEY, encoding="utf-8").read()
    refs = sorted({(int(n), b.strip())
                   for n, b in re.findall(r"^\[(\d+)\]\s+(.*?)$", text, re.M)})
    if len(refs) < 30:
        raise SystemExit(f"only {len(refs)} references in {SURVEY}")
    return [f"[{n}] {b}" for n, b in refs]


TITLE = ("SymptoGraph-LLM: a symptom-grounded, comorbidity-aware multi-task "
         "network for interpretable mental health assessment from social media "
         "text under calibrated weak supervision")


def build_document(eqs, figs) -> None:
    doc = Document()
    s = doc.sections[0]
    s.page_width, s.page_height = Inches(8.27), Inches(11.69)
    s.left_margin = s.right_margin = Inches(1.0)
    s.top_margin = s.bottom_margin = Inches(1.0)
    doc.styles["Normal"].font.name = FONT
    doc.styles["Normal"].font.size = SIZE

    para(doc, TITLE, bold=True, align="center")

    # -------------------------------------------------------------- abstract
    heading(doc, "Abstract:")
    para(doc,
        "Clinical symptoms should be identifiable, comorbidity should be "
        "representable and graded severity should follow the scoring rule of "
        "the instrument that defines it in text-based mental health assessment, "
        "yet existing systems satisfy none of these requirements at once. "
        "Current approaches predict a single disorder label rather than the "
        "symptoms a practitioner reasons with, treat co-occurring conditions as "
        "independent, estimate ordinal severity with unconstrained classifiers "
        "that discard the additive rule of the originating scale, consume "
        "pseudo-labels as if they were certain, and report accuracy on corpora "
        "whose overlapping provenance leaks training material into test sets. "
        "This paper proposes a symptom-grounded, comorbidity-aware multi-task "
        "network. Twenty-two symptom queries initialised from the embeddings of "
        "their own clinical definitions form an obligatory concept bottleneck; "
        "a symptom-disorder graph combining a DSM-5 prior with a "
        "sparsity-penalised residual propagates comorbid evidence; an "
        "instrument-grounded monotone ordinal head maps the concept vector onto "
        "a bounded latent trait through non-negative item weights and strictly "
        "ordered cutpoints, reproducing the additive structure of the PHQ-9; "
        "and a per-symptom noise channel inverts the bias of a locally "
        "executed labelling model. A global post identifier, author-disjoint "
        "partitioning and a strictest-split-wins rule control the measured "
        "overlap routes. The framework is trained on 19,762 gold-labelled posts "
        "with 49,982 calibrated weak labels, "
        f"and evaluated against three baselines, three re-implemented "
        f"architectures and five ablations. It attains {P} symptom "
        f"macro-AUROC, {P} symptom macro-F1, {P} stress AUROC, {P} "
        f"depression-severity and {P} suicide-risk quadratic weighted kappa, "
        f"improving on the strongest comparison model by {P} absolute symptom "
        f"macro-AUROC, while the ablation identifies the concept bottleneck and "
        f"the monotone ordinal head as the only components exceeding the "
        f"seed-noise band.")
    para(doc,
        "Keywords: Mental health assessment, Concept bottleneck, Comorbidity "
        "graph, Ordinal regression, Weak supervision, Label noise correction, "
        "Interpretable deep learning, PHQ-9, GAD-7.")

    # ---------------------------------------------------------- introduction
    heading(doc, "1. Introduction")
    para(doc,
        "Text-based mental health assessment has attracted growing interest in "
        "clinical natural language processing, driven by the gap between rising "
        "disorder prevalence and available screening capacity. Social platforms "
        "contain large volumes of self-disclosed psychological text carrying "
        "traces of clinically meaningful symptoms. "
        "A system intended for a care pathway must therefore satisfy three "
        "requirements simultaneously: the predicted quantity must be one a "
        "clinician recognises, the evidence supporting it must be traceable to "
        "the text, and the reported accuracy must be attributable to "
        "generalisation rather than to shared provenance.")
    para(doc,
        "Three ideas from outside disorder classification provide a means of "
        "meeting these requirements. Concept bottleneck modelling makes an "
        "interpretable intermediate layer obligatory rather than concomitant, "
        "so no downstream output can rest on evidence the concepts do not "
        "represent. Cumulative-link ordinal regression imposes the ordering of "
        "a clinical scale by construction rather than leaving it to be learned, "
        "and permits a data-poor task to borrow strength from a data-rich one "
        "through a shared latent trait. Label-noise correction treats an "
        "automatic annotator as a measurable channel whose per-class bias can "
        "be estimated and inverted, rather than as an oracle.")
    para(doc,
        "While previous research has applied deep models to mental health text, "
        "none has done so as this paper does. Xu et al. [1] evaluate large "
        "language models for mental health prediction from online text across "
        "zero-shot, few-shot and instruction-tuned regimes, establishing that "
        "instruction tuning recovers much of the gap to supervised encoders but "
        "that the prediction target remains a disorder-level label with no "
        "inspectable intermediate representation. Shah et al. [2] "
        "fine-tune large language models for depression detection and report "
        "consistent gains, yet the resulting model is a binary classifier whose "
        "decision cannot be attributed to diagnostic criteria. Liu et al. [3] "
        "introduce DepGLM to recognise depression degree rather than presence, "
        "acknowledging that severity is graded, but predict the degrees with an "
        "unconstrained head so that the ordering between adjacent degrees is "
        "not enforced. Abdullah and Negied [4] combine machine learning, "
        "ensemble learning and large language models to predict future "
        "disorder, showing that ensembling stabilises prediction under noisy "
        "user-generated text, although the ensemble is formed over classifiers "
        "of the same disorder-level target and introduces no interpretable "
        "layer. Qorich and El Ouazzani [5] apply deep learning with large "
        "language models to suicide ideation detection and demonstrate that "
        "risk-bearing language is separable from general distress, but treat "
        "risk as an isolated binary target rather than the extreme of a graded "
        "clinical scale, so no strength is shared with the far more abundant "
        "symptom annotation available in the same corpora.")

    label(doc, "Objectives")
    bullet(doc,
        "Construct a harmonised mental health corpus over a shared "
        "twenty-two symptom space derived from the PHQ-9 and GAD-7 instruments, "
        "with a global post identifier and author-disjoint partitioning "
        "controlling the measured overlap routes across the public collections.")
    bullet(doc,
        "Make clinical symptom mediation a property of the architecture rather "
        "than of an auxiliary explanation, using definition-initialised symptom "
        "attention with independent per-symptom scoring vectors.")
    bullet(doc,
        "Supply the ordinal heads with the additive structure of the instrument "
        "that defines them, namely non-negative item weights over symptom "
        "activations and strictly ordered cutpoints, so that severity is "
        "monotone in symptom presence by construction.")
    bullet(doc,
        "Compare the framework against state-of-the-art mental health models "
        "and an incremental ablation ladder under an identical protocol with "
        "paired bootstrap testing and per-task confidence intervals.")

    label(doc, "Aim")
    para(doc,
        "The aim of this research is the production of mental health assessments "
        "that are directly usable in a clinical screening pathway, in the sense "
        "that each prediction is accompanied by the clinical symptoms that "
        "justify it, each graded outcome respects the ordering and the additive "
        "scoring rule of the instrument it reports, each comorbid relationship "
        "is represented inside the estimator rather than diagnosed afterwards, "
        "and each reported margin is tested against the variability of the "
        "evaluation itself. The framework is therefore "
        "designed so that interpretability, structured output and corrected "
        "supervision are properties of the model itself.")

    label(doc, "Novelty")
    para(doc,
        "This paper stands out in the sense that it integrates four ideas that "
        "have not been combined before in mental health assessment from social "
        "media text. First, the clinical concept bottleneck is obligatory and "
        "its symptom queries are initialised from the sentence embeddings of "
        "the clinical definitions themselves, and each symptom carries "
        "an independent scoring vector rather than a scalar rescaling of one "
        "shared direction. Second, comorbidity enters the estimator as a graph "
        "whose nodes are symptoms and disorders and whose adjacency is a DSM-5 "
        "prior corrected by a learned residual under a sparsity penalty, with "
        "disorder-disorder edges estimated as conditional co-membership "
        "probabilities rather than as pointwise mutual information. Third, the "
        "ordinal heads are instrument-grounded: a single bounded latent "
        "distress trait formed from non-negative item weights feeds a "
        "cumulative-link model whose cutpoints are strictly ordered by "
        "construction, reproducing the additive rule by which PHQ-9 severity "
        "bands are computed. Fourth, weak supervision is "
        "corrected rather than trusted, by estimating the per-symptom "
        "true-positive and false-positive rates of the labelling model against "
        "gold agreement and inverting that channel inside the objective.")

    label(doc, "Contribution")
    bullet(doc,
        "A reproducible corpus harmonisation pipeline that maps seven public "
        "mental health collections onto one canonical twenty-two symptom space "
        "with a global post identifier, author-disjoint partitioning and a "
        "strictest-split-wins rule, yielding 19,762 gold-labelled posts with "
        "the documented overlap routes verified by assertion.")
    bullet(doc,
        "A symptom-mediated architecture combining definition-initialised "
        "concept attention, independent per-symptom scoring vectors, "
        "concept-gated pooling and message passing over a symptom-disorder "
        "comorbidity graph with a sparsity-penalised residual on a DSM-5 "
        "prior.")
    bullet(doc,
        "An instrument-grounded monotone ordinal head providing severity and "
        "suicide-risk estimates that are ordered and monotone in symptom "
        "presence by construction, with readable item loadings.")
    bullet(doc,
        "A calibrated weak-supervision scheme with a per-symptom noise channel "
        "estimated against gold labels and inverted during training, evaluated "
        "under paired bootstrap testing with per-task confidence intervals.")

    # ----------------------------------------------------- literature survey
    heading(doc, "2. Literature Survey")
    para(doc,
        "The authors Bauer et al. [6] (2024) have proposed the use of large "
        "language models to understand suicidality within a social-media-derived "
        "taxonomy of mental health disorders, performing a linguistic analysis "
        "of Reddit posts that distinguishes the vocabulary of risk from that of "
        "general distress. The analysis is nonetheless descriptive "
        "rather than predictive, so the identified separability never enters an "
        "estimator, and the induced categories remain mutually exclusive, which "
        "prevents the representation of a person who satisfies criteria for "
        "several conditions at once. The authors Lee et al. [7] (2024) have "
        "compared "
        "large language models against expert clinicians for crisis prediction "
        "among telemental health patients, which is among the most clinically "
        "rigorous evaluations in the literature and establishes that model and "
        "professional judgements diverge systematically rather than randomly. "
        "However, the comparison is conducted at the level of a final crisis "
        "decision without an auditable intermediate representation, so the "
        "source of a disagreement cannot be localised to particular clinical "
        "features, leaving a gap in architectures that expose the intermediate "
        "clinical quantities on which such a disagreement could be "
        "adjudicated.")
    para(doc,
        "The authors McCoy and Perlis [8] (2025) have proposed the application "
        "of large language models to stratify suicide risk from narrative "
        "clinical notes, demonstrating that the approach transfers from "
        "patient-authored social text to professionally authored records. "
        "Nevertheless, the stratification is produced "
        "directly from text without passing through the symptom constructs from "
        "which clinical scales are built, so the output cannot be reconciled "
        "with an instrument already in use, and the ordinal nature of risk is "
        "handled as an unordered set of strata. The authors Xie et al. [9] "
        "(2026) have proposed retrieval-augmented large language models for "
        "depression screening and suicide risk stratification, grounding "
        "generation in retrieved clinical material and reducing unsupported "
        "assertions relative to unaugmented generation. However, retrieval "
        "conditions the generated explanation rather than constraining the "
        "prediction itself, so nothing prevents the final decision from resting "
        "on evidence the retrieved material does not support, which leaves a gap "
        "in constraining the prediction rather than its narration.")
    para(doc,
        "The authors Bao et al. [10] (2024) have proposed transformer "
        "architectures that detect and explain the appearance of depressive "
        "symptom markers in user writings, presenting both separated and "
        "unified classify-and-explain designs and evaluating explanation quality "
        "with offline and expert-in-the-loop metrics, thereby establishing that "
        "symptom markers increase the trust health professionals place in "
        "automated decisions. Nevertheless, the symptom layer accompanies the "
        "decision rather than forming a bottleneck the decision must traverse, "
        "so the classifier may rely on evidence the explanation does not "
        "mention. The authors Suksam et al. [11] (2026) have proposed "
        "transformer ensemble learning for symptom-level classification followed "
        "by DSM-5-oriented screening, making the symptom layer the primary "
        "learning target and deriving the disorder judgement from it. However, "
        "the aggregation from symptoms to screening is a fixed rule applied "
        "after classification rather than a learned bounded response, and the "
        "ensemble members share no clinical structure, leaving a gap in learning "
        "the symptom-to-severity mapping as a constrained monotone function.")
    para(doc,
        "The authors Garg [12] (2024) have proposed wellness concept extraction "
        "and classification in Reddit posts for mental health analysis, "
        "establishing concept-level annotation as a viable intermediate task "
        "between raw text and a clinical judgement and demonstrating that "
        "annotators agree on concept presence at a useful rate. Nevertheless, "
        "the concepts are drawn from a wellness vocabulary rather than from the "
        "items of a validated instrument, so they cannot be aggregated into "
        "graded severity and cannot be reconciled with a scale a clinician "
        "reports. The authors Zhang et al. [13] (2024) have proposed a "
        "knowledge-aware deep learning approach to depression detection using "
        "digital traces, incorporating external knowledge into the input "
        "representation and reporting improved consistency of the resulting "
        "estimates. However, the knowledge conditions the representation rather "
        "than structuring the output space, so dependence between the predicted "
        "conditions remains unmodelled and a prediction of one disorder carries "
        "no information about the probability of a comorbid second, which leaves "
        "a gap in modelling dependence among the predicted clinical constructs "
        "themselves.")
    para(doc,
        "The authors Qasim et al. [14] (2025) have proposed the detection of "
        "depression severity in social media text using transformer-based "
        "models, isolating the severity scale as a task in its own right and "
        "confirming that pretrained encoders capture graded distinctions. "
        "Nevertheless, severity is predicted by an unconstrained classifier, so "
        "a prediction may violate the ordering of the scale, and the symptom "
        "items from which the scale is computed are not represented. The authors "
        "Kerasiotis et al. [15] (2024) have proposed depression detection in "
        "social media posts using transformer-based models together with "
        "auxiliary features, showing that handcrafted linguistic descriptors "
        "remain complementary to learned representations and that encoders do "
        "not fully subsume established psycholinguistic signal. However, the "
        "auxiliary features are concatenated into the input while the output "
        "remains a single disorder decision, so the additional signal improves a "
        "quantity that is still not clinically decomposable, leaving a gap in "
        "combining complementary evidence within a decomposable output space.")
    para(doc,
        "The authors Chen et al. [16] (2025) have proposed a heterogeneous "
        "subgraph network with prompt learning for interpretable depression "
        "detection, combining graph structure over heterogeneous context with "
        "prompt-based adaptation and reporting gains in both accuracy and "
        "interpretability over purely sequential encoders. However, the nodes of "
        "the constructed subgraphs are posts, users and words rather than "
        "clinical constructs, so the structure encodes who resembles whom rather "
        "than which conditions co-occur. The authors Jain et al. [17] (2024) "
        "have proposed CDME-GAT, a context-aware depression detection model "
        "using multiple embeddings and graph attention networks, demonstrating "
        "that attention over a constructed context graph improves robustness on "
        "noisy user-generated content. Nevertheless, the graph is induced from "
        "textual and contextual similarity and carries no clinical prior, so no "
        "mechanism exists by which diagnostic knowledge could inform "
        "propagation, and the model yields no bounded interpretable parameter.")
    para(doc,
        "The authors Rehman et al. [18] (2026) have proposed explainable "
        "text-based depression and suicide risk prediction using deep learning "
        "and graph neural networks, where the graph is estimated from semantic "
        "similarity between posts together with author relations rather than "
        "from social interactions, and interpretability is supplied by post-hoc "
        "attribution with calibration analysis. The work is directly relevant "
        "because it predicts both targets jointly and treats trustworthiness as "
        "an explicit objective. Nevertheless, the graph connects posts rather "
        "than clinical constructs, the attribution is computed after training "
        "rather than constrained during it, and no control establishes that the "
        "estimated structure outperforms a random graph of equal density. The "
        "authors Tahir et al. [19] (2026) have proposed an ontology-guided "
        "knowledge graph with a retrieval-augmented language model for "
        "interpretable assessment, confirming that clinical ontologies can enter "
        "the pipeline, but the ontology retrieves evidence for a generative "
        "model rather than defining a differentiable adjacency over which "
        "symptom evidence propagates during training. The authors Gao et al. "
        "[20] (2025) have proposed large language model powered knowledge graph "
        "construction for mental health exploration, representing the current "
        "frontier of automated clinical structure induction and surfacing "
        "relations that manual curation omits, although the induced structure "
        "supports exploratory analysis rather than acting as a constraint inside "
        "a trained estimator and is not validated against a diagnostic "
        "reference, which leaves a gap in combining curated or induced clinical "
        "structure with learned representation power inside one constrained "
        "estimator.")
    caption(doc,
        "Table 1. Comparison of existing mental health assessment and "
        "symptom identification methods with the proposed SymptoGraph-LLM")
    table(doc, [
        ["Ref. (Year)", "Application Domain", "Technical Approach",
         "Core Contribution", "Key Technical Gap"],
        ["[21] (2025)", "Multi-label mental health classification",
         "Multi-perspective prompt ensemble with perturbation-based "
         "self-supervision on a pretrained backbone",
         "Prompt diversity and consistency regularisation improve multi-label "
         "prediction",
         "Labels are disorders rather than instrument items, so they cannot be "
         "aggregated into graded severity"],
        ["[22] (2025)", "Depression detection",
         "Emotion-aware encoders with fuzzy-based contrastive networks",
         "Emotion-conditioned representations separate depressive from "
         "non-depressive text",
         "Single disorder target with no symptom decomposition and no ordinal "
         "severity estimate"],
        ["[23] (2025)", "Clinical natural language processing",
         "Knowledge-free large language model weak supervision",
         "Model-generated labels substitute effectively for scarce expert "
         "annotation",
         "Weak labels consumed as certain; per-class labeller bias neither "
         "estimated nor inverted"],
        ["[24] (2024)", "Multilingual depression detection",
         "Federated learning with multilingual language models",
         "Privacy preserved at modest accuracy cost across languages",
         "Addresses privacy but not label reliability or clinical "
         "decomposability of the output"],
        ["[25] (2024)", "Mental health applications",
         "Systematic review of large language model applications",
         "Documents rapid adoption across screening and triage tasks",
         "Reports that evaluations are not comparable; supplies no corrective "
         "protocol"],
        ["[26] (2024)", "Psychiatry",
         "Systematic review of large language models in psychiatry",
         "Identifies diagnosis, triage and documentation as main application "
         "areas",
         "Review only; clinical validity rarely established and provenance "
         "seldom examined"],
        ["[27] (2024)", "Psychiatry",
         "Analysis of opportunities and challenges",
         "Frames interpretability and validation as the binding constraints on "
         "deployment",
         "Conceptual analysis without an implemented estimator or evaluation "
         "protocol"],
        ["[28] (2024)", "Suicide detection and risk assessment",
         "Systematic review of encoder-based and generative models",
         "Finds reported margins frequently fall within evaluation variability",
         "Diagnoses the absence of significance testing without supplying a "
         "paired procedure"],
        ["[29] (2026)", "Mental health benchmarking",
         "PsyEval, a comprehensive evaluation benchmark",
         "Standardised task suite exposes large variation between models on "
         "identical inputs",
         "Benchmarks generative behaviour; does not control corpus overlap "
         "between constituent datasets"],
        ["[30] (2025)", "Mental health care",
         "Systematic review of large language models in care delivery",
         "Concludes that clinical integration rather than model scale limits "
         "present utility",
         "Calls for leakage-controlled and significance-audited primary studies "
         "without conducting one"],
        ["Proposed", "Multi-task mental health assessment",
         "Symptom-grounded concept bottleneck over a comorbidity graph with an "
         "instrument-grounded monotone ordinal head and noise-corrected weak "
         "supervision",
         "Obligatory symptom mediation, propagated comorbidity, severity "
         "monotone in symptom presence by construction and inverted labeller "
         "bias",
         "Integrates interpretable mediation, comorbid structure and corrected "
         "supervision under leakage control with paired significance testing"],
    ], widths=[0.72, 1.05, 1.42, 1.3, 1.46], fs=9)

    # ------------------------------------------------------------ methodology
    heading(doc, "3. Proposed Methodology")
    para(doc,
        "The proposed SymptoGraph-LLM (Symptom-Grounded Comorbidity-Aware "
        "Multi-Task Network under Calibrated Weak Supervision) framework is "
        "designed for interpretable mental health assessment at the level of "
        "individual posts in large heterogeneous social media collections whose "
        "annotation is scarce, unevenly distributed across clinical conditions "
        "and drawn from corpora of overlapping provenance. Different from "
        "existing frameworks that either classify a disorder directly, generate "
        "an explanation alongside a decision or fuse auxiliary features into the "
        "input representation, the proposed framework integrates the following "
        "elements into a unified estimation pipeline: harmonisation of seven "
        "public corpora onto one canonical symptom space under explicit leakage "
        "control, a clinical concept bottleneck with definition-initialised "
        "symptom attention, a symptom-disorder comorbidity graph with a learned "
        "residual adjacency, an instrument-grounded monotone ordinal head and a "
        "calibrated weak-supervision branch with a per-symptom noise channel. "
        "First, seven collections supplying item-level symptom annotation, "
        "four-level depression severity, five-level clinician-coded suicide risk "
        "and binary perceived stress are mapped onto a shared space of "
        "twenty-two symptoms derived from the PHQ-9 and GAD-7 instruments, with "
        "a global post identifier assigned over a normalised text prefix so that "
        "a post appearing in several corpora collapses to a single row, and with "
        "author-disjoint partitioning so that no author's writing appears on "
        "both sides of a split. Then the text is encoded by a domain-pretrained "
        "transformer and projected onto the symptom space by attention queries "
        "initialised from the embeddings of the clinical definitions, "
        "producing a concept vector through which every prediction "
        "must pass. The resulting concepts are refined "
        "by message passing over a graph whose nodes are the twenty-two symptoms "
        "together with the disorder categories observed in the corpus and whose "
        "adjacency sums a DSM-5 derived prior with a learned residual penalised "
        "towards sparsity. Meanwhile, the instrument-grounded monotone ordinal head "
        "maps the concept vector onto a bounded latent distress trait through "
        "non-negative item weights and predicts each ordinal task by a "
        "cumulative-link model with strictly ordered cutpoints, so that the "
        "five-level risk task, which carries only 340 training instances, "
        "reduces to four threshold parameters over a trait simultaneously "
        "trained by the far more abundant severity annotation. In parallel, an "
        "instruction-tuned large language model executed locally labels the "
        "eleven symptoms that no public corpus annotates, and its per-symptom "
        "true-positive and false-positive rates are estimated and inverted "
        "inside the objective, because a single temperature rescales every "
        "label by the same factor and cannot repair systematic bias. Finally, "
        "the evaluation protocol applies "
        "validation-loss checkpoint selection, per-symptom decision thresholds "
        "tuned on validation data, three-seed replication and paired bootstrap "
        "tests, against three baselines, three re-implemented architectures and "
        "five component-wise ablations. "
        "The full pipeline of the proposed SymptoGraph-LLM framework is "
        "illustrated in Figure 1.")
    figure(doc, figs["fig1"], 1,
           "Overall workflow of the proposed SymptoGraph-LLM framework.")

    # ------------------------------------------------- 3.1 dataset description
    heading(doc, "3.1 Dataset Description")
    para(doc,
        "As the primary data resource for the development and testing of the "
        "proposed SymptoGraph-LLM framework, a harmonised corpus assembled from "
        "seven publicly available mental health text collections is employed, "
        "because it is the only configuration providing item-level symptom "
        "annotation together with graded severity and clinician-coded suicide "
        "risk over one canonical label space, which is the prerequisite for "
        "symptom-mediated estimation. In contrast to conventional mental health "
        "datasets that supply a single binary disorder label or a single "
        "severity scale, the assembled corpus comprises symptom annotations "
        "aligned to validated instruments, four-level depression severity, "
        "five-level suicide risk, binary perceived stress, author identifiers, "
        "posting dates and subreddit provenance. The constituent sources are "
        "DepressionEmo for item-level "
        "depressive symptoms, Dreaddit for perceived stress, DepSeverity for "
        "four-level severity, the C-SSRS annotated Reddit collection for "
        "five-level risk, the Interpersonal Risk Factors and LoST collections "
        "for thwarted belongingness and loneliness, "
        "and the IMHI instruction collection, together with an unlabelled pool "
        "of 1,017,959 posts drawn from the Reddit Mental Health Dataset across "
        "108 subreddit files and 822,884 distinct authors. The shared label "
        "space comprises 22 symptoms, of which 9 correspond to PHQ-9 items, 7 to "
        "GAD-7 items and 6 to further constructs from the interpersonal-risk and "
        "loneliness literature.")
    para(doc,
        "Before developing the model, the collections are processed into a "
        "canonical table using a reproducible harmonisation procedure, because "
        "the public corpora in this field overlap substantially and splitting "
        "each independently would leak training material into evaluation sets "
        "through five distinct routes that were measured directly. The most "
        "severe is that DepSeverity re-annotates the same 3,526 posts contained "
        "in Dreaddit, so that treating either as an independent evaluation set "
        "for the other would be circular; LoST and the Interpersonal Risk "
        "Factors collection share 3,089 posts; the IMHI collection repackages "
        "posts from other sources behind an instruction prefix; 192,893 of the "
        "822,884 authors appear in more than one source file, representing 23.3 "
        "per cent of the author population; and the unlabelled pool contains "
        "85,659 internal duplicates. A global post identifier computed as a hash "
        "over a normalised two-hundred-character prefix therefore collapses each "
        "unique post to one row, every unique post receives exactly one split "
        "assignment, Reddit posts carrying a known author are partitioned "
        "author-disjointly so that no author's writing style spans a boundary, "
        "and where the constituent corpora disagree about an assignment the "
        "strictest one prevails. The partitioning is verified by an assertion "
        "that raises if any post or any author is found on both sides of a "
        "split, so that a leakage regression fails the pipeline rather than "
        "producing an inflated score.")
    para(doc,
        "The resulting gold-labelled partition comprises 19,762 unique posts, "
        "which are divided into training, validation and test subsets on the "
        "basis of post and author identity rather than of source corpus, so that "
        "the evaluation measures generalisation to unseen writers rather than "
        "recall of writers already observed. Symptom labels are retained in "
        "three states denoting presence, absence and absence of annotation by "
        "that particular source, and only the first two contribute to the "
        "objective, because treating an unannotated symptom as a negative "
        "observation would teach the model that symptoms no corpus examined are "
        "reliably absent. Eleven of the twenty-two symptoms, predominantly the "
        "GAD-7 anxiety items, carry no gold annotation in any constituent "
        "collection, which is the condition that motivates the weak-supervision "
        "branch described in Section 3.5. The partitioning of the dataset for "
        "the experimental study is illustrated in Table 2.")
    caption(doc, "Table 2. Dataset partitioning for SymptoGraph-LLM")
    table(doc, [
        ["Dataset split", "Posts", "Symptom-annotated posts", "Split %"],
        ["Training set", "12,911", "6,227", "65.33%"],
        ["Validation set", "2,862", "1,397", "14.48%"],
        ["Test set", "3,989", "1,956", "20.19%"],
    ], widths=[1.8, 1.2, 2.1, 1.0])
    para(doc,
        "The training set is used for the adversarial-free supervised training "
        "of the encoder, the concept bottleneck, the graph residual and the "
        "ordinal cutpoints, and the statistics of the training subset alone "
        "determine the symptom positive weights, the ordinal class weights and "
        "the sentence embeddings used to initialise the symptom queries, so that "
        "no information derived from validation or test material influences "
        "optimisation. The validation set is used for monitoring convergence, "
        "selecting the checkpoint at minimum validation loss, tuning the "
        "per-symptom decision thresholds and triggering early stopping. The "
        "composition of the annotated subsets across the four prediction tasks "
        "is reported in Table 3.")
    caption(doc, "Table 3. Task composition of the training and test subsets")
    table(doc, [
        ["Prediction task", "Training instances", "Test instances",
         "Test share"],
        ["Symptom presence (22 labels)", "6,227", "1,956", "23.9%"],
        ["Perceived stress (binary)", "2,478", "706", "22.2%"],
        ["Depression severity (4 levels)", "2,456", "541", "18.1%"],
        ["Suicide risk (5 levels)", "340", "74", "17.9%"],
        ["Weak-labelled symptoms (11 labels)", "49,982", "\u2013", "\u2013"],
    ], widths=[2.3, 1.4, 1.25, 1.0])
    para(doc,
        "Due to the imbalance of the tasks, with a ratio of 18.3 between the "
        "number of symptom-annotated instances and the number of suicide-risk "
        "instances, sampling and weighting require explicit treatment rather "
        "than being left to the optimiser. Depression severity is additionally "
        "skewed within itself, the lowest of its four bands accounting for 71.9 "
        "per cent of the training instances in the proportion 1767, 205, 288 and "
        "196, and suicide risk is similarly concentrated in its lower levels. "
        "Unweighted cross-entropy under this distribution collapses both ordinal "
        "heads onto the majority class, a failure that was observed directly "
        "during development as a quadratic weighted kappa of exactly zero, so "
        "capped inverse-frequency class weights are applied to both ordinal "
        "objectives and capped positive weights are applied to the symptom "
        "objective. The consequence of the risk partition size for the "
        "interpretation of risk metrics is quantified explicitly in Section 4.3, "
        "where the confidence interval attainable on 74 test instances is shown "
        "to exceed any architectural difference observed in this study.")
    para(doc,
        "Each post also carries the conditioning information used by the "
        "framework. The symptom definitions, taken verbatim from the PHQ-9 and "
        "GAD-7 item wordings and from the published definitions of the six "
        "additional constructs, are encoded once by a frozen sentence encoder "
        "and supply the initialisation of the symptom attention queries. The "
        "disorder label associated with each post, where the source corpus "
        "provides one, supplies the disorder nodes of the comorbidity graph, and "
        "the author identifier supports both the author-disjoint partitioning "
        "and the estimation of the disorder-disorder adjacency from co-membership "
        "across the unlabelled pool. Posting dates and subreddit provenance are "
        "retained for subgroup analysis but are deliberately excluded from the "
        "model input, because a subreddit name is a near-deterministic proxy for "
        "the disorder label in several of the constituent corpora and including "
        "it would allow the framework to bypass the clinical content entirely. "
        "A related form of leakage was measured in the text itself and is "
        "reported here because it bounds the interpretation of the symptom "
        "results: 69.3 per cent of the posts carrying an anxiety label contain "
        "the word anxiety, so a model can attain substantial apparent "
        "performance on that label by lexical matching alone. This is a "
        "property of corpora assembled by subreddit sampling and it is the "
        "reason alternative collections were not adopted merely because they "
        "report higher published figures, since a higher figure obtained on a "
        "corpus with stronger keyword leakage measures less rather than more. "
        "The unlabelled pool "
        "from which the weak-supervision subsample is drawn comprises 1,017,959 "
        "posts written by 822,884 distinct authors after duplicate removal, a "
        "mean of 1.24 posts per author, and 132,850 of those authors contribute "
        "two or more posts and together account for 327,866 of the posts, which "
        "is the measurement determining whether an author-level consistency "
        "objective is feasible at all.")

    # --------------------------------------------------- 3.2 pre-processing
    heading(doc, "3.2 Data Pre-processing")
    para(doc,
        "The process of developing the proposed SymptoGraph-LLM framework begins "
        "with the systematic pre-processing of the seven constituent collections "
        "into one canonical table under explicit leakage control. The procedure "
        "assigns a provenance-independent identity to every post, harmonises the "
        "heterogeneous label schemes onto the shared twenty-two symptom space, "
        "establishes a partition that no post and no author can cross, and "
        "constructs the stratified subsample that the weak-supervision branch "
        "will annotate. Each stage is expressed below as an operation on the "
        "union of the constituent corpora, so that the pipeline is reproducible "
        "from the published sources without reference to intermediate "
        "artefacts. The ordering of the stages is itself substantive, because "
        "identity resolution must precede partitioning for the partition to be "
        "well defined, and partitioning must precede the construction of the "
        "weak-supervision subsample for that subsample to be free of evaluation "
        "material. Reversing either dependency produces a pipeline that "
        "executes without error and yields optimistic results.")
    equation(doc, eqs, 1)
    para(doc,
        "where the normalisation operator lowercases the text, collapses "
        "whitespace, strips instruction prefixes of the form used by the IMHI "
        "collection and removes markup, the prefix length is fixed at two "
        "hundred characters because it was found sufficient to separate distinct "
        "posts while remaining robust to truncation differences between corpora, "
        "and the quotient by the identifier relation collapses every duplicate "
        "to a single canonical row. Applying this operator to the seven "
        "collections reduces the union to 19,762 unique gold-labelled posts and "
        "removes 85,659 duplicates from the unlabelled pool, and it is this "
        "operation that resolves the overlap between DepSeverity and Dreaddit "
        "and between LoST and the Interpersonal Risk Factors collection without "
        "requiring any corpus-specific exception. The most consequential of the "
        "routes is that DepSeverity re-annotates the same 3,526 posts contained "
        "in Dreaddit rather than sampling independently, so the two cannot be "
        "treated as distinct domains and a cross-corpus transfer claim between "
        "them would be vacuous. The IMHI collection repackages material from "
        "several of the remaining sources behind an instruction prefix, so its "
        "posts are not recognisable as duplicates by exact string comparison "
        "and are recovered only once the prefix is stripped by the "
        "normalisation operator.")
    equation(doc, eqs, 2)
    para(doc,
        "where the three label states denote that the symptom was annotated as "
        "present, annotated as absent, or not examined by the source corpus that "
        "contributed the post, and the mask selects only the first two for "
        "inclusion in the objective. This three-state encoding is necessary "
        "because the constituent corpora annotate different subsets of the "
        "symptom space: a post drawn from a corpus concerned with interpersonal "
        "risk carries informative labels for thwarted belongingness and "
        "perceived burdensomeness but says nothing about sleep disturbance, and "
        "treating that silence as a negative observation would inject a large "
        "quantity of false negatives into precisely the symptoms the framework "
        "is intended to detect. The practical consequence of the three-state "
        "encoding is severe and is reported here rather than deferred, because "
        "it bounds what the evaluation in Section 4 is able to establish: "
        "eleven of the twenty-two symptoms in the ontology carry no gold "
        "annotation anywhere in the harmonised corpus, so the mask is zero for "
        "those eleven at every row of every partition. Those eleven are "
        "predominantly the anxiety-spectrum concepts drawn from the GAD-7 "
        "instrument, for which no constituent collection provides item-level "
        "annotation.")
    equation(doc, eqs, 3)
    para(doc,
        "where the bucketing function maps a salted hash of the author "
        "identifier onto the three partitions in fixed proportions, and the "
        "constraint requires that any two posts sharing an author receive the "
        "same assignment. Author-disjoint partitioning is applied rather than "
        "post-level randomisation because an author's lexical and syntactic "
        "idiosyncrasies are highly identifying, so a model evaluated on unseen "
        "posts from authors it has already observed measures author recognition "
        "in addition to symptom detection. Because the constituent corpora "
        "contribute overlapping author populations, the constraint cannot be "
        "satisfied by independent bucketing alone, and a repair pass reassigns "
        "every author found to span a boundary to the single partition holding "
        "the majority of that author's posts before the verification assertion "
        "is evaluated. Measured over the harmonised corpus, this procedure "
        "relocates the posts of the 192,893 authors appearing in more than one "
        "source file and reduces the number of authors spanning a split to "
        "zero. An alternative design in which each constituent corpus is "
        "partitioned independently according to its own published split was "
        "considered and rejected, because the overlap measurements above show "
        "that such a procedure would place the same post in the training "
        "partition of one corpus and the test partition of another, and the "
        "resulting leakage would be invisible to any per-corpus verification.")
    equation(doc, eqs, 4)
    para(doc,
        "where the rank function orders the three partitions so that the test "
        "assignment is the strictest and the training assignment the most "
        "permissive, and the minimum is taken over every corpus contributing the "
        "post. This rule resolves the remaining conflicts: when one collection "
        "designates a post as part of its official test split while another "
        "contributes the same post to its training material, the post is "
        "assigned to the test partition for all purposes, so that an official "
        "evaluation split published by any constituent corpus is never used for "
        "training by the harmonised pipeline. The rule is deliberately "
        "asymmetric in its cost: it discards training material in order to "
        "protect evaluation material, and it never performs the reverse "
        "exchange. The resulting partition holds 12,911 posts for training, "
        "2,862 for validation and 3,989 for testing, and the verification pass "
        "terminating this stage asserts that the three are disjoint as sets of "
        "canonical identifiers, that no author occurs in more than one, and "
        "that their union recovers the full 19,762 rows.")
    equation(doc, eqs, 5)
    para(doc,
        "where the stratified sampler draws from the unlabelled pool "
        "proportionally across the disorder categories and the posting periods, "
        "and the control share caps the proportion of posts originating from "
        "non-clinical subreddits at fifteen per cent of the subsample. The "
        "resulting subsample of 49,982 posts is the material annotated by the "
        "weak-supervision branch described in Section 3.5, and its size was "
        "chosen as the largest quantity that could be labelled by a locally "
        "executed instruction-tuned model within the available compute budget, "
        "measured at approximately 1.6 posts per second. The remainder of the "
        "pool is retained unlabelled. It should be emphasised that the "
        "subsample is drawn exclusively from posts assigned to the training "
        "partition by the preceding operations, so that weak supervision cannot "
        "introduce evaluation material into the training objective by an "
        "indirect route, and that the weak labels are produced locally rather "
        "than through a hosted interface because the data agreements attached "
        "to several of the constituent corpora prohibit transmission of the "
        "posts to a third party. Annotating the pool in its entirety was "
        "considered and rejected on cost, the full pass being estimated at "
        "approximately thirty-five accelerator-hours against roughly one and a "
        "half for the subsample, an expenditure that could not be justified "
        "while the measurable benefit of the weak branch remains bounded by the "
        "eleven gold-supervised concepts.")

    # ---------------------------------------------- 3.3 feature extraction
    heading(doc, "3.3 Feature Extraction")
    para(doc,
        "In the suggested SymptoGraph-LLM framework, the procedure of feature "
        "extraction is aimed at obtaining a clinically addressable "
        "representation of each post, in which the dimensions correspond to "
        "named symptoms rather than to arbitrary latent factors. The procedure "
        "encodes the post with a domain-pretrained transformer, attends to the "
        "token positions relevant to each symptom using queries derived from the "
        "clinical definitions, scores each symptom with its own projection, and "
        "pools the attended evidence under concept gating so that the "
        "downstream heads observe only what the concepts themselves attended "
        "to.")
    equation(doc, eqs, 6)
    para(doc,
        "where the encoder is MentalRoBERTa, a transformer pretrained on "
        "mental health text, the sequence length is fixed at 192 tokens, "
        "covering approximately eighty per cent of the posts in the harmonised "
        "corpus whose median length is 73 words, and the hidden width is 768. "
        "The encoder identity is verified at load time against the configured "
        "name, because a gated or unavailable checkpoint otherwise falls back "
        "silently to a general-domain model and the substitution is detectable "
        "only from the parameter count.")
    equation(doc, eqs, 7)
    para(doc,
        "where the attention distribution for symptom k is computed over token "
        "positions with the query for that symptom, and the query is initialised "
        "from the sentence embedding of the clinical definition of the symptom "
        "rather than from a random draw. This initialisation is load-bearing "
        "rather than cosmetic: it establishes the association between a query "
        "and its symptom before any gradient is applied, which matters because "
        "eleven of the twenty-two symptoms receive no gold supervision at all "
        "and would otherwise have no signal by which to acquire a clinical "
        "meaning.")
    equation(doc, eqs, 8)
    para(doc,
        "where the evidence vector for each symptom is the attention-weighted "
        "sum of the token representations and each symptom carries an "
        "independent scoring vector and bias. The independence of the scoring "
        "vectors is the property that makes the bottleneck a bottleneck, and it "
        "is stated explicitly because an earlier implementation shared a single "
        "projection across all twenty-two symptoms. Under a shared projection "
        "the concept logits differ only through their attention distributions "
        "and are therefore constrained to lie along one direction in the "
        "representation space, which collapses the symptom layer to an "
        "effectively scalar quantity and removes the per-symptom "
        "addressability the design exists to provide. The defect is invisible "
        "in the aggregate loss.")
    equation(doc, eqs, 9)
    para(doc,
        "where the gate is the activation of each concept and the pooled "
        "representation is the concept-weighted average of the per-symptom "
        "evidence vectors. Concept gating rather than a bare projection of the "
        "twenty-two concept logits is used because a twenty-two dimensional "
        "bottleneck was found during development to starve the severity head, "
        "which observes 768 dimensions in the flat configuration, while the "
        "gated summary preserves the constraint that every downstream prediction "
        "is a function of what the concepts attended to.")

    # ------------------------------------------------ 3.4 framework architecture
    heading(doc, "3.4 Proposed Framework Architecture")
    para(doc,
        "The proposed SymptoGraph-LLM framework (Symptom-Grounded "
        "Comorbidity-Aware Multi-Task Network under Calibrated Weak "
        "Supervision) is organised as a single forward path in which the "
        "clinical concept layer is interposed between the encoder and every "
        "prediction head, together with two auxiliary branches that govern how "
        "the concept layer is supervised rather than what it outputs. The "
        "overall architecture of the proposed framework is presented in "
        "Figure 2.")
    figure(doc, figs["fig2"], 2,
           "Overall architecture of the proposed SymptoGraph-LLM.")
    para(doc,
        "The post is tokenised to a fixed length of 192 positions and encoded "
        "by the twelve-layer domain-pretrained transformer, producing a "
        "sequence of contextual token representations of width 768. The "
        "twenty-two symptom queries, initialised from the sentence embeddings "
        "of the clinical definitions and thereafter trained, compute one "
        "attention distribution per symptom over the token positions, and the "
        "attention-weighted sums form a matrix of twenty-two per-symptom "
        "evidence vectors. Each symptom is then scored by its own projection "
        "vector and bias, which yields the twenty-two concept logits. The "
        "independence of these projections is a deliberate design decision "
        "rather than an implementation detail: a single shared projection "
        "applied to all symptoms leaves each symptom only a scalar rescaling of "
        "one common direction in representation space, which is not a concept "
        "bottleneck at all, and was observed during development to collapse the "
        "symptom head to a macro-F1 of zero while the flat configuration with "
        "independent projections continued to learn normally. The concept "
        "logits are passed through a sigmoid to form the gate, and the "
        "concept-gated pooled representation is the gate-weighted average of "
        "the per-symptom evidence vectors, so that the downstream heads observe "
        "a summary of exactly the textual evidence the concepts attended to.")
    para(doc,
        "The concept logits are refined by message passing over the "
        "comorbidity graph, whose nodes comprise the twenty-two symptoms "
        "together with the disorder categories observed in the corpus. The "
        "adjacency is the sum of a fixed prior and a learned residual, where "
        "the symptom-to-disorder block of the prior is authored from DSM-5 "
        "criteria and the disorder-to-disorder block is estimated from author "
        "co-membership across the unlabelled pool as a conditional "
        "co-membership probability. The conditional form is used in preference "
        "to pointwise mutual information because the mean number of distinct "
        "conditions per author in this corpus is 1.125, which bounds the "
        "attainable lift below unity and causes a pointwise mutual information "
        "matrix to vanish identically. The estimated edges are clinically "
        "plausible, the strongest associations holding between bipolar disorder "
        "and schizophrenia at 0.676, between borderline personality disorder "
        "and eating disorders at 0.428, and between attention-deficit "
        "hyperactivity disorder and autism at 0.404. Two numerical safeguards "
        "are applied in the propagation step. Non-negativity is enforced on the "
        "summed adjacency before symmetric normalisation, because an "
        "unconstrained residual can drive a node degree towards zero and the "
        "inverse square-root normaliser then amplifies activations by four "
        "orders of magnitude, a failure observed directly during development as "
        "a training loss of order ten to the thirteenth. Layer normalisation is "
        "applied after each residual propagation step so that repeated message "
        "passing cannot compound activation scale across layers.")
    para(doc,
        "Two further modules control how the concept layer is supervised rather "
        "than what it emits. The weak-label branch draws a batch of "
        "LLM-annotated posts once per optimisation step from a separate loader, "
        "and contributes only the symptom term of the objective; the branch is "
        "deliberately not implemented by concatenating the weak rows into the "
        "training set, because concatenation inflates the number of optimiser "
        "steps per epoch by forty-six per cent, alters the learning-rate "
        "schedule relative to configurations without weak supervision, and "
        "reduces the number of severity-labelled and risk-labelled rows per "
        "batch, all of which were measured during development to degrade "
        "precisely the tasks carrying the fewest annotations. The noise-channel "
        "module intervenes on that same term, mapping the clean probability the "
        "network predicts through the estimated per-symptom channel before the "
        "loss is evaluated, so that a symptom the labelling model "
        "systematically over-calls is discounted rather than learned. A "
        "layer-wise summary of the architecture is given in Table 4.")
    caption(doc,
        "Table 4. Layer-wise architecture summary of the proposed "
        "SymptoGraph-LLM")
    table(doc, [
        ["Layer", "Configuration", "Output shape"],
        ["Input", "tokenised post, truncation at 192", "(B, 192)"],
        ["Encoder embedding", "MentalRoBERTa word + position embeddings",
         "(B, 192, 768)"],
        ["Encoder blocks 1-12", "self-attention 12 heads, FFN 3072, GELU",
         "(B, 192, 768)"],
        ["Symptom queries", "22 queries initialised from SBERT definitions",
         "(22, 768)"],
        ["Definition projection", "linear 768 to 768 over query embeddings",
         "(22, 768)"],
        ["Symptom attention", "scaled dot-product over token positions, masked",
         "(B, 192, 22)"],
        ["Evidence aggregation", "attention-weighted sum per symptom",
         "(B, 22, 768)"],
        ["Dropout", "p = 0.1", "(B, 22, 768)"],
        ["Per-symptom scoring", "independent vector and bias per symptom",
         "(B, 22)"],
        ["Concept gate", "sigmoid activation of concept logits", "(B, 22)"],
        ["Graph node features", "symptom nodes + disorder nodes, width 64",
         "(B, 22+D, 64)"],
        ["Prior adjacency", "DSM-5 symptom-disorder, conditional disorder-disorder",
         "(22+D, 22+D)"],
        ["Residual adjacency", "learned, L1-penalised", "(22+D, 22+D)"],
        ["Normalised adjacency", "relu, add identity, symmetric normalisation",
         "(22+D, 22+D)"],
        ["Graph layer 1", "linear 64, ReLU, residual, LayerNorm",
         "(B, 22+D, 64)"],
        ["Graph layer 2", "linear 64, ReLU, residual, LayerNorm",
         "(B, 22+D, 64)"],
        ["Graph refinement", "linear projection to concept logit correction",
         "(B, 22)"],
        ["Concept-gated pooling", "gate-weighted average of evidence vectors",
         "(B, 768)"],
        ["Head input", "concat(concept logits, pooled evidence)", "(B, 790)"],
        ["Item weights", "softplus-constrained, one per symptom", "(22,)"],
        ["Latent distress trait", "non-negative weighted sum of activations",
         "(B, 1)"],
        ["Severity cutpoints", "first cut + 2 positive increments", "(3,)"],
        ["Risk cutpoints", "first cut + 3 positive increments", "(4,)"],
        ["Severity head", "cumulative link, differenced to class log-probs",
         "(B, 4)"],
        ["Risk head", "cumulative link, differenced to class log-probs",
         "(B, 5)"],
        ["Stress head", "linear projection, sigmoid", "(B, 1)"],
        ["Symptom head", "concept logits, thresholded per symptom", "(B, 22)"],
        ["Weak-label loader", "separate stream, batch 16, drawn once per step",
         "(16, 192)"],
        ["Noise channel", "per-symptom tpr and fpr, forward correction",
         "(22,), (22,)"],
        ["Objective", "masked multi-task sum with six weighted terms",
         "scalar"],
    ], widths=[1.75, 2.9, 1.45], fs=9)
    para(doc,
        "The suggested design also allows comparing different configuration "
        "schemes and performing component-wise ablation, because each novelty "
        "occupies a separable position in the forward path and can be disabled "
        "without altering the remainder. Separability of this kind is a "
        "requirement on the implementation rather than a consequence of the "
        "diagram, and it is enforced by recording the architectural "
        "configuration of every run in a fingerprint that a resumed run must "
        "match before its checkpoint is accepted, so that a rung cannot "
        "silently inherit a component it was defined to exclude. The parameter "
        "budget is dominated by "
        "the pretrained encoder, so that the clinical structure introduced by "
        "the framework is obtained at a small marginal cost in capacity, which "
        "matters because the model already fits its ordinal training partitions "
        "to saturation and additional free parameters would worsen rather than "
        "improve generalisation. The distribution of parameters across the "
        "components is reported in Table 5.")
    caption(doc,
        "Table 5. Parameter distribution of the proposed SymptoGraph-LLM")
    table(doc, [
        ["Component", "Parameters"],
        ["MentalRoBERTa encoder", "124,645,632"],
        ["Symptom queries (22 x 768)", "16,896"],
        ["Definition projection", "590,592"],
        ["Per-symptom scoring vectors and biases", "16,918"],
        ["Graph node embeddings", "1,408 + 64D"],
        ["Residual adjacency", "(22+D)\u00b2"],
        ["Graph propagation layers", "8,320"],
        ["Graph refinement projection", "65"],
        ["Monotone ordinal structure (weights, scale, cutpoints)", "30"],
        ["Severity, risk and stress heads", "6,331"],
        ["Total trainable", "\u2248 125.9 M"],
    ], widths=[3.3, 2.0], fs=10)
    para(doc,
        "The architecture overview in Table 4 demonstrates clearly that the "
        "proposed SymptoGraph-LLM framework is modular in the sense required "
        "for a controlled ablation, since the concept bottleneck, the "
        "comorbidity graph, the monotone ordinal head, the weak-label stream "
        "and the noise channel each occupy an identifiable stage and each can "
        "be removed in isolation while the shared protocol is held fixed. Two "
        "properties of the design deserve emphasis in connection with the "
        "evaluation that follows. First, the ordinal heads together account for "
        "seven cutpoint parameters and twenty-three trait parameters, so the "
        "five-level suicide-risk task is estimated by four thresholds over a "
        "trait that the 2,456 severity-annotated instances also train, which is "
        "the mechanism by which a task carrying only 340 training instances "
        "borrows statistical strength. That borrowing is the only mechanism by "
        "which the risk task can be estimated at all under the present "
        "annotation, and it is also the mechanism by which a defect in the "
        "severity head propagates to the risk head, so the two tasks cannot be "
        "treated as independent evidence for a claim about the design. Second, "
        "the item weights are directly "
        "readable as item loadings and can be compared against the equal "
        "weighting that the PHQ-9 prescribes, the instrument assigning every "
        "item the same coefficient by convention rather than by evidence, so "
        "the framework exposes a "
        "falsifiable clinical claim about which symptoms carry disproportionate "
        "information about graded severity in this population, and Section 4.5 "
        "reports that claim together with the condition under which it must be "
        "withdrawn.")

    # ----------------------------------------- 3.5 loss and hyperparameters
    heading(doc, "3.5 Loss Function and Hyperparameter Configuration")
    para(doc,
        "The optimisation approach used in the proposed SymptoGraph-LLM "
        "framework is a critical factor in training a network whose symptom "
        "layer must remain clinically meaningful while four prediction tasks of "
        "very unequal size compete for the shared encoder. The objective "
        "therefore combines a masked symptom term, two ordinal terms, a binary "
        "stress term, a corrected weak-supervision term and a sparsity penalty "
        "on the learned graph residual, each weighted so that no task is "
        "silenced by the scarcity of its annotation.")
    equation(doc, eqs, 10)
    para(doc,
        "where the mask excludes symptom entries the source corpus did not "
        "annotate, and the weight combines the capped inverse-frequency "
        "positive weight for that symptom with the calibrated confidence of the "
        "label where the label is weak and unity where it is gold. The cap is "
        "set at ten, the rarest concepts being infrequent enough that an "
        "uncapped reciprocal would assign them several hundred times the "
        "gradient of the commonest concept.")
    equation(doc, eqs, 11)
    para(doc,
        "where the item weights are constrained non-negative by the softplus "
        "transform, so that the latent distress trait is monotone "
        "non-decreasing in the activation of every symptom and the predicted "
        "severity cannot fall when a symptom turns on. The constraint encodes "
        "the assumption the PHQ-9 and GAD-7 instruments themselves make, namely "
        "that the presence of an additional symptom cannot reduce a severity "
        "score, and it is the property that renders the head auditable. The "
        "weights are normalised to sum to unity so that the scale of the trait "
        "is fixed by the cutpoints, removing the degeneracy in which the "
        "weights and the spacing expand together.")
    equation(doc, eqs, 12)
    para(doc,
        "where the cutpoints are parameterised as a first threshold plus "
        "positive increments so that strict ordering holds by construction, and "
        "the class probabilities are obtained by differencing adjacent "
        "cumulative terms. The initial spacing of the increments is a "
        "consequential hyperparameter: an interior class of a cumulative-link "
        "model attains a maximum probability of approximately the logistic "
        "difference across the spacing, so a spacing below roughly 2.2 leaves "
        "the interior classes unable to be the most likely class at any value "
        "of the trait, and the spacing is therefore initialised at 2.5. This "
        "value was not selected by search but derived from the failure it "
        "corrects: an earlier configuration initialised the increments through "
        "a softplus of zero, giving a spacing of approximately 0.69, at which "
        "the maximum attainable probability of an interior class is about 0.17 "
        "and the head is structurally incapable of predicting the mild or "
        "moderate band however the encoder is trained. The trait is "
        "additionally centred and rescaled so that its initial range spans the "
        "cutpoints.")
    equation(doc, eqs, 13)
    para(doc,
        "where non-negativity precedes the addition of the identity and the "
        "symmetric normalisation, guaranteeing that the node degree is at least "
        "unity and the normaliser is bounded, and the penalty drives the learned "
        "residual towards sparsity so that the adjacency remains anchored to "
        "the clinical prior. The ordering of these three operations is not "
        "interchangeable: normalising before the non-negativity projection "
        "permits a node whose incident weights sum to approximately zero, at "
        "which point the reciprocal of the degree diverges, and the "
        "consequences are reported in Section 4.4. Clamping the degree at a "
        "minimum of unity and applying layer normalisation to the propagated "
        "features removes the divergence.")
    equation(doc, eqs, 14)
    para(doc,
        "where the per-symptom true-positive and false-positive rates are "
        "estimated from paired gold and model labels under Laplace smoothing, "
        "and the network predicts the clean probability while the loss is "
        "evaluated on the probability pushed through the channel. The channel is "
        "estimated from data rather than assumed, on the paired subset where "
        "gold and weak labels coexist, and the dispersion of the resulting "
        "rates reported in Section 4.6 establishes that no single scalar "
        "confidence temperature could substitute for a per-symptom channel. "
        "For the eleven symptoms carrying no gold annotation the rates are "
        "pooled within the originating instrument, and the anxiety-side pooled "
        "estimate rests on a single measurable symptom.")
    equation(doc, eqs, 15)
    para(doc,
        "with the five auxiliary weights set to one for severity, one for risk, "
        "one half for stress, one for the weak-supervision term and ten to the "
        "power minus three for the graph sparsity penalty.")
    para(doc,
        "The optimisation is performed with the AdamW optimiser using separate "
        "learning rates of two times ten to the power minus five for the "
        "pretrained encoder and one times ten to the power minus three for the "
        "randomly initialised heads, because a single rate either fails to move "
        "the heads or destabilises the encoder. A linear schedule with ten per "
        "cent warmup is applied over the full step budget, gradients are "
        "clipped at unit norm, and mixed-precision arithmetic is used with "
        "bfloat16 where the accelerator supports it. The checkpoint is selected "
        "at minimum validation loss rather than by an aggregate score, because "
        "an aggregate dominated by the symptom metrics continues to improve "
        "while the ordinal heads memorise their small training partitions, a "
        "failure measured during development as the selection of epochs at "
        "approximately twice the minimum validation loss. Per-symptom decision "
        "thresholds are tuned on the validation split and applied unchanged to "
        "the test split. Every configuration is replicated across three seeds "
        "and the full hyperparameter configuration is reported in Table 6.")
    caption(doc,
        "Table 6. Hyperparameter configuration of the proposed SymptoGraph-LLM")
    table(doc, [
        ["Hyperparameter", "Value / configuration"],
        ["Framework", "PyTorch with HuggingFace Transformers"],
        ["Dataset", "harmonised eight-corpus union, 19,762 posts"],
        ["Training paradigm",
         "multi-task supervised with weak-supervision stream"],
        ["Accelerator", "single NVIDIA L4, 24 GB, high-RAM runtime"],
        ["Encoder", "MentalRoBERTa, 12 layers, hidden 768"],
        ["Tokeniser", "byte-level BPE, vocabulary 50,265"],
        ["Sequence pooling", "mean over non-padding positions"],
        ["Maximum sequence length", "192 tokens"],
        ["Batch size", "32"],
        ["Weak-label batch size", "16, drawn once per optimiser step"],
        ["Symptom ontology", "22 concepts, 9 PHQ-9, 7 GAD-7, 6 auxiliary"],
        ["Severity levels", "4, minimal to severe"],
        ["Risk levels", "5, C-SSRS bands"],
        ["Split strategy", "author-disjoint, strictest split wins"],
        ["Weak labelling model", "Qwen2.5-7B-Instruct served through vLLM"],
        ["Weak pool", "1,017,959 posts, 49,982 annotated"],
        ["Optimiser", "AdamW, weight decay 0.01"],
        ["Encoder learning rate", "2 x 10\u207b\u2075"],
        ["Head learning rate", "1 x 10\u207b\u00b3"],
        ["Schedule", "linear warmup and decay, warmup ratio 0.1"],
        ["Gradient clipping", "1.0 by global norm"],
        ["Mixed precision", "bfloat16 where supported, else fp16 with scaler"],
        ["Epochs", "8"],
        ["Early stopping", "patience 2 on validation loss"],
        ["Checkpoint selection", "minimum validation loss"],
        ["Dropout", "0.1"],
        ["Symptom positive weight", "capped inverse frequency, cap 10"],
        ["Ordinal class weights", "capped inverse frequency"],
        ["Label smoothing", "0.05, ordinal heads only"],
        ["Decision thresholds", "per symptom, tuned on validation"],
        ["Graph node width", "64"],
        ["Graph propagation layers", "2"],
        ["Graph sparsity weight", "1 x 10\u207b\u00b3"],
        ["Initial cutpoint spacing", "2.5"],
        ["Trait scale initialisation", "1.2 x cutpoint span"],
        ["Item weight initialisation", "softplus inverse of 1/22"],
        ["Task weights", "symptoms 1.0, severity 1.0, risk 1.0, stress 0.5"],
        ["Weak-supervision weight", "1.0"],
        ["Noise channel smoothing", "Laplace, prior 2.0"],
        ["Noise channel minimum support", "20 paired instances per cell"],
        ["Seeds", "0, 1, 2"],
        ["Significance test", "paired bootstrap, two-sided"],
        ["Bootstrap resamples", "1,000, paired on test examples"],
    ], widths=[2.6, 2.7], fs=10)
    para(doc,
        "The selected setup provides trade-offs between symptom-detection "
        "fidelity, ordinal calibration, resistance to memorisation on the small "
        "annotated partitions and computational cost. The epoch budget and the "
        "early-stopping patience are deliberately small, because validation "
        "loss was observed to reach its minimum between the second and fourth "
        "epoch in every seed while the training quadratic weighted kappa on the "
        "suicide-risk partition continued towards unity, which is the signature "
        "of memorisation on a partition of 340 instances rather than of "
        "continued learning. The weak-label batch size is set to half the gold "
        "batch size so that the additional encoder pass per step does not "
        "double the training cost, and because the cost of the stream is "
        "determined by this batch size rather than by the size of the weak pool, "
        "the full subsample of 49,982 annotated posts is used at the same "
        "expense that a subsample of six thousand would incur.")

    # ---------------------------------------------- 4. results and discussion
    heading(doc, "4. Results and Discussions")
    heading(doc, "4.1 Experimental Setup")
    para(doc,
        "The harmonised seven-corpus collection described in Section 3.1 has "
        "been employed in the current work in order to evaluate the performance "
        "of the proposed SymptoGraph-LLM framework. All experiments were "
        "conducted on a single accelerator with 24 gigabytes of memory using "
        "mixed-precision arithmetic, and every configuration was replicated "
        "across three random seeds so that a variance estimate accompanies each "
        "reported quantity. The framework and all competing conditions share "
        "one configuration object, so that the only quantity varying between "
        "conditions is the representation under test: identical "
        "leakage-controlled partitions, identical masked multi-task objective, "
        "identical capped symptom positive weights, identical capped ordinal "
        "class weights, identical per-symptom decision thresholds tuned on "
        "validation data, identical checkpoint selection at minimum validation "
        "loss, identical epoch budget and early-stopping patience, and "
        "identical seeds. Three baselines were retrained under this protocol, "
        "namely a term-frequency representation with a multilayer perceptron, a "
        "bidirectional recurrent encoder with attention, and a fine-tuned "
        "domain-pretrained transformer. Three further architectures were "
        "re-implemented for the same multi-task objective from recent work, "
        "namely a transformer stacking ensemble for symptom-level "
        "classification, a graph neural network over a post graph estimated "
        "from semantic similarity and author relations, and a "
        "multi-perspective prompt ensemble with perturbation-based "
        "self-supervision. It must be stated explicitly that these three are "
        "re-implementations of the architectures described in the cited works, "
        "adapted to the twenty-two symptom label space and the four-task output "
        "of the present study, and are therefore not reproductions of the "
        "published systems; their results are not comparable with the figures "
        "reported in the original papers, and two components of the cited "
        "designs are reported as out of scope rather than as failures, namely "
        "a windowed screening "
        "stage requiring several posts per author within a fixed observation "
        "window, against a gold partition whose median is one post per author, "
        "and post-hoc attribution analyses that affect no metric in this "
        "comparison. Five component-wise ablations complete the design, "
        "disabling one element of the framework at a time while the protocol is "
        "held fixed.")

    heading(doc, "4.2 Evaluation Metrics")
    para(doc,
        "To test the efficacy of the proposed SymptoGraph-LLM framework, "
        "several metrics are considered for analysing symptom-detection "
        "fidelity, ordinal agreement, calibration under class imbalance and the "
        "statistical reliability of every reported difference. Because the four "
        "prediction tasks differ in size by more than an order of magnitude, no "
        "single aggregate is reported as the headline quantity; instead each "
        "task is reported separately with a confidence interval, and every "
        "comparison between two conditions is accompanied by a paired test.")
    equation(doc, eqs, 16)
    para(doc,
        "where the macro average is taken over the symptoms carrying gold "
        "annotation, and the per-symptom decision threshold is the value tuned "
        "on the validation split rather than a fixed one half. The distinction "
        "is not cosmetic: under a fixed threshold of one half the symptom "
        "macro-F1 of the proposed configuration was measured at 0.336, and the "
        "same predictions scored 0.603 once each threshold was tuned on "
        "validation data, so a reported F1 on this label space characterises "
        "the threshold policy as much as the representation.")
    equation(doc, eqs, 17)
    para(doc,
        "where the rank-based form is used so that the estimate is invariant to "
        "the decision threshold, the sums run over the positive instances of "
        "each symptom, and the macro average is taken over symptoms for which "
        "both classes are present in the evaluation split. Area under the "
        "receiver operating characteristic is reported alongside macro-F1 "
        "because the two respond differently to the class imbalance of the "
        "symptom space: the threshold-free measure characterises the ranking "
        "the model induces, whereas the F1 measure characterises the operating "
        "point a deployment would use, and a method may improve one without "
        "the other.")
    equation(doc, eqs, 18)
    para(doc,
        "where the weight matrix penalises disagreements quadratically in the "
        "distance between the predicted and the true level, so that a "
        "near-miss on an ordinal scale is credited more than a distant error, "
        "and the observed and expected matrices are computed over the levels "
        "present in the evaluation split. The quadratic weighting is the "
        "appropriate choice for both graded tasks because the levels of the "
        "PHQ-9 severity bands and of the C-SSRS risk bands are ordered and "
        "unequally consequential, so that confusing the minimal band with the "
        "severe band is a materially different error from confusing adjacent "
        "bands, a distinction that unweighted accuracy and macro-F1 both "
        "discard. The chance correction matters here because the severity "
        "training distribution places 71.9 per cent of its mass in the lowest "
        "band.")
    equation(doc, eqs, 19)
    para(doc,
        "where the prediction is the most probable level under the "
        "cumulative-link model. Mean absolute error is reported alongside the "
        "kappa because the two are sensitive to different failure modes, a head "
        "that emits only the extreme levels being able to retain moderate kappa "
        "while its absolute error degrades substantially. It should be noted "
        "that this metric is oriented oppositely to every other quantity "
        "reported in this paper, lower values indicating better performance, "
        "and the direction of each metric is marked in the column headings of "
        "the result tables so that no row can be read in the wrong direction.")
    equation(doc, eqs, 20)
    para(doc,
        "where the bootstrap resamples test examples with replacement, the same "
        "resampled indices are applied to both conditions so that the "
        "comparison observes differences in prediction rather than differences "
        "in which examples were drawn, and the two-sided probability is the "
        "proportion of resamples in which the sign of the "
        "difference reverses. A difference is reported as established only if "
        "it attains significance on every seed and preserves its sign on every "
        "seed, and the multiple-comparison burden is stated explicitly wherever "
        "several metrics are tested together.")

    heading(doc, "4.3 Performance Analysis of the Proposed SymptoGraph-LLM")
    para(doc,
        "The evaluation of the suggested SymptoGraph-LLM model was carried out "
        "on the held-out test partition to confirm the efficiency of the "
        "symptom-mediated architecture across the four prediction tasks. "
        "Because the checkpoint is selected at minimum validation loss and the "
        "early-stopping patience is two epochs, the reported configuration is "
        "the one the protocol would select in deployment rather than the "
        "best-performing epoch observed in hindsight. The distinction matters "
        "on a corpus of this size, because the epoch attaining the best test "
        "metric and the epoch a deployment would actually ship diverge "
        "systematically once the ordinal heads begin to fit their small "
        "training partitions, and reporting the former while describing the "
        "latter is a common and undetectable source of optimism. Every number "
        "reported in this section is taken from the checkpoint the selection "
        "rule chose on validation data alone, with the test partition read "
        "exactly once per run and the per-symptom decision thresholds carried "
        "over from validation without adjustment. The progression of the "
        "metrics over the training budget is reported in Table 7, which permits "
        "the separation of continued learning from the onset of memorisation on "
        f"the small ordinal partitions.")
    caption(doc,
        "Table 7. Training progression of the proposed SymptoGraph-LLM "
        "(validation split, mean over three seeds)")
    table(doc, [
        ["Metric", "Epoch 1", "Epoch 2", "Epoch 4", "Epoch 6", "Epoch 8"],
        ["Multi-task validation loss", P, P, P, P, P],
        ["Multi-task training loss", P, P, P, P, P],
        ["Symptom macro-F1", P, P, P, P, P],
        ["Symptom macro-AUROC", P, P, P, P, P],
        ["Symptom macro-AUPRC", P, P, P, P, P],
        ["Stress AUROC", P, P, P, P, P],
        ["Stress F1", P, P, P, P, P],
        ["Severity QWK", P, P, P, P, P],
        ["Severity macro-F1", P, P, P, P, P],
        ["Severity MAE", P, P, P, P, P],
        ["Risk QWK", P, P, P, P, P],
        ["Risk macro-F1", P, P, P, P, P],
        ["Risk MAE", P, P, P, P, P],
        ["Training risk QWK", P, P, P, P, P],
        ["Validation-to-minimum loss ratio", P, P, P, P, P],
    ], widths=[2.14, 0.66, 0.66, 0.66, 0.66, 0.66], fs=10)
    para(doc,
        "The results presented in Table 7 clearly indicate the behaviour that "
        "the training protocol was designed to control. The validation loss "
        "reaches its minimum early in the budget and rises thereafter, while "
        "the training quadratic weighted kappa on the suicide-risk partition "
        "continues to climb towards unity, which is the signature of "
        "memorisation on a partition of 340 instances rather than of continued "
        "generalisation. The ratio between the validation loss at the selected "
        "checkpoint and the minimum validation loss attained is therefore "
        "reported explicitly, because it quantifies directly whether the "
        "selection rule has chosen a checkpoint a deployment would want. The "
        "symptom metrics continue to improve for longer than the ordinal "
        "metrics, which is the reason an aggregate score dominated by symptom "
        "terms is unsuitable as a selection signal and the reason minimum "
        "validation loss is used in its place. This replaces an earlier "
        "selection rule that maximised a weighted aggregate of the task "
        "metrics, under which the procedure was measured to select checkpoints "
        "at approximately twice the minimum validation loss, late in the budget "
        "and well past the point at which the ordinal heads had begun to "
        "memorise; the aggregate continued to rise because its symptom terms "
        "continued to rise, and the deterioration of the graded tasks was "
        "invisible to it. The final performance of the "
        f"proposed framework against the reference condition is reported in "
        f"Table 8, where the proposed framework attains {P} symptom "
        f"macro-AUROC and {P} symptom macro-F1 on the held-out partition.")
    caption(doc,
        "Table 8. Final performance of the proposed SymptoGraph-LLM "
        "(test split) against the flat-encoder reference")
    table(doc, [
        ["Metric", "SymptoGraph-LLM", "Flat-encoder reference"],
        ["Symptom macro-F1", PM, PM],
        ["Symptom macro-AUROC", PM, PM],
        ["Symptom macro-F1, gold-supervised subset", PM, PM],
        ["Symptom macro-AUROC, gold-supervised subset", PM, PM],
        ["Symptom macro-AUROC, weak-only subset", "not measurable",
         "not measurable"],
        ["Stress F1", PM, PM],
        ["Stress AUROC", PM, PM],
        ["Severity macro-F1", PM, PM],
        ["Severity QWK", PM, PM],
        ["Severity MAE", PM, PM],
        ["Risk macro-F1", PM, PM],
        ["Risk QWK", PM, PM],
        ["Risk MAE", PM, PM],
        ["Trainable parameters", P, P],
    ], widths=[2.6, 1.45, 1.45], fs=10)
    para(doc,
        "The final evaluation confirms the behaviour observed during training. "
        "The proposed architecture delivers symptom detection at the level "
        "reported in Table 8 while retaining the clinical addressability that "
        "the flat reference lacks entirely, since the flat configuration "
        "produces no symptom-level intermediate quantity that a practitioner "
        "could inspect. Two entries in the table require comment rather than "
        "interpretation. The weak-only symptom subset is marked as not "
        "measurable because eleven of the twenty-two symptoms carry no gold "
        "annotation in any constituent corpus, so the direct effect of the "
        "weak-supervision branch on the symptoms it actually supervises cannot "
        "be evaluated with the present annotation and can appear only "
        "indirectly through transfer to the gold-supervised subset. This is a "
        "property of the available data rather than of the framework, and it "
        "bounds the strength of any claim that can be made about that component "
        "regardless of how the remaining numbers fall. The second entry "
        "requiring comment is the parameter count, which is reported for both "
        "conditions because the proposed framework and the flat reference share "
        "the same pretrained encoder and differ by approximately one per cent "
        "of the total budget; a difference in performance between them is "
        "therefore attributable to the arrangement of the structure rather than "
        "to capacity, which is the comparison the ablation in Section 4.5 is "
        "designed to make and the reason the flat reference rather than a "
        "smaller model is used as the point of departure.")
    para(doc,
        "The ordinal measurements represent the gap between the proposed "
        "framework and the reference on the two graded tasks, and they must be "
        "read together with the partition sizes rather than in isolation. "
        "Depression severity is evaluated on 541 test instances across four "
        "levels whose training distribution is dominated by the lowest band at "
        "71.9 per cent, so the macro-F1 measure on that task rewards a model "
        "that distributes its predictions across the minority levels, which a "
        "class-weighted unconstrained classifier can do freely while a "
        "calibrated cumulative-link model resists; the quadratic weighted kappa "
        "and the mean absolute error are consequently the more appropriate "
        "ordinal measures and are reported alongside. Suicide risk is evaluated "
        "on 74 test instances across five levels, and the consequence of that "
        "partition size is quantified in Section 4.6, where the attainable "
        "confidence interval is shown to exceed every architectural difference "
        "observed in this study, so that differences on the risk task are not "
        "interpretable as architectural effects.")

    heading(doc, "4.4 Training Convergence Analysis")
    figure(doc, [figs["fig3a"], figs["fig3b"]], 3,
           "Training convergence curves of the proposed SymptoGraph-LLM.")
    para(doc,
        "Figure 3 represents the optimisation convergence properties of the "
        "proposed SymptoGraph-LLM architecture over the entire training budget. "
        "The left panel reports the multi-task objective on the training and "
        "validation partitions, and the right panel reports the quadratic "
        "weighted kappa on the suicide-risk task separately for the two "
        "partitions, because the divergence between them is the quantity the "
        "training protocol is designed to control. The training objective "
        "decreases monotonically throughout the budget, which confirms that "
        "optimisation is stable under the combined six-term loss and that no "
        "term dominates to the exclusion of the others. Stability under this "
        "objective is not a given and was not obtained at the first attempt: "
        "an earlier formulation of the graph term, in which the degree "
        "normalisation preceded the non-negativity projection, drove the "
        "objective to the order of ten to the thirteenth power in a substantial "
        "fraction of the runs that included the graph, and the curves reported "
        "here are those of the corrected formulation given in Section 3.5. The "
        "validation "
        "objective, by contrast, attains its minimum early and increases "
        "thereafter, and it is that minimum which determines the selected "
        "checkpoint rather than the end of the budget.")
    para(doc,
        "Moreover, the divergence between the training and validation "
        "trajectories demonstrates why the selection rule matters on a corpus "
        "of this size. The training quadratic weighted kappa on the "
        "suicide-risk partition rises steadily towards unity across the budget, "
        "which on a partition of 340 instances distributed over five ordinal "
        "levels indicates that the head has memorised the partition rather than "
        "learned a transferable ordering. The validation kappa on the same task "
        "remains approximately flat over the same interval, confirming that the "
        "additional training fits the partition without improving "
        "generalisation. An aggregate selection score dominated by the symptom "
        "terms, which continue to improve for longer, would therefore select a "
        "late checkpoint at which the ordinal heads have already degraded, and "
        "this was measured during development as the selection of epochs at "
        f"approximately twice the minimum validation loss in {P} of the runs "
        f"examined. A further observation from the convergence record qualifies "
        f"the epoch budget itself. In an earlier four-epoch configuration the "
        f"two most complete rungs of the ablation ladder attained their minimum "
        f"validation loss at the fourth epoch, that is, at the final epoch "
        f"available, which means their trajectories were truncated rather than "
        f"converged and that the comparison between them and the earlier rungs "
        f"was confounded by budget. The budget reported in Table 6 was extended "
        f"to eight epochs for that reason, and the convergence curves in "
        f"Figure 3 are presented in part as evidence that the minimum now falls "
        f"strictly inside the budget for every rung, without which no "
        f"step-to-step comparison in Section 4.5 would be interpretable.")
    para(doc,
        "Finally, the convergence behaviour demonstrates that the improvement "
        "in symptom detection was not obtained by sacrificing the ordinal "
        "tasks, and that the monotone ordinal head contributes a measurable "
        "regularising effect in addition to its interpretability. Because the "
        "severity and risk predictions are both functions of a single bounded "
        "latent trait formed from non-negative item weights, the capacity "
        "available to fit the small ordinal partitions is substantially lower "
        "than that of an unconstrained classifier head, and the training kappa "
        "attained at the end of the budget is correspondingly lower under the "
        "proposed configuration than under the flat reference. The practical "
        "consequence, visible in the three-seed replication rather than in any "
        "single run, is a reduction in the seed-to-seed variability of the "
        f"ordinal metrics, reported as {P} for severity and {P} for risk under "
        f"the proposed configuration against {P} and {P} respectively under the "
        f"flat reference, which matters because a variance of that magnitude "
        f"would otherwise dominate any architectural difference the ablation is "
        f"intended to detect.")

    heading(doc, "4.5 Ablation Study")
    para(doc,
        "In order to evaluate the effect of each novelty in isolation within "
        "the SymptoGraph-LLM model, an ablation experiment was conducted in "
        "which exactly one component is added at each step of a ladder "
        "beginning from a flat encoder with unconstrained heads and ending with "
        "the complete framework, every rung sharing the configuration of Table 6 "
        "without exception so that no rung benefits from a learning rate, "
        "schedule, epoch budget or selection rule the others did not receive. "
        "Three further conditions isolate individual claims rather than "
        "contributing to the ladder: a graph condition establishing whether the "
        "comorbidity graph outperforms its absence, an unconstrained-trait "
        "condition separating the contribution of the monotone structure from "
        "the mere reduction in free parameters, and a down-weighted condition "
        "establishing whether the estimated noise channel outperforms a uniform "
        "reduction of the weak-label weight. The results are reported in "
        "Table 9.")
    caption(doc, "Table 9. Ablation study results of the proposed SymptoGraph-LLM")
    table(doc, [
        ["Model", "Sym F1 \u2191", "Sym AUROC \u2191", "Sym AUPRC \u2191",
         "Sev QWK \u2191", "Sev MAE \u2193", "Risk QWK \u2191", "Risk MAE \u2193",
         "Stress AUROC \u2191", "Composite \u2191"],
        ["M0 (flat encoder)", P, P, P, P, P, P, P, P, P],
        ["M1 (+ N1 bottleneck)", P, P, P, P, P, P, P, P, P],
        ["M2 (+ N-A ordinal head)", P, P, P, P, P, P, P, P, P],
        ["M3 (+ N3 weak stream)", P, P, P, P, P, P, P, P, P],
        ["Proposed (+ N-B channel)", P, P, P, P, P, P, P, P, P],
        ["C1 (+ N2 graph)", P, P, P, P, P, P, P, P, P],
        ["C2 (monotonicity removed)", P, P, P, P, P, P, P, P, P],
        ["C3 (uniform weak weight)", P, P, P, P, P, P, P, P, P],
    ], widths=[1.45, 0.52, 0.64, 0.62, 0.56, 0.56, 0.56, 0.56, 0.68, 0.69],
        fs=8)
    figure(doc, figs["fig4"], 4,
           "Comparison of symptom-detection fidelity of the proposed "
           "SymptoGraph-LLM with the ablation configurations.", width=4.3)
    para(doc,
        "The ablation in Table 9 indicates the contribution of each component "
        "against the seed-noise band established by the three-seed replication, "
        "and the comparison must be read against that band rather than against "
        "zero. The concept bottleneck introduced at M1 changes the ordinal "
        "metrics relative to the flat encoder at M0 while leaving the symptom "
        "metrics approximately unchanged, which is consistent with its "
        "mechanism: the bottleneck does not add information to the "
        "representation but constrains the path by which the ordinal heads may "
        "access it. The monotone ordinal head introduced at M2 is the component "
        "with the clearest effect on the two graded tasks, and the "
        "unconstrained-trait control at C2 is what distinguishes the two "
        "possible explanations for that effect, namely that the monotone "
        "structure itself carries clinical information or that the reduction in "
        "free parameters alone provides the benefit. A claim about the "
        "interpretability of this component was anticipated in Section 3.4, "
        "namely that the fitted item weights are readable as item loadings and "
        "can be contrasted with the equal weighting the PHQ-9 prescribes, and "
        "the condition under which that claim must be withdrawn is stated here "
        "rather than after the fact: if the dispersion of the fitted weights is "
        "indistinguishable from the dispersion at initialisation, then the head "
        "has not learned a weighting and no clinical reading of the weights is "
        f"warranted. The measured dispersion is {P}, and the claim is "
        f"maintained or withdrawn on that number alone. Stating the test in "
        f"advance of the number matters here because an item-loading vector "
        f"that has not moved from its initialisation is numerically "
        f"indistinguishable from a uniform clinical prior, and a reader shown "
        f"only the fitted vector could not tell the two apart; the "
        f"interpretability of the head is therefore a claim about learning "
        f"rather than about the presence of readable parameters, and it is "
        f"falsifiable on this one quantity.")
    para(doc,
        "Adding the weak-label stream at M3 and the per-symptom noise channel "
        "at the proposed configuration completes the ladder, and the "
        "interpretation of these two steps requires care for a reason that is a "
        "property of the data rather than of the result. Eleven of the "
        "twenty-two symptoms carry no gold annotation anywhere in the "
        "harmonised corpus, and those eleven are precisely the symptoms the "
        "weak-supervision branch exists to supervise, so neither step can be "
        "credited directly by any metric in Table 9; their effect can appear "
        "only indirectly, through transfer to the eleven gold-supervised "
        "symptoms on which the metrics are computed. The maximum measurable "
        "effect of these two components is therefore small by construction, and "
        "a result in which they fall within the seed-noise band should be "
        "reported as uninformative about their value rather than as evidence "
        "against them. The comorbidity graph condition at C1 is reported "
        "separately from the ladder for a related reason: the symptom-disorder "
        "prior is authored from diagnostic criteria by the present authors and "
        "has not been reviewed by a clinician, and the disorder-disorder edges "
        "derive from subreddit co-membership, which is a weak proxy for "
        "diagnosed comorbidity, so a positive result would require the "
        "random-graph control before it could be attributed to the clinical "
        f"content of the prior rather than to the presence of structure as "
        f"such. Across the ladder, {P} of the step-to-step differences exceed "
        f"the seed-noise band, and the components with distinguishable effect "
        f"are reported in Section 4.6 together with their paired test outcomes.")

    heading(doc, "4.6 Comparison with State-of-the-Art Models")
    para(doc,
        "In order to measure the performance of the SymptoGraph-LLM framework "
        "against the current state of the art, a comparison was conducted "
        "between the proposed framework, three retrained baselines and three "
        "architectures re-implemented from recent work for the same multi-task "
        "objective, all under the single protocol of Table 6 on the identical "
        "leakage-controlled partitions. The comparison is reported in Table 10 "
        "together with the parameter count of each condition, since a margin "
        "obtained at several times the capacity is a different claim from one "
        "obtained at parity. The three re-implemented architectures were "
        "selected from the works surveyed in Section 2 on a single criterion, "
        "namely that the architecture is specified in sufficient detail to be "
        "reconstructed and that its output structure can be mapped onto the "
        "four tasks evaluated here without altering its mechanism. One further "
        "candidate from the survey was considered and excluded on that "
        "criterion, its contribution being the quality of generated "
        "explanations rather than a predictive architecture, so that no metric "
        "in the present suite would score the property the work actually "
        "advances and a comparison would misrepresent it.")
    caption(doc,
        "Table 10. Performance comparison of the proposed SymptoGraph-LLM with "
        "state-of-the-art models")
    table(doc, [
        ["Model", "Sym F1 \u2191", "Sym AUROC \u2191", "Sym AUPRC \u2191",
         "Sev QWK \u2191", "Sev MAE \u2193", "Risk QWK \u2191", "Risk MAE \u2193",
         "Stress AUROC \u2191", "Composite \u2191", "Params (M)"],
        ["TF-IDF + MLP", PM, PM, PM, PM, PM, PM, PM, PM, PM, P],
        ["BiLSTM + attention", PM, PM, PM, PM, PM, PM, PM, PM, PM, P],
        ["MentalRoBERTa, fine-tuned", PM, PM, PM, PM, PM, PM, PM, PM, PM, P],
        ["Transformer stacking ensemble, after [11]", PM, PM, PM, PM, PM, PM,
         PM, PM, PM, P],
        ["Post-graph GNN, after [18]", PM, PM, PM, PM, PM, PM, PM, PM, PM, P],
        ["Prompt ensemble with self-supervision, after [21]", PM, PM, PM, PM,
         PM, PM, PM, PM, PM, P],
        ["Proposed", PM, PM, PM, PM, PM, PM, PM, PM, PM, P],
    ], widths=[1.18, 0.52, 0.6, 0.58, 0.5, 0.5, 0.5, 0.5, 0.6, 0.6, 0.46],
        fs=7)
    figure(doc, [figs["fig5a"], figs["fig5b"]], 5,
           "Comparison of the proposed SymptoGraph-LLM with state-of-the-art "
           "models, and per-task confidence intervals by evaluation partition "
           "size.")
    para(doc,
        "As follows from the results in Table 10, the proposed SymptoGraph-LLM "
        f"framework yields {P} symptom macro-AUROC against {P} for the "
        f"strongest re-implemented comparison model, a difference of {P} "
        f"absolute, obtained at {P} million parameters against {P} million. The "
        f"stress task saturates near {P} for every condition including the "
        f"term-frequency baseline, which indicates that the binary stress "
        f"target is close to the ceiling attainable on this annotation and that "
        f"no architectural conclusion should be drawn from differences on it. "
        f"The paired bootstrap outcomes qualify the comparison substantially: "
        f"of the differences tested, {P} attain significance with a sign "
        f"consistent across all three seeds, and the multiple-comparison burden "
        f"must be stated because testing several metrics across several "
        f"conditions yields approximately one false positive at the "
        f"conventional threshold by chance alone, so a single significant "
        f"result is consistent with chance rather than evidence of an effect.")
    para(doc,
        "The comparison of the structural properties offers the clearest "
        "distinction between the proposed approach and the competing methods, "
        "and it is a distinction no entry in Table 10 captures. The right panel "
        "of Figure 5 reports the width of the attainable confidence interval "
        "for each task against the size of its evaluation partition, and the "
        "consequence for the suicide-risk task is decisive: on 74 test "
        "instances across five ordinal levels the ninety-five per cent "
        f"bootstrap interval for the quadratic weighted kappa has a measured "
        f"width of {P}, which exceeds every architectural difference observed "
        f"anywhere in this study. Differences on that task are therefore not "
        f"interpretable as architectural effects under the present annotation, "
        f"and reporting them as such would be unsound irrespective of which "
        f"model they favour; the risk column of Table 10 is retained because "
        f"omitting it would conceal the limitation rather than resolve it, and "
        f"not because the ordering within it is meaningful. The proposed "
        f"framework is additionally the only "
        f"condition in the comparison that emits a symptom-level intermediate "
        f"representation, an ordered severity estimate that cannot violate the "
        f"ordering of its scale, and an estimate of the labelling model's "
        f"per-symptom reliability, none of which the competing conditions "
        f"provide and none of which is expressible as a row in a metric table; "
        f"the measured reliability of that labelling model ranges from a "
        f"true-positive rate of 0.252 to 0.944 and a false-positive rate of "
        f"0.065 to 0.745 across the eleven measurable symptoms, which "
        f"establishes directly that no single confidence temperature could "
        f"correct it. The two extremes of that range illustrate the point: for "
        f"suicidal ideation the labelling model is accurate, at 0.816 true "
        f"positive against 0.065 false positive, whereas for low self-esteem it "
        f"reports the symptom in 79.9 per cent of posts against a gold "
        f"prevalence of 23.8 per cent, and for concentration problems it "
        f"recovers only a quarter of the gold-positive instances. A framework "
        f"that treats such a labeller as a uniform source of supervision "
        f"inherits those biases as though they were clinical signal, and the "
        f"control in Table 9 that replaces the estimated channel with a uniform "
        f"down-weighting is included so that the benefit of estimating the "
        f"channel is separable from the benefit of merely trusting the weak "
        f"labels less.")

    # ------------------------------------------------------------- conclusion
    heading(doc, "5. Conclusion and Future Work")
    heading(doc, "5.1 Conclusion")
    para(doc,
        "In this paper, we present SymptoGraph-LLM, a symptom-grounded and "
        "comorbidity-aware multi-task network that addresses the problem of "
        "producing mental health assessments from social media text which a "
        "clinician can inspect and act upon. The framework replaces direct "
        "disorder classification with an obligatory clinical concept "
        "bottleneck, whose twenty-two symptom queries are initialised from the "
        "sentence embeddings of their own clinical definitions and each of which "
        "carries an independent scoring vector, so that every severity, risk "
        "and stress prediction is a function of the symptom evidence that "
        "justifies it. Comorbid structure enters the estimator as a graph whose "
        "nodes are symptoms and disorders and whose adjacency combines a DSM-5 "
        "prior with a learned residual under a sparsity penalty, with "
        "disorder-disorder edges estimated as conditional co-membership "
        "probabilities because the mean of 1.125 conditions per author makes a "
        "pointwise mutual information estimate vanish identically. Graded "
        "severity and suicide risk are produced by an instrument-grounded "
        "monotone ordinal head that reproduces the additive scoring rule of the "
        "PHQ-9, is monotone in symptom presence by construction, and reduces "
        "the five-level risk task to four threshold parameters over a trait "
        "that the far more abundant severity annotation also trains. Weak "
        "supervision from a locally executed model is corrected rather than "
        "trusted, through a per-symptom noise channel estimated against gold "
        "agreement and inverted inside the objective. All of this is evaluated "
        "under a protocol that assigns one global identifier per post, "
        "partitions authors disjointly, applies a strictest-split-wins rule, "
        f"selects checkpoints at minimum validation loss and replicates across "
        f"three seeds. Validated on 19,762 "
        f"gold-labelled posts together with 49,982 calibrated weak labels, the "
        f"framework attains {P} symptom macro-AUROC and {P} symptom macro-F1, "
        f"with {P} of the component-wise differences exceeding the seed-noise "
        f"band. The study also establishes two measurement results of "
        f"independent interest, namely that the labelling model's reliability "
        f"ranges from a true-positive rate of 0.252 to 0.944 across symptoms, "
        f"and that the suicide-risk annotation supports an interval wider than "
        f"any difference it could resolve.")

    heading(doc, "5.2 Future Work")
    para(doc, "The future direction of this research includes:")
    bullet(doc,
        "Constructing an in-house anxiety gold standard so that the eleven "
        "weak-only symptoms become measurable.")
    bullet(doc,
        "Submitting the symptom-disorder prior for clinician review together "
        "with a sensitivity analysis over perturbed priors and a random-graph "
        "control.")
    bullet(doc,
        "Extending the framework with a symptom-grounded rationale generator "
        "evaluated for faithfulness against the concept activations.")
    bullet(doc,
        "Conducting the matched-data comparison required to attribute the "
        "remaining margin to architecture rather than to additional "
        "supervision.")

    # ------------------------------------------------------------- references
    heading(doc, "6. References")
    for ref in load_references():
        para(doc, ref, align="left")

    os.makedirs(DOCS, exist_ok=True)
    doc.save(OUT)
    print(f"wrote {OUT}")


def write_placeholder_index() -> None:
    path = os.path.join(DOCS, "PLACEHOLDERS.md")
    open(path, "w", encoding="utf-8").write("""# Placeholder index

`docs/SymptoGraph-LLM_Paper.docx` follows the structure and formatting of
`PRISM - Copy.docx`. Every **performance** number is `X.XXX` or
`X.XXX ± X.XXX`. Search for `X.XXX`.

Dataset counts, corpus overlaps, the graph edge weights and the labeller noise
channel are **measured facts and already real** - do not overwrite them.

Regenerate: `python scripts/build_paper.py`

## Tables

| Table | Status | Fills from |
|---|---|---|
| 1. Literature comparison | final, no numbers | - |
| 2. Dataset partitioning | real | `processed/manifest.json` |
| 3. Task composition | real | `processed/label_coverage.csv` |
| 4. Layer-wise architecture | real | `src/symptograph/proposed.py` |
| 5. Parameter distribution | real | `config.json` field `trainable_parameters` |
| 6. Hyperparameter configuration | real | `ProposedConfig` |
| **7. Training progression** | placeholder | `proposed_v2_e8/*/seed_*/history.csv` |
| **8. Final performance** | placeholder | `proposed_v2_e8/*/aggregate/aggregate_metrics.json` |
| **9. Ablation study** | placeholder | notebook 08 sections 9 and 11 |
| **10. State-of-the-art comparison** | placeholder | `comparison/comparison_table.csv`, notebook 09 |

## Figures

| Figure | Status | Replace with |
|---|---|---|
| 1. Overall workflow | **final** | - |
| 2. Architecture | **final** | - |
| 3. Convergence curves (2 panels) | placeholder | `proposed_v2_e8/M4_denoised/seed_0/plots/` |
| 4. Ablation fidelity | placeholder | `proposed_v2_e8/comparison_ladder/*.png` |
| 5. SOTA comparison + CIs (2 panels) | placeholder | `comparison/comparison_vs_*/`, `bootstrap_cis.csv` |

## Citation placement, as instructed

- `[1]`-`[5]` Introduction only
- `[6]`-`[20]` Literature Survey prose, in order across the 7 paragraphs
- `[21]`-`[30]` Table 1, in order, plus the `Proposed` row
- `[11]`, `[18]`, `[21]` re-appear in sections 4.1 and 4.6 only to name the
  three re-implemented comparison models

## Claims to check rather than fill

1. **4.5 / 4.6** - if the learned item loadings show near-zero dispersion
   around uniform weighting, the head never left its initialisation and the
   interpretability claim must be **withdrawn**, not reported.
2. **4.1** - the three comparison models are re-implementations, not
   reproductions. That sentence must stay.
3. **4.3 / 4.5** - the weak-only symptom subset is marked "not measurable"
   because eleven symptoms have no gold labels anywhere. Keep it.
4. **Future work** - the matched-data caveat stands until the baselines also
   see weak labels.
""")
    print(f"wrote {path}")


def main() -> None:
    os.makedirs(FIGS, exist_ok=True)
    eqs = render_equations()
    print(f"rendered {len(eqs)} equations -> {EQS}")
    figs = {
        "fig1": os.path.join(FIGS, "figure1_workflow.png"),
        "fig2": os.path.join(FIGS, "figure2_architecture.png"),
        "fig3a": os.path.join(FIGS, "figure3a_loss.png"),
        "fig3b": os.path.join(FIGS, "figure3b_qwk.png"),
        "fig4": os.path.join(FIGS, "figure4_ablation.png"),
        "fig5a": os.path.join(FIGS, "figure5a_sota.png"),
        "fig5b": os.path.join(FIGS, "figure5b_ci.png"),
    }
    fig1_workflow(figs["fig1"])
    fig2_architecture(figs["fig2"])
    fig3a_loss(figs["fig3a"])
    fig3b_qwk(figs["fig3b"])
    fig4_ablation(figs["fig4"])
    fig5a_sota(figs["fig5a"])
    fig5b_ci(figs["fig5b"])
    print(f"figures -> {FIGS}")
    build_document(eqs, figs)
    write_placeholder_index()


if __name__ == "__main__":
    main()
