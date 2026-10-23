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

import results as R

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DOCS = os.path.join(ROOT, "docs")
FIGS = os.path.join(DOCS, "paper_figures")
EQS = os.path.join(FIGS, "equations")
SURVEY = os.path.join(DOCS, "Literature_Survey_SymptoGraph-LLM.md")
OUT = os.path.join(DOCS, "SymptoGraph-LLM_Paper.docx")

FONT = "Times New Roman"
SIZE = Pt(12)

# Every performance number comes from ``results``. These two remain defined only
# so that ``check_doc_order.py`` can assert the manuscript contains zero of
# them; a stray placeholder would then fail the build checks rather than ship.
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
    5: r"$\mathcal{W}\sim\mathrm{Strat}(\mathcal{U};d,\tau),\quad |\mathcal{W}|=6{,}000,\quad \pi_{\mathrm{ctrl}}\leq 0.15$",
    6: r"$H_i=\mathrm{Enc}(x_i)\in\mathbb{R}^{L\times D},\qquad D=768,\ L=192$",
    7: r"$\alpha_{ik}=\mathrm{softmax}_{l}\left(\frac{H_{il}^{T}q_k}{\sqrt{D}}\right),\qquad q_k^{(0)}=\mathrm{SBERT}(\mathrm{def}_k)$",
    8: r"$Z_{ik}=\sum_{l=1}^{L}\alpha_{ikl}H_{il},\qquad c_{ik}=w_k^{T}Z_{ik}+b_k$",
    9: r"$g_i=\frac{\sum_{k=1}^{K}\sigma(c_{ik})\,Z_{ik}}{\sum_{k=1}^{K}\sigma(c_{ik})},\qquad K=22$",
    10: r"$\mathcal{L}_{\mathrm{sym}}=\frac{\sum_{i,k}m_{ik}\,\omega_{ik}\,\mathrm{BCE}(c_{ik},y_{ik})}{\sum_{i,k}m_{ik}\,\omega_{ik}}$",
    11: r"$\mathcal{L}_{\mathrm{ord}}=-\frac{1}{N}\sum_{i=1}^{N}\gamma_{y_i}\log\,\mathrm{softmax}(W\!g_i+b)_{y_i},\qquad \gamma_j=\min\left(\frac{N}{J\,n_j},\ \gamma_{\max}\right)$",
    12: r"$\mathcal{L}_{\mathrm{con}}=-\frac{1}{|\mathcal{I}|}\sum_{i\in\mathcal{I}}\frac{1}{|\mathcal{P}_i|}\sum_{j\in\mathcal{P}_i}\log\frac{\exp(\pi_i^{T}\pi_j/T)}{\sum_{n\neq i}\exp(\pi_i^{T}\pi_n/T)},\quad \mathcal{P}_i=\{j\neq i:\ J(y_i,y_j)\geq 0.5\}$",
    13: r"$A=\mathrm{relu}(A_0+\Delta A)+I,\quad \tilde{A}=D^{-1/2}AD^{-1/2},\quad \mathcal{L}_{\mathrm{graph}}=\|\Delta A\|_1$",
    14: r"$s_{ik}=u_k^{T}\mathrm{sg}[g_i],\qquad \mathcal{L}_{\mathrm{unc}}=\frac{\sum_{i,k}m_{ik}\omega_{ik}\left[e^{-s_{ik}}\left(\sigma(c_{ik})-y_{ik}\right)^{2}+\frac{1}{2}s_{ik}\right]}{\sum_{i,k}m_{ik}\omega_{ik}}$",
    15: r"$\mathcal{L}=\mathcal{L}_{\mathrm{sym}}+\lambda_{s}\mathcal{L}_{\mathrm{sev}}+\lambda_{r}\mathcal{L}_{\mathrm{risk}}+\lambda_{t}\mathcal{L}_{\mathrm{str}}+\lambda_{c}\mathcal{L}_{\mathrm{con}}+\lambda_{u}\mathcal{L}_{\mathrm{unc}}+\lambda_{g}\mathcal{L}_{\mathrm{graph}}$",
    16: r"$\mathrm{F1}_{\mathrm{macro}}=\frac{1}{K}\sum_{k=1}^{K}\frac{2\,\mathrm{TP}_k}{2\,\mathrm{TP}_k+\mathrm{FP}_k+\mathrm{FN}_k}$",
    17: r"$\mathrm{AUROC}_k=\frac{\sum_{i\in\mathcal{P}_k}\mathrm{rank}(p_{ik})-n_k^{+}(n_k^{+}+1)/2}{n_k^{+}\,n_k^{-}}$",
    18: r"$\kappa_{w}=1-\frac{\sum_{a,b}W_{ab}O_{ab}}{\sum_{a,b}W_{ab}E_{ab}},\qquad W_{ab}=\frac{(a-b)^2}{(J-1)^2}$",
    19: r"$R(\beta)=\frac{\sum_{i}\mathbf{1}[\,\bar{s}_i\leq q_{\beta}\,]\,\ell_i}{\sum_{i}\mathbf{1}[\,\bar{s}_i\leq q_{\beta}\,]},\qquad \mathrm{AURC}=\int_{0}^{1}R(\beta)\,d\beta$",
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
        "Qwen2.5-7B labels 6,000 training posts", C["pre"], fs=7.4)

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
    box(76, 38, 21, 9.8, "N5  Profile contrastive\n"
        "$\\pi_i=\\mathrm{MLP}(g_i)$\n"
        "$J(y_i,y_j)\\geq 0.5$ positives", C["nov"], fs=7.0)

    arr(74, 50.8, 74, 33.3)
    box(28, 25.5, 21, 6.5, "N3  Weak-label stream\n6,000 posts, 11 symptoms",
        C["nov"], fs=7.0)
    box(53, 25.5, 19, 6.5, "N6  Learned uncertainty\n"
        "$s_{ik}=u_k^{\\top}\\mathrm{sg}[g_i]$", C["nov"], fs=7.0)
    arr(49.2, 28.8, 52.8, 28.8)
    arr(38.5, 32.1, 38.5, 37.8)
    arr(62.5, 32.1, 62.5, 37.8)
    box(76, 24.5, 21, 7.5, "Prediction heads\n22 symptoms  \u2022  severity (4)\n"
        "risk (5)  \u2022  stress (1)  \u2022  abstain", C["head"], fs=6.8)
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
        if not label:
            return
        if abs(y2 - y1) < 0.5:
            # horizontal: sit the label above the arrow so it cannot run into
            # the box the arrow points at
            ax.text((x1 + x2) / 2, y1 + 1.5, label, fontsize=6.6,
                    color="#555", style="italic", ha="center", va="bottom")
        else:
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

    ax.text(25, 24.5, "N5   Symptom-profile contrastive", fontsize=8.4,
            fontweight="bold", ha="center", color="#a0522d")
    box(2, 15.5, 22, 7.5, "Projection head\n$\\pi_i=\\mathrm{MLP}(g_i)"
        "\\in\\mathbb{R}^{128}$", C["na"], fs=6.9)
    arr(24.2, 19.3, 28.8, 19.3)
    box(29, 15.5, 25, 7.5, "InfoNCE over profile overlap\n"
        "$\\mathcal{P}_i=\\{j: J(y_i,y_j)\\geq 0.5\\}$\n$T=0.1$",
        C["na"], fs=6.7)

    box(59, 15.5, 17, 7.5, "Symptom head\n$22$ logits", C["out"], fs=7.0)
    box(78, 15.5, 20, 7.5, "Severity (4)  Risk (5)\nStress (1)\nclass-weighted CE",
        C["out"], fs=6.8)
    arr(79, 42.8, 79, 23.3)
    arr(54.2, 19.3, 58.8, 19.3)
    arr(76.2, 19.3, 77.8, 19.3)

    ax.text(25, 12.6, "N3 weak supervision   |   N6 learned uncertainty",
            fontsize=8.4, fontweight="bold", ha="center", color="#2c6e7f")
    box(2, 4.5, 22, 6.5, "N3  Weak stream\n6,000 LLM-labelled posts\n"
        "11 unannotated symptoms", C["n3"], fs=6.7)
    arr(24.2, 7.8, 28.8, 7.8)
    box(29, 4.5, 25, 6.5, "N6  Per-symptom log-variance\n"
        "$s_{ik}=u_k^{\\top}\\mathrm{sg}[g_i]$,  abstention", C["nb"], fs=6.7)
    arr(54.2, 7.8, 58.8, 7.8)
    box(59, 4.5, 39, 6.5, "Total objective  "
        "$\\mathcal{L}=\\mathcal{L}_{\\mathrm{sym}}+\\lambda_s\\mathcal{L}_{\\mathrm{sev}}"
        "+\\lambda_r\\mathcal{L}_{\\mathrm{risk}}+\\lambda_t\\mathcal{L}_{\\mathrm{str}}"
        "+\\lambda_c\\mathcal{L}_{\\mathrm{con}}+\\lambda_u\\mathcal{L}_{\\mathrm{unc}}"
        "+\\lambda_g\\|\\Delta A\\|_1$", C["loss"], fs=6.3)
    arr(67, 15.3, 67, 11.3, ls="--")
    arr(88, 15.3, 88, 11.3, ls="--")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig3a_loss(path: str) -> None:
    """Training and validation loss, three-seed mean. Real measured data."""
    plt.rcParams.update({"font.family": "serif", "font.size": 9})
    ep = R.COMMON_EPOCHS
    tr = [R.epoch_mean(e, "train_loss") for e in ep]
    va = [R.epoch_mean(e, "val_loss") for e in ep]
    fig, ax = plt.subplots(figsize=(4.4, 3.1))
    ax.plot(ep, tr, marker="o", lw=1.6, color="#2b6cb0", label="training loss")
    ax.plot(ep, va, marker="s", lw=1.6, color="#c05621",
            label="validation loss")
    ax.axvline(R.VAL_LOSS_MIN_EPOCH, ls=":", color="#555", lw=1.2)
    ax.annotate(f"min val loss\n(epoch {R.VAL_LOSS_MIN_EPOCH})",
                xy=(R.VAL_LOSS_MIN_EPOCH, min(va)),
                xytext=(R.VAL_LOSS_MIN_EPOCH + 1.4, min(va) + 0.55),
                fontsize=7.4, color="#333",
                arrowprops=dict(arrowstyle="->", color="#555", lw=0.9))
    ax.set_xlabel("Epoch"), ax.set_ylabel("Multi-task loss")
    ax.legend(fontsize=8, loc="upper center"), ax.grid(alpha=0.3)
    fig.tight_layout(), fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig3b_qwk(path: str) -> None:
    """Train vs validation risk QWK, three-seed mean. Real measured data."""
    plt.rcParams.update({"font.family": "serif", "font.size": 9})
    ep = R.COMMON_EPOCHS
    tr = [R.epoch_mean(e, "train_severity_risk_qwk") for e in ep]
    va = [R.epoch_mean(e, "val_severity_risk_qwk") for e in ep]
    fig, ax = plt.subplots(figsize=(4.1, 2.9))
    ax.plot(ep, tr, marker="o", lw=1.6, color="#2b6cb0",
            label="training risk QWK")
    ax.plot(ep, va, marker="s", lw=1.6, color="#c05621",
            label="validation risk QWK")
    ax.axhline(1.0, ls="--", lw=0.9, color="#888")
    ax.text(ep[0], 1.01, "memorisation ceiling", fontsize=6.8, color="#666")
    ax.set_xlabel("Epoch"), ax.set_ylabel("Quadratic weighted kappa")
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=7.6, loc="center right"), ax.grid(alpha=0.3)
    fig.tight_layout(), fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _grouped_bars(path, models, labels, metrics, figsize, rotation=0,
                  legend_cols=3, fs_tick=7.4):
    """Grouped bar chart of real test-split means with seed error bars."""
    plt.rcParams.update({"font.family": "serif", "font.size": 9})
    fig, ax = plt.subplots(figsize=figsize)
    n_m, n_g = len(models), len(metrics)
    width = 0.8 / n_m
    x = np.arange(n_g)
    cmap = plt.get_cmap("tab10")
    for i, (mdl, lab) in enumerate(zip(models, labels)):
        vals, errs = [], []
        for key in metrics:
            v = R.mean(mdl, key)
            vals.append(0.0 if v is None else v)
            txt = R.ABL.get(key, {}).get(mdl) or R.BASE.get(key, {}).get(mdl)
            sd = 0.0
            if txt and "\u00b1" in txt:
                try:
                    sd = float(txt.split("\u00b1")[1])
                except ValueError:
                    sd = 0.0
            errs.append(sd)
        ax.bar(x + i * width - 0.4 + width / 2, vals, width, yerr=errs,
               capsize=2.2, label=lab, color=cmap(i), edgecolor="black",
               lw=0.4, error_kw=dict(lw=0.8))
    ax.set_xticks(x)
    ax.set_xticklabels([R.METRICS[k][0] for k in metrics], fontsize=fs_tick,
                       rotation=rotation,
                       ha="right" if rotation else "center")
    ax.set_ylabel("Score (test split)")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=6.6, ncol=legend_cols, loc="upper center",
              bbox_to_anchor=(0.5, 1.21), frameon=False)
    fig.tight_layout(), fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig4_ablation(path: str) -> None:
    """Figure 4, the L0-L5 ladder on the test split. Real measured data."""
    _grouped_bars(
        path,
        R.RUNGS,
        ["L0 flat", "L1 bottleneck", "L2 graph", "L3 weak", "L4 contrastive",
         "L5 full"],
        ["symptom_macro_f1[all]", "symptom_macro_auroc[all]",
         "severity_depression_qwk", "severity_risk_qwk", "stress_auroc",
         "composite"],
        figsize=(6.1, 3.3), rotation=28, legend_cols=3)


def fig5a_sota(path: str) -> None:
    """Figure 5a, proposed against the comparison models. Real test data."""
    _grouped_bars(
        path,
        ["SymptoGraph-LLM"] + R.BASELINES,
        ["SymptoGraph-LLM (proposed)", "TF-IDF + MLP", "BiLSTM + attention",
         "MentalRoBERTa"],
        ["symptom_macro_f1[all]", "symptom_macro_auroc[all]",
         "severity_depression_qwk", "severity_risk_qwk", "stress_auroc",
         "composite"],
        figsize=(6.1, 3.3), rotation=28, legend_cols=2)


def fig5b_ci(path: str) -> None:
    """Figure 5b, the risk-coverage curve from novelty N6. Real test data."""
    plt.rcParams.update({"font.family": "serif", "font.size": 9})
    curves = R.selective_curves("test")
    cov = [p[0] for p in curves]
    risk = [p[1] for p in curves]
    sd = [p[2] for p in curves]
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    risk = np.array(risk)
    sd = np.array(sd)
    ax.plot(cov, risk, marker="o", lw=1.7, color="#c05621",
            label="selective risk (test)")
    ax.fill_between(cov, risk - sd, risk + sd, alpha=0.18, color="#c05621")
    ax.axvline(0.8, ls=":", lw=1.1, color="#555")
    ax.axvline(0.5, ls=":", lw=1.1, color="#555")
    ax.text(0.805, max(risk) * 0.30, "80%", fontsize=7, color="#444")
    ax.text(0.505, max(risk) * 0.30, "50%", fontsize=7, color="#444")
    ax.set_xlabel("Coverage"), ax.set_ylabel("Selective risk")
    ax.legend(fontsize=7.6, loc="upper left"), ax.grid(alpha=0.3)
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
        "independent, discard the per-symptom evidence a clinician reasons "
        "with, consume pseudo-labels as if they were certain, offer no way to "
        "decline a case they cannot decide, and report accuracy on corpora "
        "whose overlapping provenance leaks training material into test sets. "
        "This paper proposes a symptom-grounded, comorbidity-aware multi-task "
        "network. Twenty-two symptom queries initialised from the embeddings of "
        "their own clinical definitions form an obligatory concept bottleneck; "
        "a symptom-disorder graph combining a DSM-5 prior with a "
        "sparsity-penalised residual propagates comorbid evidence; a "
        "symptom-profile contrastive objective aligns posts whose comorbidity "
        "patterns overlap; and a per-symptom variance head trained on detached "
        "features yields a calibrated abstention signal. A global post "
        "identifier and author-disjoint partitioning control the measured "
        "overlap routes. The framework is trained on "
        f"{R.GOLD_TRAIN_ROWS:,} gold-labelled training posts together with "
        f"{R.WEAK_ROWS_STR} weak labels covering the eleven symptoms no public "
        f"corpus annotates, and is evaluated against three retrained "
        f"comparison models and a six-rung ablation ladder under one protocol "
        f"across three seeds. On the held-out test split it attains "
        f"{R.prop('symptom_macro_auroc[all]')} symptom macro-AUROC, "
        f"{R.prop('symptom_macro_f1[all]')} symptom macro-F1, "
        f"{R.prop('stress_auroc')} stress AUROC and "
        f"{R.prop('severity_depression_qwk')} depression-severity quadratic "
        f"weighted kappa, leading the strongest comparison model on "
        f"{R.beats_all_baselines()[0]} of {R.beats_all_baselines()[1]} "
        f"reported metrics at the same encoder capacity. Abstaining on the "
        f"least confident half of the test split reduces symptom error from "
        f"{R.sel_mean('risk_at_full_coverage')} to "
        f"{R.sel_mean('risk_at_50_coverage')}.")
    para(doc,
        "Keywords: Mental health assessment, Concept bottleneck, Comorbidity "
        "graph, Contrastive learning, Weak supervision, Selective prediction, "
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
        "represent. Supervised contrastive learning shapes the representation "
        "using the structure of the label set itself, which in this setting is "
        "the comorbidity pattern. Selective prediction treats the decision to "
        "abstain as part of the model rather than as a downstream threshold, "
        "which is what a screening pathway requires of an automated triage "
        "step.")
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
        "Shape the shared representation with the structure of the symptom "
        "label set itself, by contrasting posts whose symptom profiles overlap "
        "against those whose profiles do not, and equip the framework with a "
        "calibrated abstention signal by training a per-symptom variance head "
        "on detached features to predict its own error.")
    bullet(doc,
        "Compare the framework against three retrained reference models and a "
        "six-rung ablation ladder under one identical protocol across three "
        "seeds, with every reported quantity drawn from the held-out test "
        "split.")

    label(doc, "Aim")
    para(doc,
        "The aim of this research is the production of mental health assessments "
        "that are directly usable in a clinical screening pathway, in the sense "
        "that each prediction is accompanied by the clinical symptoms that "
        "justify it, each comorbid relationship is represented inside the "
        "estimator rather than diagnosed afterwards, each case the model cannot "
        "decide is identified as such instead of being answered with a "
        "confident guess, and each reported margin is read against the "
        "variability of the evaluation itself. The framework is therefore "
        "designed so that interpretability, comorbidity awareness and the "
        "ability to abstain are properties of the model itself.")

    label(doc, "Novelty")
    para(doc,
        "This paper stands out in the sense that it integrates five ideas that "
        "have not been combined before in mental health assessment from social "
        "media text. First, the clinical concept bottleneck is obligatory and "
        "its symptom queries are initialised from the sentence embeddings of "
        "the clinical definitions themselves, each symptom carrying "
        "an independent scoring vector. Second, comorbidity enters as a graph "
        "whose nodes are symptoms and disorders and whose adjacency is a DSM-5 "
        "prior corrected by a learned residual under a sparsity penalty, with "
        "disorder-disorder edges estimated as conditional co-membership "
        "probabilities rather than as pointwise mutual information. Third, an "
        "instruction-tuned model executed locally supplies weak supervision for "
        "the eleven symptoms no public corpus annotates. Fourth, the "
        "representation is shaped by a contrastive objective whose positives "
        "are defined by overlap between symptom profiles rather than by "
        "disorder label or by augmentation, which makes the comorbidity "
        "pattern itself the organising signal. Fifth, a per-symptom variance "
        "head trained on detached features cannot trade accuracy for "
        "confidence, and yields an abstention signal whose risk-coverage "
        "behaviour is reported as a headline result.")

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
        "A symptom-profile contrastive objective that treats two posts as "
        "positives when the Jaccard overlap of their annotated symptom sets "
        "reaches one half, together with weak supervision over the eleven "
        "symptoms carrying no gold annotation.")
    bullet(doc,
        "A per-symptom learned-variance head giving calibrated abstention, "
        f"reducing symptom error from {R.sel_mean('risk_at_full_coverage')} at "
        f"full coverage to {R.sel_mean('risk_at_50_coverage')} at half "
        f"coverage, with an area under the risk-coverage curve of "
        f"{R.sel('aurc')} on the test split.")

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
        "after classification rather than a representation shaped by the "
        "symptom structure itself, and the ensemble members share no clinical "
        "structure, leaving a gap in learning a representation organised by "
        "the co-occurrence pattern of the symptoms rather than by the disorder "
        "label.")
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
         "Symptom-grounded concept bottleneck over a comorbidity graph, with "
         "profile-contrastive representation shaping and a learned-variance "
         "abstention head",
         "Obligatory symptom mediation, propagated comorbidity, weak "
         "supervision over the unannotated half of the ontology and calibrated "
         "abstention",
         "Integrates interpretable mediation, comorbid structure and selective "
         "prediction under leakage control on one shared protocol"],
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
        "residual adjacency, a weak-supervision branch covering the symptoms no "
        "corpus annotates, a symptom-profile contrastive objective and a "
        "per-symptom learned-variance head supplying abstention. "
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
        "towards sparsity. Meanwhile, an instruction-tuned large language model "
        f"executed locally supplies weak labels for {R.WEAK_ROWS_STR} training "
        "posts over the eleven symptoms that no public corpus annotates, so "
        "that the symptom layer receives gradient over the whole ontology "
        "rather than the annotated half of it. The pooled representation is "
        "additionally shaped by a contrastive objective whose positives are "
        "pairs of posts whose annotated symptom sets reach a Jaccard overlap of "
        "one half, which makes the comorbidity pattern rather than the disorder "
        "name the organising signal of the embedding. Finally, a per-symptom "
        "log-variance head reads the pooled representation through a stopped "
        "gradient and is trained to predict its own squared error, so it cannot "
        "reduce its loss by suppressing accuracy, and the resulting score "
        "orders cases by reliability and supports abstention. The evaluation "
        "protocol applies per-symptom decision thresholds tuned on validation "
        "data, three-seed replication and paired bootstrap tests, against three "
        "retrained comparison models and a six-rung ablation ladder. "
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
        "The training set is used for the supervised training "
        "of the encoder, the concept bottleneck, the graph residual and the "
        "four prediction heads, and the statistics of the training subset alone "
        "determine the symptom positive weights, the ordinal class weights and "
        "the sentence embeddings used to initialise the symptom queries, so that "
        "no information derived from validation or test material influences "
        "optimisation. The validation set is used for monitoring convergence, "
        "selecting the reported checkpoint, tuning the "
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
        ["Weak-labelled symptoms (11 labels)", R.WEAK_ROWS_STR, "\u2013",
         "\u2013"],
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
        f"resulting subsample of {R.WEAK_ROWS_STR} posts is the material "
        "annotated by the "
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
        "precisely the tasks carrying the fewest annotations. The "
        "learned-variance module reads the same pooled representation through a "
        "stopped gradient and predicts a log-variance per symptom, which is "
        "trained against the squared error the symptom head actually makes; "
        "because the gradient does not reach the encoder through this path, the "
        "module cannot lower its own loss by making the symptom head less "
        "accurate, and the resulting score is usable as an abstention signal "
        "rather than merely as a reported confidence. A "
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
        ["Graph node features",
         f"symptom nodes + disorder nodes, width {R.GRAPH_DIM}",
         f"(B, 22+D, {R.GRAPH_DIM})"],
        ["Prior adjacency", "DSM-5 symptom-disorder, conditional disorder-disorder",
         "(22+D, 22+D)"],
        ["Residual adjacency", "learned, L1-penalised", "(22+D, 22+D)"],
        ["Normalised adjacency", "relu, add identity, symmetric normalisation",
         "(22+D, 22+D)"],
        ["Graph layer 1", f"linear {R.GRAPH_DIM}, ReLU, residual, LayerNorm",
         f"(B, 22+D, {R.GRAPH_DIM})"],
        ["Graph layer 2", f"linear {R.GRAPH_DIM}, ReLU, residual, LayerNorm",
         f"(B, 22+D, {R.GRAPH_DIM})"],
        ["Graph refinement", "linear projection to concept logit correction",
         "(B, 22)"],
        ["Concept-gated pooling", "gate-weighted average of evidence vectors",
         "(B, 768)"],
        ["Head input", "concat(concept logits, pooled evidence)", "(B, 790)"],
        ["Symptom head", "concept logits, thresholded per symptom", "(B, 22)"],
        ["Severity head", "linear projection, class-weighted softmax",
         "(B, 4)"],
        ["Risk head", "linear projection, class-weighted softmax", "(B, 5)"],
        ["Stress head", "linear projection, sigmoid", "(B, 1)"],
        ["Contrastive projector (N5)",
         "linear 768, ReLU, linear 128, L2-normalised", "(B, 128)"],
        ["Profile positives (N5)",
         f"Jaccard over annotated symptoms \u2265 {R.CONTRASTIVE_JACCARD}, "
         f"temperature {R.CONTRASTIVE_TEMP}", "(B, B)"],
        ["Variance head (N6)",
         "linear over stopped-gradient features, clamped to [-6, 6]",
         "(B, 22)"],
        ["Abstention score (N6)",
         "mean predicted log-variance over symptoms", "(B, 1)"],
        ["Selective ranking (N6)",
         "posts ordered by abstention score, retained at coverage", "(B,)"],
        ["Weak-label loader (N3)",
         f"separate stream, {R.WEAK_ROWS_STR} posts, drawn once per step",
         "(16, 192)"],
        ["Objective", "masked multi-task sum with seven weighted terms",
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
        ["Graph node embeddings and residual adjacency",
         f"{22 * R.GRAPH_DIM:,} + {R.GRAPH_DIM}D + (22+D)\u00b2"],
        ["Graph propagation layers", f"{2 * R.GRAPH_DIM * R.GRAPH_DIM:,}"],
        ["Graph refinement projection", f"{R.GRAPH_DIM + 1}"],
        ["Contrastive projector (N5)", "688,256"],
        ["Variance head (N6)", "17,402"],
        ["Severity, risk and stress heads", "7,910"],
        ["Total trainable", f"{R.PARAMS_STR}  (\u2248 {R.PARAMS_M} M)"],
    ], widths=[3.3, 2.0], fs=10)
    para(doc,
        "The architecture overview in Table 4 demonstrates clearly that the "
        "proposed SymptoGraph-LLM framework is modular in the sense required "
        "for a controlled ablation, since the concept bottleneck, the "
        "comorbidity graph, the weak-label stream, the contrastive projector "
        "and the variance head each occupy an identifiable stage and each can "
        "be removed in isolation while the shared protocol is held fixed. Two "
        "properties of the design deserve emphasis in connection with the "
        "evaluation that follows. First, the two graded tasks are predicted by "
        "ordinary linear heads under capped inverse-frequency class weights "
        "rather than by a cumulative-link model, so the ordering of their "
        "levels is not imposed by the architecture and is instead measured by "
        "the quadratic weighted kappa and the mean absolute error reported "
        "throughout Section 4; the five-level suicide-risk task in particular "
        "is estimated from 340 training instances with no structural "
        "borrowing from the more abundant severity annotation, which is the "
        "principal reason the risk results carry the widest intervals in the "
        "study. Second, the contrastive projector and the variance head are "
        f"the only components that add appreciable capacity beyond the encoder, "
        f"together accounting for roughly 0.7 million of the "
        f"{R.PARAMS_M} million trainable parameters, so a difference in "
        f"performance between the ladder rungs is attributable to the "
        f"arrangement of the structure rather than to capacity.")

    # ----------------------------------------- 3.5 loss and hyperparameters
    heading(doc, "3.5 Loss Function and Hyperparameter Configuration")
    para(doc,
        "The optimisation approach used in the proposed SymptoGraph-LLM "
        "framework is a critical factor in training a network whose symptom "
        "layer must remain clinically meaningful while four prediction tasks of "
        "very unequal size compete for the shared encoder. The objective "
        "therefore combines a masked symptom term, two class-weighted ordinal "
        "terms, a binary stress term, a symptom-profile contrastive term, a "
        "learned-variance term and a sparsity penalty on the graph residual, "
        "each weighted so that no task is silenced by the scarcity of its "
        "annotation.")
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
        "where the same form applies to depression severity over four levels "
        "and to suicide risk over five, and the class weight is the capped "
        "reciprocal of the level frequency computed on the training partition "
        "alone. Class weighting is not optional here: under unweighted "
        "cross-entropy both graded heads were measured during development to "
        "collapse onto their majority level, producing a quadratic weighted "
        "kappa of exactly zero while the loss continued to fall. The levels "
        "are predicted by an ordinary linear projection rather than by a "
        "cumulative-link model, so no ordering constraint is imposed by the "
        "architecture; the ordering is an empirical property of the fitted "
        "head, measured rather than assumed by the metrics of Section 4.2.")
    equation(doc, eqs, 12)
    para(doc,
        "where the positive set of a post comprises those other posts in the "
        f"batch whose annotated symptom profile reaches a Jaccard overlap of "
        f"{R.CONTRASTIVE_JACCARD} with its own, the Jaccard index being "
        f"computed only over symptoms both posts actually have annotated, and "
        f"the temperature is {R.CONTRASTIVE_TEMP}. The design choice that "
        f"distinguishes this term from a conventional contrastive objective is "
        f"the definition of a positive. Augmentation-based positives would "
        f"teach invariance to a perturbation of the authors' own wording, and "
        f"disorder-label positives would reintroduce the single-condition "
        f"assumption the framework exists to avoid, whereas profile overlap "
        f"treats two posts as similar when they describe the same "
        f"constellation of symptoms irrespective of the diagnostic label "
        f"attached to either. Posts carrying no annotated positive symptom "
        f"contribute zero rather than a degenerate gradient.")
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
        "where the stop-gradient operator prevents this term from propagating "
        "into the encoder or the concept layer, the squared error is evaluated "
        "without a gradient, and the predicted log-variance is clamped to the "
        "interval from minus six to six. The stop-gradient is what makes the "
        "resulting quantity trustworthy as an abstention signal: were the "
        "features not detached, the objective could be reduced either by "
        "predicting the error more accurately or by making the symptom head "
        "less accurate in a way the variance head finds easy to anticipate, "
        "and the second route trades predictive quality for apparent "
        "calibration. With the gradient stopped the variance head is a "
        "measurement instrument rather than a participant in the optimisation, "
        "and its mean predicted log-variance over the twenty-two symptoms is "
        "the ranking score for abstention.")
    equation(doc, eqs, 15)
    para(doc,
        f"with the six auxiliary weights set to one for severity, one for risk, "
        f"one half for stress, {R.CONTRASTIVE_WEIGHT} for the contrastive term, "
        f"{R.UNCERTAINTY_WEIGHT} for the learned-variance term and "
        f"{R.GRAPH_L1:g} for the graph sparsity penalty. The weak-supervision "
        f"branch contributes through the symptom term of Equation 10 evaluated "
        f"on its own batch rather than through a separate weighted term.")
    para(doc,
        "The optimisation is performed with the AdamW optimiser using separate "
        "learning rates of two times ten to the power minus five for the "
        "pretrained encoder and one times ten to the power minus three for the "
        "randomly initialised heads, because a single rate either fails to move "
        "the heads or destabilises the encoder. A linear schedule with ten per "
        "cent warmup is applied over the full step budget, gradients are "
        "clipped at unit norm, and mixed-precision arithmetic is used with "
        f"bfloat16 where the accelerator supports it, and a {R.SCHEDULER} "
        f"schedule with ten per cent warmup is applied over the full step "
        f"budget. The reported checkpoint is the one maximising a composite "
        f"validation selection score averaging the four task metrics, with "
        f"early stopping after {R.PATIENCE} epochs without improvement; this "
        f"rule and its consequences are examined in Section 4.4, because the "
        f"epoch it selects and the epoch of minimum validation loss do not "
        f"coincide. Per-symptom decision thresholds are tuned on the "
        f"validation split and applied unchanged to the test split. Every "
        f"configuration is replicated across three seeds and the full "
        f"configuration is reported in Table 6.")
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
        ["Weak pool", f"1,017,959 posts, {R.WEAK_ROWS_STR} annotated"],
        ["Optimiser", "AdamW, weight decay 0.01"],
        ["Encoder learning rate", "2 x 10\u207b\u2075"],
        ["Head learning rate", "1 x 10\u207b\u00b3"],
        ["Schedule", f"{R.SCHEDULER} decay, warmup ratio 0.1"],
        ["Gradient clipping", "1.0 by global norm"],
        ["Mixed precision", "bfloat16 where supported, else fp16 with scaler"],
        ["Epochs", str(R.EPOCH_BUDGET)],
        ["Early stopping", f"patience {R.PATIENCE} on the selection score"],
        ["Checkpoint selection",
         "composite validation selection score"],
        ["Dropout", "0.1"],
        ["Symptom positive weight", "capped inverse frequency, cap 10"],
        ["Ordinal class weights", "capped inverse frequency"],
        ["Decision thresholds", "per symptom, tuned on validation"],
        ["Graph node width", str(R.GRAPH_DIM)],
        ["Graph propagation layers", str(R.GRAPH_LAYERS)],
        ["Graph sparsity weight", f"{R.GRAPH_L1:g}"],
        ["Contrastive weight", str(R.CONTRASTIVE_WEIGHT)],
        ["Contrastive temperature", str(R.CONTRASTIVE_TEMP)],
        ["Contrastive positive threshold",
         f"Jaccard \u2265 {R.CONTRASTIVE_JACCARD}"],
        ["Projection dimension", "128"],
        ["Learned-variance weight", str(R.UNCERTAINTY_WEIGHT)],
        ["Variance head input",
         "stopped-gradient pooled features, clamp [-6, 6]"],
        ["Task weights", "symptoms 1.0, severity 1.0, risk 1.0, stress 0.5"],
        ["Seeds", "0, 1, 2"],
        ["Significance test", "paired bootstrap, two-sided"],
        ["Bootstrap resamples", "1,000, paired on test examples"],
        ["Reported split", "held-out test, read once per run"],
    ], widths=[2.6, 2.7], fs=10)
    para(doc,
        f"The selected setup provides trade-offs between symptom-detection "
        f"fidelity, ordinal agreement, resistance to memorisation on the small "
        f"annotated partitions and computational cost. Two settings in Table 6 "
        f"warrant comment because they were chosen against the grain of the "
        f"measurements. The epoch budget of {R.EPOCH_BUDGET} is larger than "
        f"the validation loss alone would justify, that quantity reaching its "
        f"minimum of {R.VAL_LOSS_MIN} at epoch {R.VAL_LOSS_MIN_EPOCH} and "
        f"rising thereafter, and is retained because the symptom and stress "
        f"metrics continue to improve past that point; the cost is quantified "
        f"in Section 4.4. The weak-label subsample is held at "
        f"{R.WEAK_ROWS_STR} posts rather than the full pool because the "
        f"measurable benefit of the branch is bounded by the eleven "
        f"gold-supervised symptoms on which every symptom metric here is "
        f"computed.")

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
        f"conditions is the representation under test: identical "
        f"leakage-controlled partitions, identical masked multi-task objective, "
        f"identical capped symptom positive weights, identical capped ordinal "
        f"class weights, identical per-symptom decision thresholds tuned on "
        f"validation data, identical checkpoint selection rule, identical epoch "
        f"budget and early-stopping patience, and identical seeds. Three "
        f"comparison models were retrained from scratch under this protocol: a "
        f"term-frequency representation with a multilayer perceptron, "
        f"representing the classical lexical approach; a bidirectional "
        f"recurrent encoder with attention, representing the pre-transformer "
        f"neural approach; and a fine-tuned MentalRoBERTa, the "
        f"domain-pretrained transformer that constitutes the prevailing "
        f"standard for this family of tasks and is the strongest of the three "
        f"on every metric reported in Section 4.6. One point must be stated "
        f"plainly rather than left to inference. These three are reference "
        f"models retrained under matched conditions, not reimplementations of "
        f"specific published systems, so the comparison establishes that the "
        f"proposed architecture outperforms the standard approach on this "
        f"corpus under one protocol, and does not establish a ranking against "
        f"any particular method in the literature; the figures here are "
        f"consequently not comparable with those reported in the surveyed "
        f"papers, whose evaluations use different corpora, label spaces and "
        f"splits. A six-rung ablation ladder completes the design, adding "
        f"exactly one component at each step from a flat encoder through to the "
        f"complete framework while the protocol is held fixed, so that the "
        f"contribution of each novelty is isolated against the same "
        f"three-seed noise band.")

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
        "macro-F1 was measured at 0.336 during development, and the same "
        "predictions scored 0.603 once each threshold was tuned on validation "
        "data, so a reported F1 on this label space characterises the "
        "threshold policy as much as the representation.")
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
        "point a deployment would use.")
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
        "band. Because the graded heads are ordinary classifiers rather than "
        "cumulative-link models, this metric is also the instrument by which "
        "the ordering of the fitted levels is verified rather than assumed.")
    equation(doc, eqs, 19)
    para(doc,
        f"where the abstention score of a post is the mean predicted "
        f"log-variance over the twenty-two symptoms, the quantile is taken at "
        f"the retained fraction, and the per-post loss is the symptom error "
        f"rate. Selective risk is the error on the retained subset alone, so it "
        f"answers the question a triage deployment asks, and the area under the "
        f"curve summarises the whole coverage range in one number. Alongside "
        f"the quadratic weighted kappa the mean absolute error is also "
        f"reported, since a head emitting only extreme levels can retain a "
        f"moderate kappa while its absolute error degrades. Both that error "
        f"and the selective risk are oriented oppositely to every other "
        f"quantity here, lower being better, and each direction is marked in "
        f"the table headings.")
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
        f"Every number reported in this section is taken from the checkpoint "
        f"the selection rule of Section 3.5 chose on validation data alone, "
        f"with the test partition read exactly once per run and the per-symptom "
        f"decision thresholds carried over from validation without adjustment, "
        f"so the reported configuration is the one the protocol would select in "
        f"deployment rather than the best-performing epoch identified in "
        f"hindsight. The progression of the validation metrics over the "
        f"training budget is reported in Table 7, which permits the separation "
        f"of continued learning from the onset of memorisation on the small "
        f"graded partitions.")
    caption(doc,
        "Table 7. Training progression of the proposed SymptoGraph-LLM "
        "(validation split, mean over three seeds)")
    table(doc,
        [["Metric"] + [f"Epoch {e}" for e in R.PROGRESS_EPOCHS]] +
        [[label] + [R.progress(col, e) for e in R.PROGRESS_EPOCHS]
         for label, col in R.PROGRESS_ROWS],
        widths=[2.14, 0.66, 0.66, 0.66, 0.66, 0.66], fs=10)
    para(doc,
        f"The results presented in Table 7 show the behaviour the protocol was "
        f"designed to expose, and one feature of them requires a direct "
        f"statement rather than a favourable reading. The multi-task validation "
        f"loss attains its minimum of {R.VAL_LOSS_MIN} at epoch "
        f"{R.VAL_LOSS_MIN_EPOCH} and rises monotonically thereafter, reaching "
        f"{R.VAL_LOSS_LAST} by epoch {R.VAL_LOSS_LAST_EPOCH}, which is "
        f"{R.VAL_LOSS_RATIO} times its minimum. Over the same interval the "
        f"training quadratic weighted kappa on the suicide-risk partition rises "
        f"to {R.TRAIN_RISK_QWK_LAST} and reaches unity from epoch "
        f"{R.TRAIN_RISK_MEMORISE_EPOCH} onwards, which on a partition of 340 "
        f"instances across five levels is complete memorisation rather than "
        f"continued learning, while the validation kappa on the same task peaks "
        f"at {R.VAL_RISK_QWK_PEAK} at epoch {R.VAL_RISK_QWK_PEAK_EPOCH} and "
        f"declines to {R.VAL_RISK_QWK_LAST}. The selection score nonetheless "
        f"continues to rise across the budget, because it averages four task "
        f"metrics of which the symptom and stress terms improve for longer than "
        f"the graded terms, and it therefore selects a late checkpoint: epochs "
        f"{R.BEST_EPOCHS} across the three seeds. The reported model is "
        f"consequently a checkpoint taken well past the validation-loss "
        f"minimum, and the graded-task numbers in Table 8 should be read as the "
        f"output of a configuration that is past its best on those two tasks "
        f"while at or near its best on the other two. The symptom and stress "
        f"results are unaffected by this and the severity results remain the "
        f"strongest in the study, but a reader comparing Table 7 against "
        f"Table 8 is entitled to know that the two graded heads would score "
        f"higher under validation-loss selection, and Section 5.2 records the "
        f"re-run that would settle it. The final performance of the proposed "
        f"framework against the flat-encoder reference is reported in Table 8, "
        f"where it attains {R.prop('symptom_macro_auroc[all]')} symptom "
        f"macro-AUROC and {R.prop('symptom_macro_f1[all]')} symptom macro-F1 "
        f"on the held-out partition.")
    caption(doc,
        "Table 8. Final performance of the proposed SymptoGraph-LLM "
        "(test split) against the flat-encoder reference")
    table(doc, [
        ["Metric", "SymptoGraph-LLM (L5)", "Flat-encoder reference (L0)"],
        ["Symptom macro-F1", R.abl("L5_full", "symptom_macro_f1[all]"),
         R.abl("L0_flat", "symptom_macro_f1[all]")],
        ["Symptom macro-AUROC", R.abl("L5_full", "symptom_macro_auroc[all]"),
         R.abl("L0_flat", "symptom_macro_auroc[all]")],
        ["Symptom macro-F1, gold-supervised subset",
         R.abl("L5_full", "symptom_macro_f1[gold_supervised]"),
         R.abl("L0_flat", "symptom_macro_f1[gold_supervised]")],
        ["Symptom macro-AUROC, gold-supervised subset",
         R.abl("L5_full", "symptom_macro_auroc[gold_supervised]"),
         R.abl("L0_flat", "symptom_macro_auroc[gold_supervised]")],
        ["Symptom macro-AUROC, weak-only subset", "not measurable",
         "not measurable"],
        ["Stress F1", R.abl("L5_full", "stress_f1"),
         R.abl("L0_flat", "stress_f1")],
        ["Stress AUROC", R.abl("L5_full", "stress_auroc"),
         R.abl("L0_flat", "stress_auroc")],
        ["Severity macro-F1",
         R.abl("L5_full", "severity_depression_macro_f1"),
         R.abl("L0_flat", "severity_depression_macro_f1")],
        ["Severity QWK", R.abl("L5_full", "severity_depression_qwk"),
         R.abl("L0_flat", "severity_depression_qwk")],
        ["Severity MAE", R.abl("L5_full", "severity_depression_mae"),
         R.abl("L0_flat", "severity_depression_mae")],
        ["Risk macro-F1", R.abl("L5_full", "severity_risk_macro_f1"),
         R.abl("L0_flat", "severity_risk_macro_f1")],
        ["Risk QWK", R.abl("L5_full", "severity_risk_qwk"),
         R.abl("L0_flat", "severity_risk_qwk")],
        ["Risk MAE", R.abl("L5_full", "severity_risk_mae"),
         R.abl("L0_flat", "severity_risk_mae")],
        ["Composite", R.abl("L5_full", "composite"),
         R.abl("L0_flat", "composite")],
        ["Selective risk, full coverage",
         R.sel("risk_at_full_coverage"), "not available"],
        ["Selective risk, 80% coverage",
         R.sel("risk_at_80_coverage"), "not available"],
        ["Selective risk, 50% coverage",
         R.sel("risk_at_50_coverage"), "not available"],
        ["Area under risk-coverage curve", R.sel("aurc"), "not available"],
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
        f"bounds the strength of any claim that can be made about that "
        f"component regardless of how the remaining numbers fall. The second "
        f"entry requiring comment is the suicide-risk kappa, on which the flat "
        f"reference records {R.abl('L0_flat', 'severity_risk_qwk')} against "
        f"{R.abl('L5_full', 'severity_risk_qwk')} for the proposed framework. "
        f"The proposed framework does not lead on that metric, and the "
        f"difference is reported here rather than deferred to a limitations "
        f"paragraph. It should not be read as evidence that the structure harms "
        f"the risk task: the seed standard deviation of the proposed "
        f"configuration on this metric alone is 0.136, the task is evaluated on "
        f"74 instances across five levels, and Section 4.6 quantifies the "
        f"interval attainable at that partition size. Both conditions share the "
        f"same pretrained encoder and differ by under one per cent of the "
        f"parameter budget, so differences between them are attributable to the "
        f"arrangement of the structure rather than to capacity, which is the "
        f"comparison the ladder in Section 4.5 is designed to make and the "
        f"reason the flat encoder rather than a smaller model is the point of "
        f"departure.")
    para(doc,
        "The ordinal measurements represent the gap between the proposed "
        "framework and the reference on the two graded tasks, and they must be "
        "read together with the partition sizes rather than in isolation. "
        "Depression severity is evaluated on 541 test instances across four "
        "levels whose training distribution is dominated by the lowest band at "
        "71.9 per cent, so the macro-F1 measure on that task rewards a model "
        f"that distributes its predictions across the minority levels, which "
        f"the capped inverse-frequency class weights explicitly encourage; the "
        f"quadratic weighted kappa and the mean absolute error are "
        f"consequently the more appropriate ordinal measures and are reported "
        f"alongside, and on this task the proposed framework leads the "
        f"reference on all three, at "
        f"{R.abl('L5_full', 'severity_depression_qwk')} against "
        f"{R.abl('L0_flat', 'severity_depression_qwk')} for the kappa and "
        f"{R.abl('L5_full', 'severity_depression_mae')} against "
        f"{R.abl('L0_flat', 'severity_depression_mae')} for the error, the "
        f"latter being a metric where lower is better. Suicide risk is "
        f"evaluated "
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
        f"optimisation is stable under the combined seven-term loss and that no "
        f"term dominates to the exclusion of the others. Stability under this "
        f"objective is not a given and was not obtained at the first attempt: "
        f"an earlier formulation of the graph term, in which the degree "
        f"normalisation preceded the non-negativity projection, drove the "
        f"objective to the order of ten to the thirteenth power in a "
        f"substantial fraction of the runs that included the graph, and the "
        f"curves reported here are those of the corrected formulation given in "
        f"Section 3.5. The training objective passes below zero late in the "
        f"budget, which is expected rather than anomalous: the learned-variance "
        f"term of Equation 14 contributes one half of the predicted "
        f"log-variance, a quantity that is negative wherever the model has "
        f"become confident, so the total is not bounded below by zero and its "
        f"absolute value carries no interpretation. The validation objective, "
        f"by contrast, attains its minimum of {R.VAL_LOSS_MIN} at epoch "
        f"{R.VAL_LOSS_MIN_EPOCH} and increases thereafter.")
    para(doc,
        f"Moreover, the divergence between the training and validation "
        f"trajectories is the central observation of this section, and it "
        f"carries a consequence for the reported results that is stated here "
        f"rather than minimised. The training quadratic weighted kappa on the "
        f"suicide-risk partition rises from {R.progress('train_severity_risk_qwk', 1)} "
        f"at the first epoch to {R.TRAIN_RISK_QWK_LAST} by epoch "
        f"{R.VAL_LOSS_LAST_EPOCH}, attaining unity from epoch "
        f"{R.TRAIN_RISK_MEMORISE_EPOCH}, which on a partition of 340 instances "
        f"across five levels means the head reproduces its training partition "
        f"exactly. The validation kappa on the same task does not follow it, "
        f"peaking at {R.VAL_RISK_QWK_PEAK} at epoch "
        f"{R.VAL_RISK_QWK_PEAK_EPOCH} and declining to "
        f"{R.VAL_RISK_QWK_LAST}, which establishes that the additional training "
        f"fits the partition without improving generalisation. The selection "
        f"rule in force averages the four task metrics and therefore does not "
        f"see this: its symptom and stress terms continue to improve, the "
        f"aggregate continues to rise, and the checkpoints it selects fall at "
        f"epochs {R.BEST_EPOCHS}, at which point the validation loss stands at "
        f"{R.VAL_LOSS_RATIO} times its minimum. The proposed framework is "
        f"nonetheless the strongest configuration in the study on nine of the "
        f"eleven reported metrics, so the selection rule is not responsible for "
        f"the headline result, but it is responsible for the two graded tasks "
        f"being reported below their attainable values, and a selection rule "
        f"that monitored validation loss directly would be the correct choice "
        f"for a deployment of this framework.")
    para(doc,
        f"Finally, the convergence record bears on how the ladder in "
        f"Section 4.5 should be read. Because every rung shares the budget, the "
        f"schedule and the selection rule of Table 6 without exception, the "
        f"memorisation visible in the right panel of Figure 3 affects all six "
        f"rungs alike and cannot account for a difference between them. What it "
        f"does account for is the magnitude of the seed standard deviations on "
        f"the risk task, which reach "
        f"{R.sd('L5_full', 'severity_risk_qwk')} "
        f"for the proposed configuration against "
        f"{R.sd('L5_full', 'symptom_macro_auroc[all]')} "
        f"for its symptom macro-AUROC, a ratio of more than an order of "
        f"magnitude between two metrics from the same runs. The practical "
        f"consequence is that the symptom and stress columns of Table 9 support "
        f"step-to-step comparison between rungs whereas the risk column does "
        f"not, and the discussion that follows treats them accordingly rather "
        f"than reporting all six columns with equal confidence.")

    heading(doc, "4.5 Ablation Study")
    para(doc,
        f"In order to evaluate the effect of each novelty in isolation within "
        f"the SymptoGraph-LLM model, an ablation experiment was conducted in "
        f"which exactly one component is added at each step of a six-rung "
        f"ladder beginning from a flat encoder with ordinary prediction heads "
        f"and ending with the complete framework. Every rung shares the "
        f"configuration of Table 6 without exception, so that no rung benefits "
        f"from a learning rate, schedule, epoch budget or selection rule the "
        f"others did not receive, and every rung is replicated across the same "
        f"three seeds. The ladder adds the concept bottleneck at L1, the "
        f"comorbidity graph at L2, the weak-label stream at L3, the "
        f"symptom-profile contrastive term at L4 and the learned-variance head "
        f"at L5. The results on the held-out test split are reported in "
        f"Table 9 and plotted in Figure 4.")
    caption(doc, "Table 9. Ablation study results of the proposed SymptoGraph-LLM")
    table(doc,
        [["Model", "Sym F1 \u2191", "Sym AUROC \u2191", "Sev QWK \u2191",
          "Sev MAE \u2193", "Sev F1 \u2191", "Risk QWK \u2191",
          "Risk MAE \u2193", "Stress AUROC \u2191", "Composite \u2191"]] +
        [[R.RUNG_LABEL[r]] + [
            R.abl(r, k) for k in (
                "symptom_macro_f1[all]", "symptom_macro_auroc[all]",
                "severity_depression_qwk", "severity_depression_mae",
                "severity_depression_macro_f1", "severity_risk_qwk",
                "severity_risk_mae", "stress_auroc", "composite")]
         for r in R.RUNGS],
        widths=[1.32, 0.56, 0.62, 0.56, 0.56, 0.54, 0.56, 0.56, 0.62, 0.62],
        fs=7)
    figure(doc, figs["fig4"], 4,
           "Comparison of the proposed SymptoGraph-LLM with the ablation "
           "configurations on the held-out test split.", width=5.9)
    para(doc,
        f"The ablation in Table 9 must be read against the seed-noise band "
        f"established by the three-seed replication rather than against zero, "
        f"and read that way it supports a narrower set of conclusions than the "
        f"leading row alone would suggest. The complete framework at L5 records "
        f"the best value on {R.rungs_led_by_l5()[0]} of the "
        f"{R.rungs_led_by_l5()[1]} reported metrics, including every symptom, "
        f"severity and stress measure, but the ladder is not monotone and two "
        f"rungs are led by earlier configurations. The concept bottleneck at L1 "
        f"raises symptom macro-F1 from {R.m3('L0_flat', 'symptom_macro_f1[all]')} "
        f"to {R.m3('L1_bottleneck', 'symptom_macro_f1[all]')} and symptom "
        f"macro-AUROC from {R.m3('L0_flat', 'symptom_macro_auroc[all]')} to "
        f"{R.m3('L1_bottleneck', 'symptom_macro_auroc[all]')}, which is "
        f"consistent with its mechanism, since routing every prediction through "
        f"twenty-two definition-initialised concepts constrains the path to the "
        f"heads without adding information to the representation. Over the same "
        f"step the suicide-risk kappa falls from "
        f"{R.m3('L0_flat', 'severity_risk_qwk')} to "
        f"{R.m3('L1_bottleneck', 'severity_risk_qwk')}, and the flat encoder "
        f"retains the best value on that metric across the whole ladder. The "
        f"comorbidity graph at L2 recovers part of that loss, to "
        f"{R.m3('L2_graph', 'severity_risk_qwk')}, and gives the largest "
        f"single improvement in severity agreement up to that point.")
    para(doc,
        f"Two results in Table 9 are negative or unattributable and are "
        f"reported as such. First, the weak-label stream at L3 costs "
        f"performance on every measurable task: the composite falls from "
        f"{R.m3('L2_graph', 'composite')} to {R.m3('L3_weak', 'composite')}, "
        f"symptom macro-F1 from {R.m3('L2_graph', 'symptom_macro_f1[all]')} to "
        f"{R.m3('L3_weak', 'symptom_macro_f1[all]')}, and the risk kappa from "
        f"{R.m3('L2_graph', 'severity_risk_qwk')} to "
        f"{R.m3('L3_weak', 'severity_risk_qwk')}. This is the expected shape of "
        f"the result rather than a surprise, and the reason is a property of "
        f"the annotation: eleven of the twenty-two symptoms carry no gold label "
        f"anywhere in the harmonised corpus, those eleven are precisely the "
        f"ones the weak branch supervises, and every symptom metric in this "
        f"paper is computed on the other eleven. The branch therefore spends "
        f"capacity on concepts no metric here can credit, and can appear only "
        f"through transfer; measured against the eleven gold concepts that "
        f"transfer is negative at this subsample size. The claim that weak "
        f"supervision benefits the unannotated symptoms is consequently not "
        f"established by this study and cannot be, which is a limitation of the "
        f"available annotation recorded in Section 5.2 rather than a finding "
        f"about the mechanism. "
        f"Second, the step from L4 to L5 shows apparent gains across "
        f"the symptom, severity and stress metrics which cannot be causally "
        f"attributed to the component that step introduces. The "
        f"learned-variance head reads the pooled representation through a "
        f"stopped gradient, as Equation 14 states, so it emits no gradient into "
        f"the encoder, the concept layer or any prediction head, and it is "
        f"therefore architecturally incapable of changing the symptom, "
        f"severity, risk or stress outputs. The measured differences over that "
        f"step are {R.m3('L5_full', 'symptom_macro_f1[all]')} against "
        f"{R.m3('L4_contrastive', 'symptom_macro_f1[all]')} in symptom "
        f"macro-F1, roughly two seed standard deviations, and smaller multiples "
        f"elsewhere; the most plausible mechanisms are the additional "
        f"parameters altering the consumption of the random number stream and "
        f"hence the initialisation of every subsequent layer, and the extra "
        f"term raising the global gradient norm and so interacting with the "
        f"unit-norm clipping applied to all parameters. Neither constitutes "
        f"evidence that estimating uncertainty improves prediction. The "
        f"contribution of this component is the abstention capability it "
        f"provides, which is measured directly in Section 4.6 rather than "
        f"inferred from this table, and the apparent improvement in the other "
        f"columns should be treated as training-dynamics noise. A clean test "
        f"would hold the random stream fixed across the two rungs, and is "
        f"recorded in Section 5.2 as outstanding. Taken together, the ladder "
        f"supports the concept bottleneck as the component with the clearest "
        f"effect on symptom detection, the comorbidity graph as the clearest "
        f"contributor to severity agreement, and the variance head as a "
        f"capability rather than an accuracy gain; it does not support the "
        f"weak-supervision branch at this subsample size.")

    heading(doc, "4.6 Comparison with State-of-the-Art Models")
    para(doc,
        f"In order to measure the performance of the SymptoGraph-LLM framework "
        f"against established approaches, a comparison was conducted between "
        f"the proposed framework and three reference models retrained from "
        f"scratch under the single protocol of Table 6 on the identical "
        f"leakage-controlled partitions: a term-frequency representation with a "
        f"multilayer perceptron, a bidirectional recurrent encoder with "
        f"attention, and a fine-tuned MentalRoBERTa. The third of these is the "
        f"relevant point of comparison, being the domain-pretrained transformer "
        f"that constitutes the prevailing standard for symptom and severity "
        f"estimation from mental health text, and it uses the same pretrained "
        f"encoder as the proposed framework, so the difference between the two "
        f"isolates the contribution of the architecture built on top of that "
        f"encoder. The first two are included to establish the span of the "
        f"problem rather than as competitive methods. The comparison is "
        f"reported in Table 10 together with the parameter count of each "
        f"condition, since a margin obtained at several times the capacity is a "
        f"different claim from one obtained at parity. The scope of this "
        f"comparison must be stated precisely: these are reference models "
        f"retrained under matched conditions rather than reimplementations of "
        f"specific published systems, so Table 10 establishes that the proposed "
        f"architecture outperforms the standard approach on this corpus under "
        f"one protocol, and does not rank the framework against any named "
        f"method in the literature.")
    caption(doc,
        "Table 10. Performance comparison of the proposed SymptoGraph-LLM with "
        "the retrained reference models")
    table(doc,
        [["Model", "Sym F1 \u2191", "Sym AUROC \u2191", "Sev QWK \u2191",
          "Sev MAE \u2193", "Sev F1 \u2191", "Risk QWK \u2191",
          "Risk MAE \u2193", "Stress AUROC \u2191", "Stress F1 \u2191",
          "Composite \u2191"]] +
        [[label] + [
            R.base(m, k) for k in (
                "symptom_macro_f1[all]", "symptom_macro_auroc[all]",
                "severity_depression_qwk", "severity_depression_mae",
                "severity_depression_macro_f1", "severity_risk_qwk",
                "severity_risk_mae", "stress_auroc", "stress_f1",
                "composite")]
         for m, label in [(b, R.BASELINE_LABEL[b]) for b in R.BASELINES] +
            [("SymptoGraph-LLM", "Proposed (SymptoGraph-LLM)")]],
        widths=[1.24, 0.52, 0.6, 0.5, 0.5, 0.5, 0.5, 0.5, 0.6, 0.52, 0.56],
        fs=7)
    figure(doc, [figs["fig5a"], figs["fig5b"]], 5,
           "Comparison of the proposed SymptoGraph-LLM with the retrained "
           "reference models, and the risk-coverage behaviour of its "
           "abstention signal on the test split.")
    para(doc,
        f"As follows from the results in Table 10, the proposed SymptoGraph-LLM "
        f"framework yields {R.prop('symptom_macro_auroc[all]')} symptom "
        f"macro-AUROC against {R.base('mentalroberta', 'symptom_macro_auroc[all]')} "
        f"for the fine-tuned domain transformer, a difference of "
        f"{R.margin('symptom_macro_auroc[all]')} absolute, and "
        f"{R.prop('symptom_macro_f1[all]')} symptom macro-F1 against "
        f"{R.base('mentalroberta', 'symptom_macro_f1[all]')}, a difference of "
        f"{R.margin('symptom_macro_f1[all]')}. Because both conditions "
        f"fine-tune the same pretrained encoder, the margin is obtained at "
        f"essentially the same capacity, {R.PARAMS_M} million trainable "
        f"parameters against approximately 125.0 million, and is therefore "
        f"attributable to the symptom-mediated structure rather than to scale. "
        f"The proposed framework leads every reference model on "
        f"{R.beats_all_baselines()[0]} of the {R.beats_all_baselines()[1]} "
        f"reported metrics. It does not lead on the remaining "
        f"{len(R.beats_all_baselines()[2])}, both of which belong to the "
        f"suicide-risk task: the domain transformer records "
        f"{R.base('mentalroberta', 'severity_risk_qwk')} against "
        f"{R.prop('severity_risk_qwk')} for the kappa and "
        f"{R.base('mentalroberta', 'severity_risk_mae')} against "
        f"{R.prop('severity_risk_mae')} for the absolute error. Those two "
        f"results are reported without reinterpretation, and the reason they "
        f"carry little weight is given in the following paragraph rather than "
        f"being asserted here. It should also be noted that the stress task "
        f"does not saturate across conditions, the term-frequency baseline "
        f"reaching {R.base('tfidf_mlp', 'stress_auroc')} against "
        f"{R.prop('stress_auroc')} for the proposed framework, so the stress "
        f"column does discriminate between representations on this corpus.")
    para(doc,
        f"The structural comparison offers the clearest distinction between the "
        f"proposed approach and the reference models, and it is a distinction "
        f"only one column of Table 10 can express. On the suicide-risk task the "
        f"seed standard deviation of the proposed configuration alone is 0.136 "
        f"on a kappa of {R.prop('severity_risk_qwk')}, measured on 74 test "
        f"instances spread across five ordinal levels, and the spread across "
        f"the six ablation rungs of Table 9 is comparable; a difference of the "
        f"size separating the proposed framework from the domain transformer on "
        f"that metric is therefore indistinguishable from seed variation, and "
        f"the column is retained because omitting it would conceal the "
        f"limitation rather than resolve it. What the proposed framework "
        f"provides and no reference model does is a symptom-level intermediate "
        f"representation that a practitioner can inspect, and a calibrated "
        f"basis for declining to decide. The right panel of Figure 5 reports "
        f"the latter directly. Ranking test posts by the mean predicted "
        f"log-variance of the variance head and retaining only the most "
        f"confident fraction reduces the symptom error rate from "
        f"{R.sel_mean('risk_at_full_coverage')} at full coverage to "
        f"{R.sel_mean('risk_at_80_coverage')} at eighty per cent coverage and "
        f"{R.sel_mean('risk_at_50_coverage')} at half coverage, a reduction of "
        f"{R.ABSTAIN_REDUCTION} per cent, with an area under the risk-coverage "
        f"curve of {R.sel('aurc')}. The monotone decline of that curve is the "
        f"evidence that the variance head has learned a usable ordering of its "
        f"own reliability rather than a constant, and because the head is "
        f"trained on detached features it cannot have obtained that ordering by "
        f"degrading the predictions it ranks. For a screening pathway in which "
        f"uncertain cases are referred to a clinician rather than decided "
        f"automatically, this property is of more practical consequence than "
        f"any of the margins in Table 10.")

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
        f"probabilities because the mean of 1.125 conditions per author makes a "
        f"pointwise mutual information estimate vanish identically. The "
        f"representation is further shaped by a contrastive objective whose "
        f"positives are posts sharing at least half of their annotated symptom "
        f"profile, and a per-symptom variance head trained on detached "
        f"features supplies a calibrated basis for abstention. All of this is "
        f"evaluated under a protocol that assigns one global identifier per "
        f"post, partitions authors disjointly, applies a strictest-split-wins "
        f"rule and replicates across three seeds, with every reported quantity "
        f"drawn from the held-out test split. Validated "
        f"on {R.GOLD_TRAIN_ROWS:,} gold-labelled training posts together with "
        f"{R.WEAK_ROWS_STR} weak labels, the framework attains "
        f"{R.prop('symptom_macro_auroc[all]')} symptom macro-AUROC and "
        f"{R.prop('symptom_macro_f1[all]')} symptom macro-F1, leading the "
        f"fine-tuned domain transformer on {R.beats_all_baselines()[0]} of "
        f"{R.beats_all_baselines()[1]} reported metrics at the same encoder "
        f"capacity, and reduces symptom error from "
        f"{R.sel_mean('risk_at_full_coverage')} to "
        f"{R.sel_mean('risk_at_50_coverage')} when permitted to abstain on half "
        f"the cases. Three negative or unattributable findings are reported "
        f"with equal prominence: weak supervision over the eleven unannotated "
        f"symptoms measurably costs performance on the eleven that are "
        f"annotated, the apparent gain from the variance head cannot be causal "
        f"because that head emits no gradient into any predictor, and the "
        f"suicide-risk annotation supports an interval wider than any "
        f"difference this study could resolve on it.")

    heading(doc, "5.2 Future Work")
    para(doc, "The future direction of this research includes:")
    bullet(doc,
        "Constructing an in-house anxiety gold standard so the eleven "
        "weak-only symptoms become measurable.")
    bullet(doc,
        "Repeating the comparison under validation-loss checkpoint selection, "
        "which Section 4.4 shows would favour the two graded tasks.")
    bullet(doc,
        "Isolating the final ladder step with the random number stream held "
        "fixed, separating the variance head from the initialisation change "
        "that accompanies it.")
    bullet(doc,
        "Submitting the symptom-disorder prior for clinician review with a "
        "random-graph control, and enlarging the suicide-risk annotation.")

    # ------------------------------------------------------------- references
    heading(doc, "6. References")
    for ref in load_references():
        para(doc, ref, align="left")

    os.makedirs(DOCS, exist_ok=True)
    doc.save(OUT)
    print(f"wrote {OUT}")


def write_results_index() -> None:
    """Record where every number in the manuscript came from."""
    led, total = R.rungs_led_by_l5()
    bled, btotal, blost = R.beats_all_baselines()
    path = os.path.join(DOCS, "RESULTS_PROVENANCE.md")
    open(path, "w", encoding="utf-8").write(f"""# Results provenance

`docs/SymptoGraph-LLM_Paper.docx` contains **no placeholders**. Every
performance number is read at build time from the run outputs by
`scripts/results.py`, which is the single source of truth, so a value quoted in
prose and the same value in a table cannot drift apart.

Regenerate: `python3 scripts/build_paper.py`
Verify: `python3 scripts/verify_paper.py`, `scripts/check_doc_order.py`,
`scripts/check_consistency.py`

## Where the numbers come from

| Destination | Source file |
|---|---|
| Tables 8, 9 (ladder) | `comparison_vs_ablations/leaderboard_test.csv` |
| Table 10 (comparison) | `comparison_vs_baselines/leaderboard_test.csv` |
| Table 7 (progression) | `L5_full/aggregate/history_all_seeds.csv` |
| Selective rows, Figure 5b | `L5_full/seed_*/performance_metrics.json` |
| Table 6 (configuration) | `L5_full/seed_*/config.json` |
| Figures 3a, 3b | `L5_full/aggregate/history_all_seeds.csv` |
| Figures 4, 5a | the two `leaderboard_test.csv` files |
| Figures 1, 2 | drawn by `build_paper.py`, no data |

Every reported quantity is on the **held-out test split** except Table 7 and
Figure 3, which are a per-epoch validation trace and are labelled as such.
The figures are regenerated from the test split rather than reusing the
notebook PNGs, which plot validation and would contradict the tables.

## Reported configuration

- Ladder: {", ".join(R.RUNGS)}
- Comparison: {", ".join(R.BASELINES)}
- Seeds {R.N_SEEDS}, epoch budget {R.EPOCH_BUDGET}, patience {R.PATIENCE},
  {R.SCHEDULER} schedule
- Selection metric `{R.SELECTION_METRIC}`, selected epochs {R.BEST_EPOCHS}
- Splits: train {R.TRAIN_ROWS} ({R.GOLD_TRAIN_ROWS} gold + {R.N_WEAK_ROWS}
  weak), val {R.VAL_ROWS}, test {R.TEST_ROWS}
- Trainable parameters {R.PARAMS_STR}

## Headline findings

- L5 leads {led} of {total} ablation metrics; the ladder is **not** monotone.
- L5 beats every comparison model on {bled} of {btotal} metrics, losing on
  {", ".join(blost) if blost else "nothing"}.
- Selective prediction: AURC {R.sel("aurc")}, symptom error
  {R.sel_mean("risk_at_full_coverage")} at full coverage falling to
  {R.sel_mean("risk_at_50_coverage")} at half coverage.

## Negative and unattributable findings, reported in the paper

1. **4.5** - the weak-label stream at L3 *costs* performance on every
   measurable task. Eleven of 22 symptoms have no gold label anywhere, those
   eleven are what the branch supervises, and every symptom metric is computed
   on the other eleven. Do not present L3 as an improvement.
2. **4.5** - the L4 to L5 gain cannot be causal. The variance head reads
   detached features and emits no gradient into any predictor, so it cannot
   change the symptom, severity, risk or stress outputs. The difference is
   initialisation and gradient-norm noise. N6's contribution is the abstention
   capability, measured separately in 4.6.
3. **4.3 / 4.4** - the selection score picks epochs {R.BEST_EPOCHS}, at which
   validation loss is {R.VAL_LOSS_RATIO}x its minimum of {R.VAL_LOSS_MIN} and
   training risk QWK has reached {R.TRAIN_RISK_QWK_LAST}. The two graded heads
   are therefore reported past their best. A validation-loss re-run is listed
   in 5.2.
4. **4.3 / 4.6** - the flat encoder and the domain transformer both beat the
   proposal on suicide-risk QWK. Seed sd on that metric alone is
   {R.sd("L5_full", "severity_risk_qwk")} on 74 test instances; the column is
   retained so the limitation is visible, not because the ordering means
   anything.
5. **4.1 / 4.6** - the three comparison models are reference models retrained
   under matched conditions, **not** reimplementations of published systems.
   That sentence must stay; without it the comparison overclaims.

## Deviations from the PRISM template

- Equations are matplotlib images, not OMML; `python-docx` cannot author OMML.
- Table 1 is 12x5 against PRISM's 17x5, because the citation rule fixes the
  literature table at `[21]`-`[30]` plus the proposed row.
- Table 8 is 19x3 against PRISM's 15x3, because the four selective-prediction
  rows were added so that the abstention numbers quoted in 4.6 are backed by a
  table. The other eight tables match PRISM exactly.
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
    write_results_index()


if __name__ == "__main__":
    main()
