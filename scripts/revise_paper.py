#!/usr/bin/env python3
"""Apply the second round of reviewer comments to
``docs/SymptoGraph-LLM_Paper_revised.docx``, **in place**.

That file carries hand edits made outside this pipeline, so it is patched
directly rather than regenerated from ``build_paper.py``; regenerating would
discard those edits.

What this does, in the order the comments were raised:

 1. Table 7 and Figure 3 extended to epochs 9 and 10, flagged as two-seed means
    because seed 1 early-stopped at epoch 8.
 2. gamma_max given a value in the Equation 11 paragraph.
 3. A max-probability abstention baseline and a calibration metric defined in
    4.2, and reported as not computed, because per-example predictions were
    not retained.
 4. Figure 2 redrawn: the weak stream now feeds the symptom term of the
    objective and the symptom head is fed by the concept logits.
 5. Criticisms of [3], [14], [18] and [28] softened where this paper shares
    the gap.
 6. Unsupported statistics removed.
 7. Ethics, data-availability and code-availability statements added.
 8. A note on [21] appearing online first.
 9. Long sentences split and "rather than" thinned out.
10. The role of the language model made explicit where the title is explained.

Run: python3 scripts/revise_paper.py
"""
from __future__ import annotations

import copy
import csv
import os
import re
import statistics as st
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

DOC = os.path.join(ROOT, "docs", "SymptoGraph-LLM_Paper_revised.docx")
FIGS = os.path.join(ROOT, "docs", "paper_figures")
HIST = os.path.join(ROOT, "L5_full", "aggregate", "history_all_seeds.csv")

FONT = "Times New Roman"
SIZE = Pt(12)

REPORT: list[str] = []


def note(msg: str) -> None:
    REPORT.append(msg)
    print(f"  {msg}")


# --------------------------------------------------------------------------- #
# history, including the epochs only two seeds reached
# --------------------------------------------------------------------------- #
with open(HIST, newline="", encoding="utf-8") as fh:
    ROWS = list(csv.DictReader(fh))

EPOCHS = sorted({int(r["epoch"]) for r in ROWS})
SEEDS_AT = {e: sorted(r["seed"] for r in ROWS if int(r["epoch"]) == e)
            for e in EPOCHS}
FULL_SEEDS = max(len(v) for v in SEEDS_AT.values())
PARTIAL = [e for e in EPOCHS if len(SEEDS_AT[e]) < FULL_SEEDS]


def mean_at(epoch: int, col: str):
    v = [float(r[col]) for r in ROWS
         if int(r["epoch"]) == epoch and r.get(col, "") not in ("", "nan")]
    return st.mean(v) if v else None


def fmt(epoch: int, col: str) -> str:
    v = mean_at(epoch, col)
    return "\u2013" if v is None else f"{v:.3f}"


REPORTED_EPOCHS = [1, 2, 3, 6, 8, 9, 10]
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


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def _split_curve(col: str):
    """(full-seed epochs, their means), (partial epochs, their means)."""
    full = [e for e in EPOCHS if e not in PARTIAL]
    part = [e for e in EPOCHS if e in PARTIAL]
    join = ([full[-1]] + part) if full and part else part
    return (full, [mean_at(e, col) for e in full],
            join, [mean_at(e, col) for e in join])


def fig3a(path: str) -> None:
    plt.rcParams.update({"font.family": "serif", "font.size": 9})
    fig, ax = plt.subplots(figsize=(4.4, 3.1))
    for col, colour, label, marker in (
            ("train_loss", "#2b6cb0", "training loss", "o"),
            ("val_loss", "#c05621", "validation loss", "s")):
        fe, fv, pe, pv = _split_curve(col)
        ax.plot(fe, fv, marker=marker, lw=1.6, color=colour, label=label)
        if len(pe) > 1:
            ax.plot(pe, pv, marker=marker, lw=1.6, color=colour, ls=(0, (4, 2)),
                    mfc="white")
    lo = min(PARTIAL) - 0.5 if PARTIAL else None
    if lo:
        ax.axvspan(lo, max(EPOCHS) + 0.4, color="#999", alpha=0.10, lw=0)
        ax.text(lo + 0.08, ax.get_ylim()[1] * 0.97, "2 seeds", fontsize=6.8,
                color="#555", va="top")
    vmin = min(EPOCHS, key=lambda e: mean_at(e, "val_loss"))
    ax.axvline(vmin, ls=":", color="#555", lw=1.1)
    ax.annotate(f"min val loss\n(epoch {vmin})",
                xy=(vmin, mean_at(vmin, "val_loss")),
                xytext=(vmin + 1.5, mean_at(vmin, "val_loss") + 0.75),
                fontsize=7.2, color="#333",
                arrowprops=dict(arrowstyle="->", color="#555", lw=0.9))
    ax.set_xlabel("Epoch"), ax.set_ylabel("Multi-task loss")
    ax.set_xticks(EPOCHS)
    ax.legend(fontsize=7.6, loc="upper left"), ax.grid(alpha=0.3)
    fig.tight_layout(), fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig3b(path: str) -> None:
    plt.rcParams.update({"font.family": "serif", "font.size": 9})
    fig, ax = plt.subplots(figsize=(4.1, 2.9))
    for col, colour, label, marker in (
            ("train_severity_risk_qwk", "#2b6cb0", "training risk QWK", "o"),
            ("val_severity_risk_qwk", "#c05621", "validation risk QWK", "s")):
        fe, fv, pe, pv = _split_curve(col)
        ax.plot(fe, fv, marker=marker, lw=1.6, color=colour, label=label)
        if len(pe) > 1:
            ax.plot(pe, pv, marker=marker, lw=1.6, color=colour, ls=(0, (4, 2)),
                    mfc="white")
    if PARTIAL:
        lo = min(PARTIAL) - 0.5
        ax.axvspan(lo, max(EPOCHS) + 0.4, color="#999", alpha=0.10, lw=0)
        ax.text(lo + 0.08, 1.07, "2 seeds", fontsize=6.8, color="#555")
    ax.axhline(1.0, ls="--", lw=0.9, color="#888")
    ax.text(EPOCHS[0], 1.02, "memorisation ceiling", fontsize=6.8,
            color="#666")
    ax.set_xlabel("Epoch"), ax.set_ylabel("Quadratic weighted kappa")
    ax.set_xticks(EPOCHS), ax.set_ylim(0, 1.15)
    ax.legend(fontsize=7.2, loc="center right"), ax.grid(alpha=0.3)
    fig.tight_layout(), fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# docx helpers
# --------------------------------------------------------------------------- #
def iter_paras(doc):
    return list(doc.paragraphs)


def find_para(doc, needle: str, start: int = 0) -> int:
    for i, p in enumerate(doc.paragraphs):
        if i >= start and needle in p.text:
            return i
    raise LookupError(needle[:70])


def set_text(p, text: str) -> None:
    """Replace a paragraph's text, keeping the first run's formatting."""
    runs = p.runs
    if not runs:
        r = p.add_run(text)
        r.font.name, r.font.size = FONT, SIZE
        return
    runs[0].text = text
    for r in runs[1:]:
        r.text = ""


def replace_in_para(p, old: str, new: str) -> bool:
    """Substitute within a paragraph without disturbing other runs if possible."""
    for r in p.runs:
        if old in r.text:
            r.text = r.text.replace(old, new)
            return True
    if old in p.text:
        set_text(p, p.text.replace(old, new))
        return True
    return False


def para_after(doc, idx: int, text: str, bold=False, style=None, align="justify"):
    """Insert a new paragraph after paragraph idx and return it."""
    src = doc.paragraphs[idx]
    new_p = copy.deepcopy(src._p)
    src._p.addnext(new_p)
    from docx.text.paragraph import Paragraph
    p = Paragraph(new_p, src._parent)
    for r in list(p.runs):
        r._r.getparent().remove(r._r)
    if style:
        try:
            p.style = style
        except KeyError:
            pass
    p.alignment = {"justify": WD_ALIGN_PARAGRAPH.JUSTIFY,
                   "center": WD_ALIGN_PARAGRAPH.CENTER,
                   "left": WD_ALIGN_PARAGRAPH.LEFT}[align]
    r = p.add_run(text)
    r.bold = bold
    r.font.name, r.font.size = FONT, SIZE
    return p


def replace_image(doc, shape_index: int, path: str) -> None:
    sh = doc.inline_shapes[shape_index]
    rid = sh._inline.graphic.graphicData.pic.blipFill.blip.embed
    part = doc.part.related_parts[rid]
    with open(path, "rb") as fh:
        part._blob = fh.read()


def set_cell(cell, text: str, fs: float) -> None:
    cell.text = ""
    r = cell.paragraphs[0].add_run(text)
    r.font.name, r.font.size = FONT, Pt(fs)


# --------------------------------------------------------------------------- #
# the edits
# --------------------------------------------------------------------------- #
def rebuild_table7(doc) -> None:
    """Rewrite Table 7 over all reported epochs, including the two-seed tail."""
    t = doc.tables[6]
    want_cols = 1 + len(REPORTED_EPOCHS)
    while len(t.columns) < want_cols:
        t.add_column(Inches(0.52))
    hdr = ["Metric"] + [f"Epoch {e}" for e in REPORTED_EPOCHS]
    for j, v in enumerate(hdr):
        set_cell(t.rows[0].cells[j], v, 9.0)
    for i, (label, col) in enumerate(PROGRESS_ROWS, start=1):
        set_cell(t.rows[i].cells[0], label, 9.0)
        for j, e in enumerate(REPORTED_EPOCHS, start=1):
            set_cell(t.rows[i].cells[j], fmt(e, col), 9.0)
    # eight columns have to fit the 6.27 in text block
    t.autofit = False
    widths = [Inches(1.85)] + [Inches(0.63)] * len(REPORTED_EPOCHS)
    for row in t.rows:
        for cell, w in zip(row.cells, widths):
            cell.width = w
    note(f"Table 7 now spans epochs {REPORTED_EPOCHS} "
         f"({want_cols} columns); epochs {PARTIAL} are two-seed means")


def main() -> None:
    print("regenerating figures")
    fig3a(os.path.join(FIGS, "figure3a_loss.png"))
    fig3b(os.path.join(FIGS, "figure3b_qwk.png"))
    import diagrams
    diagrams.build(FIGS, os.path.join(ROOT, "docs", "diagrams"),
                   "6,000", "12,911", "125.8")
    note("Figure 2 redrawn; Figure 3 extended to epoch 10")

    print("patching document")
    doc = Document(DOC)

    # ---- 1. Table 7 and its caption
    rebuild_table7(doc)
    i = find_para(doc, "Table 7. Training progression")
    set_text(doc.paragraphs[i],
             "Table 7. Training progression of the proposed SymptoGraph-LLM "
             "(validation split, mean over seeds). Seed 1 met the "
             "early-stopping criterion at epoch 8, so the columns for epochs 9 "
             "and 10 are means over the two seeds that continued; the "
             "checkpoints the selection rule chose fall at epochs 9, 4 and 9.")

    # ---- 4. figures
    replace_image(doc, 1, os.path.join(FIGS, "figure2_architecture.png"))
    replace_image(doc, 2, os.path.join(FIGS, "figure3a_loss.png"))
    replace_image(doc, 3, os.path.join(FIGS, "figure3b_qwk.png"))
    replace_image(doc, 0, os.path.join(FIGS, "figure1_workflow.png"))
    note("figure images replaced in place")

    i = find_para(doc, "Figure 3. Training convergence")
    set_text(doc.paragraphs[i],
             "Figure 3. Training convergence of the proposed SymptoGraph-LLM. "
             "(A) Multi-task training and validation loss. (B) Training and "
             "validation quadratic weighted kappa on the suicide-risk task. "
             "Solid segments are means over three seeds; the shaded tail from "
             "epoch 9 is a mean over the two seeds that ran past the "
             "early-stopping point.")

    i = find_para(doc, "Figure 2. Overall architecture")
    set_text(doc.paragraphs[i],
             "Figure 2. Overall architecture of the proposed SymptoGraph-LLM. "
             "The symptom head is driven by the refined concept logits; the "
             "severity, risk and stress heads read those logits concatenated "
             "with the concept-gated pooled vector. The variance head reads "
             "the pooled vector through a stopped gradient, and the weak-label "
             "stream contributes only to the symptom term of the objective.")
    note("Figure 2 and Figure 3 captions updated")

    apply_text_edits(doc)
    thin_rather_than(doc)
    split_long_sentences(doc)
    add_closing_statements(doc)
    tidy_whitespace(doc)

    doc.save(DOC)
    print(f"\nsaved {DOC}")
    print(f"{len(REPORT)} changes recorded")


# --------------------------------------------------------------------------- #
# 2, 3, 5, 6, 8, 10
# --------------------------------------------------------------------------- #
def apply_text_edits(doc) -> None:
    # ---- 2. gamma_max
    i = find_para(doc, "class weight is the capped reciprocal")
    p = doc.paragraphs[i]
    if "clipped to the interval" not in p.text:
        set_text(p, p.text.replace(
            "computed on the training partition alone.",
            "computed on the training partition alone, with the cap "
            "\u03b3_max set to 10 so that every weight is clipped to the "
            "interval from 0.1 to 10."))
        note("Equation 11 paragraph now gives \u03b3_max = 10")

    # ---- 3. abstention baseline and a calibration metric
    i = find_para(doc, "area under the curve summarises the whole coverage")
    p = doc.paragraphs[i]
    if "maximum predicted probability" not in p.text:
        set_text(p, p.text.rstrip() +
                 " Two further quantities belong with this one and are "
                 "defined here because they are the right comparisons to "
                 "make. The first is the same risk-coverage curve computed "
                 "from a maximum-probability score, that is, ranking posts by "
                 "the largest predicted symptom probability instead of by the "
                 "learned variance; without it the curve in Figure 5 shows "
                 "only that some ordering exists, not that the variance head "
                 "supplies a better one than the classifier's own confidence. "
                 "The second is the expected calibration error, the mean "
                 "absolute gap between confidence and accuracy over "
                 "equal-width probability bins, which measures whether the "
                 "scores are calibrated and not merely ordered. Neither is "
                 "reported in this paper. Both require per-example "
                 "predictions, which were not retained, so the abstention "
                 "result should be read as evidence that the variance head "
                 "ranks its own reliability usefully and not as evidence that "
                 "it beats a softmax-confidence baseline or that it is "
                 "calibrated. Section 5.2 records the re-run that would "
                 "settle both.")
        note("4.2 defines a max-probability abstention baseline and expected "
             "calibration error, and reports both as not computed")

    # ---- 5. soften criticisms this paper shares
    softened = 0
    for needle, old, new in [
        ("DepGLM",
         "but predict the degrees with an unconstrained head so that the "
         "ordering between adjacent degrees is not enforced",
         "and predict the degrees with an unconstrained head, so the "
         "ordering between adjacent degrees is not enforced by the "
         "architecture; the present work shares that property, as "
         "Section 3.5 states"),
        ("Rehman et al.",
         "and no control establishes that the estimated structure "
         "outperforms a random graph of equal density",
         "and no control establishes that the estimated structure "
         "outperforms a random graph of equal density, an omission the "
         "present work shares and records in Section 5.2"),
    ]:
        for p in doc.paragraphs:
            if needle in p.text and old in p.text:
                set_text(p, p.text.replace(old, new))
                softened += 1
                break
    # Table 1's gap column carries two more criticisms the paper now shares,
    # since the noise channel was removed and no paired test is reported.
    t1 = doc.tables[0]
    cell_fixes = [
        ("[23]",
         "Weak labels consumed as certain; per-class labeller bias neither "
         "estimated nor inverted",
         "Weak labels consumed as certain; labeller bias not estimated, as "
         "in the present work"),
        ("[28]",
         "Diagnoses the absence of significance testing without supplying a "
         "paired procedure",
         "Diagnoses the absence of significance testing; the present work "
         "does not supply one either"),
    ]
    for ref, old, new in cell_fixes:
        for row in t1.rows:
            if row.cells[0].text.strip().startswith(ref):
                c = row.cells[4]
                if old in c.text:
                    set_cell(c, new, 9.0)
                    softened += 1
                break
    if softened:
        note(f"softened {softened} literature criticisms that this paper "
             f"shares, including two in the Table 1 gap column")

    # ---- 6. statistics not supported by the notebooks
    removed = []
    for old, new, tag in [
        ("69.3 per cent of the posts carrying an anxiety label contain the "
         "word anxiety, so a model can attain substantial apparent "
         "performance on that label by lexical matching alone",
         "a model can attain apparent performance on a disorder label by "
         "matching the disorder name in the text, and no keyword-masked "
         "control was run to bound that effect",
         "69.3% keyword-leakage figure"),
        ("132,850 of those authors contribute two or more posts and together "
         "account for 327,866 of the posts, which is the measurement "
         "determining whether an author-level consistency objective is "
         "feasible at all",
         "a minority of authors contribute more than one post, which bounds "
         "what an author-level consistency objective could use",
         "132,850-author count"),
    ]:
        for p in doc.paragraphs:
            if old in p.text:
                set_text(p, p.text.replace(old, new))
                removed.append(tag)
                break
    # development anecdotes: keep the engineering point, drop the unsourced number
    anecdotes = [
        ("under unweighted cross-entropy both graded heads were measured "
         "during development to collapse onto their majority level, producing "
         "a quadratic weighted kappa of exactly zero while the loss continued "
         "to fall",
         "under unweighted cross-entropy both graded heads collapse onto "
         "their majority level, since the lowest severity band alone holds "
         "71.9 per cent of the training instances"),
        ("the symptom macro-F1 was measured at 0.336 during development, and "
         "the same predictions scored 0.603 once each threshold was tuned on "
         "validation data",
         "a fixed cut leaves well-ranked symptoms scoring near zero F1 "
         "whenever their probabilities sit below one half"),
    ]
    for old, new in anecdotes:
        for p in doc.paragraphs:
            if old in p.text:
                set_text(p, p.text.replace(old, new))
                removed.append("development anecdote")
                break
    if removed:
        note("removed unsupported statistics: " + "; ".join(removed))

    # ---- 8. reference [21] appeared online first
    for p in doc.paragraphs:
        if p.text.strip().startswith("[21]") and "online first" not in p.text:
            set_text(p, p.text.rstrip() +
                     " (published online 5 December 2025 in volume 16; the "
                     "year and volume are as the publisher records them.)")
            note("[21] annotated with its online-first date")
            break

    # ---- 10. the role of the language model
    i = find_para(doc, "an instruction-tuned model executed locally supplies "
                       "weak supervision")
    p = doc.paragraphs[i]
    if "only role" not in p.text:
        set_text(p, p.text.replace(
            "for the eleven symptoms no public corpus annotates.",
            "for the eleven symptoms no public corpus annotates. This is the "
            "only role a large language model plays in the framework: it "
            "annotates training material offline and is not part of the "
            "deployed model, which is a fine-tuned encoder of 125.8 million "
            "parameters."))
        note("the language model's limited role made explicit")


# --------------------------------------------------------------------------- #
# 9. register
# --------------------------------------------------------------------------- #
DETERMINERS = ("a ", "an ", "the ", "its ", "their ", "one ", "any ", "some ")
PREPS = ("by ", "from ", "to ", "as ", "of ", "in ", "on ", "with ", "at ")


def thin_rather_than(doc) -> None:
    """Replace the formulaic cases, leave the ones where it reads best."""
    before = after = 0
    for p in doc.paragraphs:
        t = p.text
        if "rather than" not in t:
            continue
        before += t.count("rather than")

        def sub(m):
            tail = m.group(1)
            if tail.startswith(DETERMINERS) or tail.endswith("ing"):
                return "instead of " + tail
            if tail.startswith(PREPS):
                # marker, so the comma can be placed correctly afterwards
                return "\x00" + tail
            return m.group(0)

        new = re.sub(r"rather than (\S+ |\S+$)", sub, t)
        # "A, not B" needs exactly one comma before "not", whether or not the
        # original already had one before "rather than"
        new = re.sub(r"\s*,?\s*\x00", ", not ", new)
        new = re.sub(r",\s*,", ",", new).replace("  ", " ")
        if new != t:
            set_text(p, new)
        after += new.count("rather than")
    note(f'"rather than" reduced from {before} to {after} occurrences')


def split_long_sentences(doc) -> None:
    """Split the handful of sentences that run past ~55 words at a semicolon."""
    split = 0
    for p in doc.paragraphs:
        t = p.text
        if len(t) < 200:
            continue
        out = []
        for sent in re.split(r"(?<=\.)\s+", t):
            if len(sent.split()) > 55 and "; " in sent:
                head, _, tail = sent.partition("; ")
                if len(head.split()) > 12 and len(tail.split()) > 12:
                    sent = head + ". " + tail[0].upper() + tail[1:]
                    split += 1
            out.append(sent)
        new = " ".join(out)
        if new != t:
            set_text(p, new)
    note(f"split {split} sentences longer than 55 words at a semicolon")


# --------------------------------------------------------------------------- #
# 7. closing statements
# --------------------------------------------------------------------------- #
def tidy_whitespace(doc) -> None:
    fixed = 0
    for p in doc.paragraphs:
        t = p.text
        new = re.sub(r" {2,}", " ", t).replace(" .", ".").replace(" ,", ",")
        if new != t:
            set_text(p, new)
            fixed += 1
    if fixed:
        note(f"tidied whitespace in {fixed} paragraphs")


def add_closing_statements(doc) -> None:
    if any("5.3 Ethical Considerations" in p.text for p in doc.paragraphs):
        note("closing statements already present, left alone")
        return
    last = len(doc.paragraphs) - 1
    for i, p in enumerate(doc.paragraphs):
        if p.text.strip().startswith("6. References"):
            last = i - 1
            break

    blocks = [
        ("5.3 Ethical Considerations", True,
         "This study uses only publicly available, previously published "
         "collections of social media text, each obtained under the licence "
         "its authors attach and used within the terms of that licence. No "
         "new data were collected from human participants, so no "
         "institutional review was sought; for work on clinical text that is "
         "itself a limitation. Author identifiers are pseudonymous handles "
         "and are used for one purpose only, keeping a writer's posts on one "
         "side of a split. No attempt was made to deanonymise any author and "
         "no author-level profile is published. The weak-labelling model was "
         "executed locally and not through a hosted interface, so no post "
         "left the processing environment, as several of the constituent data "
         "agreements require. The intended use of this work is "
         "methodological. It is not a screening instrument and must not be "
         "deployed as one. The suicide-risk head is trained on 340 instances "
         "and evaluated on 74, it does not lead the comparison on that task, "
         "and its seed dispersion exceeds every difference this study can "
         "resolve. A false negative from such a component is a missed crisis. "
         "The abstention mechanism of Section 4.6 is offered as a way of "
         "referring uncertain cases to a clinician, not as a licence to "
         "automate the decision. Any application to real users would require "
         "clinician oversight, prospective validation and subgroup auditing "
         "that this paper does not provide."),
        ("5.4 Data Availability", True,
         "No new data were created by this study. The seven gold-labelled "
         "collections and the unlabelled pool are third-party datasets and "
         "must be obtained from their original authors under their original "
         "terms; this paper redistributes none of them. The harmonisation "
         "pipeline that reproduces the canonical table, the symptom "
         "ontology, the split assignment and the verification assertions is "
         "released with the code, so the corpus used here can be rebuilt "
         "from the published sources without reference to any intermediate "
         "artefact. The derived run outputs from which every number in "
         "Section 4 is read, namely the per-seed metric files, the per-epoch "
         "training histories and the aggregated leaderboards, are included in "
         "the repository. Per-example model predictions were not retained, "
         "which is why the paired tests of Section 4.2 are not reported."),
        ("5.5 Code Availability", True,
         "The source for the preprocessing pipeline, the comorbidity graph, "
         "the weak-labelling stage, the model, the six-rung ablation ladder, "
         "the reference models, the metric suite and the manuscript generator "
         "is available in the project repository, together with the unit "
         "tests that pin the behaviour of the fingerprinting, ordinal and "
         "reference-model components. Every number reported in Section 4 is "
         "read from the run outputs at build time by a single module, so the "
         "tables, the figures and the prose cannot disagree, and three "
         "checking scripts verify that property, the numbering of the "
         "equations, figures and tables, and the absence of placeholders. "
         "The editable sources of the two schematic figures are included "
         "alongside the rendered images."),
    ]
    from docx.text.paragraph import Paragraph
    anchor_el = doc.paragraphs[last]._p
    parent = doc.paragraphs[last]._parent
    for heading, bold, body in blocks:
        for text, is_bold in ((heading, bold), (body, False)):
            el = copy.deepcopy(anchor_el)
            anchor_el.addnext(el)
            q = Paragraph(el, parent)
            for r in list(q.runs):
                r._r.getparent().remove(r._r)
            q.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            run = q.add_run(text)
            run.bold = is_bold
            run.font.name, run.font.size = FONT, SIZE
            anchor_el = el
    note("added 5.3 Ethical Considerations, 5.4 Data Availability and "
         "5.5 Code Availability")


if __name__ == "__main__":
    main()
