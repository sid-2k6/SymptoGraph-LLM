#!/usr/bin/env python3
"""The manuscript's two schematic figures, in the visual language of the
reference document.

Each diagram is declared once as data and emitted twice:

    docs/diagrams/*.drawio   editable source, openable at app.diagrams.net
    docs/paper_figures/*.png 300 dpi raster for the manuscript

Emitting both from one declaration is the point: the editable source and the
figure in the paper cannot drift apart. No draw.io binary is available in this
environment, so the PNG is drawn by matplotlib using the same shape vocabulary
the reference uses (extruded boxes, cylinders for stores, hexagons for named
outputs, dashed regions for novelty groupings).

Coordinates are draw.io units (points), origin top-left, which is why the
matplotlib renderer inverts y.
"""
from __future__ import annotations

import html
import os
from dataclasses import dataclass, field

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.path import Path as MPath

# --------------------------------------------------------------------------- #
# palette, taken from the reference figures
# --------------------------------------------------------------------------- #
PALETTE = {
    "navy":   ("#1f3864", "#d9e2f3"),
    "green":  ("#2e6b4f", "#dff0e4"),
    "purple": ("#5b3a87", "#e5dcf2"),
    "gold":   ("#9c7a10", "#fdf2d0"),
    "red":    ("#9e2b25", "#fadddb"),
    "teal":   ("#1f6b6b", "#d7eded"),
    "plum":   ("#7b2d5e", "#f7dcea"),
    "slate":  ("#33475b", "#e3e9ee"),
}
EDGE = "#2c3e50"
FONT = "serif"


@dataclass
class Shape:
    kind: str                 # box | cube | cylinder | hexagon | bar | header
    x: float
    y: float
    w: float
    h: float
    text: str
    colour: str = "slate"
    bold: bool = False
    fs: float = 9.0
    ident: str = ""


@dataclass
class Region:
    x: float
    y: float
    w: float
    h: float
    label: str
    colour: str = "navy"


@dataclass
class Edge:
    pts: list[tuple[float, float]]
    label: str = ""
    dashed: bool = False
    arrow: bool = True


@dataclass
class Diagram:
    name: str
    width: float
    height: float
    shapes: list[Shape] = field(default_factory=list)
    regions: list[Region] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)

    def add(self, s: Shape) -> Shape:
        if not s.ident:
            s.ident = f"n{len(self.shapes) + 2}"
        self.shapes.append(s)
        return s


# --------------------------------------------------------------------------- #
# matplotlib renderer
# --------------------------------------------------------------------------- #
def _cube(ax, s: Shape, line, fill, d=7.0):
    """Extruded box: front face plus top and right parallelograms."""
    x, y, w, h = s.x, -s.y - s.h, s.w, s.h
    ax.add_patch(mpatches.Rectangle((x, y), w, h, facecolor=fill,
                                    edgecolor=line, lw=1.0, zorder=3))
    top = [(x, y + h), (x + d, y + h + d), (x + w + d, y + h + d),
           (x + w, y + h)]
    rgt = [(x + w, y), (x + w + d, y + d), (x + w + d, y + h + d),
           (x + w, y + h)]
    for poly, shade in ((top, 0.92), (rgt, 0.84)):
        ax.add_patch(mpatches.Polygon(poly, closed=True, facecolor=fill,
                                      edgecolor=line, lw=0.9, zorder=2,
                                      alpha=shade))
    ax.text(x + w / 2, y + h / 2, s.text, ha="center", va="center",
            fontsize=s.fs, family=FONT, zorder=4,
            fontweight="bold" if s.bold else "normal")


def _cylinder(ax, s: Shape, line, fill):
    x, y, w, h = s.x, -s.y - s.h, s.w, s.h
    e = min(14.0, h * 0.26)
    ax.add_patch(mpatches.Rectangle((x, y + e / 2), w, h - e, facecolor=fill,
                                    edgecolor="none", zorder=3))
    for cy in (y + h - e / 2, y + e / 2):
        ax.add_patch(mpatches.Ellipse((x + w / 2, cy), w, e, facecolor=fill,
                                      edgecolor=line, lw=1.0, zorder=3))
    ax.plot([x, x], [y + e / 2, y + h - e / 2], color=line, lw=1.0, zorder=4)
    ax.plot([x + w, x + w], [y + e / 2, y + h - e / 2], color=line, lw=1.0,
            zorder=4)
    ax.text(x + w / 2, y + h / 2 - e * 0.1, s.text, ha="center", va="center",
            fontsize=s.fs, family=FONT, zorder=5,
            fontweight="bold" if s.bold else "normal")


def _hexagon(ax, s: Shape, line, fill):
    x, y, w, h = s.x, -s.y - s.h, s.w, s.h
    c = min(18.0, w * 0.17)
    pts = [(x + c, y), (x + w - c, y), (x + w, y + h / 2),
           (x + w - c, y + h), (x + c, y + h), (x, y + h / 2)]
    ax.add_patch(mpatches.Polygon(pts, closed=True, facecolor=fill,
                                  edgecolor=line, lw=1.2, zorder=3))
    ax.text(x + w / 2, y + h / 2, s.text, ha="center", va="center",
            fontsize=s.fs, family=FONT, color=line, zorder=4,
            fontweight="bold")


def _plain(ax, s: Shape, line, fill, rounded=True, header=False):
    x, y, w, h = s.x, -s.y - s.h, s.w, s.h
    style = "round,pad=0,rounding_size=4" if rounded else "square,pad=0"
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle=style, facecolor=fill, edgecolor=line,
        lw=1.1, zorder=3, mutation_aspect=1))
    ax.text(x + w / 2, y + h / 2, s.text, ha="center", va="center",
            fontsize=s.fs, family=FONT, zorder=4,
            color="white" if header else "black",
            fontweight="bold" if (s.bold or header) else "normal")


def render_png(d: Diagram, path: str, dpi: int = 300) -> None:
    scale = 100.0
    fig, ax = plt.subplots(figsize=(d.width / scale, d.height / scale))
    ax.set_xlim(0, d.width)
    ax.set_ylim(-d.height, 0)
    ax.axis("off")
    ax.set_aspect("equal")

    for r in d.regions:
        line, _ = PALETTE[r.colour]
        ax.add_patch(mpatches.FancyBboxPatch(
            (r.x, -r.y - r.h), r.w, r.h,
            boxstyle="round,pad=0,rounding_size=6",
            facecolor="none", edgecolor=line, lw=1.3, linestyle=(0, (6, 4)),
            zorder=1))
        if r.label:
            # hang the caption clear below the dashed border, as the
            # reference does, rather than letting it sit on the line
            ax.text(r.x + 5, -r.y - r.h - 6, r.label, fontsize=8.6,
                    family=FONT, style="italic", color=line, va="top",
                    ha="left", zorder=5)

    for e in d.edges:
        xs = [p[0] for p in e.pts]
        ys = [-p[1] for p in e.pts]
        ax.plot(xs, ys, color=EDGE, lw=1.1, zorder=2,
                linestyle=(0, (4, 3)) if e.dashed else "-",
                solid_capstyle="round")
        if e.arrow and len(e.pts) >= 2:
            ax.annotate("", xy=(xs[-1], ys[-1]), xytext=(xs[-2], ys[-2]),
                        arrowprops=dict(arrowstyle="-|>", color=EDGE, lw=1.1,
                                        shrinkA=0, shrinkB=0), zorder=3)
        if e.label:
            mid = len(e.pts) // 2
            ax.text(xs[mid] + 3, ys[mid] + 4, e.label, fontsize=6.8,
                    family=FONT, style="italic", color="#555", zorder=5)

    for s in d.shapes:
        line, fill = PALETTE[s.colour]
        if s.kind == "cube":
            _cube(ax, s, line, fill)
        elif s.kind == "cylinder":
            _cylinder(ax, s, line, fill)
        elif s.kind == "hexagon":
            _hexagon(ax, s, line, fill)
        elif s.kind == "header":
            _plain(ax, s, line, line, rounded=False, header=True)
        elif s.kind == "bar":
            _plain(ax, s, line, fill, rounded=False)
        else:
            _plain(ax, s, line, fill)

    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white",
                pad_inches=0.04)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# draw.io renderer
# --------------------------------------------------------------------------- #
DRAWIO_STYLE = {
    "box": "rounded=1;arcSize=8;whiteSpace=wrap;html=1;",
    "cube": "shape=cube;whiteSpace=wrap;html=1;boundedPerimeter=1;"
            "backgroundOutline=1;darkOpacity=0.05;darkOpacity2=0.1;size=8;",
    "cylinder": "shape=cylinder3;whiteSpace=wrap;html=1;boundedPerimeter=1;"
                "size=12;",
    "hexagon": "shape=hexagon;perimeter=hexagonPerimeter2;whiteSpace=wrap;"
               "html=1;fixedSize=1;",
    "bar": "rounded=0;whiteSpace=wrap;html=1;",
    "header": "rounded=0;whiteSpace=wrap;html=1;",
}


def render_drawio(d: Diagram, path: str) -> None:
    cells: list[str] = []

    for i, r in enumerate(d.regions):
        line, _ = PALETTE[r.colour]
        cells.append(
            f'<mxCell id="r{i}" value="{html.escape(r.label)}" '
            f'style="rounded=1;arcSize=4;dashed=1;dashPattern=8 4;'
            f'fillColor=none;strokeColor={line};strokeWidth=1.5;'
            f'verticalAlign=bottom;align=left;spacingLeft=6;'
            f'spacingBottom=-20;fontStyle=2;fontSize=11;fontColor={line};'
            f'html=1;" vertex="1" parent="1">'
            f'<mxGeometry x="{r.x:.0f}" y="{r.y:.0f}" '
            f'width="{r.w:.0f}" height="{r.h:.0f}" as="geometry"/></mxCell>')

    for s in d.shapes:
        line, fill = PALETTE[s.colour]
        base = DRAWIO_STYLE.get(s.kind, DRAWIO_STYLE["box"])
        if s.kind == "header":
            style = (f"{base}fillColor={line};strokeColor={line};"
                     f"fontColor=#ffffff;fontStyle=1;")
        else:
            style = (f"{base}fillColor={fill};strokeColor={line};"
                     f"fontStyle={'1' if s.bold else '0'};")
        style += f"fontSize={s.fs + 2:.0f};fontFamily=Times New Roman;"
        cells.append(
            f'<mxCell id="{s.ident}" value="{html.escape(s.text)}" '
            f'style="{style}" vertex="1" parent="1">'
            f'<mxGeometry x="{s.x:.0f}" y="{s.y:.0f}" '
            f'width="{s.w:.0f}" height="{s.h:.0f}" as="geometry"/></mxCell>')

    for i, e in enumerate(d.edges):
        dash = "dashed=1;dashPattern=6 3;" if e.dashed else ""
        pts = "".join(
            f'<mxPoint x="{x:.0f}" y="{y:.0f}"/>' for x, y in e.pts[1:-1])
        waypoints = (f'<Array as="points">{pts}</Array>' if pts else "")
        x0, y0 = e.pts[0]
        x1, y1 = e.pts[-1]
        cells.append(
            f'<mxCell id="e{i}" value="{html.escape(e.label)}" '
            f'style="edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;'
            f'strokeColor={EDGE};strokeWidth=1.2;{dash}'
            f'endArrow={"blockThin" if e.arrow else "none"};endFill=1;'
            f'fontSize=9;fontStyle=2;fontFamily=Times New Roman;" '
            f'edge="1" parent="1">'
            f'<mxGeometry relative="1" as="geometry">'
            f'<mxPoint x="{x0:.0f}" y="{y0:.0f}" as="sourcePoint"/>'
            f'<mxPoint x="{x1:.0f}" y="{y1:.0f}" as="targetPoint"/>'
            f'{waypoints}</mxGeometry></mxCell>')

    body = "\n        ".join(cells)
    xml = f"""<mxfile host="app.diagrams.net" type="device">
  <diagram name="{html.escape(d.name)}" id="{d.name.replace(' ', '-')}">
    <mxGraphModel dx="{d.width:.0f}" dy="{d.height:.0f}" grid="0"
        gridSize="10" guides="1" tooltips="1" connect="1" arrows="1"
        fold="1" page="1" pageScale="1" pageWidth="{d.width:.0f}"
        pageHeight="{d.height:.0f}" math="0" shadow="0">
      <root>
        <mxCell id="0"/>
        <mxCell id="1" parent="0"/>
        {body}
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(xml)


# --------------------------------------------------------------------------- #
# Figure 1: the pipeline, as a serpentine of numbered stages
# --------------------------------------------------------------------------- #
def workflow(weak_rows: str, gold_train: str) -> Diagram:
    COLW, GAP, PAD = 206, 34, 12
    HDR, SUBH, SUBG = 40, 52, 9
    COLH = HDR + SUBG + 3 * (SUBH + SUBG) + SUBH
    top_y, bot_y = 26, 26 + COLH + 52
    d = Diagram("SymptoGraph-LLM workflow",
                24 + 5 * COLW + 4 * GAP + 24, bot_y + COLH + 32)

    stages = [
        ("1. Corpora", "navy",
         ["7 public collections", "22-symptom ontology",
          "1,017,959-post pool"], "19,762 posts"),
        ("2. Harmonise", "green",
         ["200-char prefix hash", "3-state symptom mask",
          "overlap routes merged"], "one row per post"),
        ("3. Partition", "purple",
         ["author-disjoint buckets", "strictest split wins",
          "asserted leakage-free"], f"{gold_train} train"),
        ("4. Weak labels", "gold",
         ["Qwen2.5-7B via vLLM", "11 unannotated symptoms",
          "train split only"], f"{weak_rows} posts"),
        ("5. Encode", "red",
         ["MentalRoBERTa", "192 tokens", "hidden 768"], "token states"),
    ]
    back = [
        ("10. Reporting", "teal",
         ["test split read once", "provenance recorded",
          "negative results kept"], "tables 7-10"),
        ("9. Analysis", "navy",
         ["L0-L5 ladder", "3 reference models", "risk-coverage curve"],
         "9 of 11 metrics"),
        ("8. Objective", "red",
         ["masked symptom BCE", "class-weighted ordinal",
          "contrastive + variance"], "7 weighted terms"),
        ("7. Heads", "purple",
         ["22 symptoms", "severity 4, risk 5", "stress 1, abstain"],
         "predictions"),
        ("6. Structure", "green",
         ["concept bottleneck", "comorbidity graph",
          "profile contrastive"], "concept vector"),
    ]

    def column(x, y, title, colour, items, out):
        line, fill = PALETTE[colour]
        d.add(Shape("header", x, y, COLW, HDR, title, colour, bold=True,
                    fs=10.5))
        cy = y + HDR + SUBG
        for it in items:
            d.add(Shape("box", x + PAD, cy, COLW - 2 * PAD, SUBH, it, colour,
                        fs=8.5))
            cy += SUBH + SUBG
        d.add(Shape("hexagon", x + PAD, cy, COLW - 2 * PAD, SUBH, out,
                    colour, fs=9.2))
        bot = cy + SUBH
        d.regions.append(Region(x - 6, y - 6, COLW + 12, bot - y + 12, "",
                                colour))
        return bot

    xs = [24 + i * (COLW + GAP) for i in range(5)]
    for x, (t, c, items, out) in zip(xs, stages):
        column(x, top_y, t, c, items, out)
    for x, (t, c, items, out) in zip(xs, back):
        column(x, bot_y, t, c, items, out)

    # Connectors run in the gaps between regions, never across a box.
    mid_top = top_y + COLH / 2
    mid_bot = bot_y + COLH / 2
    for i in range(4):
        d.edges.append(Edge([(xs[i] + COLW + 7, mid_top),
                             (xs[i + 1] - 7, mid_top)]))
    # stage 5 wraps down into stage 6, which sits at the right of the lower row
    d.edges.append(Edge([(xs[4] + COLW / 2, top_y + COLH + 7),
                         (xs[4] + COLW / 2, bot_y - 7)], dashed=True))
    for i in range(4, 0, -1):
        d.edges.append(Edge([(xs[i] - 7, mid_bot),
                             (xs[i - 1] + COLW + 7, mid_bot)]))
    return d


# --------------------------------------------------------------------------- #
# Figure 2: the architecture, as extruded stages inside novelty regions
# --------------------------------------------------------------------------- #
def architecture(weak_rows: str, params_m: str) -> Diagram:
    d = Diagram("SymptoGraph-LLM architecture", 1270, 900)
    BW, BH = 168, 66

    def row(y, items, colour, region_label, x0=38, gap=26, kinds=None,
            region_extra=0.0):
        xs = []
        for i, txt in enumerate(items):
            x = x0 + i * (BW + gap)
            kind = (kinds[i] if kinds else "cube")
            d.add(Shape(kind, x, y, BW, BH, txt, colour, fs=8.6))
            xs.append(x)
        if region_label is not None:
            w = (len(items) - 1) * (BW + gap) + BW + region_extra
            d.regions.append(Region(x0 - 14, y - 14, w + 28, BH + 28,
                                    region_label, colour))
        return xs

    # encoding
    y1 = 30
    e = row(y1, ["Post text  $x_i$", "Tokenise, 192", "MentalRoBERTa\n12 layers",
                 "$H_i$  192 x 768"], "navy", "Text Encoding")
    for i in range(3):
        d.edges.append(Edge([(e[i] + BW, y1 + BH / 2),
                             (e[i + 1], y1 + BH / 2)]))

    # N1 bottleneck
    y2 = 164
    b = row(y2, ["Symptom queries\nSBERT definitions", "Definition\nprojection",
                 "Symptom attention\n$\\alpha_{ik}$",
                 "Per-symptom scoring\n$c_{ik}=w_k^{\\top}Z_{ik}$"],
            "plum", "Clinical Concept Bottleneck (N1)",
            region_extra=BW + 26)
    d.add(Shape("hexagon", b[3] + BW + 26, y2, BW, BH,
                "Concept vector\n$c_i \\in R^{22}$", "plum", fs=8.6))
    for i in range(3):
        d.edges.append(Edge([(b[i] + BW, y2 + BH / 2),
                             (b[i + 1], y2 + BH / 2)]))
    d.edges.append(Edge([(b[3] + BW, y2 + BH / 2),
                         (b[3] + BW + 26, y2 + BH / 2)]))
    d.edges.append(Edge([(e[3] + BW / 2, y1 + BH),
                         (e[3] + BW / 2, y2 - 14)], dashed=True))

    # N2 graph
    y3 = 300
    g = row(y3, ["DSM-5 prior $A_0$", "Learned residual\n$\\Delta A$, $L_1$",
                 "Normalise\n$relu$, $+I$, $D^{-1/2}$",
                 "Propagate\n2 layers, LayerNorm"],
            "green", "Comorbidity Graph (N2)",
            kinds=["cylinder", "cube", "cube", "cube"],
            region_extra=BW + 26)
    for i in range(3):
        d.edges.append(Edge([(g[i] + BW, y3 + BH / 2),
                             (g[i + 1], y3 + BH / 2)]))
    d.add(Shape("hexagon", g[3] + BW + 26, y3, BW, BH,
                "Refined concepts", "green", fs=8.8))
    d.edges.append(Edge([(g[3] + BW, y3 + BH / 2),
                         (g[3] + BW + 26, y3 + BH / 2)]))

    # Heads sit directly under the graph row so the concept logits reach the
    # symptom head by a short vertical drop. The symptom head is rightmost,
    # aligned under the refined-concepts hexagon, because it is fed by the
    # concept logits and not by the pooled vector.
    y4 = 440
    hx = g[3] + BW + 26
    pool = d.add(Shape("cube", 38, y4, BW, BH,
                       "Concept-gated pool\n$g_i \\in R^{768}$", "purple",
                       fs=8.6))
    sev = d.add(Shape("cube", 38 + BW + 26, y4, BW, BH,
                      "Severity 4 / Risk 5\nclass-weighted CE", "purple",
                      fs=8.6))
    stress = d.add(Shape("cube", 38 + 2 * (BW + 26), y4, BW, BH,
                         "Stress head\nsigmoid", "purple", fs=8.6))
    sym = d.add(Shape("cube", hx, y4, BW, BH,
                      "Symptom head\n22 logits", "purple", fs=8.6))
    d.regions.append(Region(38 - 14, y4 - 14, hx + BW - 38 + 28, BH + 28,
                            "Prediction Heads", "purple"))
    # refined concepts drop straight into the symptom head
    d.edges.append(Edge([(hx + BW / 2, y3 + BH), (hx + BW / 2, y4 - 14)],
                        label="concept logits"))
    d.edges.append(Edge([(pool.x + BW, y4 + BH / 2), (sev.x, y4 + BH / 2)]))
    d.edges.append(Edge([(sev.x + BW, y4 + BH / 2), (stress.x, y4 + BH / 2)]))

    # N5 on the left, N6 on the right, both below the heads
    y5 = 578
    c = row(y5, ["Projection head\n$\\pi_i \\in R^{128}$",
                 "InfoNCE, $T=0.1$\n$J(y_i,y_j)\\geq 0.5$"],
            "gold", "Symptom-Profile Contrastive (N5)")
    d.edges.append(Edge([(c[0] + BW, y5 + BH / 2), (c[1], y5 + BH / 2)]))

    u = row(y5, ["Variance head\n$u_k^{\\top}sg[g_i]$",
                 "Abstention score"],
            "teal", "Learned Uncertainty (N6)",
            x0=hx - BW - 26, kinds=["cube", "hexagon"])
    d.edges.append(Edge([(u[0] + BW, y5 + BH / 2), (u[1], y5 + BH / 2)]))
    # the variance head reads the pooled vector through a stopped gradient
    d.edges.append(Edge([(pool.x + BW / 2, y4 + BH),
                         (pool.x + BW / 2, y5 - 32),
                         (u[0] + BW / 2, y5 - 32),
                         (u[0] + BW / 2, y5 - 14)],
                        label="stop-gradient", dashed=True))
    d.edges.append(Edge([(pool.x + BW / 2, y4 + BH),
                         (pool.x + BW / 2, y5 - 14)], arrow=False))

    # objective bar, with the weak stream feeding its symptom term
    y6 = 726
    weak = d.add(Shape("cylinder", hx + 72, y6 - 4, BW, 64,
                       f"N3  Weak stream\n{weak_rows} posts, 11\n"
                       f"unlabelled symptoms", "navy", fs=7.8))
    d.add(Shape("bar", 38, y6, hx - 38 - 26, 56,
                "Multi-task objective   "
                "$L = L_{sym} + \\lambda_s L_{sev} + \\lambda_r L_{risk} + "
                "\\lambda_t L_{str} + \\lambda_c L_{con} + \\lambda_u L_{unc} "
                "+ \\lambda_g \\|\\Delta A\\|_1$",
                "slate", bold=False, fs=8.6))
    d.edges.append(Edge([(weak.x, y6 + 28), (hx - 26, y6 + 28)],
                        label="to $L_{sym}$"))
    for src in (sym.x + BW / 2, sev.x + BW / 2, stress.x + BW / 2):
        d.edges.append(Edge([(src, y4 + BH), (src, y6 - 6)], dashed=True))
    d.edges.append(Edge([(c[1] + BW / 2, y5 + BH), (c[1] + BW / 2, y6 - 6)],
                        dashed=True))
    d.add(Shape("box", 38, y6 + 84, 240, 44,
                f"Update {params_m} M parameters", "slate", fs=9.0,
                bold=True))
    d.edges.append(Edge([(158, y6 + 56), (158, y6 + 84)]))
    return d


# --------------------------------------------------------------------------- #
def build(figs_dir: str, drawio_dir: str, weak_rows: str, gold_train: str,
          params_m: str) -> dict[str, str]:
    os.makedirs(figs_dir, exist_ok=True)
    os.makedirs(drawio_dir, exist_ok=True)
    out = {}
    for key, diagram in (
            ("fig1", workflow(weak_rows, gold_train)),
            ("fig2", architecture(weak_rows, params_m))):
        stem = "figure1_workflow" if key == "fig1" else "figure2_architecture"
        png = os.path.join(figs_dir, f"{stem}.png")
        render_png(diagram, png)
        render_drawio(diagram, os.path.join(drawio_dir, f"{stem}.drawio"))
        out[key] = png
    return out


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    paths = build(os.path.join(root, "docs", "paper_figures"),
                  os.path.join(root, "docs", "diagrams"),
                  "6,000", "12,911", "125.8")
    for k, v in paths.items():
        print(f"{k} -> {v}")
