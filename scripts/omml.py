#!/usr/bin/env python3
"""Convert a constrained LaTeX subset into OMML, the equation markup Word uses.

The manuscript previously shipped equations as PNG images because python-docx
cannot author OMML. Images are not editable in Word and do not reflow, so this
module builds the OMML XML directly instead. The supported subset is exactly
what the manuscript's equations use; anything outside it raises
``UnsupportedLatex`` rather than being silently dropped, so a malformed
equation fails the build instead of shipping wrong.

Supported
    \\frac, \\sqrt, _, ^, \\left...\\right, | |, \\| \\|
    \\sum \\int \\cup \\prod with optional limits, \\min \\max \\log \\exp \\Pr
    \\mathrm \\mathcal \\mathbb \\mathbf \\text, \\tilde \\bar \\hat
    Greek letters, relations, arrows, spacing commands

Usage
    >>> xml = latex_to_omml(r"$a=\\frac{b}{c}$")
"""
from __future__ import annotations

import re

M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
MATH_FONT = "Cambria Math"


class UnsupportedLatex(ValueError):
    """Raised when the input uses a construct this converter does not handle."""


# --------------------------------------------------------------------------- #
# symbols
# --------------------------------------------------------------------------- #
GREEK = {
    "alpha": "\u03b1", "beta": "\u03b2", "gamma": "\u03b3", "delta": "\u03b4",
    "epsilon": "\u03b5", "zeta": "\u03b6", "eta": "\u03b7", "theta": "\u03b8",
    "iota": "\u03b9", "kappa": "\u03ba", "lambda": "\u03bb", "mu": "\u03bc",
    "nu": "\u03bd", "xi": "\u03be", "pi": "\u03c0", "rho": "\u03c1",
    "sigma": "\u03c3", "tau": "\u03c4", "upsilon": "\u03c5", "phi": "\u03c6",
    "chi": "\u03c7", "psi": "\u03c8", "omega": "\u03c9",
    "Gamma": "\u0393", "Delta": "\u0394", "Theta": "\u0398",
    "Lambda": "\u039b", "Xi": "\u039e", "Pi": "\u03a0", "Sigma": "\u03a3",
    "Phi": "\u03a6", "Psi": "\u03a8", "Omega": "\u03a9",
}

RELATIONS = {
    "leq": "\u2264", "geq": "\u2265", "neq": "\u2260", "approx": "\u2248",
    "equiv": "\u2261", "sim": "\u223c", "propto": "\u221d",
    "times": "\u00d7", "cdot": "\u22c5", "pm": "\u00b1", "div": "\u00f7",
    "in": "\u2208", "notin": "\u2209", "subset": "\u2282",
    "subseteq": "\u2286", "cap": "\u2229", "setminus": "\u2216",
    "rightarrow": "\u2192", "leftarrow": "\u2190", "to": "\u2192",
    "Rightarrow": "\u21d2", "Leftarrow": "\u21d0",
    "leftrightarrow": "\u2194", "Leftrightarrow": "\u21d4",
    "infty": "\u221e", "partial": "\u2202", "nabla": "\u2207",
    "forall": "\u2200", "exists": "\u2203", "ldots": "\u2026",
    "cdots": "\u22ef", "langle": "\u27e8", "rangle": "\u27e9",
    "top": "\u22a4", "perp": "\u22a5", "star": "\u22c6",
    "ell": "\u2113", "hbar": "\u210f", "Re": "\u211c", "Im": "\u2111",
    "aleph": "\u2135", "emptyset": "\u2205", "varnothing": "\u2205",
    "oplus": "\u2295", "otimes": "\u2297", "circ": "\u2218",
    "lfloor": "\u230a", "rfloor": "\u230b", "lceil": "\u2308",
    "rceil": "\u2309", "parallel": "\u2225", "angle": "\u2220",
    "varepsilon": "\u03b5", "vartheta": "\u03d1", "varphi": "\u03c6",
    "varsigma": "\u03c2", "varrho": "\u03f1", "mid": "\u2223",
}

# n-ary operators: command -> (glyph, grows with limits above/below)
NARY = {
    "sum": ("\u2211", True),
    "prod": ("\u220f", True),
    "cup": ("\u22c3", True),
    "cap": ("\u22c2", True),
    "int": ("\u222b", False),
    "iint": ("\u222c", False),
    "oint": ("\u222e", False),
}

# upright function names that may carry a subscript limit
FUNCTIONS = {
    "min", "max", "argmin", "argmax", "log", "ln", "exp", "sin", "cos",
    "tan", "det", "dim", "ker", "deg", "gcd", "sup", "inf", "lim", "Pr",
    "softmax", "relu", "sigmoid", "diag", "tr", "rank", "sign",
}

# spacing commands, rendered as literal space or nothing
SPACING = {
    "quad": "\u2003", "qquad": "\u2003\u2003", ",": "\u2009", ";": "\u2009",
    ":": "\u2009", "!": "", " ": " ", "thinspace": "\u2009",
}

ACCENTS = {"tilde": "\u0303", "bar": "\u0304", "hat": "\u0302",
           "vec": "\u20d7", "dot": "\u0307"}

# style commands -> (m:sty value, m:scr value or None)
STYLES = {
    "mathrm": ("p", None),
    "text": ("p", None),
    "operatorname": ("p", None),
    "mathbf": ("b", None),
    "mathit": ("i", None),
    "mathcal": ("p", "script"),
    "mathscr": ("p", "script"),
    "mathbb": ("p", "double-struck"),
    "mathfrak": ("p", "fraktur"),
}

DELIMS = {"(": ")", "[": "]", "\\{": "\\}", "|": "|", "\\|": "\\|",
          "\\langle": "\\rangle"}

DELIM_GLYPH = {"\\{": "{", "\\}": "}", "\\|": "\u2016", "\\langle": "\u27e8",
               "\\rangle": "\u27e9", "\\lvert": "|", "\\rvert": "|",
               "\\lVert": "\u2016", "\\rVert": "\u2016", ".": ""}


def esc(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


# --------------------------------------------------------------------------- #
# tokenizer
# --------------------------------------------------------------------------- #
TOKEN_RE = re.compile(r"""
      (?P<cmd>\\[A-Za-z]+)
    | (?P<esccmd>\\[\\{}|,;:!\ ])
    | (?P<open>\{)
    | (?P<close>\})
    | (?P<sub>_)
    | (?P<sup>\^)
    | (?P<amp>&)
    | (?P<ws>\s+)
    | (?P<char>.)
""", re.X)


def tokenize(src: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for m in TOKEN_RE.finditer(src):
        kind = m.lastgroup
        val = m.group()
        if kind == "ws":
            continue
        if kind == "cmd":
            out.append(("cmd", val[1:]))
        elif kind == "esccmd":
            out.append(("cmd", val[1:]))
        else:
            out.append((kind, val))
    return out


# --------------------------------------------------------------------------- #
# parser -> node tree
# --------------------------------------------------------------------------- #
class Node:
    pass


class Row(Node):
    def __init__(self, items): self.items = items


class Run(Node):
    def __init__(self, text, sty="i", scr=None):
        self.text, self.sty, self.scr = text, sty, scr


class Frac(Node):
    def __init__(self, num, den): self.num, self.den = num, den


class Script(Node):
    def __init__(self, base, sub=None, sup=None):
        self.base, self.sub, self.sup = base, sub, sup


class Nary(Node):
    def __init__(self, glyph, sub, sup, body, limloc):
        self.glyph, self.sub, self.sup = glyph, sub, sup
        self.body, self.limloc = body, limloc


class Rad(Node):
    def __init__(self, body, index=None): self.body, self.index = body, index


class Delim(Node):
    def __init__(self, op, cl, body): self.op, self.cl, self.body = op, cl, body


class Accent(Node):
    def __init__(self, body, mark): self.body, self.mark = body, mark


class Parser:
    def __init__(self, tokens):
        self.t, self.i = tokens, 0

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else None

    def next(self):
        tok = self.peek()
        if tok is None:
            raise UnsupportedLatex("unexpected end of equation")
        self.i += 1
        return tok

    def parse(self, stop_at_close=False) -> Row:
        items: list[Node] = []
        while True:
            tok = self.peek()
            if tok is None:
                break
            if tok[0] == "close":
                if stop_at_close:
                    break
                raise UnsupportedLatex("unbalanced }")
            items.append(self.atom_with_scripts())
        return Row(items)

    def group(self) -> Node:
        """Parse one {...} group, or a single atom if no brace follows."""
        tok = self.peek()
        if tok and tok[0] == "open":
            self.next()
            inner = self.parse(stop_at_close=True)
            nxt = self.next()
            if nxt[0] != "close":
                raise UnsupportedLatex("expected }")
            return inner
        return self.atom()

    def atom_with_scripts(self) -> Node:
        base = self.atom()
        sub = sup = None
        while True:
            tok = self.peek()
            if tok and tok[0] == "sub" and sub is None:
                self.next()
                sub = self.group()
            elif tok and tok[0] == "sup" and sup is None:
                self.next()
                sup = self.group()
            else:
                break
        if sub is None and sup is None:
            return base
        return Script(base, sub, sup)

    def atom(self) -> Node:
        kind, val = self.next()

        if kind == "open":
            self.i -= 1
            return self.group()

        if kind == "char":
            if val in "([":
                close = ")" if val == "(" else "]"
                return self.read_delim(val, close)
            if val == "|":
                return self.read_delim("|", "|")
            return Run(val, sty=self._char_style(val))

        if kind in ("sub", "sup"):
            raise UnsupportedLatex(f"stray {val}")

        if kind == "amp":
            raise UnsupportedLatex("alignment & is not supported")

        # --- commands
        if val == "frac" or val == "dfrac" or val == "tfrac":
            return Frac(self.group(), self.group())
        if val == "sqrt":
            return Rad(self.group())
        if val in STYLES:
            sty, scr = STYLES[val]
            return self.styled(self.group(), sty, scr)
        if val in ACCENTS:
            return Accent(self.group(), ACCENTS[val])
        if val in NARY:
            return self.read_nary(val)
        if val in FUNCTIONS:
            return self.read_function(val)
        if val in GREEK:
            return Run(GREEK[val], sty="i")
        if val in RELATIONS:
            return Run(RELATIONS[val], sty="p")
        if val in SPACING:
            txt = SPACING[val]
            return Row([]) if txt == "" else Run(txt, sty="p")
        if val == "left":
            return self.read_left_right()
        if val == "right":
            raise UnsupportedLatex("\\right without \\left")
        if val in ("{", "}"):
            return Run(val, sty="p")
        if val == "|":
            return self.read_delim("\\|", "\\|")
        if val in ("lvert", "rvert"):
            return Run("|", sty="p")
        if val in ("lVert", "rVert"):
            return Run("\u2016", sty="p")
        if val in ("big", "Big", "bigg", "Bigg", "biggl", "biggr",
                   "displaystyle", "limits", "nolimits"):
            return Row([])

        raise UnsupportedLatex(f"unsupported command \\{val}")

    @staticmethod
    def _char_style(ch: str) -> str:
        # digits, punctuation and operators upright; letters italic
        return "i" if ch.isalpha() else "p"

    def styled(self, node: Node, sty: str, scr: str | None) -> Node:
        """Push a style down onto every Run in a subtree."""
        if isinstance(node, Run):
            return Run(node.text, sty, scr)
        if isinstance(node, Row):
            return Row([self.styled(x, sty, scr) for x in node.items])
        return node

    def read_delim(self, op: str, cl: str) -> Node:
        """Read a bracketed group opened by a bare ( [ or |."""
        depth = 1
        items: list[Node] = []
        while True:
            tok = self.peek()
            if tok is None:
                # unbalanced: treat the opener as a literal character
                return Run(DELIM_GLYPH.get(op, op), sty="p")
            if tok[0] == "char" and tok[1] == cl and op != cl:
                depth -= 1
                if depth == 0:
                    self.next()
                    break
            if tok[0] == "char" and tok[1] == op and op != cl:
                depth += 1
            if op == cl and tok[0] == "char" and tok[1] == cl:
                self.next()
                break
            if tok[0] == "cmd" and tok[1] == cl.lstrip("\\"):
                self.next()
                break
            items.append(self.atom_with_scripts())
        return Delim(DELIM_GLYPH.get(op, op), DELIM_GLYPH.get(cl, cl),
                     Row(items))

    def read_left_right(self) -> Node:
        kind, val = self.next()
        op = val if kind == "char" else "\\" + val
        items: list[Node] = []
        while True:
            tok = self.peek()
            if tok is None:
                raise UnsupportedLatex("\\left without \\right")
            if tok[0] == "cmd" and tok[1] == "right":
                self.next()
                k2, v2 = self.next()
                cl = v2 if k2 == "char" else "\\" + v2
                break
            items.append(self.atom_with_scripts())
        return Delim(DELIM_GLYPH.get(op, op), DELIM_GLYPH.get(cl, cl),
                     Row(items))

    def read_nary(self, name: str) -> Node:
        glyph, grows = NARY[name]
        sub = sup = None
        while True:
            tok = self.peek()
            if tok and tok[0] == "sub" and sub is None:
                self.next()
                sub = self.group()
            elif tok and tok[0] == "sup" and sup is None:
                self.next()
                sup = self.group()
            else:
                break
        body = Row([])
        limloc = "undOvr" if grows else "subSup"
        return Nary(glyph, sub, sup, body, limloc)

    def read_function(self, name: str) -> Node:
        """An upright function name, optionally with a sub/superscript limit."""
        label = Run(name, sty="p")
        sub = sup = None
        while True:
            tok = self.peek()
            if tok and tok[0] == "sub" and sub is None:
                self.next()
                sub = self.group()
            elif tok and tok[0] == "sup" and sup is None:
                self.next()
                sup = self.group()
            else:
                break
        if sub is None and sup is None:
            return label
        return Script(label, sub, sup)


# --------------------------------------------------------------------------- #
# renderer
# --------------------------------------------------------------------------- #
def _rpr(sty: str, scr: str | None) -> str:
    bits = []
    if scr:
        bits.append(f'<m:scr m:val="{scr}"/>')
    if sty:
        bits.append(f'<m:sty m:val="{sty}"/>')
    inner = "".join(bits)
    return f"<m:rPr>{inner}</m:rPr>" if inner else ""


def _wrpr() -> str:
    return (f'<w:rPr><w:rFonts w:ascii="{MATH_FONT}" '
            f'w:hAnsi="{MATH_FONT}"/></w:rPr>')


def _ctrl() -> str:
    return (f'<m:ctrlPr><w:rPr><w:rFonts w:ascii="{MATH_FONT}" '
            f'w:hAnsi="{MATH_FONT}"/><w:i/></w:rPr></m:ctrlPr>')


def render(node: Node) -> str:
    if isinstance(node, Row):
        return "".join(render(x) for x in node.items)

    if isinstance(node, Run):
        if node.text == "":
            return ""
        return (f"<m:r>{_rpr(node.sty, node.scr)}{_wrpr()}"
                f"<m:t xml:space=\"preserve\">{esc(node.text)}</m:t></m:r>")

    if isinstance(node, Frac):
        return (f"<m:f><m:fPr>{_ctrl()}</m:fPr>"
                f"<m:num>{render(node.num)}</m:num>"
                f"<m:den>{render(node.den)}</m:den></m:f>")

    if isinstance(node, Rad):
        deg = render(node.index) if node.index else ""
        hide = "" if node.index else '<m:degHide m:val="1"/>'
        return (f"<m:rad><m:radPr>{hide}{_ctrl()}</m:radPr>"
                f"<m:deg>{deg}</m:deg><m:e>{render(node.body)}</m:e></m:rad>")

    if isinstance(node, Script):
        base = render(node.base)
        if node.sub is not None and node.sup is not None:
            return (f"<m:sSubSup><m:sSubSupPr>{_ctrl()}</m:sSubSupPr>"
                    f"<m:e>{base}</m:e>"
                    f"<m:sub>{render(node.sub)}</m:sub>"
                    f"<m:sup>{render(node.sup)}</m:sup></m:sSubSup>")
        if node.sub is not None:
            return (f"<m:sSub><m:sSubPr>{_ctrl()}</m:sSubPr>"
                    f"<m:e>{base}</m:e>"
                    f"<m:sub>{render(node.sub)}</m:sub></m:sSub>")
        return (f"<m:sSup><m:sSupPr>{_ctrl()}</m:sSupPr>"
                f"<m:e>{base}</m:e>"
                f"<m:sup>{render(node.sup)}</m:sup></m:sSup>")

    if isinstance(node, Nary):
        sub = render(node.sub) if node.sub is not None else ""
        sup = render(node.sup) if node.sup is not None else ""
        hide_sub = '<m:subHide m:val="1"/>' if node.sub is None else ""
        hide_sup = '<m:supHide m:val="1"/>' if node.sup is None else ""
        return (f"<m:nary><m:naryPr>"
                f'<m:chr m:val="{esc(node.glyph)}"/>'
                f'<m:limLoc m:val="{node.limloc}"/>'
                f"{hide_sub}{hide_sup}{_ctrl()}</m:naryPr>"
                f"<m:sub>{sub}</m:sub><m:sup>{sup}</m:sup>"
                f"<m:e>{render(node.body)}</m:e></m:nary>")

    if isinstance(node, Delim):
        op = esc(node.op) if node.op else ""
        cl = esc(node.cl) if node.cl else ""
        return (f"<m:d><m:dPr>"
                f'<m:begChr m:val="{op}"/><m:endChr m:val="{cl}"/>'
                f"{_ctrl()}</m:dPr>"
                f"<m:e>{render(node.body)}</m:e></m:d>")

    if isinstance(node, Accent):
        return (f"<m:acc><m:accPr>"
                f'<m:chr m:val="{esc(node.mark)}"/>{_ctrl()}</m:accPr>'
                f"<m:e>{render(node.body)}</m:e></m:acc>")

    raise UnsupportedLatex(f"cannot render {type(node).__name__}")


def latex_to_omml(latex: str) -> str:
    """Return the ``<m:oMath>`` element for one display equation."""
    src = latex.strip()
    if src.startswith("$") and src.endswith("$"):
        src = src[1:-1]
    body = render(Parser(tokenize(src)).parse())
    return f"<m:oMath>{body}</m:oMath>"


def is_well_formed(omml: str) -> bool:
    """Parse the fragment with the m: and w: namespaces bound, as Word sees it."""
    import xml.etree.ElementTree as ET
    wrapped = (f'<root xmlns:m="{M}" xmlns:w="{W}">{omml}</root>')
    try:
        ET.fromstring(wrapped)
        return True
    except ET.ParseError:
        return False


if __name__ == "__main__":
    tests = [
        r"$a=\frac{b}{c}$",
        r"$\mathcal{L}_{\mathrm{sym}}=\sum_{i=1}^{N}\alpha_i$",
        r"$\tilde{A}=D^{-1/2}AD^{-1/2}$",
        r"$\mathrm{AURC}=\int_{0}^{1}R(\beta)\,d\beta$",
        r"$\sigma(c)\leq\sqrt{D}$",
        r"$R(\beta)=\frac{\sum_i \mathbf{1}[\bar{s}_i\leq q_\beta]\ell_i}"
        r"{\sum_i \mathbf{1}[\bar{s}_i\leq q_\beta]}$",
    ]
    bad = 0
    for t in tests:
        out = latex_to_omml(t)
        ok = is_well_formed(out)
        bad += not ok
        print(f"  {'OK  ' if ok else 'FAIL'} {len(out):5d} chars  {t[:58]}")
    raise SystemExit(1 if bad else 0)
