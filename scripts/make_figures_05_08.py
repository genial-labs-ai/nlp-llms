"""Write the schematic figures of Lectures 5-8 as hand-laid-out SVG.

  images/05-self-attention.svg      one query reads the keys and values (Lecture 5, section 1)
  images/05-causal-mask.svg         lower-triangular attention weights (section 3)
  images/05-multi-head.svg          project, split into heads, attend, concatenate (section 4)
  images/05-transformer-block.svg   pre-norm block and the decoder-only model (section 6)
  images/06-bpe-merges.svg          BPE merges on the toy corpus (Lecture 6, section 2)
  images/06-pretrain-finetune.svg   the two-stage transfer recipe (section 3)
  images/06-clm-vs-mlm.svg          causal versus masked language modeling (section 5)
  images/07-chat-template-mask.svg  chat template and response mask (Lecture 7, section 2)
  images/07-lora-update.svg         the LoRA bypass W0 x + (alpha/r) B A x (section 5)
  images/08-message-list.svg        the stateless message list (Lecture 8, section 2)
  images/08-validate-retry.svg      parse, validate, retry (section 4)
  images/08-tool-loop.svg           the tool-calling loop as a state machine (section 5)

These are schematics, with one exception: the attention weights in
05-causal-mask.svg are measured. They are read from images/05-attention-heads.json,
the weights of the trained Lab 5 mini-GPT on the six characters "To be,"
(layer 3, head 3; see MEASURED_HEAD below), and the figure and its caption say so.
The weights drawn in 05-self-attention.svg are illustrative and labeled as such.

The style block and arrow markers are copied from the Day 1 schematics
(images/02-*.svg to 04-*.svg). Each figure also gets an opaque white backdrop,
so it stays readable if the site ever gains a dark theme. Output is
deterministic: running the script twice gives identical files.

Run:  python scripts/make_figures_05_08.py      (standard library only)
"""

# SVG markup, the copied Day 1 style block and the alt-text strings are kept on
# one line each, so they read as they appear in the output.
# ruff: noqa: E501

from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parent.parent
IMAGES = ROOT / "images"
# The measured head drawn in 05-causal-mask.svg: (layer, head), counted from 1.
# Layer 3, head 3 spreads its weight over several earlier characters, so the
# lower triangle is visibly filled; the near one-hot heads of layer 1 show the
# mask less clearly.
MEASURED_HEAD = (3, 3)

NAVY, ACCENT, INK, MUTED = "#16324f", "#b3541e", "#1f2933", "#52606d"
RULE, DIM, PALE = "#c5ced8", "#9aa5b1", "#f4f6f9"
# Four head colors (stroke/text, light fill). Each head is also numbered, so
# color is never the only cue.
HEADS = [
    ("#1f5a96", "#dbe7f5"),
    ("#b3541e", "#f8e2d3"),
    ("#23734d", "#d9efe3"),
    ("#6a4c9c", "#e8e0f3"),
]

STYLE = """\
    text { font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 15px; fill: #1f2933; }
    .m { font-family: "Source Serif 4", Georgia, "Times New Roman", serif; }
    .s { font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
    .i { font-style: italic; }
    .b { font-weight: 600; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; }
    .note { font-size: 13.5px; fill: #52606d; }
    .acc { fill: #b3541e; }
    .box { fill: #eef2f6; stroke: #16324f; stroke-width: 1.5; }
    .abox { fill: #f8e9de; stroke: #b3541e; stroke-width: 1.5; }
    .dim { fill: #f4f6f9; stroke: #9aa5b1; stroke-width: 1.5; stroke-dasharray: 4 3; }
    .ln { fill: none; stroke: #16324f; stroke-width: 1.5; }
    .aln { fill: none; stroke: #b3541e; stroke-width: 2.5; }
    .dln { fill: none; stroke: #9aa5b1; stroke-width: 1.5; stroke-dasharray: 4 3; }
    .rule { fill: none; stroke: #c5ced8; stroke-width: 1.5; }
    .op { fill: #ffffff; stroke: #16324f; stroke-width: 1.5; }
    .panel { fill: #fafbfc; stroke: #c5ced8; stroke-width: 1.5; }"""


def marker(mid: str, color: str, size: int = 7) -> str:
    return (
        f'<marker id="{mid}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="{size}" '
        f'markerHeight="{size}" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{color}"/></marker>'
    )


MARKERS = [marker("a", NAVY), marker("b", ACCENT, 5), marker("c", MUTED, 6), marker("d", DIM, 6)]
THICK = marker("e", ACCENT, 3)  # for the 4px residual stream


def svg(
    w: int, h: int, title: str, desc: str, body: list[str], defs: list[str] | None = None
) -> str:
    d = "\n    ".join(MARKERS + (defs or []))
    inner = "\n  ".join(body)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
        f'role="img" aria-labelledby="t d">\n'
        f'  <title id="t">{escape(title)}</title>\n'
        f'  <desc id="d">{escape(desc)}</desc>\n'
        f"  <style>\n{STYLE}\n  </style>\n"
        f"  <defs>\n    {d}\n  </defs>\n"
        f'  <rect width="{w}" height="{h}" rx="8" fill="#ffffff"/>\n'
        f"  {inner}\n"
        f"</svg>\n"
    )


# ---------------------------------------------------------------- primitives


def f(v: float) -> str:
    """Format a coordinate: integers without a decimal point."""
    return str(int(v)) if float(v).is_integer() else f"{v:.1f}"


def rect(x, y, w, h, cls="box", rx=6, extra="") -> str:
    return f'<rect class="{cls}" x="{f(x)}" y="{f(y)}" width="{f(w)}" height="{f(h)}" rx="{rx}"{extra}/>'


def line(x1, y1, x2, y2, cls="ln", mk="a", extra="") -> str:
    m = f' marker-end="url(#{mk})"' if mk else ""
    return f'<line class="{cls}" x1="{f(x1)}" y1="{f(y1)}" x2="{f(x2)}" y2="{f(y2)}"{m}{extra}/>'


def path(d, cls="ln", mk="a", extra="") -> str:
    m = f' marker-end="url(#{mk})"' if mk else ""
    return f'<path class="{cls}" d="{d}"{m}{extra}/>'


def dot(x, y, r=3.2, color=NAVY) -> str:
    return f'<circle cx="{f(x)}" cy="{f(y)}" r="{r}" fill="{color}"/>'


def plus(x, y, r=12) -> str:
    return (
        f'<circle class="op" cx="{f(x)}" cy="{f(y)}" r="{r}"/>'
        f'<path class="ln" d="M{f(x - r / 2)},{f(y)} H{f(x + r / 2)} M{f(x)},{f(y - r / 2)} V{f(y + r / 2)}"/>'
    )


def text(x, y, s, cls="", anchor="middle", size=None, extra="") -> str:
    st = f' style="font-size:{size}px"' if size else ""
    c = f' class="{cls}"' if cls else ""
    return f'<text x="{f(x)}" y="{f(y)}" text-anchor="{anchor}"{c}{st}{extra}>{escape(s)}</text>'


# ----------------------------------------------------- math labels as tspans
#
# A tiny TeX-like syntax: Latin and lowercase Greek letters are italic; digits,
# operators and capital Greek are upright. _x or _{..} is a subscript, ^x or
# ^{..} a superscript, \r{..} upright serif, \t{..} upright sans (for words).
# The subscript offset (0.24 em) and size (0.68 em) match the Day 1 figures.

ITALIC = set("αβγδεζηθικλμνξπρστυφχψωℓ")


def _is_comb(c: str) -> bool:
    return unicodedata.combining(c) != 0


def _close(s: str, i: int) -> int:
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return j
    raise ValueError(f"unbalanced braces in {s!r}")


Run = tuple  # (text, italic, baseline shift in em, scale, sans)


def _parse(s: str, shift: float = 0.0, scale: float = 1.0, mode: str = "auto") -> list[Run]:
    out: list[Run] = []
    i = 0
    while i < len(s):
        if s.startswith("\\r{", i) or s.startswith("\\t{", i):
            j = _close(s, i + 2)
            out += _parse(s[i + 3 : j], shift, scale, "roman" if s[i + 1] == "r" else "sans")
            i = j + 1
            continue
        c = s[i]
        if c in "_^" and mode != "sans":
            if s[i + 1] == "{":
                j = _close(s, i + 1)
                inner, i = s[i + 2 : j], j + 1
            elif s[i + 1] == "\\":  # _\r{..}: the group is the argument
                j = _close(s, i + 3)
                inner, i = s[i + 1 : j + 1], j + 1
            else:
                k = i + 2
                while k < len(s) and _is_comb(s[k]):
                    k += 1
                inner, i = s[i + 1 : k], k
            first = scale == 1.0
            sub_scale = scale * (0.68 if first else 0.8)
            d = (0.24 if first else 0.16) if c == "_" else -0.36 * scale
            out += _parse(inner, shift + d, sub_scale, mode)
            continue
        k = i + 1
        while k < len(s) and _is_comb(s[k]):
            k += 1
        ch, i = s[i:k], k
        italic = mode == "auto" and ((ch[0].isascii() and ch[0].isalpha()) or ch[0] in ITALIC)
        out.append((ch, italic, shift, scale, mode == "sans"))
    merged: list[Run] = []
    for run in out:
        if merged and merged[-1][1:] == run[1:]:
            merged[-1] = (merged[-1][0] + run[0],) + run[1:]
        else:
            merged.append(run)
    return merged


def M(x, y, s, size=19, anchor="middle", cls="", extra="") -> str:
    shift, parts = 0.0, []
    for txt, italic, em, scale, sans in _parse(s):
        target = em * size
        attrs = []
        if abs(target - shift) > 0.01:
            attrs.append(f'dy="{target - shift:.1f}"')
        shift = target
        if scale != 1.0:
            attrs.append(f'font-size="{scale * size:.1f}"')
        classes = " ".join(c for c, on in (("i", italic), ("s", sans)) if on)
        if classes:
            attrs.append(f'class="{classes}"')
        a = (" " + " ".join(attrs)) if attrs else ""
        parts.append(f"<tspan{a}>{escape(txt)}</tspan>")
    c = f"m {cls}".strip()
    return f'<text x="{f(x)}" y="{f(y)}" text-anchor="{anchor}" class="{c}" style="font-size:{size}px"{extra}>{"".join(parts)}</text>'


def mix(a: str, b: str, t: float) -> str:
    ca = [int(a[i : i + 2], 16) for i in (1, 3, 5)]
    cb = [int(b[i : i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(ca, cb, strict=True))


# ===================================================================== Lecture 5


def fig_self_attention() -> str:
    W, H = 760, 404
    XQ, XK, XH, XV, XO = 80, 270, 400, 530, 665
    Y = [96, 156, 216, 276, 336]
    alpha = [0.10, 0.20, 0.55, 0.15]  # illustrative; one clearly largest
    bw, bh = 54, 34
    b: list[str] = []
    # column headers
    for x, word, formula in [
        (XQ, "query", "q_4 = W_Q^⊤ h_4"),
        (XK, "keys", "k_i = W_K^⊤ h_i"),
        (XH, "vectors", "h_i\\t{ of token }w_i"),
        (XV, "values", "v_i = W_V^⊤ h_i"),
        (XO, "output", None),
    ]:
        b.append(text(x, 26, word, "note b"))
        if formula:
            b.append(M(x, 50, formula, size=15))
    # value lines (drawn first, under the boxes): width grows with the weight
    for i in range(4):
        b.append(
            line(
                XV + bw / 2,
                Y[i],
                XO - bw / 2,
                186,
                cls="aln",
                mk=None,
                extra=f' style="stroke-width:{1 + 10 * alpha[i]:.2f}"',
            )
        )
    # query-key lines, width grows with the weight
    xq, xk = XQ + bw / 2, XK - bw / 2
    for i in range(4):
        b.append(
            line(
                xq, Y[3], xk, Y[i], mk=None, extra=f' style="stroke-width:{1 + 10 * alpha[i]:.2f}"'
            )
        )
    for i in range(4):
        t = (195 - xq) / (xk - xq)
        ly = Y[3] + (Y[i] - Y[3]) * t
        b.append(f'<rect x="178" y="{f(ly - 11)}" width="40" height="22" rx="4" fill="#ffffff"/>')
        b.append(M(197, ly + 5, f"α_{{4,{i + 1}}}", size=15))
    # rows 1-4
    for i in range(4):
        y = Y[i]
        hcls = "abox" if i == 3 else "box"
        b.append(line(XH - bw / 2, y, XK + bw / 2 + 2, y))
        b.append(line(XH + bw / 2, y, XV - bw / 2 - 2, y))
        b.append(rect(XK - bw / 2, y - bh / 2, bw, bh))
        b.append(rect(XH - bw / 2, y - bh / 2, bw, bh, hcls))
        b.append(rect(XV - bw / 2, y - bh / 2, bw, bh))
        b.append(M(XK, y + 6, f"k_{i + 1}"))
        b.append(M(XH, y + 6, f"h_{i + 1}"))
        b.append(M(XV, y + 6, f"v_{i + 1}"))
    # row 5: computed, but masked for the query of position 4
    y = Y[4]
    b.append(line(XH - bw / 2, y, XK + bw / 2 + 2, y, cls="dln", mk="d"))
    b.append(line(XH + bw / 2, y, XV - bw / 2 - 2, y, cls="dln", mk="d"))
    for x, lab in [(XK, "k_5"), (XH, "h_5"), (XV, "v_5")]:
        b.append(rect(x - bw / 2, y - bh / 2, bw, bh, "dim"))
        b.append(M(x, y + 6, lab, cls="note", size=19))
    b.append(
        text(
            XK - bw / 2,
            y + 40,
            "position 5 is masked for the query of position 4 (section 3)",
            "note",
            anchor="start",
        )
    )
    # the query: h_4 -> W_Q -> q_4
    b.append(path(f"M{XH},{Y[3] + bh / 2} V{Y[3] + 31} H{XQ} V{Y[3] + bh / 2 + 2}"))
    b.append(rect(150, Y[3] + 20, 42, 22, "op", rx=4))
    b.append(M(171, Y[3] + 36, "W_Q", size=15))
    b.append(rect(XQ - bw / 2, Y[3] - bh / 2, bw, bh, "abox"))
    b.append(M(XQ, Y[3] + 6, "q_4"))
    # the output
    b.append(rect(XO - bw / 2, 186 - bh / 2, bw, bh, "abox"))
    b.append(M(XO, 192, "v̄_4"))
    b.append(M(XO - 35, 232, "= Σ_{i≤4} α_{4,i} v_i", size=16, anchor="start"))
    b.append(
        M(
            24,
            394,
            "\\t{Line width grows with the weight }α_{4,i}\\t{. The weights over }i ≤ 4\\t{ sum to 1; the values drawn are illustrative.}",
            size=13.5,
            anchor="start",
            cls="note",
        )
    )
    return svg(
        W,
        H,
        "Self-attention for one position",
        "Five positions, one per row, each with a vector h_1 to h_5. Each vector is projected to a key k_i "
        "(left) and a value v_i (right). Position 4 is the asker: h_4 is projected by W_Q to the query q_4. "
        "Lines from q_4 to the keys k_1 to k_4 carry the weights alpha_{4,1} to alpha_{4,4}, drawn thicker "
        "for larger weights; the line to k_3 is the thickest. The values v_1 to v_4 are combined with the "
        "same weights into the output v-bar_4, the sum over i of alpha_{4,i} v_i. Position 5 is grayed out: "
        "it is masked for the query of position 4, so no line runs to it.",
        b,
    )


def fig_causal_mask() -> str:
    W, H = 680, 470
    chars = ["T", "o", "␣", "b", "e", ","]
    layer, head = MEASURED_HEAD
    data = json.loads((IMAGES / "05-attention-heads.json").read_text(encoding="utf-8"))
    A = data["heads"]["To be,"][layer - 1][head - 1]  # row t, column i; stored to 4 decimals
    assert len(A) == 6 and all(len(row) == 6 for row in A)
    for t, row in enumerate(A):
        assert all(v == 0 for v in row[t + 1 :]), "measured weights above the diagonal must be 0"
        assert abs(sum(row) - 1) < 1e-3, "each measured row must sum to 1"
    weights = [row[: t + 1] for t, row in enumerate(A)]
    gx, gy, c = 176, 98, 48
    n = 6
    defs = [
        '<pattern id="h" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
        f'<rect width="8" height="8" fill="{PALE}"/><line x1="0" y1="0" x2="0" y2="8" stroke="{RULE}" stroke-width="3"/></pattern>',
        f'<linearGradient id="g" x1="0" x2="1" y1="0" y2="0"><stop offset="0" stop-color="{PALE}"/><stop offset="1" stop-color="{NAVY}"/></linearGradient>',
    ]
    b: list[str] = []
    b.append(text(gx + n * c / 2, 34, "position i (is read)", "note"))
    b.append(
        f'<text transform="translate(40,{gy + n * c / 2}) rotate(-90)" text-anchor="middle" class="note">position t (asks)</text>'
    )
    for k, ch in enumerate(chars):
        b.append(text(gx + c * k + c / 2, gy - 14, ch, "mono", size=17))
        b.append(text(gx - 18, gy + c * k + c / 2 + 6, ch, "mono", size=17))
        b.append(text(gx + c * k + c / 2, 58, str(k + 1), "note", size=12))
        b.append(text(70, gy + c * k + c / 2 + 5, str(k + 1), "note", size=12))
    for t in range(n):
        for i in range(n):
            x, y = gx + c * i, gy + c * t
            if i <= t:
                w = weights[t][i]
                b.append(
                    f'<rect x="{x}" y="{y}" width="{c}" height="{c}" fill="{mix(PALE, NAVY, w)}" stroke="#ffffff" stroke-width="1.5"/>'
                )
                col = "#ffffff" if w > 0.45 else INK
                b.append(
                    f'<text x="{x + c / 2}" y="{y + c / 2 + 5}" text-anchor="middle" class="mono" style="font-size:13px;fill:{col}">{w:.2f}</text>'
                )
            else:
                b.append(
                    f'<rect x="{x}" y="{y}" width="{c}" height="{c}" fill="url(#h)" stroke="#ffffff" stroke-width="1.5"/>'
                )
    b.append(
        f'<rect x="{gx}" y="{gy}" width="{n * c}" height="{n * c}" fill="none" stroke="{RULE}" stroke-width="1"/>'
    )
    # annotation for the upper triangle
    ax, ay = gx + 4.5 * c, gy + 1.5 * c
    b.append(dot(ax, ay, 3, MUTED))
    b.append(
        path(
            f"M{ax},{ay} L{gx + n * c + 22},{gy + 30}",
            cls="rule",
            mk=None,
            extra=f' style="stroke:{MUTED};stroke-width:1"',
        )
    )
    lx = gx + n * c + 28
    b.append(M(lx, gy + 28, "−∞\\t{ before the softmax,}", size=14.5, anchor="start", cls="note"))
    b.append(text(lx, gy + 47, "weight 0 after", "note", anchor="start"))
    # row-sum bracket on the last row
    ry = gy + 5 * c
    b.append(
        path(
            f"M{gx + n * c + 6},{ry + 3} h7 v{c - 6} h-7",
            extra=f' style="stroke:{ACCENT}"',
            mk=None,
        )
    )
    b.append(text(gx + n * c + 20, ry + 21, "each row", "note acc", anchor="start"))
    b.append(text(gx + n * c + 20, ry + 38, "sums to 1", "note acc", anchor="start"))
    b.append(text(gx + n * c + 20, gy + 2.5 * c + 4, "lower triangular:", "note", anchor="start"))
    b.append(
        M(
            gx + n * c + 20,
            gy + 2.5 * c + 22,
            "\\t{position }t\\t{ reads }i ≤ t",
            size=14.5,
            anchor="start",
            cls="note",
        )
    )
    # legend
    ly = gy + n * c + 46
    b.append(M(gx, ly, "α_{t,i}", size=18, anchor="start"))
    b.append(text(gx + 48, ly - 1, "0", "note"))
    b.append(
        f'<rect x="{gx + 58}" y="{ly - 12}" width="100" height="13" fill="url(#g)" stroke="{RULE}"/>'
    )
    b.append(text(gx + 168, ly - 1, "1", "note"))
    b.append(
        text(
            W - 20,
            ly - 1,
            f"measured: Lab 5 mini-GPT, layer {layer}, head {head}",
            "note acc",
            anchor="end",
        )
    )
    return svg(
        W,
        H,
        f"The causal mask on measured attention weights (Lab 5, layer {layer}, head {head})",
        "A six by six grid for the characters T, o, space, b, e and comma. Rows are the asking position t, "
        "columns the position i that is read. Cells on and below the diagonal hold attention weights, "
        "shaded darker for larger values and printed as numbers rounded to two decimals; each row sums to 1, "
        "and row 1 is a single cell of weight 1. Cells above the diagonal are hatched: their scores are minus "
        "infinity before the softmax, so their weights are 0 after it. The weights are measured: they are "
        f"those of layer {layer}, head {head} of the trained Lab 5 mini-GPT reading these six characters.",
        b,
        defs,
    )


def _block(b: list[str], x: float, y: float, w: float, h: float, tint: bool = True) -> None:
    """A T x d block cut into four head slices, each numbered."""
    sw = w / 4
    for r in range(4):
        dark, light = HEADS[r]
        b.append(
            f'<rect x="{f(x + r * sw)}" y="{f(y)}" width="{f(sw)}" height="{f(h)}" fill="{light if tint else "#eef2f6"}" stroke="#ffffff" stroke-width="1"/>'
        )
        b.append(
            f'<text x="{f(x + r * sw + sw / 2)}" y="{f(y + h / 2 + 4)}" text-anchor="middle" class="b" style="font-size:12px;fill:{dark}">{r + 1}</text>'
        )
    b.append(
        f'<rect x="{f(x)}" y="{f(y)}" width="{f(w)}" height="{f(h)}" fill="none" stroke="{NAVY}" stroke-width="1.5"/>'
    )


def fig_multi_head() -> str:
    W, H = 760, 392
    b: list[str] = []
    defs = [marker(f"h{r}", HEADS[r][0], 6) for r in range(4)]
    # headers
    b.append(text(46, 24, "input", "note b"))
    b.append(text(160, 24, "project, cut into heads", "note b"))
    b.append(text(360, 24, "attend: one head each", "note b"))
    b.append(text(360, 41, "masked scaled dot-product", "note"))
    b.append(text(552, 24, "concatenate", "note b"))
    b.append(text(632, 24, "project", "note b"))
    b.append(text(712, 24, "output", "note b"))
    # input block H
    b.append('<rect class="box" x="14" y="168" width="64" height="64" rx="2"/>')
    b.append(M(46, 206, "H"))
    b.append(M(46, 252, "T × d", size=14, cls="note"))
    # projections
    ys = [92, 200, 308]
    b.append(line(78, 200, 90, 200, mk=None))
    b.append(line(90, ys[0], 90, ys[2], mk=None))
    for y, wl, lab in zip(ys, ["W_Q", "W_K", "W_V"], ["Q", "K", "V"], strict=True):
        b.append(line(90, y, 96, y))
        b.append(rect(98, y - 14, 42, 28, "op", rx=4))
        b.append(M(119, y + 5, wl, size=15))
        b.append(line(140, y, 156, y))
        _block(b, 158, y - 32, 64, 64)
        b.append(M(190, y - 38, lab, size=17))
    b.append(path("M158,346 v6 h16 v-6", cls="rule", mk=None))
    b.append(M(150, 368, "d_k = d / n_h", size=14, anchor="start", cls="note"))
    # split into heads
    b.append(
        path(
            "M232,60 q8,0 8,10 v118 q0,12 10,12 q-10,0 -10,12 v118 q0,10 -8,10", cls="rule", mk=None
        )
    )
    yr = [80, 160, 240, 320]
    for r in range(4):
        dark, light = HEADS[r]
        b.append(
            f'<line x1="252" y1="200" x2="288" y2="{yr[r]}" style="fill:none;stroke:{dark};stroke-width:1.5" marker-end="url(#h{r})"/>'
        )
        b.append(
            f'<rect x="290" y="{yr[r] - 24}" width="140" height="48" rx="6" style="fill:#ffffff;stroke:{dark};stroke-width:1.5"/>'
        )
        b.append(
            f'<text x="304" y="{yr[r] - 4}" text-anchor="start" class="b" style="font-size:14px;fill:{dark}">head {r + 1}</text>'
        )
        b.append(M(304, yr[r] + 16, f"A^{{({r + 1})}} =", size=15, anchor="start"))
        # lower-triangular thumbnail A^(r), 6 x 6
        tx, ty, cs = 384, yr[r] - 18, 6
        for t in range(6):
            for i in range(6):
                if i <= t:
                    op = 0.25 + 0.75 * (((t * 7 + i * 3 + r * 5) % 6) / 5)
                    b.append(
                        f'<rect x="{tx + i * cs}" y="{ty + t * cs}" width="{cs}" height="{cs}" fill="{dark}" fill-opacity="{op:.2f}"/>'
                    )
        b.append(
            f'<rect x="{tx}" y="{ty}" width="36" height="36" fill="none" stroke="{RULE}" stroke-width="1"/>'
        )
        # head output slice and route into the concatenation
        b.append(
            f'<line x1="430" y1="{yr[r]}" x2="446" y2="{yr[r]}" style="fill:none;stroke:{dark};stroke-width:1.5" marker-end="url(#h{r})"/>'
        )
        b.append(
            f'<rect x="448" y="{yr[r] - 18}" width="16" height="36" style="fill:{light};stroke:{dark};stroke-width:1.5"/>'
        )
        # enter the concatenation block at staggered heights: no crossings
        b.append(
            f'<path d="M464,{yr[r]} C490,{yr[r]} 492,{176 + 16 * r} 518,{176 + 16 * r}" style="fill:none;stroke:{dark};stroke-width:1.5" marker-end="url(#h{r})"/>'
        )
    b.append(M(456, 362, "V̄^{(r)}", size=15, cls="note"))
    b.append(path("M456,346 V342", cls="rule", mk=None))
    # concatenation, projection, output
    _block(b, 520, 168, 64, 64)
    b.append(M(552, 252, "T × d", size=14, cls="note"))
    b.append(line(584, 200, 598, 200))
    b.append(rect(600, 186, 64, 28, "op", rx=4))
    b.append(M(632, 205, "W_{\\r{proj}}", size=15))
    b.append(line(664, 200, 678, 200))
    b.append('<rect class="box" x="680" y="168" width="64" height="64" rx="2"/>')
    b.append(M(712, 205, "\\r{MHA}(H)", size=14))
    b.append(M(712, 252, "T × d", size=14, cls="note"))
    return svg(
        W,
        H,
        "Multi-head attention",
        "The input H, of size T by d, is multiplied by W_Q, W_K and W_V to give Q, K and V, each T by d and "
        "each cut into four numbered vertical slices of width d_k = d / n_h. Slice r of Q, K and V goes to "
        "head r. Each of the four heads runs masked scaled dot-product attention and produces a "
        "lower-triangular T by T weight matrix A^(r) and an output slice V-bar^(r). The four output slices "
        "are placed side by side, concatenated into a T by d block, and multiplied by W_proj to give "
        "MHA(H), again T by d.",
        b,
        defs,
    )


def fig_transformer_block() -> str:
    W, H = 720, 596
    b: list[str] = []
    # ---- left panel: one pre-norm block
    b.append(rect(16, 40, 334, 424, "panel", rx=12))
    b.append(text(183, 28, "one pre-norm block", "b"))
    SX = 92
    # residual stream (accent, like the LSTM cell state)
    b.append(line(SX, 422, SX, 264, cls="aln", mk=None, extra=' style="stroke-width:4"'))
    b.append(line(SX, 236, SX, 110, cls="aln", mk=None, extra=' style="stroke-width:4"'))
    b.append(line(SX, 82, SX, 62, cls="aln", mk="e", extra=' style="stroke-width:4"'))
    b.append(
        '<text transform="translate(44,300) rotate(-90)" text-anchor="middle" class="note acc">residual stream</text>'
    )
    b.append(M(SX, 446, "H^{(ℓ−1)}"))
    b.append(M(SX + 14, 70, "H^{(ℓ)}", anchor="start"))
    # sublayer 1: LN -> masked multi-head attention
    b.append(dot(SX, 396, 3.6, ACCENT))
    b.append(path(f"M{SX},396 H220 V374"))
    b.append(rect(180, 346, 80, 26))
    b.append(text(220, 364, "LN", size=14))
    b.append(line(220, 346, 220, 324))
    b.append(rect(146, 272, 148, 50))
    b.append(text(220, 293, "masked multi-head", size=14))
    b.append(text(220, 311, "attention", size=14))
    b.append(path(f"M220,272 V250 H{SX + 14}"))
    b.append(plus(SX, 250))
    b.append(M(SX - 14, 232, "H′", size=17, anchor="end"))
    # sublayer 2: LN -> FFN
    b.append(dot(SX, 214, 3.6, ACCENT))
    b.append(path(f"M{SX},214 H220 V192"))
    b.append(rect(180, 164, 80, 26))
    b.append(text(220, 182, "LN", size=14))
    b.append(line(220, 164, 220, 144))
    b.append(rect(146, 110, 148, 32))
    b.append(text(220, 131, "feed-forward (FFN)", size=14))
    b.append(path(f"M220,110 V96 H{SX + 14}"))
    b.append(plus(SX, 96))
    b.append(text(270, 364, "sublayer 1", "note", anchor="start"))
    b.append(text(270, 182, "sublayer 2", "note", anchor="start"))
    # ---- inset: post-norm sublayer
    b.append(rect(16, 478, 334, 104, "panel", rx=12))
    b.append(
        text(30, 500, "post-norm (original): LN sits on the main line", "note", anchor="start")
    )
    my = 556
    b.append(text(34, my + 5, "in", "note", anchor="start"))
    b.append(line(54, my, 228, my, cls="aln", mk=None))
    b.append(dot(76, my, 3.6, ACCENT))
    b.append(path(f"M76,{my} V522 H118"))
    b.append(rect(120, 510, 80, 24))
    b.append(text(160, 527, "sublayer", size=13.5))
    b.append(path(f"M200,522 H240 V{my - 14}"))
    b.append(plus(240, my))
    b.append(line(252, my, 270, my, cls="aln", mk=None))
    b.append(rect(272, my - 13, 40, 26, "abox"))
    b.append(text(292, my + 5, "LN", size=14))
    b.append(line(312, my, 334, my, cls="aln", mk="b"))
    b.append(text(326, my - 16, "out", "note"))
    # ---- right panel: the model
    b.append(rect(380, 40, 324, 424, "panel", rx=12))
    b.append(text(542, 28, "the model: decoder-only", "b"))
    CX = 542
    b.append(M(CX, 452, "w_1 … w_T", size=18))
    b.append(line(CX, 434, CX, 414))
    b.append(rect(432, 366, 220, 46))
    b.append(text(CX, 384, "token embedding + position", size=14))
    b.append(M(CX, 404, "h_t^{(0)} = e_{w_t} + p_t", size=16))
    b.append(line(CX, 366, CX, 340))
    for k in (2, 1):
        b.append(rect(462 + 6 * k, 280 - 6 * k, 160, 58, "box"))
    b.append(rect(462, 280, 160, 58, "box"))
    b.append(text(CX, 306, "pre-norm block", size=14))
    b.append(text(CX, 325, "(left panel)", "note"))
    b.append(M(664, 306, "× n_ℓ", size=18, anchor="start"))
    b.append(line(CX, 268, CX, 248))
    b.append(rect(492, 220, 100, 26))
    b.append(text(CX, 238, "final LN", size=14))
    b.append(line(CX, 220, CX, 200))
    b.append(rect(462, 170, 160, 28))
    b.append(M(CX, 190, "\\t{output layer }W_o, b_o", size=15))
    b.append(line(CX, 170, CX, 150))
    b.append(rect(502, 122, 80, 26))
    b.append(text(CX, 140, "softmax", size=14))
    b.append(line(CX, 122, CX, 102))
    b.append(M(CX, 88, "ŷ_t\\t{: distribution over }w_{t+1}", size=16))
    # dashed link: the stacked block is the left panel
    b.append(
        path(
            "M350,300 C400,300 420,309 460,309",
            cls="rule",
            mk=None,
            extra=' style="stroke-dasharray:4 3"',
        )
    )
    return svg(
        W,
        H,
        "A pre-norm transformer block and the decoder-only model",
        "Left: one pre-norm block. A thick residual stream runs from H^(l-1) at the bottom to H^(l) at the "
        "top. The first branch leaves the stream, passes through layer normalization and masked multi-head "
        "attention, and is added back at a plus sign, giving H-prime. The second branch passes through layer "
        "normalization and the feed-forward network and is added back the same way. Below, an inset shows "
        "the original post-norm arrangement of one sublayer, where the sublayer output is added to the main "
        "line and the sum then passes through layer normalization on the main line itself. Right: the "
        "decoder-only model, bottom to top: tokens w_1 to w_T, token embedding plus position giving "
        "h_t^(0) = e_{w_t} + p_t, a stack of n_l pre-norm blocks, a final layer normalization, the output "
        "layer W_o, b_o, a softmax, and y-hat_t, the distribution over the next token w_{t+1}.",
        b,
        [THICK],
    )


# ===================================================================== Lecture 6


def _symbols(b: list[str], x: float, y: float, syms: list[str], cw: float = 9.0) -> float:
    for s in syms:
        w = 10 + cw * len(s)
        b.append(rect(x, y - 13, w, 26, "abox" if len(s) > 1 else "op", rx=4))
        b.append(text(x + w / 2, y + 5, s, "mono", size=15))
        x += w + 3
    return x


def fig_bpe_merges() -> str:
    W, H = 720, 350
    words = [("low", 5), ("lower", 2), ("newest", 6), ("widest", 3)]
    cols = [
        ("start", "|V| = 11", [list("low_"), list("lower_"), list("newest_"), list("widest_")]),
        (
            "after merge 3",
            "|V| = 14",
            [list("low_"), list("lower_"), ["n", "e", "w", "est_"], ["w", "i", "d", "est_"]],
        ),
        (
            "after merge 5",
            "|V| = 16",
            [
                ["low", "_"],
                ["low", "e", "r", "_"],
                ["n", "e", "w", "est_"],
                ["w", "i", "d", "est_"],
            ],
        ),
    ]
    CX = [150, 340, 530]
    Y = [96, 142, 188, 234]
    b: list[str] = []
    b.append(text(24, 40, "word, count", "note b", anchor="start"))
    for (w, n), y in zip(words, Y, strict=True):
        b.append(text(24, y + 5, w, "mono", anchor="start", size=15))
        b.append(M(118, y + 5, f"× {n}", size=15, anchor="end", cls="note"))
    for k, ((title, vocab, rows), x) in enumerate(zip(cols, CX, strict=True)):
        b.append(text(x + 70, 40, title, "b", size=14.5))
        b.append(M(x + 70, 60, vocab, size=14, cls="note"))
        for syms, y in zip(rows, Y, strict=True):
            _symbols(b, x, y, syms)
        if k < 2:
            b.append(
                line(
                    x + 132,
                    35,
                    CX[k + 1] + 8,
                    35,
                    cls="ln",
                    mk="c",
                    extra=f' style="stroke:{MUTED};stroke-width:1.2"',
                )
            )
    # merge list, grouped under the column each group produces
    b.append(text(24, 288, "merges, in order", "note", anchor="start"))
    for x, items in [
        (CX[1], ["1  e + s", "2  es + t", "3  est + _"]),
        (CX[2], ["4  l + o", "5  lo + w"]),
    ]:
        b.append(path(f"M{x},262 v6 H{x + 150} v-6", cls="rule", mk=None))
        for j, s in enumerate(items):
            b.append(text(x + 4, 288 + 20 * j, s, "mono", anchor="start", size=14))
    b.append(rect(24, 318, 22, 16, "abox", rx=3))
    b.append(text(52, 331, "a symbol made by a merge", "note", anchor="start"))
    return svg(
        W,
        H,
        "BPE merges on the toy corpus",
        "Four words, low (count 5), lower (2), newest (6) and widest (3), each drawn as a row of symbol boxes "
        "ending in the end-of-word symbol, underscore. Column 1, start: every symbol is one character and the "
        "vocabulary has 11 symbols. Column 2, after merge 3: e, s, t and underscore are fused into one box, "
        "est_, in newest and widest; 14 symbols. Column 3, after merge 5: l, o and w are fused into one box, "
        "low, in low and lower; 16 symbols. Below, the merge list in order: 1 e+s, 2 es+t, 3 est+_, 4 l+o, "
        "5 lo+w.",
        b,
    )


def _stack(b: list[str], x: float, y: float, w: float, h: float, n: int) -> None:
    for k in range(n - 1, 0, -1):
        b.append(rect(x + 5 * k, y - 5 * k, w, h, "op", rx=4))
    b.append(rect(x, y, w, h, "op", rx=4))


def fig_pretrain_finetune() -> str:
    W, H = 720, 336
    b: list[str] = []
    b.append(rect(16, 40, 314, 280, "panel", rx=12))
    b.append(rect(390, 40, 314, 280, "panel", rx=12))
    b.append(text(173, 28, "pretraining (once)", "b"))
    b.append(text(547, 28, "fine-tuning (per task)", "b"))
    tstyle = ' style="fill:#dce5ef;stroke:#16324f;stroke-width:1.5"'
    # left
    _stack(b, 58, 252, 200, 48, 4)
    b.append(text(158, 272, "large unlabeled corpus", size=14))
    b.append(text(158, 290, "billions of words, no labels", "note"))
    b.append(line(173, 238, 173, 218))
    b.append(rect(88, 156, 170, 60, extra=tstyle))
    b.append(text(173, 182, "transformer", size=15))
    b.append(M(173, 204, "\\t{trained: }θ_\\r{pre}", size=15))
    b.append(line(173, 156, 173, 138))
    b.append(rect(98, 104, 150, 32, "box"))
    b.append(text(173, 125, "language-model head", size=13.5))
    b.append(line(173, 104, 173, 86))
    b.append(text(173, 76, "predict the next or masked tokens", "note"))
    # right
    _stack(b, 472, 258, 150, 40, 2)
    b.append(text(547, 276, "4,800 labeled", size=14))
    b.append(text(547, 292, "abstracts", size=14))
    b.append(line(547, 248, 547, 218))
    b.append(rect(462, 156, 170, 60, extra=tstyle))
    b.append(text(547, 182, "transformer", size=15))
    b.append(M(547, 204, "\\t{starts from }θ_\\r{pre}", size=15))
    b.append(line(547, 156, 547, 138))
    b.append(rect(472, 104, 150, 32, "abox"))
    b.append(M(547, 125, "\\t{classifier }W, b\\t{ (new)}", size=14))
    b.append(line(547, 104, 547, 86))
    b.append(text(547, 76, "topic label", "note"))
    # copy arrow
    b.append(line(258, 186, 460, 186, cls="aln", mk="b"))
    b.append(M(360, 174, "\\t{copy }θ_\\r{pre}", size=15))
    return svg(
        W,
        H,
        "Pretrain once, fine-tune per task",
        "Two panels. Pretraining, done once: a large unlabeled text corpus, billions of words with no labels, "
        "feeds a transformer topped by a language-model head that predicts the next or masked tokens; "
        "training gives the parameters theta-pre. An arrow labeled copy theta-pre leads to the second panel. "
        "Fine-tuning, done per task: the same transformer, starting from theta-pre, gets a small new "
        "classifier layer W, b on top that outputs a topic label, and the whole model is trained on 4,800 "
        "labeled abstracts.",
        b,
    )


def fig_clm_vs_mlm() -> str:
    W, H = 720, 392
    b: list[str] = []
    panels = [
        (16, "causal LM (GPT)", ["the", "cat", "sat", "on", "the"], "transformer, causal mask"),
        (368, "masked LM (BERT)", ["the", "cat", "[MASK]", "on", "the"], "transformer, no mask"),
    ]
    targets = ["cat", "sat", "on", "the", "mat"]
    for p, (x0, title, toks, bar) in enumerate(panels):
        b.append(rect(x0, 40, 336, 340, "panel", rx=12))
        b.append(text(x0 + 168, 28, title, "b"))
        xs = [x0 + 48 + 60 * j for j in range(5)]
        b.append(rect(x0 + 20, 142, 296, 36, "box"))
        b.append(text(x0 + 168, 165, bar, size=14))
        for j, (x, tk) in enumerate(zip(xs, toks, strict=True)):
            w = 12 + 7.8 * len(tk)
            cls = "abox" if tk == "[MASK]" else "op"
            b.append(rect(x - w / 2, 250, w, 26, cls, rx=4))
            b.append(text(x, 268, tk, "mono", size=13))
            b.append(line(x, 250, x, 180))
            if p == 0 or j == 2:
                b.append(line(x, 142, x, 114, cls="aln", mk="b"))
                b.append(text(x, 104, targets[j] if p == 0 else "sat", "mono acc", size=14))
            else:
                b.append(text(x, 108, "no loss", "note", size=12))
        # what position 3 reads: arcs below the tokens, arrowheads at the read positions
        reads = [0, 1] if p == 0 else [0, 1, 3, 4]
        for j in reads:
            d = abs(j - 2)
            b.append(
                path(
                    f"M{xs[2]},278 Q{(xs[2] + xs[j]) / 2},{290 + 22 * d} {xs[j]},280",
                    cls="ln",
                    mk="c",
                    extra=f' style="stroke:{MUTED};stroke-width:1.2"',
                )
            )
        who = "sat" if p == 0 else "[MASK]"
        side = "the left only" if p == 0 else "both sides"
        b.append(text(x0 + 168, 342, f"position 3 ({who}) reads {side}", "note"))
        b.append(
            M(
                x0 + 168,
                366,
                "T\\t{ targets per sequence}"
                if p == 0
                else "\\t{about }0.15 T\\t{ targets per sequence}",
                size=14.5,
                cls="acc",
            )
        )
    return svg(
        W,
        H,
        "Causal versus masked language modeling",
        "Two panels over the sentence the cat sat on the. Left, causal LM (GPT): a transformer with a causal "
        "mask outputs, above every position, the next token: cat, sat, on, the, mat. Arcs below the tokens "
        "show that position 3, sat, reads only positions to its left. Caption: T targets per sequence. Right, "
        "masked LM (BERT): the input is the cat [MASK] on the; a transformer with no mask outputs a target "
        "only above the masked position, the original token sat, and the other positions are marked no loss. "
        "Arcs show that the masked position reads both sides. Caption: about 0.15 T targets per sequence.",
        b,
    )


# ===================================================================== Lecture 7


def _tokens(
    b: list[str], x: float, y: float, toks: list[tuple[str, str, str]], cw: float = 6.9
) -> list[tuple[float, float, str]]:
    """Draw a strip of tokens; kind is 'special', 'plain' or 'loss'. Returns (x0, x1, mask) per token."""
    out = []
    for s, kind, mask in toks:
        w = 10 + cw * len(s)
        special = kind.startswith("special")
        loss = mask.startswith("1")
        if special:
            st = f' style="fill:#ffffff;stroke:{ACCENT if loss else NAVY};stroke-width:1.5"'
        else:
            st = f' style="fill:{"#f8e9de" if loss else "#eef2f6"};stroke:{ACCENT if loss else "#eef2f6"};stroke-width:1.5"'
        b.append(rect(x, y, w, 26, "", rx=3, extra=st))
        b.append(text(x + w / 2, y + 17, s, "mono", size=11.5))
        out.append((x, x + w, mask))
        x += w + 4
    return out


def fig_chat_template_mask() -> str:
    W, H = 760, 244
    b: list[str] = []
    b.append(text(16, 32, "messages", "note b", anchor="start"))
    for y, role, content in [
        (44, "user", "Name the capital of France."),
        (104, "assistant", "Paris."),
    ]:
        b.append(rect(16, y, 190, 48, "op", rx=6))
        b.append(text(28, y + 18, role, "note b", anchor="start", size=12.5))
        b.append(text(28, y + 38, content, anchor="start", size=13.5))
    b.append(line(214, 104, 266, 104))
    b.append(text(240, 124, "chat", "note"))
    b.append(text(240, 140, "template", "note"))
    x0 = 276
    rows = [
        (
            44,
            [
                ("<|im_start|>", "special", "0"),
                ("user", "plain", "0"),
                ("\\n", "plain", "0"),
                ("Name the capital of France.", "plain", "0 … 0"),
                ("<|im_end|>", "special", "0"),
                ("\\n", "plain", "0"),
            ],
        ),
        (
            116,
            [
                ("<|im_start|>", "special", "0"),
                ("assistant", "plain", "0"),
                ("\\n", "plain", "0"),
                ("Paris.", "plain", "1 … 1"),
                ("<|im_end|>", "special", "1"),
            ],
        ),
    ]
    loss_span = None
    for y, toks in rows:
        spans = _tokens(b, x0, y, toks)
        b.append(M(x0 - 8, y + 48, "m_t", size=16, anchor="end", cls="note"))
        for xa, xb, mk in spans:
            cls = "mono acc" if mk.startswith("1") else "mono note"
            b.append(text((xa + xb) / 2, y + 48, mk, cls, size=13))
        ones = [(xa, xb) for xa, xb, mk in spans if mk.startswith("1")]
        if ones:
            loss_span = (ones[0][0], ones[-1][1])
    xa, xb = loss_span
    b.append(path(f"M{f(xa)},176 v6 H{f(xb)} v-6", extra=f' style="stroke:{ACCENT}"', mk=None))
    b.append(text((xa + xb) / 2, 200, "loss is computed here", "note acc"))
    b.append(rect(16, 214, 22, 16, "op", rx=3))
    b.append(text(44, 227, "special token", "note", anchor="start"))
    b.append('<rect x="150" y="214" width="22" height="16" rx="3" style="fill:#eef2f6"/>')
    b.append(text(178, 227, "ordinary text", "note", anchor="start"))
    return svg(
        W,
        H,
        "A chat template and its response mask",
        "Left: two messages, role user with content Name the capital of France., and role assistant with "
        "content Paris. An arrow labeled chat template leads to one token sequence, drawn in two lines: "
        "im_start, user, newline, the instruction text, im_end, newline; then im_start, assistant, newline, "
        "Paris., im_end. Special tokens are outlined boxes and ordinary text is filled. Under each token is "
        "its mask value m_t: 0 for every token up to and including the newline after assistant, and 1 for the "
        "response text and the closing im_end. A bracket under the 1s reads: loss is computed here.",
        b,
    )


def fig_lora_update() -> str:
    W, H = 680, 456
    b: list[str] = []
    b.append(M(24, 34, "h = W_0 x + (α / r) B A x", size=17, anchor="start"))
    # input and split
    b.append(M(330, 440, "x", size=19))
    b.append(M(344, 440, "\\t{(}d_\\r{in}\\t{ numbers)}", size=14, anchor="start", cls="note"))
    b.append(line(330, 422, 330, 394, mk=None))
    b.append(dot(330, 394, 3.6))
    b.append(path("M330,394 H200 V354"))
    b.append(path("M330,394 H460 V354"))
    # frozen W0
    b.append(rect(120, 194, 160, 158, "box", rx=4))
    b.append(M(200, 262, "W_0", size=24))
    b.append(text(200, 290, "frozen", "note"))
    b.append(M(200, 314, "d_\\r{out} × d_\\r{in}", size=15, cls="note"))
    # trainable A and B
    b.append('<polygon class="abox" points="380,352 540,352 472,284 448,284"/>')
    b.append(M(460, 336, "A", size=20))
    b.append(line(460, 284, 460, 266, cls="aln", mk=None, extra=' style="stroke-width:1.5"'))
    b.append('<polygon class="abox" points="448,264 472,264 540,194 380,194"/>')
    b.append(M(460, 222, "B", size=20))
    b.append(M(552, 318, "A: r × d_\\r{in}", size=15, anchor="start"))
    b.append(text(552, 338, "random init", "note", anchor="start"))
    b.append(M(552, 220, "B: d_\\r{out} × r", size=15, anchor="start"))
    b.append(text(552, 240, "zero init", "note", anchor="start"))
    b.append(M(478, 280, "A x\\t{: }r\\t{ numbers}", size=14, anchor="start", cls="note"))
    # scale and sum
    b.append(line(460, 194, 460, 174))
    b.append(rect(428, 148, 64, 24, "op", rx=12))
    b.append(M(460, 166, "× α / r", size=15))
    b.append(path("M460,148 V100 H344"))
    b.append(path("M200,194 V100 H316"))
    b.append(plus(330, 100))
    b.append(line(330, 88, 330, 62))
    b.append(M(330, 50, "h", size=19))
    b.append(M(344, 50, "\\t{(}d_\\r{out}\\t{ numbers)}", size=14, anchor="start", cls="note"))
    # legend
    b.append(rect(24, 400, 22, 16, "box", rx=3))
    b.append(text(52, 413, "frozen: no gradient", "note", anchor="start"))
    b.append(rect(24, 424, 22, 16, "abox", rx=3))
    b.append(text(52, 437, "trained", "note", anchor="start"))
    return svg(
        W,
        H,
        "A LoRA layer",
        "The input x, with d_in numbers, enters at the bottom and splits into two paths. Left: a large square "
        "block W_0 of size d_out by d_in, frozen. Right: a trapezoid A of size r by d_in, randomly "
        "initialized, narrows the input to r numbers, A x; a second trapezoid B of size d_out by r, "
        "initialized to zero, widens it back to d_out; the result is scaled by alpha over r. The two paths "
        "meet at a plus sign, giving the output h with d_out numbers: h = W_0 x + (alpha / r) B A x. Only A "
        "and B are trained.",
        b,
    )


# ===================================================================== Lecture 8


def fig_message_list() -> str:
    W, H = 720, 336
    b: list[str] = []
    b.append(rect(16, 36, 234, 284, "panel", rx=12))
    b.append(text(133, 60, "client (your code)", "b", size=14))
    b.append(rect(470, 36, 234, 284, "panel", rx=12))
    b.append(text(587, 60, "provider (server)", "b", size=14))
    for k, role in enumerate(["system", "user", "assistant", "user"]):
        y = 76 + 40 * k
        b.append(rect(36, y, 194, 32, "op", rx=5))
        b.append(text(48, y + 21, role, "b", anchor="start", size=13))
        b.append(f'<rect x="130" y="{y + 13}" width="84" height="6" rx="3" fill="{RULE}"/>')
    b.append(rect(36, 240, 194, 32, "abox", rx=5))
    b.append(text(48, 261, "assistant", "b", anchor="start", size=13))
    b.append(text(218, 261, "the reply", "note acc", anchor="end", size=12.5))
    b.append(text(133, 300, "the list lives here and grows", "note"))
    # provider pipeline
    for k, lab in enumerate(["chat template", "tokens", "transformer"]):
        y = 84 + 56 * k
        b.append(rect(512, y, 150, 32, "box"))
        b.append(text(587, y + 21, lab, size=14))
        if k < 2:
            b.append(line(587, y + 32, 587, y + 54))
    b.append(text(587, 284, "keeps nothing", "note"))
    b.append(text(587, 301, "between requests", "note"))
    # request and response
    b.append(line(250, 100, 510, 100))
    b.append(text(360, 90, "request: the whole list", "note"))
    b.append(path("M587,228 V256 H232", cls="aln", mk="b"))
    b.append(text(352, 226, "response: text + stop reason", "note acc"))
    b.append(M(352, 245, "\\t{+ usage (}n_\\r{in}\\t{, }n_\\r{out}\\t{)}", size=14, cls="acc"))
    return svg(
        W,
        H,
        "The message list: stateless requests",
        "Left, the client: a vertical stack of message cards labeled system, user, assistant, user. An arrow "
        "labeled request: the whole list goes to the provider, which applies the chat template, turns the "
        "text into tokens and runs the transformer. An arrow labeled response: text, stop reason and usage "
        "(n_in, n_out) comes back, and the reply is appended to the client's stack as a new assistant card. "
        "The provider keeps nothing between requests; the list lives on the client and grows.",
        b,
    )


def fig_validate_retry() -> str:
    W, H = 720, 330
    b: list[str] = []
    boxes = [
        (16, 100, ["messages"]),
        (150, 90, ["model"]),
        (274, 100, ["parse JSON"]),
        (408, 130, ["validate against", "the schema"]),
        (580, 120, ["typed object"]),
    ]
    y0, bh = 132, 46
    for x, w, labs in boxes:
        b.append(rect(x, y0, w, bh, "box"))
        if len(labs) == 1:
            b.append(text(x + w / 2, y0 + 28, labs[0], size=14))
        else:
            b.append(text(x + w / 2, y0 + 20, labs[0], size=14))
            b.append(text(x + w / 2, y0 + 37, labs[1], size=14))
    for xa, xb in [(116, 148), (240, 272), (374, 406), (538, 578)]:
        b.append(line(xa, y0 + bh / 2, xb, y0 + bh / 2))
    b.append(text(558, y0 + 14, "valid", "note"))
    b.append(text(640, y0 + 66, "for example, Event", "note"))
    # constrained decoding: a token mask on the model box
    for k in range(5):
        x = 170 + 10 * k
        masked = k in (1, 3)
        b.append(
            f'<rect x="{x}" y="96" width="9" height="12" fill="{"url(#hm)" if masked else "#ffffff"}" stroke="{NAVY}" stroke-width="1"/>'
        )
    b.append(
        line(
            195,
            110,
            195,
            y0 - 2,
            cls="ln",
            mk="c",
            extra=f' style="stroke:{MUTED};stroke-width:1.2;stroke-dasharray:3 2"',
        )
    )
    b.append(text(195, 66, "constrained decoding, provider side:", "note"))
    b.append(text(195, 84, "invalid tokens masked", "note"))
    # failure loop
    yl = 226
    b.append(path(f"M324,{y0 + bh} V{yl}", cls="aln", mk=None))
    b.append(path(f"M473,{y0 + bh} V{yl}", cls="aln", mk=None))
    b.append(path(f"M473,{yl} H66 V{y0 + bh + 2}", cls="aln", mk="b"))
    b.append(text(331, 206, "parse error", "note acc", anchor="start"))
    b.append(text(480, 206, "schema error", "note acc", anchor="start"))
    b.append(text(190, 202, "on failure: append the error", "note acc"))
    b.append(M(190, 219, "\\t{and call again, attempt }r + 1 ≤ R", size=13.5, cls="acc"))
    # give up: leaves the merged failure path
    b.append(dot(200, yl, 3.6, ACCENT))
    b.append(path(f"M200,{yl} V282", cls="aln", mk="b"))
    b.append(M(200, 304, "\\t{after }R\\t{ failed attempts: return None}", size=13.5, cls="note"))
    defs = [
        '<pattern id="hm" width="4" height="4" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
        f'<rect width="4" height="4" fill="#ffffff"/><line x1="0" y1="0" x2="0" y2="4" stroke="{NAVY}" stroke-width="2"/></pattern>'
    ]
    return svg(
        W,
        H,
        "Validate and retry",
        "A left-to-right pipeline: messages, model, parse JSON, validate against the schema, typed object. "
        "Above the model, a small row of token cells with two cells hatched shows constrained decoding on the "
        "provider side: invalid tokens masked. Failure arrows leave parse JSON (a parse error) and validate "
        "(a schema error) and loop back to messages, labeled: on failure, append the error and call again, "
        "attempt r + 1 at most R. A third exit leads down: after R failed attempts, return None.",
        b,
        defs,
    )


def fig_tool_loop() -> str:
    W, H = 740, 430
    b: list[str] = []
    # sides and trust boundary
    b.append(f'<rect x="16" y="46" width="350" height="170" rx="10" fill="{PALE}"/>')
    b.append('<rect x="374" y="46" width="350" height="170" rx="10" fill="#fbf1e9"/>')
    b.append(text(30, 68, "provider side: the model", "note", anchor="start"))
    b.append(text(710, 68, "your side: your code", "note acc", anchor="end"))
    b.append(
        line(
            370,
            42,
            370,
            214,
            cls="ln",
            mk=None,
            extra=f' style="stroke:{MUTED};stroke-dasharray:6 4"',
        )
    )
    b.append(text(370, 32, "trust boundary", "note"))

    # states
    def state(x, y, name, sub, cls):
        b.append(rect(x - 80, y - 30, 160, 60, cls, rx=22))
        b.append(text(x, y - 3, name, "b", size=15))
        b.append(text(x, y + 16, sub, "note"))

    b.append(dot(50, 140, 6))
    b.append(text(50, 122, "start", "note"))
    b.append(line(56, 140, 108, 140))
    state(190, 140, "CALL", "send messages + tools", "box")
    state(550, 140, "EXECUTE", "run each tool call", "abox")
    state(190, 310, "DONE", "return the text", "op")
    state(370, 310, "FAILED", "return a failure", "op")
    b.append(line(270, 124, 468, 124))
    b.append(text(370, 115, "reply has tool calls", "note"))
    b.append(line(470, 156, 272, 156, cls="aln", mk="b"))
    b.append(text(370, 186, "append reply + results (by call ID)", "note acc"))
    b.append(line(190, 170, 190, 278))
    b.append(text(198, 236, "final text", "note", anchor="start"))
    b.append(line(240, 170, 322, 278))
    b.append(text(292, 230, "truncated or refused", "note", anchor="start"))
    b.append(line(520, 170, 420, 278))
    b.append(M(480, 238, "k = K_\\r{max}", size=15, anchor="start", cls="note"))
    # the message list grows by two per round trip
    b.append(text(16, 386, "messages", "note b", anchor="start"))

    def chip(x, s, fill, stroke):
        w = 12 + 7.2 * len(s)
        b.append(
            f'<rect x="{f(x)}" y="372" width="{f(w)}" height="22" rx="4" style="fill:{fill};stroke:{stroke};stroke-width:1.2"/>'
        )
        b.append(text(x + w / 2, 387, s, "mono", size=12))
        return x + w + 4

    x = chip(96, "system", PALE, NAVY)
    x = chip(x, "user", PALE, NAVY)
    for k in range(2):
        xa = x + 12
        x = chip(xa, "call", PALE, NAVY)
        x = chip(x, "result", "#fbf1e9", ACCENT)
        b.append(path(f"M{f(xa)},400 v5 H{f(x - 4)} v-5", cls="rule", mk=None))
        b.append(text((xa + x - 4) / 2, 420, f"round trip {k + 1}: +2", "note", size=12))
    b.append(text(x + 10, 387, "…", "note", anchor="start"))
    chip(x + 34, "final", PALE, NAVY)
    return svg(
        W,
        H,
        "The tool-calling loop as a state machine",
        "Four states. From start, the loop enters CALL, on the provider side: send the messages and the tool "
        "definitions. If the reply has tool calls, it moves to EXECUTE, on your side of a dashed trust "
        "boundary: run each tool call. EXECUTE returns to CALL after appending the reply and one result per "
        "call, matched by call ID. From CALL, a final text leads to DONE, and a truncated or refused reply "
        "leads to FAILED. From EXECUTE, reaching the step limit k = K_max leads to FAILED. Below, the message "
        "list: system, user, then a tool call and a tool result for each round trip, two entries per round "
        "trip, and finally the final answer.",
        b,
    )


FIGURES = {
    "05-self-attention.svg": fig_self_attention,
    "05-causal-mask.svg": fig_causal_mask,
    "05-multi-head.svg": fig_multi_head,
    "05-transformer-block.svg": fig_transformer_block,
    "06-bpe-merges.svg": fig_bpe_merges,
    "06-pretrain-finetune.svg": fig_pretrain_finetune,
    "06-clm-vs-mlm.svg": fig_clm_vs_mlm,
    "07-chat-template-mask.svg": fig_chat_template_mask,
    "07-lora-update.svg": fig_lora_update,
    "08-message-list.svg": fig_message_list,
    "08-validate-retry.svg": fig_validate_retry,
    "08-tool-loop.svg": fig_tool_loop,
}


def main() -> None:
    for name, fn in FIGURES.items():
        (IMAGES / name).write_text(fn(), encoding="utf-8")
        print(f"wrote images/{name}")


if __name__ == "__main__":
    main()
