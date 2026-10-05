"""Draw the measured attention map of Lecture 4 from the committed Lab 4 output.

  images/04-attention-alignment.svg  attention weights alpha[t][i] for `3 March 2021` (Figure 4.2)

The weights come from data/lab04_attention_example.json, written by the results
cell of notebooks/04-seq2seq-attention.ipynb: the dot-product attention model
of the recorded run (settings in the file), greedy decoding, one row per output
token including <eos>, one column per source character. Nothing is invented or
smoothed; each cell is shaded by its weight and weights of at least 0.10 are
printed in the cell.

The layout and the style block follow the Day 1 schematics (images/02-*.svg to
04-*.svg), so the figure keeps a <title> and a <desc> for screen readers.
Output is deterministic: running the script twice gives identical files.

Run:  python scripts/make_figures_04.py      (standard library only)
"""

# SVG markup and the copied Day 1 style block are kept on one line each, so they
# read as they appear in the output.
# ruff: noqa: E501

from __future__ import annotations

import json
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parent.parent
IMAGES = ROOT / "images"
SOURCE = ROOT / "data" / "lab04_attention_example.json"

NAVY, ACCENT, INK, MUTED, RULE, PALE = (
    "#16324f",
    "#b3541e",
    "#1f2933",
    "#52606d",
    "#c5ced8",
    "#f4f6f9",
)
PRINT_AT = 0.10  # weights at or above this are printed in their cell

STYLE = """\
    text { font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 15px; fill: #1f2933; }
    .m { font-family: "Source Serif 4", Georgia, "Times New Roman", serif; }
    .i { font-style: italic; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; }
    .note { font-size: 13.5px; fill: #52606d; }
    .acc { fill: #b3541e; }
    .rule { fill: none; stroke: #c5ced8; stroke-width: 1.5; }"""


def mix(a: str, b: str, t: float) -> str:
    """Linear blend of two hex colors, t in [0, 1]."""
    ca = [int(a[i : i + 2], 16) for i in (1, 3, 5)]
    cb = [int(b[i : i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(ca, cb, strict=True))


def text(x: float, y: float, s: str, cls: str = "", anchor: str = "middle", size=None) -> str:
    st = f' style="font-size:{size}px"' if size else ""
    c = f' class="{cls}"' if cls else ""
    return f'<text x="{x:g}" y="{y:g}" text-anchor="{anchor}"{c}{st}>{escape(s)}</text>'


def bracket(b: list[str], x: float, y0: float, y1: float, label: str) -> None:
    """A rule bracket to the left of rows y0..y1 with a label."""
    b.append(f'<path class="rule" d="M{x:g},{y0 + 4:g} h-6 V{y1 - 4:g} h6"/>')
    b.append(text(x - 12, (y0 + y1) / 2 + 5, label, "note", anchor="end"))


def describe(src: list[str], out: list[str], alpha: list[list[float]]) -> str:
    """Alt text that states, row by row, where the weight sits."""

    def name(i: int) -> str:
        ch = "space" if src[i] == " " else src[i]
        return f"{ch} (position {i + 1})"

    rows = []
    for t, row in enumerate(alpha):
        ranked = sorted(range(len(row)), key=lambda i: -row[i])
        first, second = ranked[0], ranked[1]
        part = f"{out[t]}: {row[first]:.2f} on {name(first)}"
        if row[second] >= PRINT_AT:
            part += f", {row[second]:.2f} on {name(second)}"
        rows.append(part)
    return (
        f"Heat-map of measured attention weights from Lab 4, dot-product attention model, greedy decoding. "
        f"Columns are the {len(src)} characters of the source {''.join(src)}; rows are the {len(out)} output "
        f"tokens {' '.join(out)}. Each row sums to 1. Largest weights per row: "
        + "; ".join(rows)
        + "."
    )


def fig_alignment(data: dict) -> str:
    src, out, alpha = data["source_tokens"], data["output_tokens"], data["alpha"]
    assert len(alpha) == len(out) and all(len(row) == len(src) for row in alpha)
    for row in alpha:
        assert abs(sum(row) - 1) < 0.01, "each row must sum to 1, to rounding"

    c = 36  # cell size
    gx, gy = 150, 96  # top-left corner of the grid
    n_rows, n_cols = len(out), len(src)
    W, H = 680, gy + n_rows * c + 72
    b: list[str] = [f'<rect width="{W}" height="{H}" rx="8" fill="#ffffff"/>']

    b.append(
        f'<text x="{gx + n_cols * c / 2:g}" y="26" text-anchor="middle" class="note">source position '
        f'<tspan class="m i" font-size="16">i</tspan> (one column per character)</text>'
    )
    b.append(
        f'<text transform="translate(24,{gy + n_rows * c / 2:g}) rotate(-90)" text-anchor="middle" '
        f'class="note">output step <tspan class="m i" font-size="16">t</tspan> (one row per token)</text>'
    )
    for i, ch in enumerate(src):
        x = gx + c * i + c / 2
        b.append(text(x, gy - 34, str(i + 1), "note", size=12))
        b.append(text(x, gy - 12, "␣" if ch == " " else ch, "mono", size=17))
    for t, tok in enumerate(out):
        y = gy + c * t + c / 2 + 6
        if tok == "<eos>":
            b.append(text(128, y - 1, tok, "mono", size=13))
        else:
            b.append(text(128, y, tok, "mono", size=17))

    for t, row in enumerate(alpha):
        for i, w in enumerate(row):
            x, y = gx + c * i, gy + c * t
            b.append(
                f'<rect x="{x}" y="{y}" width="{c}" height="{c}" fill="{mix(PALE, NAVY, w)}" stroke="#ffffff" stroke-width="1.5"/>'
            )
            if w >= PRINT_AT:
                col = "#ffffff" if w > 0.45 else INK
                b.append(
                    f'<text x="{x + c / 2:g}" y="{y + c / 2 + 4:g}" text-anchor="middle" class="mono" style="font-size:11px;fill:{col}">{w:.2f}</text>'
                )
    b.append(
        f'<rect x="{gx}" y="{gy}" width="{n_cols * c}" height="{n_rows * c}" fill="none" stroke="{RULE}" stroke-width="1"/>'
    )

    # Which output rows are year, month and day digits: the ISO layout YYYY-MM-DD.
    bracket(b, 108, gy, gy + 4 * c, "year")
    bracket(b, 108, gy + 5 * c, gy + 7 * c, "month")
    bracket(b, 108, gy + 8 * c, gy + 10 * c, "day")

    gr = gx + n_cols * c
    b.append(
        f'<path d="M{gr + 6},{gy + 3} h7 v{c - 6} h-7" style="fill:none;stroke:{ACCENT};stroke-width:1.5"/>'
    )
    b.append(text(gr + 20, gy + 15, "each row", "note acc", anchor="start"))
    b.append(text(gr + 20, gy + 32, "sums to 1", "note acc", anchor="start"))

    ly = gy + n_rows * c + 42
    b.append(
        f'<text x="{gx}" y="{ly}" text-anchor="start" class="m " style="font-size:18px"><tspan class="i">α</tspan>'
        f'<tspan dy="4.3" font-size="12.2" class="i">t</tspan><tspan font-size="12.2">,</tspan>'
        f'<tspan font-size="12.2" class="i">i</tspan></text>'
    )
    b.append(text(gx + 48, ly - 1, "0", "note"))
    b.append(
        f'<rect x="{gx + 58}" y="{ly - 12}" width="100" height="13" fill="url(#g)" stroke="{RULE}"/>'
    )
    b.append(text(gx + 168, ly - 1, "1", "note"))
    b.append(text(gr, ly - 1, "measured: Lab 4, dot-product attention", "note acc", anchor="end"))

    title = "Measured attention weights for 3 March 2021, Lab 4"
    defs = f'<linearGradient id="g" x1="0" x2="1" y1="0" y2="0"><stop offset="0" stop-color="{PALE}"/><stop offset="1" stop-color="{NAVY}"/></linearGradient>'
    inner = "\n  ".join(b)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        f'role="img" aria-labelledby="t d">\n'
        f'  <title id="t">{escape(title)}</title>\n'
        f'  <desc id="d">{escape(describe(src, out, alpha))}</desc>\n'
        f"  <style>\n{STYLE}\n  </style>\n"
        f"  <defs>\n    {defs}\n  </defs>\n"
        f"  {inner}\n"
        f"</svg>\n"
    )


def main() -> None:
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    (IMAGES / "04-attention-alignment.svg").write_text(fig_alignment(data), encoding="utf-8")
    src, out = data["source_tokens"], data["output_tokens"]
    verdict = "correct" if data["correct"] else "WRONG"
    print(
        f"wrote images/04-attention-alignment.svg: {data['source']!r} -> {data['prediction']!r} ({verdict})"
    )
    for t, row in enumerate(data["alpha"]):
        i = max(range(len(row)), key=row.__getitem__)
        print(f"  {out[t]:>5}  arg-max position {i + 1:>2} {src[i]!r:<5} weight {row[i]:.3f}")


if __name__ == "__main__":
    main()
