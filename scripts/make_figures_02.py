"""Draw the data-driven figure of Lecture 2 from Lab 2's saved PCA coordinates.

  images/02-embedding-pca.svg  2-D PCA of 60 SGNS word vectors in four groups

The input, images/02-embedding-pca.csv (columns word, group, pc1, pc2), is the
file `lab02_pca_points.csv` that notebooks/02-word-vectors.ipynb writes in
Section A3, copied from the run recorded in data/baselines.json
(lab02.sgns_avg_ffn). The vectors were scaled to unit length and centered
before PCA, so the plot shows directions, as cosine similarity sees them.

Colors are the four categorical slots checked with a colorblind-safety
validator; each group also has its own marker shape, and every point carries
its word, so group identity never rests on color alone.

Run:  uv run --group site --with matplotlib python scripts/make_figures_02.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
IMAGES = ROOT / "images"
NAVY, MUTED, GRID = "#16324f", "#5b6770", "#d9dee3"
GROUP_STYLE = {  # group -> (color, marker), in the lab's group order
    "Vision": ("#2a78d6", "o"),
    "Language": ("#b3541e", "s"),
    "Robotics": ("#1baf7a", "^"),
    "Numbers": ("#7a4fa3", "D"),
}
# Candidate label offsets in points, tried in order: the first that overlaps nothing wins,
# otherwise the one that overlaps least.
OFFSETS = [
    (4, 3),
    (4, -9),
    (-4, 3),
    (-4, -9),
    (0, 7),
    (0, -12),
    (7, -3),
    (-7, -3),
    (6, 9),
    (-6, 9),
    (6, -15),
    (-6, -15),
]


def load_points() -> list[tuple[str, str, float, float]]:
    with open(IMAGES / "02-embedding-pca.csv", encoding="utf-8", newline="") as f:
        return [
            (r["word"], r["group"], float(r["pc1"]), float(r["pc2"])) for r in csv.DictReader(f)
        ]


def style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def place_labels(fig, ax, points) -> None:
    """Label every point at the candidate offset that overlaps earlier labels and markers least."""
    renderer = fig.canvas.get_renderer()
    centers = [ax.transData.transform((x, y)) for _, _, x, y in points]  # marker centers, pixels
    taken = [(px - 4, py - 4, px + 4, py + 4) for px, py in centers]

    def overlap(a, b) -> float:
        w = min(a[2], b[2]) - max(a[0], b[0])
        h = min(a[3], b[3]) - max(a[1], b[1])
        return w * h if w > 0 and h > 0 else 0.0

    for word, _, x, y in points:
        best = None
        for dx, dy in OFFSETS:
            ha = "left" if dx > 0 else "right" if dx < 0 else "center"
            text = ax.annotate(
                word,
                (x, y),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=7.5,
                color=NAVY,
                ha=ha,
            )
            box = text.get_window_extent(renderer).padded(1)
            bounds = (box.x0, box.y0, box.x1, box.y1)
            cost = sum(overlap(bounds, b) for b in taken)
            if best is None or cost < best[0]:
                if best is not None:
                    best[1].remove()
                best = (cost, text, bounds)
            else:
                text.remove()
            if cost == 0:
                break
        taken.append(best[2])


def embedding_pca(points) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 6.6))
    style(ax)
    for group, (color, marker) in GROUP_STYLE.items():
        xs = [x for _, g, x, _ in points if g == group]
        ys = [y for _, g, _, y in points if g == group]
        ax.scatter(
            xs,
            ys,
            s=30,
            color=color,
            marker=marker,
            edgecolors="white",
            linewidths=0.8,
            label=group,
        )
    ax.set_xlim(-0.75, 0.68)
    ax.set_ylim(-0.66, 0.52)
    ax.set_xlabel("First principal component", color=NAVY)
    ax.set_ylabel("Second principal component", color=NAVY)
    ax.set_title(
        "SGNS word vectors from Lab 2, projected to two dimensions",
        color=NAVY,
        fontsize=11,
        loc="left",
    )
    ax.legend(frameon=False, fontsize=9, loc="lower left", labelcolor=NAVY)
    fig.tight_layout()
    place_labels(fig, ax, points)
    fig.savefig(IMAGES / "02-embedding-pca.svg", format="svg", metadata={"Date": None})
    plt.close(fig)


def main() -> None:
    plt.rcParams["svg.hashsalt"] = "nlp-llms-02"
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["svg.fonttype"] = "none"
    points = load_points()
    embedding_pca(points)
    counts = {g: sum(1 for _, pg, _, _ in points if pg == g) for g in GROUP_STYLE}
    print(
        f"02-embedding-pca.svg: {len(points)} words; "
        + ", ".join(f"{g} {n}" for g, n in counts.items())
    )


if __name__ == "__main__":
    main()
