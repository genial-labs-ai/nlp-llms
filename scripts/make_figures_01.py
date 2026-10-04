"""Draw the two data-driven figures of Lecture 1 from the committed corpus.

  images/01-zipf.svg            word frequency against rank, log-log (Figure 1.1)
  images/01-ngram-sparsity.svg  share of test n-grams never seen in training (Figure 1.2)

Both use the Tiny Shakespeare splits of data/README.md and the word tokenizer
of Lab 1 (lowercase; runs of letters, digits or apostrophes; each other
non-space symbol is its own token). Figure 1.1 ranks word tokens only, so
punctuation marks are left out; Figure 1.2 counts every token.

Run:  uv run --group site --with matplotlib python scripts/make_figures_01.py
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
IMAGES = ROOT / "images"
NAVY, ACCENT, MUTED, GRID = "#16324f", "#b3541e", "#5b6770", "#d9dee3"


def tokenize(text: str) -> list[str]:
    """The Lab 1 tokenizer."""
    return re.findall(r"[a-z0-9']+|[^a-z0-9'\s]", text.lower())


def load_splits() -> dict[str, str]:
    v = yaml.safe_load((ROOT / "_variables.yml").read_text(encoding="utf-8"))
    lm = v["datasets"]["lm"]
    text = (ROOT / "data" / lm["file"]).read_text(encoding="utf-8")
    return {name: text[start:end] for name, (start, end) in lm["splits"].items()}


def style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def save(fig, name: str) -> None:
    # A fixed hash salt and no date keep the SVG identical from run to run.
    fig.savefig(IMAGES / name, format="svg", metadata={"Date": None})
    plt.close(fig)


def zipf(train_tokens: list[str]) -> dict:
    words = [t for t in train_tokens if re.fullmatch(r"[a-z0-9']+", t)]
    ranked = Counter(words).most_common()
    freqs = [count for _, count in ranked]
    ranks = range(1, len(freqs) + 1)
    once = sum(1 for f in freqs if f == 1)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    style(ax)
    ax.loglog(ranks, freqs, color=NAVY, linewidth=2, label="Observed")
    # Reference line f = A / r through the most frequent word: slope -1.
    ax.loglog(
        [1, len(freqs)],
        [freqs[0], freqs[0] / len(freqs)],
        color=ACCENT,
        linewidth=1.5,
        linestyle="--",
        label="Slope −1",
    )
    for rank in (1, 2, 3):
        ax.plot([rank], [ranked[rank - 1][1]], marker="o", markersize=4, color=NAVY)
    top = ", ".join(f"“{word}” ({count:,})" for word, count in ranked[:3])
    ax.annotate(
        f"Top three: {top}",
        xy=(2, ranked[1][1]),
        xytext=(0, 16),
        textcoords="offset points",
        fontsize=9,
        color=NAVY,
    )
    first_once = len(freqs) - once + 1
    ax.annotate(
        f"{once:,} of {len(freqs):,} distinct words\n({once / len(freqs):.0%}) occur once",
        xy=((first_once * len(freqs)) ** 0.5, 1),
        xytext=(len(freqs) * 1.6, 45),
        ha="right",
        fontsize=9,
        color=NAVY,
        arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.8},
    )
    ax.set_ylim(0.4, 3e4)
    ax.set_xlabel("Rank r of the word (1 = most frequent)", color=NAVY)
    ax.set_ylabel("Count f(r) in the training split", color=NAVY)
    ax.set_title(
        "Word frequency against rank, Tiny Shakespeare", color=NAVY, fontsize=11, loc="left"
    )
    ax.legend(frameon=False, fontsize=9, loc="lower left", labelcolor=NAVY)
    fig.tight_layout()
    save(fig, "01-zipf.svg")
    return {"words": len(words), "distinct": len(freqs), "once": once, "top": ranked[:3]}


def unseen_share(train: list[str], test: list[str], n: int) -> float:
    """Fraction of the n-gram occurrences in `test` whose n-gram is not in `train`."""
    seen = set(zip(*[train[i:] for i in range(n)], strict=False))
    grams = list(zip(*[test[i:] for i in range(n)], strict=False))
    return sum(g not in seen for g in grams) / len(grams)


def sparsity(train_tokens: list[str], test_tokens: list[str]) -> list[float]:
    orders = [1, 2, 3, 4, 5]
    shares = [unseen_share(train_tokens, test_tokens, n) for n in orders]

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    style(ax)
    ax.grid(False, axis="x")
    ax.bar(orders, shares, width=0.55, color=NAVY)
    for n, share in zip(orders, shares, strict=True):
        ax.text(n, share + 0.02, f"{share:.0%}", ha="center", fontsize=9, color=NAVY)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xticks(orders)
    ax.set_xlabel("n-gram order n (word level)", color=NAVY)
    ax.set_ylabel("Share of test n-grams unseen in training", color=NAVY)
    ax.set_title(
        "Test n-grams that never occur in the training split",
        color=NAVY,
        fontsize=11,
        loc="left",
    )
    fig.tight_layout()
    save(fig, "01-ngram-sparsity.svg")
    return shares


def main() -> None:
    plt.rcParams["svg.hashsalt"] = "nlp-llms-01"
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["svg.fonttype"] = "none"
    splits = load_splits()
    train, test = tokenize(splits["train"]), tokenize(splits["test"])
    stats = zipf(train)
    print(
        f"01-zipf.svg: {stats['words']:,} word tokens, {stats['distinct']:,} distinct, "
        f"{stats['once']:,} occur once; top three {stats['top']}"
    )
    shares = sparsity(train, test)
    print("01-ngram-sparsity.svg: " + ", ".join(f"n={n}: {s:.4f}" for n, s in enumerate(shares, 1)))


if __name__ == "__main__":
    main()
