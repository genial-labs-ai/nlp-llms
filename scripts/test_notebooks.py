"""Execute the notebooks top to bottom in a real kernel.

Each notebook is run from a temporary directory, so nothing is written back to
notebooks/. Exercise stubs are followed by their solution cells, so a full run
exercises the solutions and every checkpoint. No API keys are needed: labs fall
back to open models when none are set.

Run:  uv run --group execute python scripts/test_notebooks.py [slug ...]

Options (used by .github/workflows/health.yml):
  --save DIR     write each executed notebook, with its outputs, to DIR
  --expect-hub   fail a notebook that ran but fell back from its open-model path
                 (see FALLBACK_MARKERS), so a green run means the Hub path ran
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = ROOT / "notebooks"
# Per-cell limit. CI raises it: CPU training cells in Labs 2 and 7 run long.
TIMEOUT_SECONDS = int(os.environ.get("NLP_LLMS_CELL_TIMEOUT", "900"))

# Text a lab prints when it could not load an open model and carried on with a
# stand-in. Labs 8, 11, 13 and 14 fall back silently on purpose (a participant
# without Hub access still finishes), so with --expect-hub these count as failures.
FALLBACK_MARKERS = (
    "Could not load ",  # Labs 8, 11, 13, 14: open chat model, encoder, reranker or NLI model
    "did not load",  # Labs 13, 14: build_retriever's encoder or cross-encoder
    "USING StubProvider",  # Labs 8, 11, 13, 14: the test double answered instead of a model
    "LSA stand-in fitted",  # Lab 13: TF-IDF + SVD instead of the dense encoder
    "OFFLINE TEST MODE",  # Labs 6, 7, 10: an offline test flag is set
)


def fallbacks(nb: nbformat.NotebookNode) -> list[str]:
    """The fallback markers found in the notebook's printed output."""
    text = "".join(
        "".join(out.get("text", ""))
        for cell in nb.cells
        if cell.cell_type == "code"
        for out in cell.get("outputs", [])
        if out.get("output_type") == "stream"
    )
    return [marker.strip() for marker in FALLBACK_MARKERS if marker in text]


def run(path: Path, save: Path | None) -> tuple[bool, float, str, list[str]]:
    nb = nbformat.read(path, as_version=4)
    started = time.monotonic()
    ok, error = True, ""
    with tempfile.TemporaryDirectory() as workdir:
        client = NotebookClient(
            nb,
            timeout=TIMEOUT_SECONDS,
            kernel_name="python3",
            resources={"metadata": {"path": workdir}},
        )
        try:
            client.execute()
        except CellExecutionError as exc:
            ok, error = False, str(exc)
    if save is not None:
        save.mkdir(parents=True, exist_ok=True)
        nbformat.write(nb, save / path.name)
    return ok, time.monotonic() - started, error, fallbacks(nb)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("slugs", nargs="*", help="notebook stems; default: all")
    parser.add_argument("--save", type=Path, help="write executed notebooks here")
    parser.add_argument("--expect-hub", action="store_true", help="fail on a fallback")
    args = parser.parse_args()

    # Notebooks run in a temporary directory, so point the data loader
    # (data/README.md) at the repository's copies instead of the network.
    os.environ.setdefault("NLP_LLMS_DATA", str(ROOT / "data"))
    wanted = set(args.slugs)
    paths = [p for p in sorted(NOTEBOOKS.glob("*.ipynb")) if not wanted or p.stem in wanted]
    missing = wanted - {p.stem for p in paths}
    if missing:
        print(f"No such notebook: {', '.join(sorted(missing))}")
        return 2
    failures = 0
    for path in paths:
        ok, seconds, error, found = run(path, args.save)
        fell_back = ok and args.expect_hub and bool(found)
        status = "FAIL" if not ok else "FALLBACK" if fell_back else "PASS"
        note = f"  fell back: {'; '.join(found)}" if found and (fell_back or not ok) else ""
        print(f"{status}  {path.name}  {seconds:.1f}s{note}", flush=True)
        if not ok or fell_back:
            failures += 1
        if not ok:
            print(error)
    print(f"{len(paths) - failures} of {len(paths)} notebooks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
