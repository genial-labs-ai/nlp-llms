"""Execute the notebooks top to bottom in a real kernel.

Each notebook is run from a temporary directory, so nothing is written back to
notebooks/. Exercise stubs are followed by their solution cells, so a full run
exercises the solutions and every checkpoint. No API keys are needed: labs fall
back to open models when none are set.

Run:  uv run --group execute python scripts/test_notebooks.py [slug ...]
"""

from __future__ import annotations

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


def run(path: Path) -> tuple[bool, float, str]:
    nb = nbformat.read(path, as_version=4)
    started = time.monotonic()
    with tempfile.TemporaryDirectory() as workdir:
        client = NotebookClient(
            nb,
            timeout=TIMEOUT_SECONDS,
            kernel_name="python3",
            resources={"metadata": {"path": workdir}},
        )
        try:
            client.execute()
        except CellExecutionError as error:
            return False, time.monotonic() - started, str(error)
    return True, time.monotonic() - started, ""


def main() -> int:
    # Notebooks run in a temporary directory, so point the data loader
    # (data/README.md) at the repository's copies instead of the network.
    os.environ.setdefault("NLP_LLMS_DATA", str(ROOT / "data"))
    wanted = set(sys.argv[1:])
    paths = [p for p in sorted(NOTEBOOKS.glob("*.ipynb")) if not wanted or p.stem in wanted]
    missing = wanted - {p.stem for p in paths}
    if missing:
        print(f"No such notebook: {', '.join(sorted(missing))}")
        return 2
    failures = 0
    for path in paths:
        ok, seconds, error = run(path)
        print(f"{'PASS' if ok else 'FAIL'}  {path.name}  {seconds:.1f}s")
        if not ok:
            failures += 1
            print(error)
    print(f"{len(paths) - failures} of {len(paths)} notebooks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
