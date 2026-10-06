"""Create a lab notebook from the template.

Usage:  uv run --group site python scripts/new_notebook.py m03 [--api]

Writes notebooks/<slug>.ipynb for the given module key with the standard
skeleton: setup cell, one worked exercise (stub, folded solution, why-it-works
note, checkpoint) and an optional stretch section. With --api it also adds the
provider cell that reads keys from Colab Secrets and falls back to an open
model. The header and footer cells are added by gen_notebooks.py.
"""

from __future__ import annotations

import argparse

import nbformat
from gen_notebooks import NOTEBOOKS, load_variables
from gen_notebooks import main as normalize_all

SETUP = """\
# Setup: installs, imports, seeds. Pin anything Colab does not preinstall, e.g.
# %pip install -q somepackage==1.2.3
import random

import numpy as np

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
"""

PROVIDER = '''\
# Provider: reads optional API keys from Colab Secrets (the key icon in the
# left sidebar) or from environment variables. With no key the lab runs on a
# small open model, so nobody is blocked.
import os


def get_secret(name: str) -> str | None:
    """Return a secret from Colab Secrets or the environment, or None."""
    try:
        from google.colab import userdata

        return userdata.get(name)
    except Exception:
        return os.environ.get(name)


KEYS = {
    "openai": get_secret("OPENAI_API_KEY"),
    "anthropic": get_secret("ANTHROPIC_API_KEY"),
}
# Set PROVIDER to "openai", "anthropic" or "open" to override the default.
PROVIDER = next((name for name, key in KEYS.items() if key), "open")
print(f"Provider: {PROVIDER}")
'''

EXERCISE_MD = """\
## Exercise 1 · Title

**Predict.** Before running anything, write down what you expect.

**Task.** Complete `my_function` so that the checkpoint below passes.
"""

EXERCISE = """\
# TODO 1: complete this function.
def my_function(x):
    raise NotImplementedError("TODO 1")
"""

SOLUTION = """\
#@title Solution 1 — try it yourself first { display-mode: "form" }
@workshop.solution(1)
def my_function(x):
    return x
"""

WHY = """\
<details>
<summary>Why this solution works</summary>

Explain the idea in two or three sentences, and name the lecture equation it
implements.

</details>
"""

CHECKPOINT = """\
# Checkpoint 1
workshop.checkpoint(1)
assert my_function(1) == 1, "my_function(1) should be 1"
"""

STRETCH_MD = """\
## Stretch (optional)

For those who finish early. Nothing later in the workshop depends on it.
"""


def code(source: str, tags: list[str], **metadata):
    cell = nbformat.v4.new_code_cell(source.rstrip("\n"))
    cell["metadata"] = {"tags": tags, **metadata}
    return cell


def markdown(source: str):
    return nbformat.v4.new_markdown_cell(source.rstrip("\n"))


def template(api: bool):
    cells = [markdown("## Setup"), code(SETUP, ["setup"])]
    if api:
        cells.append(code(PROVIDER, ["setup", "provider"]))
    cells += [
        markdown(EXERCISE_MD),
        code(EXERCISE, ["exercise"]),
        code(SOLUTION, ["solution"], cellView="form", jupyter={"source_hidden": True}),
        markdown(WHY),
        code(CHECKPOINT, ["checkpoint"]),
        markdown(STRETCH_MD),
    ]
    return nbformat.v4.new_notebook(cells=cells)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("key", help="module key in _variables.yml, e.g. m03")
    parser.add_argument("--api", action="store_true", help="add the provider cell")
    args = parser.parse_args()

    modules = load_variables()["modules"]
    if args.key not in modules:
        raise SystemExit(f"{args.key}: not a module key in _variables.yml")
    if not modules[args.key].get("notebook", True):
        raise SystemExit(f"{args.key}: has `notebook: false` in _variables.yml (no lab)")
    path = NOTEBOOKS / f"{modules[args.key]['slug']}.ipynb"
    if path.exists():
        raise SystemExit(f"{path.name} already exists")
    NOTEBOOKS.mkdir(exist_ok=True)
    nbformat.write(template(args.api), path)
    normalize_all()
    print(f"Created {path.relative_to(NOTEBOOKS.parent)}")


if __name__ == "__main__":
    main()
