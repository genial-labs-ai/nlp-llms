# Contributing

Thank you for helping improve this workshop. Corrections, clearer explanations and bug reports are all welcome.

## Where to edit

| To change | Edit | Then |
|---|---|---|
| A module's title, duration, objectives or summary | `_variables.yml` | Run both generators |
| The schedule | `_variables.yml` (`schedule`, `days`) | Run `gen_tables.py` |
| Lecture content | `lectures/NN-slug.qmd` | `quarto preview` |
| A lab | `notebooks/NN-slug.ipynb`, any cell except the first and last | Run `gen_notebooks.py` |
| Site pages | the `.qmd` file at the repository root | `quarto preview` |

Do not edit anything under `_includes/`, the first or last cell of a notebook, or the region between the `BEGIN modules` and `END modules` markers in `README.md`. They are generated, and CI fails if they differ from what the generators produce.

## Notebook rules

- A notebook runs top to bottom on a fresh free-tier Colab runtime with no API keys set.
- Anything Colab does not preinstall is installed in the setup cell with a pinned version.
- Each exercise is a `# TODO N` cell (tag `exercise`) that defines a stub, followed by a folded solution cell (tag `solution`), a short "why this works" note, and a checkpoint cell (tag `checkpoint`).
- **The solution never replaces the participant's code.** Mark each definition in a solution cell with `@workshop.solution(N)`, as its outermost decorator, below the `#@title` line. For a value rather than a function or class, use `name = workshop.solution_value(N, "name", value)`. The harness cell that `scripts/gen_notebooks.py` writes at the top of every notebook stores the reference, and binds it only when `WORKED_EXAMPLE` is ticked or `NLP_LLMS_WORKED=1` (CI's default). A stuck participant calls `workshop.use_reference(N)`.
- **A stub must not raise when it is defined.** Calling an unfinished pure stub must raise `NotImplementedError("TODO N")`. A partial stub (some lines given) may simply return a wrong value: its checkpoint must catch that.
- **The first statement of a checkpoint cell is `workshop.checkpoint(N)`**, or `workshop.checkpoint(N, label="3b")`, or `workshop.checkpoint(label="...")` for a check on code the lab provides. The cell then reports whether it checked the participant's code or the reference.
- **Run the three modes before you open a pull request:**
  - `scripts/test_notebooks.py` (worked);
  - `--learner` (the run must stop at the first TODO);
  - `--verify-checkpoints` (each exercise's first checkpoint must fail on its stub).

  `tests/test_exercises.py` checks the structure. Migrate an older lab with `scripts/migrate_exercises.py`.
- One optional stretch section, last.
- API keys come from Colab Secrets. Never write a key into a cell.
- Notebooks are committed without outputs. `gen_notebooks.py` strips them.

Start a new lab with `uv run --group site python scripts/new_notebook.py mNN`.

## Before you open a pull request

```bash
uv run --group site python scripts/gen_tables.py
uv run --group site python scripts/gen_notebooks.py
uv run --group lint ruff check scripts tests data agents-intro
uv run --group lint ruff format --check scripts tests data agents-intro
uv run --group test python -m unittest discover -s tests -v
uv run --group execute python scripts/test_notebooks.py
quarto render && uv run --group site python scripts/check_links.py
```

Without flags, `test_notebooks.py` runs each lab's open-model path, which needs the Hugging Face Hub. CI runs the offline test paths instead; the flags are listed in the notebooks job of `.github/workflows/publish.yml`.

Work on a branch and open a pull request against `main`.

## Style

American English. Plain, direct sentences. Define a term the first time it appears. Every equation in a lecture should map to a named step in its lab.

## License

By contributing you agree that your text is licensed CC BY 4.0 and your code MIT, as described in [LICENSE](LICENSE).
