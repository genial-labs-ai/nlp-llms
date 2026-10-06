# From Traditional NLP to Modern LLMs

[![Publish](https://github.com/project-delphi/nlp-llms/actions/workflows/publish.yml/badge.svg)](https://github.com/project-delphi/nlp-llms/actions/workflows/publish.yml)
[![Site](https://img.shields.io/badge/site-project--delphi.github.io%2Fnlp--llms-0a7d5a)](https://project-delphi.github.io/nlp-llms/)
[![Quarto](https://img.shields.io/badge/built%20with-Quarto-447099)](https://quarto.org)
[![License: CC BY 4.0 / MIT](https://img.shields.io/badge/license-CC%20BY%204.0%20%2F%20MIT-blue)](LICENSE)

A four-day workshop by Genial Labs on the path from traditional NLP to modern LLMs: n-grams, word vectors, attention, transformers, RLHF, RLCD and agents. Modules 1 to 14 are each a short lecture followed by a hands-on lab in Google Colab, and the capstone is a hands-on afternoon.

**Workshop site: <https://project-delphi.github.io/nlp-llms/>** · [Schedule](https://project-delphi.github.io/nlp-llms/schedule.html) · [Notebooks](https://project-delphi.github.io/nlp-llms/notebooks.html) · [References](https://project-delphi.github.io/nlp-llms/references.html)

<!-- BEGIN status -->
> **As of 2026-10-06: not yet ready to teach.** 0 of 15 labs have run end to end, with their current code, on the Colab runtime they are designed for. 7 have run end to end on their real path elsewhere, on another machine or on the CI runner (3 of them only with QUICK settings), and 0 more in part. On the GitHub CPU runner (newest run of each notebook, 2026-10-06), 16 of 16 notebooks passed; 9 of the passing runs used test doubles, which check that the code runs, not what the models do. 11 of 11 pieces of blocking work are open. See the [readiness page](https://project-delphi.github.io/nlp-llms/readiness.html).
<!-- END status -->

## Who it is for

Machine learning practitioners who are comfortable with Python, NumPy and basic ML, and have some PyTorch. API keys are optional: every lab that uses a commercial API also runs on a free open model.

## The modules

<!-- BEGIN modules -->
| # | Day | Module | Lab |
|---|---|---|---|
| 0 | 1 | [Coding agents in the terminal](https://project-delphi.github.io/nlp-llms/lectures/00-coding-agents.html) | No notebook: runs in your terminal |
| 1 | 1 | [Text as data](https://project-delphi.github.io/nlp-llms/lectures/01-text-as-data.html) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/project-delphi/nlp-llms/blob/main/notebooks/01-text-as-data.ipynb) |
| 2 | 1 | [Word vectors and neural networks](https://project-delphi.github.io/nlp-llms/lectures/02-word-vectors.html) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/project-delphi/nlp-llms/blob/main/notebooks/02-word-vectors.ipynb) |
| 3 | 1 | [Sequence models](https://project-delphi.github.io/nlp-llms/lectures/03-sequence-models.html) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/project-delphi/nlp-llms/blob/main/notebooks/03-sequence-models.ipynb) |
| 4 | 1 | [Seq2seq and attention](https://project-delphi.github.io/nlp-llms/lectures/04-seq2seq-attention.html) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/project-delphi/nlp-llms/blob/main/notebooks/04-seq2seq-attention.ipynb) |
| 5 | 2 | [The transformer](https://project-delphi.github.io/nlp-llms/lectures/05-transformer-from-scratch.html) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/project-delphi/nlp-llms/blob/main/notebooks/05-transformer-from-scratch.ipynb) |
| 6 | 2 | [Pretraining and the Hugging Face stack](https://project-delphi.github.io/nlp-llms/lectures/06-pretraining-huggingface.html) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/project-delphi/nlp-llms/blob/main/notebooks/06-pretraining-huggingface.ipynb) |
| 7 | 2 | [Fine-tuning and LoRA](https://project-delphi.github.io/nlp-llms/lectures/07-finetuning-lora.html) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/project-delphi/nlp-llms/blob/main/notebooks/07-finetuning-lora.ipynb) |
| 8 | 2 | [LLMs through APIs](https://project-delphi.github.io/nlp-llms/lectures/08-llm-apis.html) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/project-delphi/nlp-llms/blob/main/notebooks/08-llm-apis.ipynb) |
| 9 | 3 | [Reinforcement and preference learning](https://project-delphi.github.io/nlp-llms/lectures/09-preference-learning.html) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/project-delphi/nlp-llms/blob/main/notebooks/09-preference-learning.ipynb) |
| 10 | 3 | [RLHF](https://project-delphi.github.io/nlp-llms/lectures/10-rlhf.html) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/project-delphi/nlp-llms/blob/main/notebooks/10-rlhf.ipynb) |
| 11 | 3 | [Calibration](https://project-delphi.github.io/nlp-llms/lectures/11-calibration.html) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/project-delphi/nlp-llms/blob/main/notebooks/11-calibration.ipynb) |
| 12 | 3 | [RLCD and Jev](https://project-delphi.github.io/nlp-llms/lectures/12-rlcd-jev.html) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/project-delphi/nlp-llms/blob/main/notebooks/12-rlcd-jev.ipynb) |
| 13 | 4 | [Retrieval-augmented generation](https://project-delphi.github.io/nlp-llms/lectures/13-rag.html) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/project-delphi/nlp-llms/blob/main/notebooks/13-rag.ipynb) |
| 14 | 4 | [Agents](https://project-delphi.github.io/nlp-llms/lectures/14-agents.html) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/project-delphi/nlp-llms/blob/main/notebooks/14-agents.ipynb) |
| 15 | 4 | [Capstone](https://project-delphi.github.io/nlp-llms/lectures/15-capstone.html) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/project-delphi/nlp-llms/blob/main/notebooks/15-capstone.ipynb) |
<!-- END modules -->

| Day | Theme |
|---|---|
| 1 | Foundations: from counts to attention |
| 2 | Transformers and LLMs |
| 3 | Training objectives: preference and calibration |
| 4 | RAG, agents and capstone |

## Run it locally

You need [Quarto](https://quarto.org/docs/get-started/) 1.6 or later and [uv](https://docs.astral.sh/uv/).

```bash
quarto preview                                  # live preview of the site
quarto render                                   # build the site into docs/
uv run --group notebooks jupyter lab            # work on the labs
```

## Regenerating the derived files

`_variables.yml` is the single source of truth for module titles, durations, objectives and URLs. After changing it, or after adding a notebook, run:

```bash
uv run --group site python scripts/gen_tables.py      # tables in _includes/ and this README
uv run --group site python scripts/gen_notebooks.py   # notebook header and footer cells
```

Checks:

```bash
uv run --group lint ruff check scripts tests data agents-intro
uv run --group test python -m unittest discover -s tests -v
uv run --group execute python scripts/test_notebooks.py   # run every notebook
quarto render && uv run --group site python scripts/check_links.py
```

To start a new lab from the template:

```bash
uv run --group site python scripts/new_notebook.py m03        # add --api for API labs
```

## How it deploys

`.github/workflows/publish.yml` regenerates the derived files, fails if they differ from what is committed, lints, tests, renders the site and checks its links. On a push to `main`, or a manual run on `main`, it deploys the site to GitHub Pages at <https://project-delphi.github.io/nlp-llms/>. Pages must be set to deploy from GitHub Actions (Settings → Pages → Source). Forks skip the deploy job.

## Repository layout

| Path | What it holds |
|---|---|
| `_variables.yml` | Single source of truth: modules, schedule, URLs, model IDs |
| `_quarto.yml`, `custom.scss`, `fonts/` | Site configuration and theme |
| `*.qmd`, `lectures/` | Site pages; one lecture page per module |
| `notebooks/` | One Colab lab per module from 1 to 15, stored without outputs (Module 0 has none) |
| `agents-intro/` | Module 0 reference solutions for its two apps |
| `_includes/` | Generated tables. Do not edit by hand |
| `scripts/`, `tests/` | Generators, notebook runner, link check, tests |
| `PLAN.md`, `AGENTS.md` | The build plan and the rules for coding agents |

## License

Teaching text is licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); code is licensed MIT. See [LICENSE](LICENSE).
