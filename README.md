# From Traditional NLP to Modern LLMs

[![Publish](https://github.com/project-delphi/nlp-llms/actions/workflows/publish.yml/badge.svg)](https://github.com/project-delphi/nlp-llms/actions/workflows/publish.yml)
[![Quarto](https://img.shields.io/badge/built%20with-Quarto-447099)](https://quarto.org)
[![License: CC BY 4.0 / MIT](https://img.shields.io/badge/license-CC%20BY%204.0%20%2F%20MIT-blue)](LICENSE)

A four-day workshop by Genial Labs on the path from traditional NLP to modern LLMs: n-grams, word vectors, attention, transformers, RLHF, RLCD and agents. Every module is a short lecture followed by a hands-on lab in Google Colab.

> **Status: in development.** The curriculum, site and notebook pipeline are in place. Lectures and labs are being written module by module. See [PLAN.md](PLAN.md).

## Who it is for

Machine learning practitioners who are comfortable with Python, NumPy and basic ML, and have some PyTorch. API keys are optional: every lab that uses a commercial API also runs on a free open model.

## The modules

<!-- BEGIN modules -->
| # | Day | Module | Lab |
|---|---|---|---|
| 1 | 1 | [Text as data](https://project-delphi.github.io/nlp-llms/lectures/01-text-as-data.html) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/project-delphi/nlp-llms/blob/main/notebooks/01-text-as-data.ipynb) |
| 2 | 1 | [Word vectors and neural networks](https://project-delphi.github.io/nlp-llms/lectures/02-word-vectors.html) | In preparation |
| 3 | 1 | [Sequence models](https://project-delphi.github.io/nlp-llms/lectures/03-sequence-models.html) | In preparation |
| 4 | 1 | [Seq2seq and attention](https://project-delphi.github.io/nlp-llms/lectures/04-seq2seq-attention.html) | In preparation |
| 5 | 2 | [The transformer](https://project-delphi.github.io/nlp-llms/lectures/05-transformer-from-scratch.html) | In preparation |
| 6 | 2 | [Pretraining and the Hugging Face stack](https://project-delphi.github.io/nlp-llms/lectures/06-pretraining-huggingface.html) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/project-delphi/nlp-llms/blob/main/notebooks/06-pretraining-huggingface.ipynb) |
| 7 | 2 | [Fine-tuning and LoRA](https://project-delphi.github.io/nlp-llms/lectures/07-finetuning-lora.html) | In preparation |
| 8 | 2 | [LLMs through APIs](https://project-delphi.github.io/nlp-llms/lectures/08-llm-apis.html) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/project-delphi/nlp-llms/blob/main/notebooks/08-llm-apis.ipynb) |
| 9 | 3 | [Reinforcement and preference learning](https://project-delphi.github.io/nlp-llms/lectures/09-preference-learning.html) | In preparation |
| 10 | 3 | [RLHF](https://project-delphi.github.io/nlp-llms/lectures/10-rlhf.html) | In preparation |
| 11 | 3 | [Calibration](https://project-delphi.github.io/nlp-llms/lectures/11-calibration.html) | In preparation |
| 12 | 3 | [RLCD and Jev](https://project-delphi.github.io/nlp-llms/lectures/12-rlcd-jev.html) | In preparation |
| 13 | 4 | [Retrieval-augmented generation](https://project-delphi.github.io/nlp-llms/lectures/13-rag.html) | In preparation |
| 14 | 4 | [Agents](https://project-delphi.github.io/nlp-llms/lectures/14-agents.html) | In preparation |
| 15 | 4 | [Capstone](https://project-delphi.github.io/nlp-llms/lectures/15-capstone.html) | In preparation |
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
uv run --group lint ruff check scripts tests
uv run --group test python -m unittest discover -s tests -v
uv run --group execute python scripts/test_notebooks.py   # run every notebook
quarto render && uv run --group site python scripts/check_links.py
```

To start a new lab from the template:

```bash
uv run --group site python scripts/new_notebook.py m03        # add --api for API labs
```

## How it deploys

`.github/workflows/publish.yml` regenerates the derived files, fails if they differ from what is committed, lints, tests, renders the site and checks its links. On a push to `main` it deploys to GitHub Pages, once the repository variable `DEPLOY_PAGES` is set to `true` and Pages is set to deploy from GitHub Actions.

## Repository layout

| Path | What it holds |
|---|---|
| `_variables.yml` | Single source of truth: modules, schedule, URLs, model IDs |
| `_quarto.yml`, `custom.scss`, `fonts/` | Site configuration and theme |
| `*.qmd`, `lectures/` | Site pages; one lecture page per module |
| `notebooks/` | One Colab lab per module, stored without outputs |
| `_includes/` | Generated tables. Do not edit by hand |
| `scripts/`, `tests/` | Generators, notebook runner, link check, tests |
| `PLAN.md`, `AGENTS.md` | The build plan and the rules for coding agents |

## License

Teaching text is licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); code is licensed MIT. See [LICENSE](LICENSE).
