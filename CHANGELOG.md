# Changelog

All notable changes to this workshop are recorded here.

## Unreleased

### Added

- Master plan (`PLAN.md`) for a four-day, 15-module workshop, and agent guidance with four build personas (`AGENTS.md`).
- Quarto website scaffold: landing, setup, schedule, four day pages, 15 lecture stubs, notebooks index, references, teach and FAQ pages.
- `_variables.yml` as the single source of truth, with generators for the schedule, module and notebook tables and for notebook header and footer cells.
- Notebook template, `00-setup.ipynb`, a notebook runner and an internal link check.
- CI workflow: generate, drift gate, lint, test, render, link check, notebook execution, and opt-in deployment to GitHub Pages.
