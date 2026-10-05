# Changelog

All notable changes to this workshop are recorded here.

## Unreleased

### Added

- Master plan (`PLAN.md`) for a four-day, 15-module workshop, and agent guidance with four build personas (`AGENTS.md`).
- Quarto website scaffold: landing, setup, schedule, four day pages, 15 lecture stubs, notebooks index, references, teach and FAQ pages.
- `_variables.yml` as the single source of truth, with generators for the schedule, module and notebook tables and for notebook header and footer cells.
- Notebook template, `00-setup.ipynb`, a notebook runner and an internal link check.
- CI workflow: generate, drift gate, lint, test, render, link check, notebook execution, and opt-in deployment to GitHub Pages.
- Scheduled notebook health workflow (`health.yml`): a weekly hermetic run, a non-blocking run of the open-model (Hub) paths, and an on-demand run with API keys. The pull-request notebook run no longer reaches the Hugging Face Hub.
- Module 0, "Coding agents in the terminal": an optional, self-serve hour on Day 1 from 08:00 to 09:00 (also usable as pre-work), with no notebook. `_variables.yml` gains `modules.m00` (with `notebook: false`), a `self_serve` schedule slot and `days.d1.self_serve`, and `agents_intro` (three.js and agent CLI npm versions, dated). The generators, tests, navbar, day, schedule, setup and teach pages, facilitator guide and pace sheet handle a module numbered 0.
- Site redesign: landing page with a hero, day cards and the module path; designed module headers; a module sidebar and previous/next links on lecture pages; a timetable with lecture, lab and break marked; a light/dark toggle. `scripts/gen_tables.py` now also writes `_includes/facts.md`, `path.md` and `sidebar.yml`.
- Lecture pedagogy pass (Modules 1–15): a recap box, check-yourself questions with folded answers, worked numeric examples and interactive Observable JS demos, all outside the 45-minute budget. The demos load Observable Plot from jsDelivr when the page is viewed.
