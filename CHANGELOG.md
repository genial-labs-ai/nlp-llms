# Changelog

All notable changes to this workshop are recorded here.

## Unreleased

### Added

- Exercise harness: every notebook gets a generated harness cell and a summary cell. A folded solution now stores the reference and binds it only in worked mode (`WORKED_EXAMPLE`, or `NLP_LLMS_WORKED=1` in CI). Every checkpoint says whether it checked the participant's code or the reference. A stuck participant can call `workshop.use_reference(N)`. The last cell prints which checkpoints passed on whose code, and a run record. `scripts/test_notebooks.py` gains `--learner`, `--verify-checkpoints` and `--record`; `scripts/add_run_record.py` and `tests/test_exercises.py` are new.
- Readiness from evidence: a `readiness` block and per-module readiness fields in `_variables.yml`, run records in `runs/` (schema, validator, records backfilled from CI and the briefs), `scripts/readiness.py`, and a generated [readiness page](readiness.qmd) with a one-paragraph status on the landing, FAQ, teach and facilitator pages and in the README. The status says plainly that no lab has yet run on Colab.

- Master plan (`PLAN.md`) for a four-day, 15-module workshop, and agent guidance with four build personas (`AGENTS.md`).
- Quarto website scaffold: landing, setup, schedule, four day pages, 15 lecture stubs, notebooks index, references, teach and FAQ pages.
- `_variables.yml` as the single source of truth, with generators for the schedule, module and notebook tables and for notebook header and footer cells.
- Notebook template, `00-setup.ipynb`, a notebook runner and an internal link check.
- CI workflow: generate, drift gate, lint, test, render, link check, notebook execution, and opt-in deployment to GitHub Pages.
- Scheduled notebook health workflow (`health.yml`): a weekly hermetic run, a non-blocking run of the open-model (Hub) paths, and an on-demand run with API keys. The pull-request notebook run no longer reaches the Hugging Face Hub.
- Module 0, "Coding agents in the terminal": an optional, self-serve hour on Day 1 from 08:00 to 09:00 (also usable as pre-work), with no notebook. `_variables.yml` gains `modules.m00` (with `notebook: false`), a `self_serve` schedule slot and `days.d1.self_serve`, and `agents_intro` (three.js and agent CLI npm versions, dated). The generators, tests, navbar, day, schedule, setup and teach pages, facilitator guide and pace sheet handle a module numbered 0.
- Site redesign: landing page with a hero, day cards and the module path; designed module headers; a module sidebar and previous/next links on lecture pages; a timetable with lecture, lab and break marked; a light/dark toggle. `scripts/gen_tables.py` now also writes `_includes/facts.md`, `path.md` and `sidebar.yml`.
- Lecture pedagogy pass (Modules 1–15): a recap box, check-yourself questions with folded answers, worked numeric examples and interactive Observable JS demos, all outside the 45-minute budget. The demos load Observable Plot from jsDelivr when the page is viewed.
- Lecture 14 (Agents), certification and harness pass: the agent harness defined and separated from evaluation and test harnesses; an ARC-AGI worked example of the same model in different harnesses, with cost per solved task; tool design, stopping and enforcement, escalation triggers and hand-offs; a new section 9 on workflow and agentic design patterns, coordinators and subagents, and context as a budget, with an optional Claude Agent SDK and MCP mapping. Drawn from Anthropic's *Claude Certified Architect – Foundations* exam guide and engineering posts, Andrew Ng's letters and ARC Prize's results. Module 0 names the harness; Module 15's further study points to Ng's course and the certification guide; `references.qmd` lists every new source. Lab 14's stretch section has four parts: the verify node, plus a research subagent with a coverage check, compaction that keeps the facts, and a stand-alone hand-off record, each with a folded solution and a scripted checkpoint.

### Changed

- "Run all" no longer replaces participants' functions with the solutions. Unless `WORKED_EXAMPLE` is ticked, Run all stops at the first unwritten exercise, with a message saying how to go on.
- Readiness claims reconciled with the evidence. The landing page no longer says the workshop is being written. The FAQ and setup page no longer promise T4 run times. The facilitator guide, pace sheet and assessments no longer say the capstone notebook or its collector do not exist. The setup page links straight to the setup notebook in Colab. The entry check's review threshold is now the same in both places (two or more of three in an area).

- Lecture 14 timing: a desk timing (words per budgeted minute, against the other lectures) found the page needed about 75 minutes after the harness pass. The ARC-AGI worked example, tool-design habits, resume or start fresh, the stopping rule, escalation triggers and the hand-off, and the detail on subagents and context are now collapsed optional callouts with short in-budget summaries; sections 5 and 9 are 5 minutes each. The pace sheet follows. Module 14 keeps three objectives.
- Lab 14 stretch, part A: an answer from a run that retrieved no passages is delivered as `no-sources` without a verifier call, instead of receiving a verdict against nothing.
