---
title: "Pace sheet"
subtitle: "Minute by minute, day by day"
---

<!--
Instructor page, rendered by Quarto. The lecture rows are generated: scripts/gen_tables.py writes
_includes/pace-NN.md from the live plan in each lecture's front matter (scripts/live_plan.py), so
change a lecture's minutes there, never here. Lab rows come from the minutes in each notebook's
exercise headings, which follow the "As built" sections of briefs/*.md. Debrief, retrieval-practice
and synthesis rows follow the day pages (day-2.qmd to day-5.qmd) and the facilitator guide. The
capstone rows follow lecture 15's timing. Module titles and slot lengths come from _variables.yml.
-->

## How to read this sheet

{{< include /_includes/module-shape.md >}}

Each module has two tables. The first is the lecture's **live plan**, generated from the lecture page: minutes count from the start of the lecture, and "In the room" lists the checks, predictions and demos whose minutes the row includes. The second is the **lab**: minutes count from the start of the lab. "CP" is a checkpoint; Checkpoint *N* belongs to Exercise *N*. The last column says what participants should have passed by the end of that row. If more than a third of the room has not, apply the module's "behind" rule. A behind rule's minute is a lab minute unless it says lecture minute or build minute. Clock times are on the [schedule](schedule.qmd).

**Day 1:** the lab is the {{< var schedule.clocks.standard.shape.lab >}}-minute core path and ends the module. **Days 2 to 5:** the lab is {{< var schedule.clocks.long.shape.lab >}} minutes: the same core path, with 5 minutes of slack for setup and downloads in its first row, so every later row starts 5 minutes later than the notebook's own minutes suggest. If setup goes quickly, the room runs ahead and keeps the minutes. The lab table ends with the {{< var schedule.clocks.long.shape.debrief >}}-minute **debrief**: the Explain step, taken with the whole room. Each day also opens with retrieval practice and closes with a synthesis, both on the day pages. What to say in each debrief, and what to cut when a day's clock slips, are in the [facilitator guide](facilitator-guide.md#shape).

**These minutes are planning estimates.** **No lab has been timed on Colab or on a T4.** Labs 1 to 5 were timed on a shared CPU only, and Labs 2 to 5 took far longer there than these minutes allow (see the [facilitator guide](facilitator-guide.md)). The lecture plans were set on paper; no lecture has yet been given aloud against its clock (see the [readiness page](readiness.qmd#open-work)). Treat every row as a target until you have run the notebook on the room's runtime and given the lecture once.

Collapsed callouts marked **Optional**, sections marked **Reference**, and checks or demos that a live plan does not name sit outside the lecture minutes. The stretch section of every lab sits outside the lab minutes.

## Before Day 1: Module 0 · {{< var modules.m00.title >}}

Optional pre-work, planned at {{< var modules.m00.minutes >}} minutes, with no lecture; the same rows pace the optional drop-in clinic on Day 1, 08:00–09:00. Minutes count from the start. **Provisional:** these rows follow the step plan on the [Module 0 page](lectures/00-coding-agents.qmd). No one has yet timed the module on a fresh laptop, so they are a target, not a measurement. Install failures and what to do about them are in the [facilitator guide](facilitator-guide.md#module-0).

| Minutes | Segment | By the end |
|---|---|---|
| 0–10 | Install one coding agent, sign in, start it in `~/agent-apps` | the agent answers the sanity prompt and leaves the folder empty |
| 10–20 | Prompts: install git and `gh` if missing, git identity, `gh` sign-in in the browser, a public repository | `gh auth status` (run by the agent) shows the participant signed in; the repository's GitHub page shows the README |
| 20–38 | App 1 by prompts: protein structure explorer (1UBQ), its separate check, the fix, its three.js page | the check matches the page's numbers; `protein/` is committed |
| 38–55 | App 2 by prompts: RFM customer segmentation, its separate check and hand-computed recency, the fix, its three.js page | the check and the recency match; `rfm/` is committed |
| 55–60 | Publish by prompts: landing page, push, GitHub Pages on | both Pages URLs load (allow up to 10 minutes after enabling Pages) |

**Behind at minute 10:** if the agent is still not installed or signed in, pair the participant with a neighbor whose agent works. They take turns writing the prompts in the neighbor's repository, and the participant finishes their own install at home. **At 08:55:** everyone stops; the rest is homework.

## Day 1

### Opening

| Minutes | Segment | By the end |
|---|---|---|
| 0–5 | Welcome: the [intro slides](welcome.qmd) with the [opening lines](facilitator-guide.md#day-1) | |
| 5–10 | Setup check: everyone runs `00-setup.ipynb` | each participant has read the provider line it prints |

### Module 1 · {{< var modules.m01.title >}}

**Lecture** ({{< var schedule.clocks.standard.shape.lecture >}} minutes)

{{< include /_includes/pace-01.md >}}

**Lab** ({{< var schedule.clocks.standard.shape.lab >}} minutes)

| Minutes | Segment | By the end |
|---|---|---|
| 0–8 | Setup, data, Exercise 1 (tokens and vocabulary) | CP1 |
| 8–20 | Exercise 2 (n-gram counts, add-k) | CP2 |
| 20–28 | Exercise 3 (perplexity) | CP3 |
| 28–33 | Exercise 4 (sampling) | CP4 |
| 33–43 | Exercise 5 (TF-IDF, two classifiers) | CP5 |
| 43–50 | Exercise 6 (precision, recall, F1; read the errors); the baseline card | CP6 |

**Behind at minute 20:** run Exercise 4 as a demonstration. Exercises 3, 5 and 6 may not be cut.

### Module 2 · {{< var modules.m02.title >}}

**Lecture** ({{< var schedule.clocks.standard.shape.lecture >}} minutes)

{{< include /_includes/pace-02.md >}}

**Lab** ({{< var schedule.clocks.standard.shape.lab >}} minutes)

| Minutes | Segment | By the end |
|---|---|---|
| 0–4 | Setup, data, three training pairs | |
| 4–16 | Exercise 1 (the SGNS loss) | CP1 |
| 16–20 | Provided: train the embeddings; predict the loss curve | first loss $= (K+1)\log 2$ |
| 20–27 | Exercise 2 (nearest neighbors) | CP2 |
| 27–32 | Provided: analogies and the PCA picture | |
| 32–38 | Exercise 3 (averaging embeddings) | CP3 |
| 38–46 | Exercise 4 (feed-forward classifier) | CP4 |
| 46–50 | The comparison with TF-IDF; results card | |

**Behind at minute 20:** the analogy step (27–32) becomes a demonstration. Skip-gram training took 695 s on the build CPU, far over its 4 minutes; start it before the Exercise 1 discussion ends.

### Module 3 · {{< var modules.m03.title >}}

**Lecture** ({{< var schedule.clocks.standard.shape.lecture >}} minutes)

{{< include /_includes/pace-03.md >}}

**Lab** ({{< var schedule.clocks.standard.shape.lab >}} minutes)

| Minutes | Segment | By the end |
|---|---|---|
| 0–8 | Setup, data, Exercise 1 (an RNN cell by hand) | CP1 |
| 8–14 | Exercise 2 (loss, perplexity, bits per character) | CP2 |
| 14–16 | Run: the n-gram baseline, measured the same way | |
| 16–20 | Run: train the RNN language model | |
| 20–29 | Exercise 3 (vanishing gradients; clipping) | CP3 |
| 29–35 | Exercise 4 (the LSTM cell) | CP4 |
| 35–42 | Run: train the LSTM; the results table | LSTM below the trigram |
| 42–50 | Exercise 5 (sampling with temperature) | CP5 |

**Behind at minute 16:** set `QUICK = True` before the training runs (a quarter of the steps; not the recorded baselines).

### Module 4 · {{< var modules.m04.title >}}

**Lecture** ({{< var schedule.clocks.standard.shape.lecture >}} minutes)

{{< include /_includes/pace-04.md >}}

**Lab** ({{< var schedule.clocks.standard.shape.lab >}} minutes)

| Minutes | Segment | By the end |
|---|---|---|
| 0–3 | Setup; read the generated dates | |
| 3–13 | Exercise 1 (teacher-forced forward pass and loss) | CP1 |
| 13–21 | Exercise 2 (exact match by input length); predict which bucket fails | CP2 |
| 21–33 | Exercise 3 (dot-product attention); accuracy with and without attention | CP3 |
| 33–43 | Exercise 4 (the additive score) | CP4 |
| 43–50 | Exercise 5 (heat-maps, alignment hit rate) | CP5 |

**Behind at minute 21:** keep Exercises 3 and 4 (the objectives need both scores); shorten Exercise 5 to viewing the heat-maps. Training runs start while participants write their next prediction.

## Day 2

### Retrieval practice

| Minutes | Segment | By the end |
|---|---|---|
| 0–6 | Alone, on paper, notes closed: the five questions on the [Day 2 page](day-2.qmd#retrieval) | an answer to each question |
| 6–10 | Compare with a neighbor | |
| 10–15 | The two most-missed questions with the room; then the [opening lines](facilitator-guide.md#day-2) | |

### Module 5 · {{< var modules.m05.title >}}

**Lecture** ({{< var schedule.clocks.long.shape.lecture >}} minutes)

{{< include /_includes/pace-05.md >}}

**Lab** ({{< var schedule.clocks.long.shape.lab >}} minutes) **and debrief** ({{< var schedule.clocks.long.shape.debrief >}})

| Minutes | Segment | By the end |
|---|---|---|
| 0–8 | Setup; Exercise 0 (read the model); the 5 minutes of slack | |
| 8–20 | Exercise 1 (scaled dot-product attention); predict the score spread | CP1 |
| 20–28 | Exercise 2 (the causal mask; leak test) | CP2 |
| 28–34 | Exercise 3 (positions); whole-model leak test | CP3, CP3b |
| 34–48 | Exercise 4 (train the mini-GPT and the LSTM; compare) | CP4a, CP4b |
| 48–55 | Exercise 5 (looking at the heads) | CP5 |
| 55–65 | **Debrief:** mini-GPT and LSTM nats per character, full budget or `QUICK`; the misconception "the transformer wins"; the bridge to Module 6 | the room's numbers on the board |

**Behind at minute 28:** set `QUICK = True` before Exercise 4 (on by default on a CPU runtime); shorten Exercise 5.

### Module 6 · {{< var modules.m06.title >}}

**Lecture** ({{< var schedule.clocks.long.shape.lecture >}} minutes, before lunch)

{{< include /_includes/pace-06.md >}}

The lecture stops at lunch, wherever it has reached. On slow Wi-Fi, participants can open Lab 6, switch on a T4 and run the setup cell as they leave, so the checkpoints download over lunch.

**Lab** ({{< var schedule.clocks.long.shape.lab >}} minutes, after lunch) **and debrief** ({{< var schedule.clocks.long.shape.debrief >}})

| Minutes | Segment | By the end |
|---|---|---|
| 0–8 | Reconnect; setup (start the downloads, or rerun the cell if Colab reset the runtime over lunch); the 5 minutes of slack | |
| 8–18 | Exercise 1 (byte-pair encoding by hand) | CP1 |
| 18–25 | Exercise 2 (train a tokenizer; tokens per word) | CP2 |
| 25–33 | Exercise 3 (causal LM perplexity) | CP3 |
| 33–43 | Exercise 4 (masked LM: select and corrupt) | CP4 |
| 43–55 | Exercise 5 (fine-tune an encoder); the results table | CP5a, CP5b, CP5c |
| 55–65 | **Debrief:** tokens per word; the encoder's test accuracy and macro-F1 beside Labs 1 and 2, by runtime; the misconception "a pretrained model always wins by a wide margin"; the bridge to Module 7 | the room's numbers on the board |

**Behind at minute 25:** run Exercise 2 as a demonstration; if still behind at minute 33, Exercise 3 too. Exercises 1, 4 and 5 may not be cut.

### Module 7 · {{< var modules.m07.title >}}

**Lecture** ({{< var schedule.clocks.long.shape.lecture >}} minutes)

{{< include /_includes/pace-07.md >}}

**Lab** ({{< var schedule.clocks.long.shape.lab >}} minutes) **and debrief** ({{< var schedule.clocks.long.shape.debrief >}})

| Minutes | Segment | By the end |
|---|---|---|
| 0–8 | Setup; Step 0 (the base model continues); the 5 minutes of slack | |
| 8–18 | Exercise 1 (the LoRA layer) | CP1 |
| 18–23 | Exercise 2 (merging the update) | CP2 |
| 23–29 | Exercise 3 (the chat template) | CP3 |
| 29–37 | Exercise 4 (the response mask and labels) | CP4 |
| 37–47 | Exercise 5 (configure LoRA, count parameters; training starts) | CP5 |
| 47–55 | Exercise 6 (ROUGE-n; decoding settings) | CP6 |
| 55–65 | **Debrief:** trainable parameters, response perplexity before and after, ROUGE-1, whether replies stop; the misconception "lower perplexity means better answers"; the bridge to Module 8 | the room's numbers on the board |

**Behind at minute 37:** run Exercise 6's `SAMPLING` cell as a demonstration and keep `rouge_n`. The stretch is the first thing dropped on any day.

### Closing synthesis

| Minutes | Segment | By the end |
|---|---|---|
| 0–5 | In pairs: the three prompts on the [Day 2 page](day-2.qmd#synthesis) | |
| 5–15 | With the room: fixed and left open for Modules 5 to 7; the running table, from the debriefs' numbers; one claim | the table on the board, kept for Day 5 |

## Day 3

### Retrieval practice

| Minutes | Segment | By the end |
|---|---|---|
| 0–6 | Alone, on paper, notes closed: the five questions on the [Day 3 page](day-3.qmd#retrieval) | an answer to each question |
| 6–10 | Compare with a neighbor | |
| 10–15 | The two most-missed questions with the room; then the [opening lines](facilitator-guide.md#day-3) | |

### Module 8 · {{< var modules.m08.title >}}

**Lecture** ({{< var schedule.clocks.long.shape.lecture >}} minutes)

{{< include /_includes/pace-08.md >}}

**Lab** ({{< var schedule.clocks.long.shape.lab >}} minutes) **and debrief** ({{< var schedule.clocks.long.shape.debrief >}})

| Minutes | Segment | By the end |
|---|---|---|
| 0–8 | Setup; read the provider cell; the 5 minutes of slack | |
| 8–16 | Exercise 1 (parse a raw response) | CP1 |
| 16–28 | Exercise 2 (validate and retry) | CP2 |
| 28–36 | Exercise 3 (score the outputs); the evaluation run opens this exercise | CP3 |
| 36–48 | Exercise 4 (the tool loop) | CP4 |
| 48–53 | Exercise 5 (cost and latency) | CP5 |
| 53–55 | What the response did not tell you | |
| 55–65 | **Debrief:** per provider and path, schema-validity rate, exact match with its standard error, cost, latency; the misconception "structured output means correct output"; the bridge to Module 9 | the room's numbers on the board |

**Behind at minute 28:** let the evaluation run finish while participants write Exercise 3, which is where the notebook places it; the checkpoints use the `FakeProvider` and do not wait for it.

### Module 9 · {{< var modules.m09.title >}}

**Lecture** ({{< var schedule.clocks.long.shape.lecture >}} minutes, before lunch)

{{< include /_includes/pace-09.md >}}

The lecture stops at lunch, wherever it has reached.

**Lab** ({{< var schedule.clocks.long.shape.lab >}} minutes, after lunch) **and debrief** ({{< var schedule.clocks.long.shape.debrief >}})

| Minutes | Segment | By the end |
|---|---|---|
| 0–8 | Reconnect; setup; Exercise 0 (the toy environment, three pairs); the 5 minutes of slack | |
| 8–20 | Exercise 1 (returns and the REINFORCE loss) | CP1a, CP1b, CP1c |
| 20–29 | Exercise 2 (a baseline and its variance) | CP2a–CP2e |
| 29–37 | Exercise 3 (the Bradley–Terry rater) | CP3a, CP3b |
| 37–49 | Exercise 4 (reward-model loss and training) | CP4a, CP4b |
| 49–55 | Exercise 5 (normalize and save) | CP5 |
| 55–65 | **Debrief:** gradient variance without and with the baseline; the reward model's held-out accuracy beside its ceiling; the misconception "a reward model should reach 100%"; the bridge to Module 10 | the room's numbers on the board |

**Behind at minute 29:** run Exercise 2's five-seed training as a demonstration. Part B (29–55) needs the Lab 9 data files; see the facilitator guide if they are missing.

### Module 10 · {{< var modules.m10.title >}}

**Lecture** ({{< var schedule.clocks.long.shape.lecture >}} minutes)

{{< include /_includes/pace-10.md >}}

**Lab** ({{< var schedule.clocks.long.shape.lab >}} minutes) **and debrief** ({{< var schedule.clocks.long.shape.debrief >}})

| Minutes | Segment | By the end |
|---|---|---|
| 0–8 | Setup; Step 0 (reward model and gold rule on the reference's samples); the 5 minutes of slack | |
| 8–16 | Exercise 1 (log-probabilities of a response) | CP1 |
| 16–24 | Exercise 2 (per-token reward with a KL penalty) | CP2 |
| 24–35 | Exercise 3 (exact KL; RLHF with the penalty) | CP3 |
| 35–43 | Exercise 4 (remove the penalty: reward hacking) | CP4 |
| 43–55 | Exercise 5 (the DPO loss; DPO training); results card | CP5 |
| 55–65 | **Debrief:** reward gain, KL drift, gold reward and distinct bigrams with and without the penalty, and for DPO; the misconception "a higher reward-model score means a better policy"; the bridge to Module 11 | the room's numbers on the board |

**Behind at minute 24:** do not cut a training run; ask the Predict questions while training runs (estimated 1.5 minutes per run on a T4, unmeasured).

### Closing synthesis

| Minutes | Segment | By the end |
|---|---|---|
| 0–5 | In pairs: the three prompts on the [Day 3 page](day-3.qmd#synthesis) | |
| 5–15 | With the room: fixed and left open for Modules 8 to 10; today's rows of the running table; what the numbers do not say | today's rows on the board, kept for Day 5 |

## Day 4

### Retrieval practice

| Minutes | Segment | By the end |
|---|---|---|
| 0–6 | Alone, on paper, notes closed: the five questions on the [Day 4 page](day-4.qmd#retrieval) | an answer to each question |
| 6–10 | Compare with a neighbor | |
| 10–15 | The two most-missed questions with the room; then the [opening lines](facilitator-guide.md#day-4), including the honesty rule | |

### Module 11 · {{< var modules.m11.title >}}

**Lecture** ({{< var schedule.clocks.long.shape.lecture >}} minutes)

{{< include /_includes/pace-11.md >}}

**Lab** ({{< var schedule.clocks.long.shape.lab >}} minutes) **and debrief** ({{< var schedule.clocks.long.shape.debrief >}})

| Minutes | Segment | By the end |
|---|---|---|
| 0–8 | Setup and Exercise 0 (which classifier, which provider); the 5 minutes of slack | |
| 8–18 | Exercise 1 (reliability bins and ECE) | CP1 |
| 18–25 | Exercise 2 (Brier score and log loss) | CP2 |
| 25–35 | Exercise 3 (temperature scaling) | CP3 |
| 35–47 | Exercise 4 (parse a stated confidence); start `ask_all` first | CP4 |
| 47–55 | Exercise 5 (risk–coverage and the threshold), ending with the closing question: what would you let act alone? | CP5 |
| 55–65 | **Debrief:** ECE, Brier score and log loss before and after temperature scaling, with accuracy unchanged; the stated confidence's ECE; the misconception "temperature scaling makes the model more accurate"; the closing question as the bridge to Module 12 | the room's numbers on the board |

**Timing:** the closing question is part of Exercise 5's 8 minutes. If the room is late, take it into the debrief as the bridge to Module 12, whose lecture follows the break. **Behind at minute 35:** start `ask_all` before discussing Exercise 3's results.

### Module 12 · {{< var modules.m12.title >}}

**Lecture** ({{< var schedule.clocks.long.shape.lecture >}} minutes, before lunch)

{{< include /_includes/pace-12.md >}}

The lecture stops at lunch, wherever it has reached.

**Lab** ({{< var schedule.clocks.long.shape.lab >}} minutes, after lunch) **and debrief** ({{< var schedule.clocks.long.shape.debrief >}})

| Minutes | Segment | By the end |
|---|---|---|
| 0–8 | Reconnect; the [lookalike-package warning](facilitator-guide.md#the-lookalike-package-warning) before the setup cell installs anything; setup and Exercise 0 (look at the decisions); the 5 minutes of slack | |
| 8–22 | Exercise 1 (accuracy reward and Brier reward; our illustration) | CP1 |
| 22–32 | Exercise 2 (typed questions, typed answers); `decide_all` starts first | CP2 |
| 32–40 | Exercise 3 (calibration arrays: $\hat{p}$ against `confidence`) | CP3 |
| 40–50 | Exercise 4 (act, ask, escalate) | CP4 |
| 50–55 | What this lab showed and what it did not | |
| 55–65 | **Debrief:** accuracy, ECE and Brier score under the two rewards (our illustration); the thresholds and the cost per case; the misconception "Jev's `confidence` is the probability that its answer is right"; the bridge to Module 13 | the room's numbers on the board, each labeled with its path |

**Behind at minute 32:** shorten the Exercise 3 discussion. Never cut the closing cell (50–55): it carries the honesty rule.

### Module 13 · {{< var modules.m13.title >}}

**Lecture** ({{< var schedule.clocks.long.shape.lecture >}} minutes)

{{< include /_includes/pace-13.md >}}

**Lab** ({{< var schedule.clocks.long.shape.lab >}} minutes) **and debrief** ({{< var schedule.clocks.long.shape.debrief >}})

| Minutes | Segment | By the end |
|---|---|---|
| 0–9 | Setup and Exercise 0 (the path banner, corpus, three questions); the 5 minutes of slack | |
| 9–17 | Exercise 1 (recall@k and reciprocal rank) | CP1 |
| 17–26 | Exercise 2 (build the index; choose $L$ and $k$); the sweep starts first | CP2 |
| 26–34 | Exercise 3 (the same retriever in LangChain) | CP3 |
| 34–43 | Exercise 4 (rerank the candidates) | CP4 |
| 43–53 | Exercise 5 (faithfulness); answers generated | CP5 |
| 53–55 | What this lab showed and what it did not | |
| 55–65 | **Debrief:** recall@$k$ and MRR at the chosen $L$ and $k$, with and without reranking; faithfulness and the two-by-two table (plumbing probes until the question set exists); the misconception "a faithful answer is a correct answer"; the bridge to Module 14 | the room's numbers on the board |

**Behind at minute 34:** on the keyed Jev path, rerank `test` only.

### Closing synthesis

| Minutes | Segment | By the end |
|---|---|---|
| 0–5 | In pairs: the three prompts on the [Day 4 page](day-4.qmd#synthesis) | |
| 5–13 | With the room: fixed and left open for Modules 11 to 13; today's rows of the running table; ranking or values | today's rows on the board, kept for Day 5 |
| 13–15 | Announce the capstone pairs; each pair decides tonight which keys it will use | every participant knows their pair |

## Day 5

### Retrieval practice

| Minutes | Segment | By the end |
|---|---|---|
| 0–6 | Alone, on paper, notes closed: the five questions on the [Day 5 page](day-5.qmd#retrieval) | an answer to each question |
| 6–10 | Compare with a neighbor | |
| 10–15 | The two most-missed questions with the room; then the [opening lines](facilitator-guide.md#day-5) | pairs and their path classes confirmed |

### Module 14 · {{< var modules.m14.title >}}

**Lecture** ({{< var schedule.clocks.long.shape.lecture >}} minutes)

{{< include /_includes/pace-14.md >}}

Lecture 14 was the densest lecture in the desk timing of 2026-10-06, and its plan is checked on paper only. **Behind at lecture minute 38 (section 7 not yet started):** give section 9 as reading, as the lecture's live-plan note says; that recovers its 5 minutes, and the core lab does not depend on it.

**Lab** ({{< var schedule.clocks.long.shape.lab >}} minutes) **and debrief** ({{< var schedule.clocks.long.shape.debrief >}})

| Minutes | Segment | By the end |
|---|---|---|
| 0–8 | Setup and Exercise 0 (read the graph); the lookalike-package warning again; the 5 minutes of slack | |
| 8–14 | Exercise 1 (a safe calculator) | CP1 |
| 14–26 | Exercise 2 (thresholds as edges) | CP2 |
| 26–34 | Exercise 3 (the human-in-the-loop driver) | CP3 |
| 34–41 | Exercise 4 (fork and replay); the evaluation run starts first | CP4 |
| 41–51 | Exercise 5 (measure the agent) | CP5 |
| 51–55 | What this lab showed and what it did not | |
| 55–65 | **Debrief:** route accuracy; unsafe-action rate and injection success, each with its denominator, by path; the misconception "an unsafe-action rate of zero means the guard is safe"; the bridge to the capstone | the room's numbers on the board |

**Behind at minute 34:** on a CPU runtime, cut the probability-shift test to 30 items (slice the list in the evaluation-run cell).

### Module 15 · {{< var modules.m15.title >}}

{{< var modules.m15.minutes >}} minutes in the two module slots after Module 14, then the day's closing slot, where the wrap-up ends: 255 minutes in all. There is no lecture table and no lab table: the plan below follows [lecture 15's timing](lectures/15-capstone.qmd#timing) and the starter notebook, `15-capstone.ipynb`. The evaluation set does not exist yet, so the baseline and the comparison in this plan cannot be run as designed: see the [readiness page](readiness.qmd#open-work).

**Part I and Part II** (the middle slot: the brief, then 110 minutes of build across lunch). Build minutes count from the end of the brief, as in lecture 15 and the notebook; lunch falls at build minute 45.

| Minutes | Segment | By the end |
|---|---|---|
| brief, 0–10 | Part I. The brief (lecture sections 1–4): a walk through the tables | pairs formed; path class chosen |
| build, 0–20 | Setup, self-test, write `after_verify`, baseline on `dev` then `test`, read ten traces | self-test green; both baselines run |
| build, 20–30 | Choose one component; fill in the hypothesis card | hypothesis card filled |
| build, 30–45 | Make the change; rerun the self-test after every edit; iterate on `dev` | self-test still green; the notebook saved to Drive |
| | Lunch | |
| build, 45–105 | Reconnect (if Colab reset the runtime, rerun the setup and the `dev` baseline first); continue the change | self-test still green |
| build, 105–110 | Freeze the configuration; do not run `test` | configuration frozen |

**At build minute 20:** check that every pair has a `dev` baseline and has read traces. **Behind at build minute 30:** any pair still without a `dev` baseline gets help before anyone else.

**Part III** (the afternoon slot, its first 105 minutes)

| Minutes | Segment | By the end |
|---|---|---|
| 0–20 | Run `test` on the frozen system; run the last cell; hand in the file | submission handed in |
| 20–80 | Two-minute shares, in menu order, with questions after each group | |
| 80–105 | The combined table, read with the room; the three cautions read aloud | |

**More than 15 pairs:** group the shares by component, four minutes per group.

**Part IV. Wrap-up** (30 minutes: the last 15 of the afternoon slot, then the day's closing slot)

| Minutes | Segment | By the end |
|---|---|---|
| 0–10 | Section 7, five days, one line of ideas, with the boards from Days 2 to 4 | the table complete, with the room's numbers |
| 10–18 | Section 8, an evaluation checklist for agentic systems | |
| 18–25 | Section 9, open problems | each pair has named one open problem and the measurement it would start with |
| 25–30 | Section 10, further study; close | |

### Closing synthesis

On Day 5 the closing synthesis is not a separate slot: it is the wrap-up's last 15 minutes, and the prompts on the [Day 5 page](day-5.qmd#synthesis) run through the whole wrap-up. Item 1 (the whole line) is section 7's table. Item 2 (today's table) takes Lab 14's numbers from its debrief and the capstone's from Part III's combined table. Item 3 (what is still open) is section 9.
