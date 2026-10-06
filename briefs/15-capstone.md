# Lab brief: `notebooks/15-capstone.ipynb`

**Status (2026-10-06).** Since this brief was written:

- the starter notebook `notebooks/15-capstone.ipynb` has been built and passes its offline code check in CI;
- the corpus snapshot exists, with status `provisional`;
- Labs 13 and 14 have been built;
- `scripts/capstone_score.py`, `scripts/collect_capstone.py` and `data/build_capstone_eval.py` are written and tested.

The evaluation set has not been written: neither Lab 13's 80 questions nor the 45 new ones. Every run therefore scores plumbing probes. The "Inputs that do not exist yet" table and the "Not verified" section below record the state when this brief was written. For what has run since, see the [readiness page](../readiness.qmd).

From the Academic Director to the **Agentic Systems Engineer**, owner of Lab 15 (`AGENTS.md`, "Who owns what"). Lecture: `lectures/15-capstone.qmd` (same symbols and equation names: `plan-rule`, `verify-rule`, `cap-acc`, `cap-abstain`, `cap-unsupported`, `cap-cost`, `sign-test`). Lab standards: `PLAN.md` section 5. Verified Jev surface: `briefs/jev-verification.md` (JV §n). Retriever, corpus and question set: `briefs/13-rag.md`. Graph, `decide()` and its backends: `briefs/14-agents.md`. Thresholds and the honesty rule: `briefs/12-rlcd-jev.md`, `briefs/11-calibration.md`. This file is not rendered by Quarto.

**Objectives exercised** (from `_variables.yml`, `m15`): combine retrieval, an agent graph and calibrated control into one system; evaluate it for accuracy, abstention and cost; explain the design choices to others.

**Every time, size and cost below is an estimate or a target, not a measurement.** Nothing in this brief was run. No notebook, corpus snapshot or question set for Labs 13–15 exists on this branch. The only computations behind this brief are arithmetic: the sign-test values, the Chow threshold and the worked costs, checked with Python on 2026-10-05.

## The lab in one paragraph

The capstone is not a 50-minute lab. On the five-day schedule it fills Day 5 after Module 14: 240 minutes in pairs across two module slots, with the wrap-up running into the closing slot (lecture 15 has the timetable). This brief was written for the Day 4 afternoon of the four-day plan (85 minutes of building and 20 of final evaluation); the minute figures below are from that plan. The starter notebook gives every pair the same working research assistant over *Workshop Lectures v1*. A plan node asks a decision model whether the question needs two searches. A retrieve node uses Lab 13's retriever. An answer node uses Lab 13's cited-answer prompt through Lab 8's `PROVIDER` switch. A verify node asks the decision model whether every claim is supported, then delivers the answer, retrieves again with twice as many chunks, or abstains. The notebook also gives a fixed, human-written evaluation set and a fixed scoring cell. Pairs run a system self-test, write one function (`after_verify`), and run the baseline on `dev` and on `test`. They then fill in a hypothesis card and change one component from a menu, iterating on `dev`. At the end they run `test` once more on the frozen system and submit one JSON file with a printed share card. The scoring reports accuracy against key facts, abstention on answerable and on unanswerable questions, the unsupported-answer rate by a fixed reference judge, cost per question and latency. It also gives a capstone cost (@eq-cap-cost) and a paired sign test against the baseline. Checkpoints test code on scripted inputs, so they give the same verdict on every path. No model metric is asserted.

## Decision: the fixed evaluation set

### What it is

**Workshop Capstone Questions v1 = Lab 13's 80 questions, reused verbatim by ID, plus 45 new questions written for Lab 15 by people.** Every item is written and labeled by people. No language model writes, proposes, paraphrases, filters or labels any item, which is the decision-set rule of brief 11 and the question-set rule of brief 13.

| Source | Kind | `dev` | `test` | Total |
|---|---|---|---|---|
| Lab 13, `data/rag_questions_v1.jsonl` (unchanged) | lookup 40%, specific 30%, multi 15%, unanswerable 15% | 30 | 50 | 80 |
| New, `data/capstone_questions_v1.jsonl` | `unanswerable` | 9 | 18 | 27 |
| New | `reading` (answerable) | 6 | 12 | 18 |
| **Total** | | **45** | **80** | **125** |

The resulting mix per split is about 30% unanswerable: `dev` about 13–14 of 45, `test` about 25–26 of 80. The exact count depends on Lab 13's rounding of the 15% share within ±1 item.

### Why reuse and extend, rather than reuse alone or start again

1. **Reuse alone cannot measure abstention.** Lab 13's 15% unanswerable share gives 7 or 8 unanswerable `test` items. At an abstention rate near 0.7, Module 8's standard error at $N = 8$ is about 0.16, so abstention would be unmeasurable. The new items bring $|\mathcal{U}_{\text{test}}|$ to about 25, with an SE of about 0.09. That is still wide, and the lecture says so, but it can be read.
2. **The capstone's contract adds a failure Lab 13 barely tests: answering from memory.** Half of the 27 new unanswerable items (at least 13) must be **memory bait**: a fact that is stated in a reading-list paper and that a model may well know, but that is not stated anywhere in the snapshot. For example, a training detail of a reading-list paper that no lecture page quotes. These separate a grounded system from a knowledgeable one, which is the point of the verify node.
3. **The capstone's task is the reading list.** The `reading` kind joins a paper on the reading list to the module and lab that use it, which is the research-assistant use case. Lab 13's `multi` items cover some of this, but only about 7 of them are in `test`.
4. **Reusing Lab 13's items keeps comparability and costs no new labels.** Lab 13's `test` numbers and Lab 15's are on the same questions, so a participant can see what the graph added over the plain pipeline.
5. **Contamination is real but bounded.** Participants saw Lab 13's `test` metrics that morning, though not per-question answers unless they printed them. The scoring reports every metric on the `lab13` and `new` subsets separately as well as pooled, so a gain that appears only on the questions participants have seen is visible.
6. **Starting again would cost more and lose the comparison.** A fresh set of 125 items would take two people about 10 hours. Extending takes about 5 (below).

**Why not more.** A larger set costs annotator time, and it multiplies the keyed path's cost and the CPU path's run time, which is already the binding constraint (compute budget below). At this size, differences are read pairwise, which the lecture teaches.

### Item schema (new items)

As Lab 13's schema (`briefs/13-rag.md`, decision (b)), with two additions:

```json
{"id": "c001", "split": "test", "kind": "unanswerable",
 "question": "...",
 "evidence": [],
 "answer": "",
 "key_facts": [],
 "would_be": {"slug": "05-transformer-from-scratch", "where": "section 6, where training details would appear"},
 "absent_terms": ["4.5 million sentence pairs", "4.5M sentence pairs"],
 "outside_source": "Vaswani et al. 2017, section 5.1",
 "memory_bait": true,
 "author": "RA", "checker": "IN", "notes": ""}
```

- **Unanswerable items:** `would_be` records where the author looked and where the fact would have been. `absent_terms` lists strings that a memory-based answer would contain; the validator asserts that none occurs in the snapshot's text, case-insensitively, which is a mechanical check of absence. `outside_source` (memory-bait items only) names where the fact *is* stated, read by the author in the paper itself, never taken from a model's output.
- **`reading` items:** answerable, with `evidence` groups as in Lab 13, and **`key_facts` mandatory** (Lab 13 requires them on 70% of answerable items; Lab 15's new items need them on 100%, so every new answerable item is scored by program).
- `memory_bait` is `false` or absent for answerable items.

### Labeling protocol

As Lab 13 (write, blind check, resolve, report), with these additions:

- **Unanswerable, blind check:** the checker sees only the question and searches the snapshot with anything except a language model: reading, `grep`, or Lab 13's BM25. An item survives only if the checker also finds nothing and the validator's `absent_terms` check passes.
- **No model in the loop at all**, including filtering. Nobody asks a model whether it knows a memory-bait fact, or keeps the items a model gets wrong. Filtering by a model would make the set adversarial to that one model and would bias every comparison involving it.
- **Order of work:** the snapshot must be built and frozen first (constraint (a) below), then Lab 13's questions, then these.

**Who:** Romeo and one instructor, the pair proposed for the decision set and Lab 13's questions. **Effort (estimate):** about 4 minutes per item to write, more for unanswerable items because absence must be searched for, and about 3 to check: roughly 3 hours and 2 hours for 45 items.

**Validator:** `tests/test_capstone_questions.py`, which an agent may write. It checks the schema; split sizes and kind counts; that IDs do not collide with Lab 13's; that every quote is found exactly once in its page; `key_facts` on every new answerable item; `absent_terms` absent from the snapshot; the lexical-overlap flag of Lab 13; and the manifest (next section). It runs on an agent-written fixture of at most 10 items under `tests/fixtures/`, headed "agent-written fixture for exercising code; not an evaluation set; no number from it is quoted anywhere", until the real file lands. The notebook never loads the fixture as evaluation data.

**License:** CC BY 4.0, as Lab 13's set (Romeo's call; CC0 would also be clean).

### The manifest

`data/capstone_eval_v1.json`, written deterministically by a builder (`data/build_capstone_eval.py`, standard library only, seed 0):

- `lab13_sha256` and `capstone_sha256`: the hashes of the two question files; `snapshot_sha256`: the corpus hash.
- `splits`: the ordered IDs of `dev` (45) and `test` (80).
- `cpu_subset`: `dev` 12 (8 auto-checkable answerable, 4 unanswerable) and `test` 24 (16 and 8). They are drawn stratified by kind and source with `random.Random(0)`, not chosen by hand, and both sources are represented.
- `scored`: the IDs in $\mathcal{A} \cup \mathcal{U}$, which leaves out answerable items without key facts. These are reported, not scored, and their count is printed.
- `costs`: `{"wrong": 5, "abstain": 1}`; `scoring_version`: `"v1"`.

The notebook checks every hash at load and refuses to score against a file that does not match.

## The honesty rule, made concrete for this notebook

1. **Jev is described only through its verified interface** (JV §2–3): typed questions about a state go in, probabilities come out. No cell describes how Jev was trained, and RLCD is not named except in a link to lecture 12.
2. **Every record and every table row carries four backend labels:** `generator`, `decider`, `retriever`, `judge`. `Jev (<resp.model>)` appears only on rows whose decisions came from `TypeSafeClassifier`. The labels for stand-ins are `Qwen log-prob decider (not Jev)`, `stub (test double)` and `stand-in (not a neural model)`.
3. **No-key banner** above the first results: "No TypeSafe key: the planner's and verifier's probabilities come from a small open language model scored by the probability of ' yes'. They measure that model, not Jev. Do not quote them as Jev's."
4. **Offline banner** (Lab 8's): "These numbers measure the notebook's code, not any model. Do not quote them." The submission's `path_class` is `stub`, and the collector does not rank it.
5. **The share card** prints the four backend labels above the numbers.
6. **The closing cell** ("What this capstone showed and what it did not", below) is part of the core path.

## Constraints that shape the lab

### (a) Inputs that do not exist yet

| Input | Status on this branch | Needed for |
|---|---|---|
| `data/workshop_lectures_v1.jsonl.gz` and its builder (brief 13, decision (a)) | not built | everything |
| `data/rag_questions_v1.jsonl` (80 items, people) | not written | the evaluation set |
| `data/capstone_questions_v1.jsonl` (45 items, people) | not written | the evaluation set |
| `data/baselines.json` `lab13.chosen` ($L$, $k$, $k_0$) | not recorded | retriever constants |
| Labs 13 and 14 notebooks | not built | the cells this lab restates |

**Ordering constraint:** the snapshot includes `references.qmd`, and `PLAN.md` Day 10 still lists "Complete `references.qmd`". The `reading` questions need the reading list in its final form at snapshot time. Either complete the Modules 1–12 entries of `references.qmd` before building the snapshot, or write the `reading` items only against the lectures' own "Readings" sections. I recommend the first. The snapshot is frozen once questions cite it.

Until the inputs land, build against the fixture and Lab 13's offline stand-ins, and quote no number.

### (b) What Lab 15 reuses, restated verbatim (Colab notebooks share no runtime)

| From | Cell, restated verbatim unless noted | Used for |
|---|---|---|
| Lab 8 | the provider cell: `ToolCall`, `Reply`, `tool_result`, `Tool`, the three adapters, `FakeProvider`, `StubProvider`, `make_provider`, `label`, `extract`, the solution `execute`; with Lab 14's JSON-safe message convention (`raw` dumped or omitted) | the generator; `extract` for the plan split; `FakeProvider` for the self-test; `execute` for option T1 |
| Lab 12 | `expected_costs`, `action_thresholds` (solutions) | printing $\tau_{\text{verify}}$ as `action_thresholds(dict(wrong=5, ask=..., miss=..., esc=1))` collapsing to Chow's 0.8; option V2 |
| Lab 11 | `reliability_bins`, `ece`, `plot_reliability`, `choose_threshold` (solutions) | the verifier's reliability diagram on `dev`; option V1 |
| Lab 13 | the cell marked "provided; reused by Labs 14 and 15": `Passage`, `load_corpus`, `build_retriever`, `Retriever` (with `as_langchain`), `format_sources`, `ABSTAIN`; plus `generate`, `split_claims`, the NLI and stub judges, `is_correct`, `is_abstention`, `covers`, the solution `recall_at_k`, `paired_counts`, `JevRerank`, the LSA stand-in and the offline banner. Settings from `lab13.chosen`, restated as constants | the `retrieve` and `answer` nodes; the reference judge; options R1–R3 |
| Lab 14 | `decide(state, questions)` **with one deletion** (below); `QwenDecider`; `StubDecider` **with two new rules** (below); `as_lab8_tool`; `snapshot_before`; the JSON-safe state convention; `RUN_CFG = {"recursion_limit": 40}`; the `.with_retry(...)` transient-error set and fail-closed behavior | the `plan` and `verify` nodes; option T1; debugging |

**Not reused, on purpose:** Lab 12's `featurize`, `ToyDecider`, `train_toy`, `LocalDecider` and `TOY_REF`. Neither of Lab 15's two questions is a decision-set question type, so `LocalDecider` would raise `UnsupportedQuestion` for both (brief 14, the proposed amendment to Lab 12). The deletion in `decide()` is its `LocalDecider` branch, with the comment: "Lab 15 asks no decision-set question; the toy model would raise UnsupportedQuestion for both questions." This also removes the decision-set download and the toy retraining from the setup. Lab 14's `SimulatedHuman`, `run_with_human` and `agent_metrics` are not reused: the capstone has no human step, and its metrics are its own. Its record schema follows Lab 14's conventions: `id`, `backend` labels, `p`, `error`.

**The verify node is Lab 15's own** (brief 14, stretch: "Lab 15 ... must restate the node itself; it must not depend on this section"). It is written here and does not import or copy Lab 14's stretch.

### (c) The two decision questions

| Question | Name | Type | `instructions` (complete, since names are not sent, JV §2) | State |
|---|---|---|---|---|
| plan | `needs_two` | `Noul` | "Does answering this question need evidence from two different modules or documents of the workshop's lecture pages, rather than one passage?" | `{"question": q}` |
| verify | `supported` | `Noul` | "Is every factual claim in the answer supported by the passages below? Background knowledge does not count as support." | `{"question": q, "answer": draft_without_citations, "passages": [{"id": chunk_id, "text": ...}, ...]}` |

- **Keyed:** `langchain_typesafe.Noul` through `TypeSafeClassifier`, as Lab 14. On a final error, the plan falls back to one query and records the error; the verify **fails closed**: it abstains and records the error.
- **No key:** `QwenDecider`, `noul = P(" yes") / (P(" yes") + P(" no"))` from next-token logits, never a written probability (Lab 11). The prompt is the instructions, the state as text, then "Answer yes or no:".
- **Stub:** `StubDecider` gains two documented rules that never read any label, gold field or kind. `needs_two` gives 0.8 if the question contains " and " together with two of "which", "what", "where", "module", "lab", and 0.2 otherwise. `supported` gives 0.95 if at least 80% of the answer's content words of five or more letters appear in the passages, 0.5 if at least 50% do, and 0.1 otherwise. The bands were meant to make every edge of the graph run offline; as built, the stub generator copies source sentences, so `supported` always gives 0.95 on the stub path and the retry and abstain-after-verify edges run only in the self-test.
- The verify state carries the passages the draft was written from, so the verifier reads what the generator read.

### (d) The graph

```python
class CapState(TypedDict):
    q: str
    taus: dict                                   # {"plan": TAU_PLAN, "verify": TAU_VERIFY}; edges read these
    p_multi: float | None
    queries: list[str]
    k_cur: int                                   # chunks per query this attempt; doubles on retry
    passages: list[dict]                         # JSON-safe Passage fields, deduplicated by chunk_id
    draft: str | None
    p_sup: Annotated[list, operator.add]         # one value per verify attempt
    r: int                                       # retries used
    n_llm: int
    n_dec: int
    output: str | None                           # "answer" | "abstain" | "error"
    final: str | None                            # the delivered text (the ABSTAIN sentence on abstain)
    usage: Annotated[list, operator.add]         # {"who", "model", "in", "out", "seconds"} per call
    trace: Annotated[list, operator.add]

LOSS = dict(wrong=5, abstain=1)                  # scoring and the verifier's Chow threshold
TAU_PLAN = 0.5                                   # equal costs (lecture eq-plan-rule); printed
TAU_VERIFY = 1 - LOSS["abstain"] / LOSS["wrong"] # 0.8; printed, and checked against action_thresholds
R_MAX, K_MAX_LLM, K_MAX_DEC = 1, 4, 4

def plan(s), retrieve(s), answer(s), verify(s), deliver(s), abstain(s)
def after_plan(s)   -> str   # provided: "retrieve"
def after_answer(s) -> str   # provided: "abstain" if is_abstention(draft) else "verify"
def after_verify(s) -> str   # TODO 1: "deliver" | "retrieve" | "abstain" (eq-verify-rule, with budgets)
```

`plan` makes one decision call and, if splitting, one `extract` call with `class Split(BaseModel): queries: conlist(str, min_length=2, max_length=2)`. If the split is invalid, it falls back to `[q]`. `retrieve` calls `retriever.retrieve(query, k=k_cur)` per query, merges, deduplicates by `chunk_id`, and keeps rank order by first appearance. `answer` calls Lab 13's `generate`. On retry, `retrieve` doubles `k_cur` before retrieving. Each node catches its own exceptions, sets `output = "error"`, and routes to `END`. Pass `RUN_CFG` on every `invoke`, with one thread per question, `f"{qid}-{run_tag}"`.

**`run_eval(graph, split, *, subset=None, tag)`** (provided) runs every question in the split. It runs questions concurrently on keyed paths (8 at a time) and sequentially on local paths. It writes one record per question:

`id, split, source ("lab13" | "new"), kind, scored, answerable, output, final, draft_withheld, queries, p_multi, p_sup (list), r, retrieved_chunk_ids, cited, n_llm, n_dec, tokens_in, tokens_out, jev_tokens_in, usd, latency_s, generator, decider, retriever, judge, error`.

Correctness and support are **not** computed by the system. The scoring cell computes them from the records.

### (e) The scoring cell (fixed, hashed)

One cell, titled "Scoring: do not edit". The notebook prints the SHA-256 of the source of its functions (`inspect.getsource`) and writes it into the submission. The same code lives in `scripts/capstone_score.py` (proposed), and a test fails if the notebook cell and the file differ, in the pattern of `tests/test_models.py`.

```python
def score(records, items, *, loss=LOSS, judge=REFERENCE_JUDGE) -> dict
    # per record: o_i (is_correct against key_facts, Unicode-normalized, case-insensitive);
    # unsupported_i (min over sentences of judge(sentence, passages) < 0.5; split_claims from Lab 13)
    # returns, pooled and by source (lab13 / new) and by kind, each with N and Module 8's SE:
    #   acc (eq-cap-acc), acc_answered, abs_U and abs_A (eq-cap-abstain), uns (eq-cap-unsupported),
    #   cost_bar (eq-cap-cost), usd_per_q (None on local paths), calls_per_q,
    #   latency_median, latency_p95, n_scored, n_reported_only, n_errors
def compare(base, new, items, *, loss=LOSS) -> dict
    # per-question cost under eq-cap-cost; gained, lost, sign-test p (eq-sign-test); deltas of every metric
def flips(run_a, run_b, items) -> int     # questions whose per-question cost differs between two baseline runs
def submission(pair, hypothesis, base_dev, base_test, final_dev, final_test, ...) -> dict  # also writes the JSON
def share_card(sub) -> None               # prints the card
```

- **Reference judge:** the NLI cross-encoder (`models.nli`, Lab 13) on every path that can load it, and the stub judge offline. **No LLM judge is used for a scored number.** That keeps the instrument identical, free and deterministic for every pair. Print the banner "Reference judge not validated against human labels" until Lab 13's judge audit set exists. Then also print the judge's agreement with people.
- **Errors** count as wrong answers in `cost_bar` (Module 8's rule) and are counted separately.
- The sign test uses `math.comb`; no SciPy.

### (f) Paths, and how results are labeled

| Path class | When | Generator | Decider | Retriever | Judge | Eval size |
|---|---|---|---|---|---|---|
| `keyed-jev` | an LLM key and `TYPESAFE_API_KEY` | `PROVIDER` | Jev | dense (bge-small) | NLI | full |
| `keyed-llm` | an LLM key, no TypeSafe key | `PROVIDER` | Qwen log-prob (not Jev) | dense | NLI | full |
| `open-t4` | no keys; GPU; Hub reachable | Qwen | Qwen log-prob (not Jev) | dense | NLI | full, if the measured `test` run is under 10 min; otherwise `cpu_subset` |
| `open-cpu` | no keys; CPU; Hub reachable | Qwen | Qwen log-prob (not Jev) | dense | NLI | `cpu_subset` |
| `stub` | no Hub, or `NLP_LLMS_STUB=1` (the CI path) | `StubProvider` | `StubDecider` | LSA stand-in | stub judge | full (seconds) |

The path class is chosen once in setup, printed, and written into every record and into the submission. **Pairs are compared only within a path class.**

## The starter: what is provided, what pairs write, what pairs change

**Provided:** everything in (b) to (f); the self-test; `run_eval`; the scoring cell; the hypothesis card; a trace viewer (`show_trace(record)`), which prints each step with its probabilities and thresholds; a failure table, which is Module 13's two-by-two of evidence retrieved against correct, computed on `dev` from `retrieved_chunk_ids` and the gold evidence; and a folded reference sketch for every menu option.

**Pairs write (the exercise, about 5 minutes):**

| # | Participant writes | Equation | Checkpoint (deterministic, in the self-test) |
|---|---|---|---|
| 1 | `after_verify(state)`: `"deliver"` if the last $p_{\text{sup}} \ge \tau_{\text{verify}}$; else `"retrieve"` if `r < R_MAX` and both budgets allow another attempt; else `"abstain"` | `verify-rule` | with `FakeProvider` and a `ScriptedDecider`: (i) 0.85 → delivered, one verify call; (ii) 0.6 then 0.9 → retried once with `k_cur` doubled, then delivered; (iii) 0.6 then 0.6 → abstained, the draft kept in `draft_withheld`; (iv) exactly 0.8 → delivered (the tie acts, as in Modules 11 and 12); (v) a generator that returns `ABSTAIN` → abstained, and the decider is never called for `supported`; (vi) with `K_MAX_LLM = 1`, a 0.6 draft abstains without retrying |
| 2 | The hypothesis card: `HYPOTHESIS = dict(component=..., change=..., metric=..., direction=..., size=...)` | – | non-empty fields; `component` one of the menu codes or `"other"`; `run_eval` refuses `tag != "baseline"` until it is filled |

**The rest of the self-test** (provided; runs after every edit; must stay green): (vii) $p_{\text{multi}} = 0.7$ → two queries, two retrievals, merged without duplicate `chunk_id`s; 0.3 → one query, `q` itself; (viii) an invalid `Split` → falls back to `[q]`, recorded; (ix) a node that raises → `output == "error"`, and `score` counts it wrong; (x) every record has the four backend labels and no `None` among them; (xi) `score` on 10 hand-made records reproduces hand-computed `acc`, `abs_U`, `abs_A`, `uns` and `cost_bar`; (xii) `compare` on hand-made pairs gives the lecture's values: 8 gained and 1 lost → $p = 0.0391$, 7 and 2 → $0.1797$, 4 and 3 → 1; (xiii) the scoring hash equals the expected constant; (xiv) the manifest hashes match the loaded files.

**Pairs change one component.** The menu is in the lecture, section 5. The table below repeats it with what each option touches. **The effect sizes are hypotheses, not facts**, and the notebook prints them as "our guess before any run".

| Code | Touches | Reference sketch (folded) | Hypothesis printed in the notebook |
|---|---|---|---|
| R1 | `build_retriever(method="hybrid")` | one line plus the rebuild | "0–3 more `test` questions right, mostly `specific`" |
| R2 | `rerank="cross"` or `"jev"`, `candidates=20` | as Lab 13, Exercise 4 (`aretrieve` for Jev) | "first useful chunk higher; 0–4 more right; latency up" |
| R3 | `chunk_size`, `k` | constants from `lab13.chosen` and its runner-up | "recall up with $k$, accuracy may fall, cost up in proportion" |
| P1 | the `generate` prompt | stricter abstention, or quote-then-answer | "$\mathrm{Abs}_{\mathcal{U}}$ up 10–30 points; $\mathrm{Abs}_{\mathcal{A}}$ up too" |
| P2 | the `generate` prompt | one cited and one abstaining example | "as P1, fewer uncited sentences" |
| V1 | `TAU_VERIFY` | `choose_tau_verify(dev_records)`: replays the logged $p_{\text{sup}}$ against @eq-cap-cost over the observed values plus 0 and 1. It is exact only with `R_MAX = 0`, so the sketch sets that and says so | "small gain if the verifier is informative; an extreme $\tau$ if it is not" |
| V2 | `after_verify` | three regions from `action_thresholds` with retry as "ask" (suggested costs: wrong 5, ask 0.3, miss 1, esc 1) | "fewer calls on hopeless drafts, little accuracy lost" |
| V3 | `R_MAX`, the retry strategy | $R_{\max} \in \{0, 2\}$; or a rewritten query via `extract` | "a few answerable questions recovered at 30–50% more cost" |
| G1 | `TAU_PLAN`, the split prompt | | "affects about 20 `test` questions; probably below noise" |
| T1 | replace `retrieve` with Lab 14's tool-calling `agent` + `tools` nodes and a `search_docs` tool (`as_lab8_tool`, `execute`) | about 30 lines | "two-part questions up; 2–3× calls" |
| M1 | `PROVIDER` | needs both LLM keys | "largest change, in accuracy and in cost" |

**Not allowed in the scored run:** adding documents to the corpus; editing the scoring cell, the manifest, the reference judge or `LOSS`; models other than the pinned IDs; sampling on the open model (`do_sample=False`).

## The baseline

- **Instructor-recorded baselines** (before delivery, by the Lab Engineer or an instructor): the starter, unmodified, run on `dev` and `test` on every path class available (at least `keyed-jev` with Anthropic, `open-t4`, `open-cpu` and `stub`). The run is repeated once on each keyed path to measure run-to-run flips. Commit the per-question records as `data/lab15_baseline_<path>_v1.jsonl.gz` (hash in the manifest) and the summary metrics as `data/baselines.json` `lab15.baseline.<path>`. **Stub values are committed only as the CI regression value and are never quoted.**
- **Each pair's own baseline:** the first `test` run of the afternoon, `tag="baseline"`. `compare` uses the pair's own baseline (same day, same API versions). `flips(own_baseline, recorded_baseline)` gives the run-to-run floor printed on the share card. On deterministic paths (`stub`, `open-*` with greedy decoding on the same hardware type) the flip count should be 0. If it is not, the card says "baseline did not reproduce", which is a finding about the environment.
- **Reporting against it:** `compare(own_baseline_test, final_test)` gives $\Delta$ of every metric, $g$, $l$, $p$ and the flip count; and `dev` likewise, labeled "development, used for choosing".

## What is asserted on each path

| Path | Asserted | Printed, never asserted |
|---|---|---|
| every path | the self-test (i)–(xiv); harness: every scored ID yields a record; every record's `output` is one of the three values; `final == ABSTAIN` exactly when `output == "abstain"`; counters within budget; Jev or Qwen `noul` in [0, 1] | all five metrics, $\bar{\ell}$, $g$, $l$, $p$, flips, cost, latency |
| `keyed-*` | the same; `resp.model` recorded on every Jev row | measured USD from `usage`, Jev input tokens, latency. **No Jev or LLM metric is asserted, ever** |
| `open-*` | the same | every metric, labeled `(not Jev)` for the decider |
| `stub` (CI) | the same; after the recorded run, the stub baseline's `cost_bar`, `acc`, `abs_U`, `abs_A`, `uns` on `dev` and `test` against `lab15.baseline.stub` to 1e-9 (deterministic) | the same under the offline banner |

## Closing cell: "What this capstone showed and what it did not"

- Your change was measured against your own baseline on 80 `test` questions about one workshop's pages, by one fixed judge that has (or has not) been compared with people. Say which.
- With $g + l$ changed questions and the printed $p$, say whether the change was shown, and compare $g + l$ with the flip count.
- Without a TypeSafe key, the planner's and verifier's probabilities were a 0.5B open model's, not Jev's.
- The scored cost leaves out dollars and seconds; read them beside it.
- Questions: (1) Your verifier abstained on a draft the key-fact check would have marked correct. Which of the two approximations in the verifier's rule (lecture section 2) does that illustrate? (2) Name one item on the checklist of lecture section 8 that this afternoon did not test.

## What a pair submits, and how instructors compare pairs fairly

**The submission** is `capstone_<pair>_<pathclass>.json`, written by `submission(...)`. It contains: pair label (a number or first names only); path class and the four backend labels with `resp.model`; `HYPOTHESIS`; the configuration diff against the starter (constants changed, cells edited by name); the scoring hash, the manifest hash and the notebook's commit; baseline and final `score` on `dev` and `test`; `compare` and `flips`; the number of `test` runs; budget flags (calls per question, `test` run seconds, USD per question against twice the baseline); and the three-sentence share text. The share card prints the same.

**Collection:** a shared folder or an upload form, the instructor's choice. `scripts/collect_capstone.py` (proposed, for the Architect) merges the files and checks each one:

1. the scoring hash and the manifest hash equal the expected values, otherwise "not comparable";
2. `test` runs = 2, otherwise flagged "test used for development";
3. budgets respected, otherwise flagged "over budget" (shown, not ranked);
4. `stub` submissions are listed as "code check only".

It prints one table per path class, grouped by menu code, with $\Delta\bar{\ell}$, $\Delta$Acc, $\Delta\mathrm{Abs}_{\mathcal{U}}$, $\Delta\mathrm{Abs}_{\mathcal{A}}$, $\Delta$Uns, $g$/$l$/$p$, flips, $\Delta$USD and $\Delta$latency. Below the table it prints the multiple-comparisons note: "with $m$ pairs at 0.05, about $0.05m$ significant results are expected by chance".

**Fairness, by construction:** the same evaluation set (manifest hash); the same scorer and judge (scoring hash); the same costs; the same budget; the same seed and greedy decoding where decoding is controllable; comparison only within a path class; comparison against the pair's own same-day baseline; `test` run twice. Keyed generation is not deterministic, so the flip count is the noise floor on keyed paths.

## Compute and cost budget

| Part | Stub (CPU) | Open, T4 | Open, CPU (subset) | Keyed |
|---|---|---|---|---|
| Setup: installs, corpus, index (705 chunks at $L = 256$ on the v1 snapshot), Qwen and NLI downloads | under 1 min | 2–4 min (about 1 GB of Qwen weights, plus about 200 MB for bge-small, the cross-encoder and the NLI model) | 3–6 min | under 2 min (plus Qwen if no TypeSafe key) |
| Self-test | seconds | seconds | seconds | seconds |
| One `dev` run (45; CPU 12) | seconds | 3–6 min | 3–6 min | 1–3 min |
| One `test` run (80; CPU 24) | seconds | 6–10 min (measure; fall back to the subset if over 10) | 6–10 min | 2–5 min |
| A pair's afternoon: 2 `test` runs plus 3–6 `dev` runs | under 5 min | 25–50 min | 25–45 min; at most 3 `dev` runs | 10–25 min |

All are estimates (the Qwen generation speed is assumed at 20–40 tokens per second on a T4, with answers capped at 160 tokens). **The CPU path is the binding constraint.** If the engineer measures a CPU `test` subset run over 10 minutes, cut `cpu_subset` to 16 `test` items (still stratified, seed 0) before anything else, and tell me.

**Cost note** (state in the notebook; estimates from token arithmetic, **not measured invoices**; replace them with figures measured from `usage`):

- **Generator, per question:** about 1.4 calls (answer, sometimes a split or a retry) × (about 1,600 input tokens, 150 output). Anthropic `models.anthropic` at \$1 / \$5 per million (Anthropic's page, read 2026-10-05 for Lab 8): about \$0.0033 per question. One pair's afternoon of about 385 question-runs (2 × 80 `test` plus 5 × 45 `dev`) costs about \$1.30: state **"under \$2 per pair on Claude"**. OpenAI `models.openai` at \$0.10 / \$0.50 (secondary sources, as Lab 8): about \$0.13 per pair, so state **"under 25 cents per pair"**.
- **Jev, per question:** about 2.4 calls × about 1,500 input tokens (the verify state carries the passages) ≈ 3,600 tokens; at \$0.042 per million input tokens, output free (`WorkflowEvals` price table, JV §5; TypeSafe's pricing page unread), about \$0.06 per pair: **"under 10 cents per pair"**. Option R2 with Jev adds 20 calls per question (about 3M tokens, about \$0.13 per pair, and 7,700 calls).
- **A room of 15 pairs:** about \$20 on Claude, \$2 on OpenAI, \$1 on Jev, and about 14,000 Jev calls in the afternoon without R2. **Rate limits are unknown** (JV §9). Ask TypeSafe about workshop keys before delivery (PLAN section 6, "Jev access"), and limit Jev reranking (R2) to `dev` if keys are shared.
- **Open and stub paths:** free.

## Flags for the Lab Engineer

1. **Names** as the lecture: `p_multi`, `p_sup`, `tau_plan`, `tau_verify`, `R_MAX`, `LOSS`, `cost_bar`, `abs_U`, `abs_A`, `uns`, `gained`, `lost`. Thresholds are compared with `noul`, never with `confidence` (neither question is a `Choice`, so there is none).
2. **The verifier and the reference judge must be different code paths.** The scoring cell must not call `decide()`. A pair that edits `verify` must not change any scored number except through the system's outputs.
3. **Abstention is detected only by exact match** with `ABSTAIN` after whitespace normalization (Lab 13's `is_abstention`). A paraphrased refusal counts as an answer, and almost always as a wrong one. Say so in the prompt and in the notebook.
4. **The tie rule:** deliver at exactly $\tau_{\text{verify}}$ (checkpoint (iv)).
5. **Keep decision calls out of any node that could pause.** There is no `interrupt` in the core path; if a pair adds one (an option of their own), Lab 14's rule applies.
6. **State is plain data** (Lab 14, constraint (d)): no SDK objects, no dataclasses; `Passage` is stored as a dict.
7. **`recursion_limit` on every run;** the budgets are counted in the state and enforced by the edges, not by the recursion limit.
8. **Never construct `TypeSafeClassifier` or `AsyncTypeSafeClient` without a key** (JV §2–3). Never set the `typesafe_sdk` logger to DEBUG. Send no personal data in a state; the corpus has none.
9. **Package names:** only the pins of Labs 8, 13 and 14 (`typesafe-sdk==0.7.2` for option R2's `JevRerank`, `langchain-typesafe==0.0.1a3`). `tests/test_package_names.py` must cover this notebook.
10. **The CI path never touches the network** beyond the repository's data URLs.
11. **Model IDs:** repeat `models.openai`, `models.anthropic`, `models.fallback`, `models.embedding`, `models.reranker`, `models.nli` and `models.jev` verbatim (a test checks; proposed below).
12. **Report back:** which path classes ran; the measured baseline on each (all five metrics, $\bar{\ell}$, flips between two keyed runs); run time per section on CPU and T4; measured cost per question; whether the full `test` set fits 10 minutes on a T4; whether Lab 13's retriever and Lab 14's `decide()` fitted unchanged; the outcome of each menu option's reference sketch on `dev`, labeled with its path (useful to instructors, not quoted as results); anything in the lecture that the notebook contradicts.

## Proposed changes (not made; for Romeo or the Architect)

- **`_variables.yml` `datasets`:** `capstone_questions` ("Workshop Capstone Questions v1", modules `[15]`, `file: capstone_questions_v1.jsonl`, `license: "CC BY 4.0"`, `splits: {dev: 15, test: 30}`), and `capstone_eval` (the manifest, `file: capstone_eval_v1.json`). Add `15` to the `modules` list of the proposed `rag_questions` and `lectures` entries (brief 13).
- **`_variables.yml` `modules.m15.stack`:** `[LlamaIndex, LangGraph, Jev, OpenAI, Anthropic]` → `[LlamaIndex, LangChain, LangGraph, Jev, OpenAI, Anthropic, Hugging Face]` (`langchain-core` for `TypeSafeClassifier` and runnables; sentence-transformers and Qwen on the open path).
- **`data/`:** `build_capstone_eval.py` and the manifest (an agent may write both once the question files exist); `capstone_questions_v1.jsonl` (people); `lab15_baseline_<path>_v1.jsonl.gz` and `baselines.json` `lab15.baseline.*` after the recorded runs; a `data/README.md` section.
- **`scripts/`:** `capstone_score.py` (the scoring functions, restated verbatim in the notebook) and `collect_capstone.py` (the instructor's merge and checks).
- **`tests/`:** `test_capstone_questions.py` (validator, fixture until the real file lands); a check that the notebook's scoring cell equals `scripts/capstone_score.py`; `tests/test_models.py` `USES["15-capstone"] = ["openai", "anthropic", "fallback", "embedding", "reranker", "nli", "jev"]`.
- **`PLAN.md` section 4, Module 15:** Build → "… Jev for routing and for a final 'is this answer supported by the sources?' check with a threshold set from stated costs (without a key: a small open model scored on yes/no, labeled 'not Jev'). Participants work in pairs: run the system self-test and the fixed evaluation set for a baseline, write a hypothesis, then improve one component of their choice (retrieval, prompts, routing, thresholds, a new tool), iterating on `dev`." Evaluate and share → "re-run the evaluation on `test` (once); report accuracy, abstention on answerable and unanswerable questions, unsupported-answer rate, cost and latency against the baseline, with the gained/lost counts and a sign test; each pair gives a two-minute account from the notebook's share card." Add: "**Evaluation set:** Lab 13's 80 questions plus 45 new human-written questions (`data/capstone_questions_v1.jsonl`; spec in `briefs/15-capstone.md`), about 30% unanswerable; `dev` 45, `test` 80." Add **Readings:** "Kapoor et al. 2024 (AI Agents That Matter); Rajpurkar et al. 2018 (SQuAD 2.0); Kamath et al. 2020 (selective QA); Dror et al. 2018 (significance testing); Liang et al. 2023 (HELM)."
- **`PLAN.md` section 6, new rows.**
  - Item: "Capstone evaluation set needs human authors". Risk: "45 new questions (27 unanswerable, at least 13 of them memory bait) must be written and blind-checked by people after the corpus snapshot is frozen; without them abstention is measured on 7 or 8 test items". Mitigation: "Romeo and one instructor, about 3 h and 2 h, after Lab 13's 80; validator checks absence mechanically".
  - Item: "Snapshot ordering". Risk: "`references.qmd` is still incomplete, and the snapshot freezes it for Labs 13 and 15". Mitigation: "complete the Modules 1–12 entries of `references.qmd` before building `workshop_lectures_v1`".
  - Item: "Capstone on CPU". Risk: "the open path on CPU may not fit a `test` run in 10 minutes". Mitigation: "a stratified 24-item `cpu_subset` (cut to 16 if needed), compared only within its path class".
  - Item: "Shared keys in the capstone". Risk: "15 pairs on one TypeSafe key make about 14,000 Jev calls in one afternoon; limits unknown". Mitigation: "workshop keys from TypeSafe; R2 with Jev limited to `dev`".
- **`PLAN.md` section 7, Day 9:** tick "Draft the Module 15 capstone brief and wrap-up" with the note "(lecture and `briefs/15-capstone.md`; not rendered: Quarto not installed)". Under "Code `15-capstone.ipynb`", add "(blocked on the corpus snapshot, Lab 13's questions and the 45 capstone questions; build against the fixture meanwhile)".
- **Lecture 13, section 8:** "Seven gained and two lost is a finding; four and three is not." Under a two-sided sign test, 7–2 gives $p = 0.18$, so it is not a finding at the 0.05 level; 8–1 gives 0.039. Proposed: "Eight gained and one lost is a finding (a sign test gives $p = 0.04$); seven and two is suggestive ($p = 0.18$); four and three is nothing." Lecture 15 teaches the test.
- **`references.qmd`:** a Module 15 / Day 4 entry for the five readings.
- **Figure 15.1** (spec in the lecture): `images/15-capstone-graph.svg`, for the Architect.
- **`teach.qmd` or the facilitator guide (Day 10):** the instructor's afternoon (the build timetable, the circulation notes of lecture 5, the collector, the order of shares).

## Not verified by the Director

- **Nothing in this lab has been built or run.** No notebook, corpus snapshot, question file or baseline exists. Every metric behavior, run time and cost above is an estimate.
- **No live Jev, OpenAI or Anthropic call; no Qwen, bge-small or NLI run** (no keys; the Hub is blocked from the build container). Whether the 0.5B open model carries any signal as a planner or verifier is unknown; option V1 is designed to reveal it.
- **The interfaces of Labs 13 and 14** are taken from their briefs, not from built notebooks. If either notebook departs from its brief, adapt (b) to (d) here.
- **The NLI model as the reference judge** on lecture pages with notation and LaTeX is unvalidated (brief 13, (e)). Until the judge audit exists, the unsupported-answer rate is printed as unvalidated.
- **The effect sizes in the menu** are my guesses, labeled as such in the notebook.
- **Framework documentation sites** (LangChain, LlamaIndex, TypeSafe) were unreachable; this brief relies on the source checks recorded in briefs 13, 14 and JV.
- **Citations** in the lecture: the five readings were checked by web search (titles, authors, venues, summaries) on 2026-10-05, not against the papers. The TMLR venue of Kapoor et al. is from a search summary only.
- `quarto render` was not run: Quarto is not installed in the build container.
