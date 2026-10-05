# Lab brief: `notebooks/12-rlcd-jev.ipynb`

From the Academic Director to the two owners of Lab 12 (`AGENTS.md`, "Who owns what"): the **Neural Lab Engineer** builds Exercise 1 (the toy calibration-reward model and its training code) and the `featurize` / `ToyDecider` / `LocalDecider` cell; the **Agentic Systems Engineer** builds Exercises 2 to 4, the keyed Jev path and the stretch. Lecture: `lectures/12-rlcd-jev.qmd` (same symbols and equation names: `rlhf-recall`, `outcome-objective`, `acc-reward`, `acc-optimum`, `brier-reward`, `chosen-prob`, `adapter-conf`, `three-costs`, `three-thresholds`, `three-empirical`). Lab standards: `PLAN.md` section 5. Verified Jev API surface: `briefs/jev-verification.md` (cited below as JV §n). Decision set and Lab 11 machinery: `briefs/11-calibration.md`. This file is not rendered by Quarto.

**Objectives exercised** (from `_variables.yml`, `m12`): state how RLCD's objective differs from RLHF's, and what is and is not public about it; call Jev for typed decisions; use its confidence to set action thresholds. The third objective is met with **probabilities, not the `confidence` field** (lecture section 6); a wording change is proposed at the end.

**Every time, size and cost below is an estimate or a target, not a measurement,** unless marked *checked*. *Checked* means I ran it on 2026-10-05 in the build container (CPU, Python 3.12, `torch` 2.14.1, scikit-learn 1.9.1, NumPy 2.5.3, and `typesafe-sdk` 0.7.2 loaded from the local package cache), not on Colab. No live Jev call has been made by anyone on the build: there is no key.

## The lab in one paragraph

Participants first train a small decision model on the decision set's training split twice, once with an accuracy-only reward and once with a Brier-score reward, and watch the first one's probabilities run toward 1 while the second one's stay calibrated: **our illustration, not TypeSafe's method**. They then ask a decider typed questions about every `dev` and `test` item of Workshop Desk Decisions v1: Jev through `typesafe-sdk` when a `TYPESAFE_API_KEY` is set, otherwise a local stand-in built on the Brier-trained toy model that returns the same `SystemOneResponse` type. They draw its reliability diagram from the probability of the chosen answer, beside Lab 11's language model, and see what goes wrong if Jev's `confidence` field is used instead. Finally they derive act / ask / escalate thresholds from stated costs, analytically and from `dev`, and report the cost per case on `test`. Every checkpoint tests the participant's function on hand-made or synthetic inputs, so it gives the same verdict on both paths. No calibration result of Jev, or of any language model, is asserted.

## The honesty rule, made concrete for this notebook

`AGENTS.md` forbids presenting the toy, or any guess, as TypeSafe's method. In this notebook that means:

1. **A banner cell opens Exercise 1**, verbatim in substance: "This exercise is our illustration of what a calibration-targeted reward can do, built on the proper scoring rules of Module 11. TypeSafe has not published how RLCD works. The only statement about training we found in a TypeSafe source is: 'System One models are trained for calibrated decisions; validate their performance in the target domain' (`typesafe-ai/skills`, `SKILL.md`). Nothing in this exercise is evidence about how Jev was trained."
2. **Every plot title, table header and printed row** produced by the toy model says `toy model (our illustration)`. Every row produced by `LocalDecider` says `local toy decider (not Jev)`. The string `Jev` appears in a result label only when `JEV_PATH == "keyed"` and the row came from `TypeSafeClient`.
3. **No sentence in the notebook describes RLCD's reward, data or algorithm.** The notebook may repeat the lecture's three-column summary (TypeSafe-stated / third-party / ours) by linking to lecture section 2; it must not restate third-party claims as facts.
4. **On the no-key path**, a banner above Exercise 2's results: "No TypeSafe key: the answers below come from the workshop's toy model wrapped to look like Jev's API. They measure our toy model, not Jev. Do not quote them as Jev's."
5. **The closing cell** ("What this lab showed and what it did not", below) is part of the core path, not optional.
6. **Damani et al.'s RLCR** may be mentioned (lecture section 3, optional callout) only as published work by other authors, with the sentence "RLCR is not RLCD".

## Constraints that shape the lab

### (a) The decision set file does not exist yet

`data/decisions_v1.jsonl.gz` (spec: `briefs/11-calibration.md`, "Shared decision set") has not been built. Lab 12 needs the **2,000-item `train` split** to train the toy model, so a 20-item fixture (Lab 11's interim plan) is not enough here.

- The `train` split is **template-only** by design, and so are 80% of `dev` and `test`. An agent may write `data/build_decisions.py` (standard library, seeded, rule-engine labels); only the 80 hand-written items and the 60-item audit need people. **Proposal:** build and commit the template part first, flagged `provisional: true` in the file's metadata until the hand items and audit land, so Labs 11, 12 and 14 can be built and run against real template items now.
- Until then, build the notebook against `build_decisions.py`'s output in a scratch location; quote no number from it in the lecture or in `data/baselines.json`.

### (b) Lab 11's language-model results must reach Lab 12

Exercise 3 compares the decider with Lab 11's language model on the same `test` items. Colab notebooks do not share a runtime, so Lab 12 cannot read Lab 11's variables. Sources, tried in order:

1. **The participant's own Lab 11 export**: `lab11_decisions_<provider>.jsonl` in `NLP_LLMS_DATA` (or uploaded through the Files pane). Lab 11 must add a one-cell export (proposed below); fields per line: `id`, `answer`, `confidence` (stated), `valid`, `provider` (label as printed in Lab 11), `model` (provider model ID), `date`.
2. **A committed reference run**: `data/lab11_reference_decisions_v1.jsonl.gz`, the same fields, recorded once by an instructor with a key (`models.anthropic` proposed; any keyed provider is fine), registered in `_variables.yml` `datasets` with hash and size. The panel title names the model and date.
3. **Neither present**: omit the comparison panel and print why.

**Never compare against Lab 11's stub (test double).** If an export's `provider` is `stub (test double)`, refuse it with a message.

### (c) One decision interface, two backends

As JV §10. The selection runs once, at the top of Exercise 2:

```python
key = get_secret("TYPESAFE_API_KEY")          # Colab Secrets, then the environment; None if absent
JEV_PATH = "keyed" if key else "local"
if JEV_PATH == "keyed":
    JEV = AsyncTypeSafeClient(api_key=key, model=JEV_MODEL)     # never construct without a key: it raises
else:
    JEV = AsyncLocalDecider(LocalDecider(TOY_REF, featurize))  # TOY_REF: the provided Brier-trained model
```

`JEV_MODEL` repeats `models.jev` from `_variables.yml` (currently `jev-latest`; a test must fail if they disagree, as for the other model IDs). Exercises 2 to 4 never know which backend answered, except through `JEV_PATH` for labels.

**The local decider always wraps `TOY_REF`**, a Brier-reward model trained by a provided cell with the *solution* reward, not the participant's Exercise 1 model. Exercises 2 to 4 must not depend on whether Exercise 1 was finished.

## Exercise 1's model: the toy decision model (Neural Lab Engineer)

### Design requirement learned from the stand-in checks

The contrast the lecture promises (lecture section 3) shows cleanly **only for a low-capacity model** trained on many more items than it has parameters. *Checked* on two stand-ins:

- **Synthetic, 2 and 5 options, 2,000 train / 2,000 test items, 8 features, labels the features cannot fully determine** (`torch` linear softmax, Adam, lr 0.05, full batch, no weight penalty, seeds 0 and 1). After 3,000 steps: test accuracy within 0.011 between the two rewards; accuracy-reward model mean probability 0.968–0.986, ECE 0.176–0.299, weight norm still growing (6.5 → 11.3 → 22.8 at 300 / 1,000 / 3,000 steps for one run); Brier-reward model converged by 300 steps, ECE 0.012–0.032. Temperature fitted on 100 held-out items: τ* = 9.9–15.9 for the accuracy-reward model, after which its ECE is 0.020–0.052 and its Brier score 0.261–0.442 against the Brier-reward model's 0.256–0.423.
- **arXiv Topics, 4,096 hashed uni- and bigram features, 2,000 training papers.** The comparison flipped with the learning rate, step count and weight penalty (for example ECE 0.111 accuracy vs 0.044 Brier at lr 0.02 / 1,000 steps, but 0.130 vs 0.182 at lr 0.005 / 200 steps). Overfitting dominated. **Do not use a high-dimensional bag-of-words model.**

### Features (provided, fixed before any evaluation)

`featurize(state, question_text) -> np.ndarray of shape (31,)`, float32. It reads only the item's structured `state` and the question text, never `label`, `rule`, `rationale` or `difficulty`. It ignores a `policy` key if present (Jev's state carries the policy text; see Exercise 2).

| Group | Dims | Features |
|---|---|---|
| Numbers | 5 | `days_before / 30` clipped to [−1, 2]; `(seats − registered) / seats`; `registered / seats`; `paid_eur / 1000` (0 if no registration); `fee_eur / 1000` |
| Flags | 4 | `remote`; registration is `null`; registration `status == "confirmed"`; the request contains `@` |
| Question type | 6 | one-hot by keyword in the question: accepted (P1), full refund (P2), transfer (P3), catering (P4), approval (P5), next step (route) |
| Interactions | 5 | `days_before / 30` times each of the five policy question-type flags |
| Request keywords | 11 | lowercased request contains: ignore / override / disregard; refund / money back; transfer; cancel; how much / how many; policy / rule; register / sign up / seat; catering / parking / room / lunch; confirm; exception / manager / just this once; a question mark |

Standardize the five numeric and five interaction columns with the `train` mean and standard deviation, stored with the model. The keyword lists come from the policy text and the template vocabulary; **do not tune them on `dev` or `test` errors**. The features are deliberately unable to express the policy exactly: `days_before` enters linearly, so interval rules (P2's 50% band) and exact boundaries are approximations, and keywords are noisy proxies for intent, especially on the held-out wordings of `dev` and `test`. That is what gives the model intermediate probabilities to calibrate.

**Target, to be measured on the real `train`/`test` split:** toy test accuracy between 0.60 and 0.90 in each family, for the Brier-reward model. Outside that range, adjust the feature set once (drop the interactions if above 0.90; add `days_before` bucket flags at 7 and 14 if below 0.60), re-measure, and tell me what you changed. Freeze it before recording any number.

### Model and training (provided except the two reward functions)

```python
class ToyDecider(nn.Module):
    """Two linear heads over 31 features: policy (yes/no) and route (5 options). ~190 parameters."""
    def __init__(self, d=31):
        super().__init__()
        self.policy = nn.Linear(d, 2)        # logits for ["yes", "no"]
        self.route = nn.Linear(d, 5)         # logits for ROUTE_OPTIONS, in that order
    def probs(self, feats, family):          # feats: (N, 31); family: "policy" | "route"
        head = self.policy if family == "policy" else self.route
        return torch.softmax(head(feats), dim=-1)            # (N, K)
```

- `train_toy(reward_fn, train_items, steps=3000, lr=0.05, seed=0, log_every=50)`: full batch, Adam, **no weight penalty** (a penalty would cap the accuracy reward's divergence and hide the lesson; say so in a comment). The objective is lecture @eq-outcome-objective: the mean over all 2,000 items of `reward_fn(probs, labels)`, each item through its family's head, maximized (the loss is its negative). Logs, every `log_every` steps: mean train reward, mean $\hat{p}$ on train per family, and the total weight norm. Returns `(model, log)`.
- Seeds: `torch.manual_seed(seed)` before constructing the model; CPU; deterministic.
- `TOY_REF = train_toy(reward_brier_solution, ...)` in a provided cell after Exercise 1 (see constraint (c)).
- Restated from Lab 11, verbatim: `reliability_bins`, `ece`, `brier`, `brier_binary`, `noise_floor`, `fit_temperature`, `risk_coverage`, `plot_reliability`.

### Exercise 1: participants write

```python
def reward_accuracy(probs, labels):
    """Expected correctness of an answer sampled from probs: probs[i, labels[i]].  (N,) tensor.
    Lecture eq-acc-reward: minus the linear score."""

def reward_brier(probs, labels):
    """Minus the multiclass Brier score per item: -sum_k (probs[i,k] - 1[labels[i]==k])**2.  (N,) tensor.
    Lecture eq-brier-reward."""
```

Both must be differentiable PyTorch functions (no `.item()`, no NumPy).

## Mapping decision items to Jev (Agentic Systems Engineer)

*Verified against `typesafe-sdk` 0.7.2 (JV §2), and in the build container with the cached 0.7.2 package: `Noul`, `Choice`, answer field names and types, `extra="forbid"` on questions, and response construction below.*

- **State:** `jev_state(item) = {"policy": POLICY_TEXT, **item["state"]}`, where `POLICY_TEXT` is `data/decisions_policy_v1.md`, the same text Lab 11 puts in its system prompt. Never include `label`, `rule`, `rationale`, `difficulty`, `source` or `id`.
- **Policy item → `Noul(instructions=item["question"])`**, one question per call, named `"decision"`. `criteria` omitted (yes/no is the natural reading); if you add `NoulCriteria(true=..., false=...)`, use the same words as Lab 11's prompt.
- **Route item → `Choice(instructions=item["question"], criteria=ROUTE_DESCRIPTIONS)`**, where `ROUTE_DESCRIPTIONS` is a dict over `ROUTE_OPTIONS` in order. **The descriptions must say exactly what Lab 11's prompt tells the language model about each option**, word for word, so that Exercise 3 compares the two models on the same information. Question names are not sent to the model (`SKILL.md`), so each description must be complete.
- **One item per call.** The state differs per item, and there is no batch endpoint. Run calls concurrently: `AsyncTypeSafeClient` under `asyncio.Semaphore(8)`, gathered with top-level `await` in the cell (IPython supports it on Colab). Default `RetryPolicy`. Catch `TypeSafeError` per item, record it, and continue: an item with an error is **invalid**, counted wrong in accuracy (denominator $N$, Lab 8's rule) and excluded from calibration metrics with $N_{\text{valid}}$ printed.
- **Per-item record** from `decide_all(JEV, items)`: `id`, `family`, `difficulty`, `answer`, `p` ($\hat{p}$), `probabilities` (dict), `kappa` (`confidence`, or `None` for yes/no), `correct`, `valid`, `model` (`resp.model`), `request_id`, `input_tokens` (`None` on the local path), `latency_s` (wall clock around the call), `error`.
- **Model ID:** print `resp.model` once per run and in every plot title on the keyed path. It can differ from `JEV_MODEL` (JV §2).
- **Logging:** never set the `typesafe_sdk` logger to DEBUG (request bodies are logged unredacted, JV §2).

## `LocalDecider`: the no-key backend, reusable by Labs 14 and 15

*Design from JV §10; the construction path was checked in the build container against the cached `typesafe-sdk` 0.7.2.*

```python
ROUTE_OPTIONS = ["retrieve", "calculate", "send_email", "ask_user", "escalate"]

class UnsupportedQuestion(NotImplementedError):
    """Raised for a question the toy model was not trained to answer."""

class LocalDecider:
    """No-key stand-in for typesafe_sdk.TypeSafeClient. NOT Jev: answers come from the
    workshop's toy decision model (Lab 12, Brier reward), our illustration."""
    model_name = "local-toy-decider-v1"

    def __init__(self, model: ToyDecider, featurize, *, round_to: int | None = None): ...

    def system_one(self, state, questions, *, model=None, retry=None, timeout=None,
                   extra_headers=None, extra_body=None, response_model=None) -> SystemOneResponse: ...

    def close(self): ...                       # no-op; with LocalDecider(...) as d: works
    def __enter__(self): ...
    def __exit__(self, *exc): ...

class AsyncLocalDecider:
    """Same, for code written against AsyncTypeSafeClient: `await d.system_one(...)`, `async with`."""
    def __init__(self, local: LocalDecider): ...
    async def system_one(self, state, questions, **kwargs) -> SystemOneResponse: ...
    async def close(self): ...
```

Behavior, all of which a unit check in the notebook (and later `tests/`) must pin:

| Input | Output |
|---|---|
| `state` | a dict with `today`, `event`, `registration`, `request` (a `policy` key is accepted and ignored); anything else raises `ValueError` |
| `questions` | a non-empty mapping, values either SDK question objects or raw dicts with a `"type"` key (as the SDK accepts); each is answered independently |
| `Noul` | `{"type": "noul", "noul": P(yes)}` from the policy head; **no `confidence` key**, as in Jev's schema |
| `Choice` with `set(criteria) == set(ROUTE_OPTIONS)` | `{"type": "choice", "choice": argmax, "probabilities": {label: p} in the criteria's order, "confidence": κ}` with κ from lecture @eq-adapter-conf. Comment in the code: "TypeSafe's emulator formula (`system-one-adapter`); Jev's own formula is not published" |
| any other `Choice`, any `Score` | raise `UnsupportedQuestion(name)`. Labs 14 and 15 catch it and send the question to their Qwen-backed decider (JV §10) |
| probabilities | full precision, summing to 1 within 1e-6 (asserted inside); `round_to=2` rounds to 0.01 to mimic the recorded Jev response (JV §2), for testing tie handling only |
| `model` argument | ignored; the response's `model` is always `"local-toy-decider-v1"` |
| `usage` | `input_tokens=None`, `output_tokens=None`: downstream cost code prints "n/a (local)" |
| construction | `SystemOneResponse.from_http_response(httpx2.Response(200, json=body, headers={"x-typesafe-request-id": f"local-{n:06d}"}, request=httpx2.Request("POST", "local://v1/systemone")))`, or `response_model.from_http_response(...)` when given |

Two facts *checked* in the build container that force the construction row:

- `SystemOneResponse.model_validate_json(...)` builds a valid response, **but `resp.request_id` then raises `TypeSafeError("The response did not include a request ID.")`**. Any downstream cell that logs the request ID would crash on the local path. Building through `from_http_response` with an `x-typesafe-request-id` header makes `.request_id` work on both paths.
- The SDK does **not** check that `probabilities` sum to 1 (`ChoiceAnswer` accepted `{"a": 0.5, "b": 0.7}`). The local decider must assert it itself; `decide_all` checks it on the keyed path with tolerance 0.03 (five probabilities rounded to 0.01).

Deterministic: no randomness anywhere. **Labs 14 and 15 restate** `featurize`, `ToyDecider`, `train_toy`, the solution `reward_brier`, `LocalDecider` and `AsyncLocalDecider` verbatim and retrain `TOY_REF` (about 5 s on CPU, estimate from the timing below), rather than loading a file. Mark the cell "provided; reused by Labs 14 and 15". Lab 13 does not use it: its no-key reranker is a cross-encoder (JV §10).

## Core path (50 minutes)

Format per exercise: Predict, Run, Explain, Check; `# TODO N` stub, folded solution (`#@title Solution N`), a short "why this works" note.

| # | Participant writes | Equation | Checkpoint (deterministic) | Printed, never asserted | Min |
|---|---|---|---|---|---|
| 0 | Nothing: run setup; read `JEV_PATH`; print the policy's first rules and three items (one policy, one route, one `injection`) | – | none | which decider will answer; `JEV_MODEL`; the decision-set version and whether it is `provisional` | 3 |
| 1 | `reward_accuracy(probs, labels)`, `reward_brier(probs, labels)` | `acc-reward`, `brier-reward`, `outcome-objective`, `acc-optimum` | (i) hand cases: probs `[[0.7, 0.3], [0.2, 0.8]]`, labels `[0, 0]` give accuracy reward `[0.7, 0.2]` and Brier reward `[-0.18, -1.28]` (atol 1e-6); (ii) uniform five-way gives Brier reward −0.8 exactly; one-hot correct gives 0 for both; (iii) Monte Carlo: 200,000 answers sampled from fixed probabilities (`torch.Generator` seeded) have mean correctness within 4 standard errors of `reward_accuracy`; (iv) `reward_brier == -brier` (Lab 11's, per item) to 1e-12; (v) both return shape `(N,)` and carry gradients | Training curves for both rewards: mean $\hat{p}$ per family and weight norm against step (the accuracy-reward curve keeps rising; the Brier curve flattens); table on `dev` and `test`: accuracy, mean $\hat{p}$, ECE (15 bins) with the noise floor for that $N$, Brier (binary, of the correctness event), per family and pooled; the same after `fit_temperature` on `dev` (τ* printed). Every row labelled `toy model (our illustration)` | 14 |
| 2 | `to_questions(item)` returning `{"decision": Noul(...) or Choice(...)}`; `chosen_answer(answer)` returning `(label, p_hat)` | `chosen-prob` | (i) questions validate, `instructions == item["question"]`, Choice criteria keys equal the item's `options` in order; no `label`/`rule`/`rationale` text appears in `jev_state(item)` or the questions; (ii) on five hand-built responses (constructed through `from_http_response`): `noul = 0.83` → `("yes", 0.83)`; `noul = 0.2` → `("no", 0.8)`; `noul = 0.5` → `("yes", 0.5)`; a Choice with probabilities `(0.6, 0.1, 0.1, 0.1, 0.1)` and `confidence = 0.5` → `("<top label>", 0.6)`, **not 0.5**; a `ScoreAnswer` → `TypeError` | Through the provided `decide_all` on `dev` + `test`: validity rate, accuracy per family, invalid items with their error; on the keyed path, `resp.model`, total input tokens, measured cost in USD, median and 95th-percentile latency; on the local path, the banner of honesty rule 4 | 10 |
| 3 | `calibration_arrays(records, use="p")` returning `(conf, correct)` over valid records; `use="kappa"` takes the `confidence` field and raises `ValueError` if asked for yes/no records only (they have none) | `chosen-prob`, `adapter-conf` | (i) fixture records give the expected arrays; (ii) **synthetic, seeded**: 100,000 five-way items with probabilities drawn from Dirichlet(1, 1, 1, 1, 1) and labels drawn from those probabilities (calibrated by construction): ECE from `p` < 0.01, ECE from `kappa` (computed with @eq-adapter-conf) > 0.10 (*checked*, seeds 0–2: 0.002–0.003 and 0.134–0.136; at 20,000 items the `p` ECE reached 0.010, too close to the bound) | Reliability diagrams on `test` from `p`: the decider (Jev or `local toy decider (not Jev)`) beside Lab 11's language model (constraint (b)), each with accuracy, ECE, binary Brier, $N_{\text{valid}}$ and the noise floor for that $N$; the route items redrawn with `kappa` on the horizontal axis; one item printed where `kappa` and `p` differ most | 8 |
| 4 | `expected_costs(p, costs)` → `(C_act, C_ask, C_esc)`; `action_thresholds(costs)` → `(tau_esc, tau_act)`, collapsing to Chow's $\lambda^*$ for both when asking never pays; `choose_thresholds(p, correct, costs)` minimizing @eq-three-empirical over pairs from the observed `p` plus 0 and a value above the maximum | `three-costs`, `three-thresholds`, `three-empirical` | (i) the lecture's worked example: `costs = dict(wrong=20, ask=0.5, miss=4, esc=3)` gives `(0.375, 0.96875)`; at `p = 0.6` the costs are `(8, 2.1, 3)`; (ii) `ask = 2.5` collapses to `(0.85, 0.85)` (both *checked* against a 100,001-point grid of the three cost lines); (iii) a hand-made eight-item case with a known best pair; (iv) the chosen pair's `dev` cost ≤ act-all, ask-all and escalate-all; (v) `tau_esc <= tau_act` always | With the worked example's costs: share of `test` items in each action and cost per case at the analytic pair and at the `dev`-chosen pair, beside act-all, ask-all, escalate-all and Chow's two-action rule; both pairs printed as numbers; the same for Lab 11's language model if its records are present | 10 |
| – | Nothing: read and answer the closing cell | – | none | – | 5 |

Minutes: 3 + 14 + 10 + 8 + 10 + 5 = 50.

**Where the slow cell goes.** On the keyed path, `decide_all` (400 calls) starts at the top of Exercise 2, before the participant writes `to_questions`; it uses the solution functions internally and says so. On the local path it takes seconds.

**Closing cell: "What this lab showed and what it did not"** (markdown, then two questions):

- Exercise 1 showed, on our toy model, that an accuracy-only reward is indifferent to probabilities and drives them toward 1, and that a proper-score reward does not. It did not show how Jev was trained: TypeSafe has not published RLCD.
- If you ran without a key, every decider number in this notebook is our toy model's.
- With a key, Jev's calibration was measured on at most 300 `test` items of one synthetic domain; the noise floor printed beside each ECE says how much of a difference is readable. TypeSafe's own advice is to validate in the target domain.
- Questions: (1) Your guard for the `send_email` tool (Lab 14) should use higher costs for a wrong action than the router. Which threshold moves, and which way? (2) Name one thing you would need to see from TypeSafe before saying anything about how RLCD works.

## What is asserted on each path

| Path | When | Asserted | Printed, never asserted |
|---|---|---|---|
| **Keyed (Jev)** | `TYPESAFE_API_KEY` set | every unit checkpoint above; harness: every item yields a record (valid, or invalid with its error); each valid answer is one of the item's options; Choice probabilities sum to 1 within 0.03; `chosen_answer`'s label equals an argmax of the probabilities within 0.01 (rounding ties) | Jev's validity, accuracy, ECE, Brier, reliability diagrams, thresholds and costs per case; `resp.model`; measured tokens, USD and latency. **No Jev accuracy or calibration value is asserted, ever** |
| **Local (no key; the CI path)** | no key | the same unit and harness checks; after the Lab Engineer's recorded run, the toy model's deterministic values (accuracy, ECE, τ* per reward, the `dev`-chosen threshold pair) against `data/baselines.json` `lab12.*`, to 1e-4, as Labs 1, 2 and 11 do with their fallbacks | the same tables and plots, every label `local toy decider (not Jev)` under the honesty banner. These numbers are never written anywhere as Jev's |
| **Lab 11 comparison panel** | an export or the reference file present | the file's hash (reference) or schema (export); refusal of a stub export | the language model's diagram and metrics, labelled with its provider and model ID |

Exercise 1's qualitative claims (the accuracy-reward model's weight norm keeps growing; its `test` ECE exceeds the Brier model's) are **printed, not asserted, until verified** on the real `train` split for seeds 0 to 4. If they hold for all five, promote them to assertions with margins taken from the worst seed, and record the five seeds' values in `briefs/12-rlcd-jev.md` under "As built". If the `train` split proves linearly separable in some family (then the Brier model's weights grow too), tell me before changing the features.

## Stretch (one section, last, optional; not required by any later lab)

**Jev against an LLM on the same decisions, through TypeSafe's emulator.** `system-one-adapter==0.2.1` (TypeSafe AI, MIT; JV §5) answers the same `system_one(state, questions)` call with an LLM and returns a `SystemOneResponse` subclass with `usage.latency`.

- Install in the stretch cell only: `%pip install -q system-one-adapter==0.2.1`. Whether it resolves alongside Lab 8's `openai` and `anthropic` pins on Colab is **unverified**; check it and pin accordingly.
- Run `SystemOneAdapterClient(llm_answer_mode="probabilities", ...)` with `provider="anthropic", model=MODELS["anthropic"]` (or `"openai"`), on the 100 `dev` items with the same `jev_state` and `to_questions`. How the adapter reads provider keys is **unverified** (JV §5 read the source, did not run it): confirm and document.
- Compare on those 100 items: accuracy, binary Brier, ECE with the noise floor (about 0.07 at $N$ = 100: "a difference smaller than this is not a finding"), median and 95th-percentile latency, input and output tokens, and cost per 1,000 decisions from each provider's prices.
- Columns: Jev (keyed only), adapter LLM (needs an LLM key), `local toy decider (not Jev)` (always). Without an LLM key, print a skip message. Label the adapter column "LLM via TypeSafe's emulator", never "Jev".
- Checkpoint: all three columns return `SystemOneResponse` instances with the same answer names; probabilities sum to 1 within 0.03.

## Compute and cost budget

| Part | No key (CPU) | Keyed |
|---|---|---|
| Setup: installs (`typesafe-sdk==0.7.2`, `pydantic` as Lab 8), decision set and policy file | under 0.5 min (estimate) | the same |
| Exercise 1: two trainings of 3,000 full-batch steps on 2,000 × 31 features, plus `TOY_REF` (a third) | about 15 s (*checked*: 4.1 s and 4.4 s per 3,000-step training on random 2,000 × 31 data, build-container CPU) | the same |
| `decide_all`, 400 items | local: under 10 s (estimate) | Jev, 8 concurrent calls: under 1.5 min (estimate; latency unmeasured, third parties report 70–500 ms) |
| Exercises 3–4 (threshold search: about 5,000 pairs on 100 `dev` items) | seconds | seconds |
| **Core path total** | **under 2 min** (estimate) | **under 3 min** (estimate) |
| Stretch: 100 adapter LLM calls, 8 concurrent | skipped | under 2 min (estimate) |

**Cost note** (state in the notebook; estimates from token arithmetic, **not measured invoices**; replace with figures measured from `usage` on a real run):

- **Jev, core path:** 400 calls × about 1,000 input tokens (the policy, about 700 tokens; records and request, about 200; question and option descriptions, about 50–100) ≈ 0.4M input tokens. At 0.042 USD per million input tokens and 0 for output (`WorkflowEvals` price table for `typesafe:jev-1.13.0`, TypeSafe-published, not the pricing page; output "currently free" per the SDK schema), about **0.02 USD: state "under 5 cents"**. The recorded example in JV §2 (448 input tokens for one state and three short questions) is consistent with this order of magnitude.
- **Stretch, adapter LLM on 100 items:** about 110k input tokens and up to 15k output tokens. Anthropic `models.anthropic` at 1 / 5 USD per million (Anthropic's pricing page, read 2026-10-05 for Lab 8): about 0.11 + 0.08 USD, state **"under 25 cents"**. OpenAI `models.openai` at 0.10 / 0.50 USD per million (secondary sources, per Lab 8's brief): state **"under 5 cents"**.
- **Local path:** free.
- A room of about 30 people on one shared key is about 12,000 Jev calls in a few minutes. Rate limits are **unknown** (JV §9). Ask TypeSafe about workshop keys (`PLAN.md` section 6, "Jev access").

## Flags for the Lab Engineers

1. **Names.** Use the lecture's: `p_hat` or `p` for $\hat{p}$, `correct` for $o$, `kappa` for Jev's `confidence`, `tau_esc` / `tau_act`, `costs = dict(wrong=..., ask=..., miss=..., esc=...)`. Do not name anything `lambda` or `T`.
2. **Never read `confidence` for calibration.** Reliability diagrams, ECE, Brier and thresholds use $\hat{p}$ from @eq-chosen-prob. `kappa` appears only in Exercise 3's redraw and in the records.
3. **Tie rule for yes/no:** "yes" when `noul >= 0.5`. State it in the notebook.
4. **Fit on `dev`, report on `test`**, for temperature and for thresholds, as in Lab 11. `train` is never shown to Jev or to an LLM.
5. **Do not send personal data** in `state`; the decision set has none by construction (invented names, `example.org`). Do not log at DEBUG.
6. **Never construct `TypeSafeClient` or `AsyncTypeSafeClient` without a key**; it raises at construction (JV §2).
7. **Package names.** Install only `typesafe-sdk==0.7.2` (and, in the stretch, `system-one-adapter==0.2.1`). `tests/test_package_names.py` fails on lookalikes; never print an unregistered name (JV §6).
8. **Rounded probabilities.** Jev's may arrive rounded to 0.01 (JV §2). Lab 11's `risk_coverage` already treats ties as one step; `choose_thresholds` must search over unique values.
9. **Invalid answers** count wrong in accuracy and are excluded from calibration metrics, with $N_{\text{valid}}$ printed (Lab 11, flag 5).
10. **The local path is the CI path.** `scripts/test_notebooks.py` runs it with no key; it must finish without network access to `api.typesafe.ai`.
11. **Report back:** the measured toy-model tables for both rewards and seeds 0–4; whether the feature set needed adjusting; run time per section on CPU (and on Colab); whether the keyed path ran, with `resp.model`, measured tokens, cost and latency; whether the stretch ran; anything in the lecture that the notebook contradicts.

## Proposed changes (not made; for Romeo or the Architect)

- **`_variables.yml` `modules.m12.objectives[2]`:** "Use its confidence to set action thresholds" → "Turn its probabilities into act, ask and escalate thresholds from stated costs". Reason: lecture section 6; the `confidence` field must not be used for thresholds.
- **`_variables.yml` `datasets`, new entry `lab11_reference`** for `data/lab11_reference_decisions_v1.jsonl.gz` (modules `[11, 12]`, hash and size on recording, `license: "CC0 1.0"`, plus `model` and `recorded` date fields).
- **`data/baselines.json`:** `lab12.toy.{accuracy,brier}.{dev,test}` (accuracy, mean $\hat{p}$, ECE, Brier, τ*) and `lab12.local_thresholds` after the recorded run.
- **Lab 11 (brief and notebook):** an export cell writing `lab11_decisions_<provider>.jsonl` with the fields of constraint (b); and `ROUTE_DESCRIPTIONS` defined once in Lab 11's prompt so Lab 12 can restate it verbatim.
- **`data/build_decisions.py`:** build and commit the template items first, marked provisional (constraint (a)).
- **`PLAN.md` section 4, Module 12, Lecture:** "typed answers (choice, score, yes/no) with confidence out" → "typed answers (yes/no, choice, score) with probabilities out (plus a concentration `confidence` for choice and score)"; "turning confidence into policy" → "turning probabilities into policy". **Readings:** "TypeSafe's public RLCD and Jev announcement and API documentation" → "TypeSafe's Jev announcement and `docs.typesafe.ai` (not yet read; see the TODO in lecture 12); `typesafe-ai/skills` `SKILL.md`; Gneiting and Raftery 2007; Damani et al. 2026 (RLCR: related published work, not TypeSafe's); Kahneman 2011 for the framing".
- **`PLAN.md` section 6, new row.** Item: "Lab 12 compares with Lab 11's LLM". Risk: "Colab notebooks share no runtime, so Lab 12 cannot see Lab 11's results". Mitigation: "Lab 11 exports its records; an instructor-recorded reference run is committed; Lab 12 omits the panel rather than compare with a stub".
- **`PLAN.md` section 6, RLCD rows:** add "Lecture 12 carries a TODO for Romeo to quote TypeSafe's announcement and docs; until then it lists only the `SKILL.md` sentence as TypeSafe-stated".
- **`PLAN.md` section 7, Day 8:** tick "Draft lecture 12: RLCD and Jev (public facts and our illustration clearly separated)", with the note "(TypeSafe's own statements are a marked TODO for Romeo; not rendered)".
- **`references.qmd`:** under Module 12, add `SKILL.md` and `typesafe-sdk`, Gneiting and Raftery 2007, Damani et al. 2026, Kahneman 2011, Stanovich and West 2000, and Melnikoff and Bargh 2018; mark TypeSafe's announcement and docs as "to be read".
- **`tests/`:** a unit test for `LocalDecider` once it lives in a module the tests can import (or a notebook-cell test in `test_notebooks.py`): the behavior table above.

## Not verified by the Director

- **Nothing in this lab has been built or run.** No notebook exists. The decision set does not exist. All toy-model numbers above come from stand-ins (synthetic data and arXiv Topics), not from the decision set.
- **No live Jev call.** Jev's accuracy, calibration, latency, rate limits, rounding, current model behind `jev-latest`, and the server's `confidence` formula are unknown. The local-path construction was checked offline against the cached `typesafe-sdk` 0.7.2 only.
- **TypeSafe's documentation and announcement are unread** (blocked from the build container on 2026-10-05, again by WebFetch today for `typesafe.ai`). Everything the lecture says TypeSafe stated comes from the SDK, `SKILL.md`, `system-one-adapter` and `WorkflowEvals` via `briefs/jev-verification.md`.
- **Third-party RLCD claims** were seen only in web-search summaries; the pages themselves were blocked.
- `system-one-adapter`'s key handling and its compatibility with Lab 8's pins are unverified.
- Prices: Jev's from TypeSafe's eval price table, not its pricing page; OpenAI's from secondary sources.
- Citations: Damani et al. (2026) checked by web search (title, ICLR 2026, abstract), not against the paper; the author list is from memory. Kahneman 2011, Stanovich and West 2000, Melnikoff and Bargh 2018 and Chow 1970 are cited from memory.
- `quarto render` was not run: Quarto is not installed in the build container.
