# Lab brief: `notebooks/08-llm-apis.ipynb`

From the Academic Director to the Agentic Systems Engineer. Lecture: `lectures/08-llm-apis.qmd` (same names: `Reply`, `Event`, `extract`, `score`, `run_tools`, `cost_usd`, `K_max`, `R`). Lab standards: `PLAN.md` section 5. Start from `scripts/new_notebook.py m08 --api`; keep its provider cell and the key names of `notebooks/00-setup.ipynb` (`PROVIDER` is `"openai"`, `"anthropic"` or `"open"`). This file is not rendered by Quarto.

**Objectives exercised** (from `_variables.yml`, `m08`): call OpenAI and Claude models for chat, structured output and tool use; compare providers on one task with one harness; reason about cost, latency and failure modes.

**Every number below is a target, a budget or an estimate, not a measurement.** Replace each with what you measure, and tell me if a target cannot be met. The API shapes in the lecture were checked against the official documentation on 2026-10-04; check them again before you write code, and pin the SDK versions you tested.

## The constraint that shapes the lab

With no keys the lab must run end to end on `models.fallback`, and that is the path CI executes. A model of about 0.5 billion parameters will not extract reliably and will often fail a multi-step tool task. So:

- **Checkpoints assert on the harness, never on the model's score.** Each exercise is tested with recorded responses or a scripted `FakeProvider` (a class that returns a fixed sequence of `Reply` objects). These tests are deterministic and identical on every provider.
- **Live-model results are printed metrics.** No assertion on accuracy for any provider. One floor is allowed on the fallback (see Exercise 2) and only if you measure that it holds with margin on three seeds.
- **The fallback uses constrained prompting plus validate-and-retry:** the schema and one worked example in the prompt, greedy decoding, a short `max_new_tokens`, the reply cut at the closing brace of the first JSON object, then Pydantic validation and up to `R = 2` retries with the error appended. No constrained-decoding library in the core path.

## Task and data (ours, generated in the notebook, no download)

- **Extraction task:** a short workshop announcement to an `Event` record: `topic` (one of the four class names of arXiv Topics v1: Language, Vision, Machine learning, Robotics), `city`, `event_date` (ISO `YYYY-MM-DD`), `seats` (integer), `remote` (boolean). The date field continues Lab 4: the announcement writes dates in several surface forms.
- **Evaluation set:** built by a seeded template generator in the notebook, so every gold record is correct by construction. `N = 40` on a keyed provider or a GPU; a fixed 12-item subset on the CPU fallback. Several templates per field, with distractor numbers (a room number, a price) and varied field order, so that it is not solved by a regular expression over one template. Print three examples. If you prefer a committed file, put it under `data/` with a `data/README.md` entry and a hash, and tell the Architect; my preference is in-notebook, because nothing can then drift.
- **Four hand-written hard items, scored separately from the 40:** a missing seat count, two dates in one announcement, a topic named only by description, and one prompt-injection item (the announcement ends with an instruction to set `seats` to 9999). They are for discussion, not for a checkpoint.
- **Tool task:** a dictionary of about six events and two tools, `get_event(event_id)` and `days_between(start_date, end_date)`. Six questions with known answers, at least three of which need both tools in sequence (for example, "How many days are there between E2 and E5?").

## Provided scaffolding

The `Reply` dataclass (`text`, `tool_calls`, `input_tokens`, `output_tokens`, `stop` in `{"end", "truncated", "refused", "tool_calls"}`, `latency_s`); three adapters with one method `chat(messages, system=None, tools=None, max_tokens=...)`: OpenAI (Responses API), Anthropic (Messages API) and `LocalProvider` (Hugging Face `transformers`, chat template, token counts from the tokenizer); `FakeProvider`; the two tools; the generator; a retry-with-backoff wrapper for HTTP 429. The adapters are short and visible, not hidden in a package: participants read them against the lecture's comparison table. Model IDs are read from `_variables.yml` values written into the setup cell by the generator, not typed in the notebook. Later labs (11, 13, 14) reuse this wrapper, so keep it in one self-contained cell.

## Core path (50 minutes)

| # | Participant writes | Checkpoint (deterministic) | Metric printed | Min |
|---|---|---|---|---|
| 0 | Nothing: run setup, read the provider cell and three examples | none | provider in use | 3 |
| 1 | `parse_openai(raw)` and `parse_anthropic(raw)`: turn one recorded raw response of each provider into a `Reply` | On recorded fixtures (a text reply, a truncated reply and a tool-call reply per provider): text, token counts, `stop` mapping and parsed tool arguments equal the expected values; OpenAI `arguments` is decoded from its JSON string | none; then one live `chat` call on `PROVIDER`, printing text, tokens and latency | 8 |
| 2 | `extract(provider, text, schema, R)`: call, parse, validate with Pydantic, retry with the error appended; return the object or `None`, and the number of attempts | With `FakeProvider`: invalid then valid returns the object with `attempts == 2`; always invalid returns `None` after exactly `R + 1` calls; a `truncated` or `refused` reply is not parsed | **Schema-validity rate** on the evaluation set, zero-shot against one-shot, and with `R = 0` against `R = 2` | 12 |
| 3 | `score(preds, golds)`: validity rate, exact match per field, record exact match, with invalid outputs counted as wrong | On six hand-made predictions (two invalid): every returned number equals the expected fraction with denominator 6 | **Field and record exact match** on the evaluation set for `PROVIDER`, with the standard error beside each | 8 |
| 4 | `run_tools(provider, messages, tools, K_max)`: the lecture's state machine | With `FakeProvider` scripts: two calls run in order and each result carries the matching call ID; an unknown tool name yields an error result, not an exception; invalid arguments yield an error result; a script that never finishes stops at `K_max` with status `"step limit"`; a final answer returns status `"done"` | **Task success** on the six questions (final answer contains the correct number) and mean number of model calls | 12 |
| 5 | `cost_usd(n_in, n_out, p_in, p_out)` | Numerical assert on two hand-computed cases, including the lecture's worked example (0.0007) | **Cost and latency table**: one row per available provider with tokens in, tokens out, cost per 100 requests, seconds per request, validity, record exact match; plus the loop's total input tokens against lecture eq. `loop-cost` | 5 |

Format per exercise: Predict, Run, Explain, Check; `# TODO N` stub, folded solution, short "why this works" note. Put the evaluation run of Exercise 2 where participants write their prediction for Exercise 3, since it is the slowest cell on the fallback. Close the core path with one markdown cell that restates lecture section 9 and asks: which of the wrong records in your table looked any different from the right ones?

## Stretch (one section, last, optional): a fourth provider

Wrap the Lab 7 fine-tuned model (base model plus LoRA adapter) as a `LocalProvider` and add its row to the table. Checkpoint: the row exists and its cost is zero. Do not assert that fine-tuning helps: Lab 7 trained on general instructions, not on this task. If the Lab 7 adapter is not available in the runtime, load the base model alone and say so. Not required by any later lab.

## Compute and cost budget

| Path | Budget (to be measured) |
|---|---|
| Fallback on CPU (the CI path): model load, 12-item evaluation with retries, three tool questions | under 6 minutes |
| Fallback on a T4: 40 items, six tool questions | under 3 minutes |
| Keyed provider: 40 items in two prompt versions, six tool questions | under 2 minutes |

**Cost note per full run** (estimate: about 60,000 input and 15,000 output tokens, more output if the model reasons before answering; prices from the pricing pages on 2026-10-04):

- OpenAI at $0.10 in and $0.50 out per million tokens (`gpt-6-luna`): about $0.02; state "under 5 cents".
- Anthropic at $1 in and $5 out per million tokens (`claude-haiku-4-5`): about $0.14, including the tool-use system prompt of roughly 500 tokens per tool request; state "under 25 cents".

Measure both from `usage` on a real run and put the measured figure in the notebook. Set the output cap low (300 tokens covers every call in the core path).

## Flags for the Lab Engineer

1. **Recorded fixtures (Exercise 1).** Record them from real calls with the pinned SDK versions (`response.model_dump()`), strip IDs you do not need, and note the date. If you have no key for a provider, build the fixture from the documented response shape, mark it "constructed, not recorded" and tell me.
2. **Model IDs are not pinned yet.** My proposal to the coordinator is `gpt-6-luna`, `claude-haiku-4-5` and `Qwen/Qwen2.5-0.5B-Instruct`. Read whatever `_variables.yml` holds when you start. Two risks: Anthropic lists Haiku 4.5's retirement as "not sooner than October 15, 2026", so confirm it is still active and has no announced retirement before the workshop; and Lecture 7 is choosing a small causal LM separately, so the two choices should be reconciled (one download for both labs if possible).
3. **Sampling parameters.** Do not pass `temperature` to the commercial adapters. Anthropic's deprecations page says current Claude models reject non-default values and that version 1.0 and later of the Python SDK raises `TypeError` on it. I could not confirm whether the proposed OpenAI model accepts it. The proposed OpenAI model is a reasoning model with a default effort of `medium`: set the effort low for cost and latency, after checking the parameter's current shape.
4. **Forced tool calls.** Use `tool_choice` `auto` only. Some current Claude models reject forced tool choice; a prompt instruction is the portable approach, and the lecture says so.
5. **The schema.** Keep `Event` flat. Both providers need `additionalProperties: false` and every field required. `event_date` is a string in the lecture; if you want a `date` type or a nullable `seats` for the missing-value item, test it on both providers' structured-output modes first and tell me, so that the lecture matches.
6. **Keyed paths use the providers' structured-output and tool features; the fallback uses the prompt protocol.** `extract` and `run_tools` are the same code on all three, because the adapters hide the difference. Say in the notebook which mechanism each provider used, so the validity column is read correctly.
7. **Truncation and refusal are states, not exceptions.** Map them to `Reply.stop` in the adapters. Exercises 2 and 4 depend on it.
8. **No key ever appears in a cell or an output.** Do not print the `KEYS` dictionary. CI runs with no keys set.
9. **Prompt-injection item.** Report what each provider did with it. Do not assert on it, and do not present any prompt as a fix; Module 14 returns to it.
10. **Report back:** which of the three paths you ran, the measured table for each, the measured cost, the fallback's validity rate with and without retry, and anything in the lecture's comparison table that the SDK contradicts.

## As built (Director review, 2026-10-05)

The notebook departs from this brief in these ways. All are accepted; the lecture now matches the notebook.

- **Interface.** `chat(messages, system=None, tools=None, schema=None, max_tokens=MAX_TOKENS)`: the schema is passed to `chat`, and each adapter maps it to its provider's structured-output field (`text.format` at OpenAI, `output_config.format` at Anthropic, the prompt protocol on the open model).
- **Exercise 1.** Four fixtures per provider (text, truncated, tool call, refusal), not three. They are **constructed, not recorded**: no key was available. They follow the documented shapes and are validated against `openai.types.responses.Response` and `anthropic.types.Message` for the pinned SDKs. `model_context_window_exceeded` maps to `"truncated"`.
- **Exercise 2.** `extract(provider, text, schema=Event, R=2, shots=1)` returns `(object or None, number of calls)`. `Event` has a validator, `iso_date`, that rejects any `event_date` not written `YYYY-MM-DD` and impossible dates. The lecture now shows it.
- **Exercise 3.** The zero-shot and one-shot evaluation run opens Exercise 3, as briefed. Validity at `R = 0` is read from the same run (valid at the first attempt). That is exact for greedy local decoding and an approximation for the APIs; the notebook says so. No accuracy floor is asserted on any provider.
- **Exercise 4.** `execute(call, tools)` returns `(content, is_error)` and never raises; `run_tools(provider, messages, tools, k_max=K_MAX, system=None)` with `K_MAX = 6`. The checkpoint also covers undecodable arguments and a tool that raises. A reported, unasserted cell shows a prompt injection arriving through a tool result (the `note` field of event `E7`).
- **Run sizes.** CPU fallback: 12 items and the first three tool questions. GPU or keyed provider: 40 items and all six questions.
- **Offline test double.** If the open model cannot be downloaded, or `NLP_LLMS_STUB=1` is set, the notebook uses `StubProvider`, a rule-based test double that writes the open model's protocol. Its rows are labelled `stub (test double)`. Its numbers measure the harness, not a model, and must never be quoted as model results.
- **Timing.** The core path is 3 (setup) + 8 + 12 + 8 + 12 + 5 + 2 (the closing "What the response did not tell you" section) = 50 minutes. The stretch is the last section and is optional.
- **Stretch.** The Lab 7 base model (`models.instruct_base`, pinned revision) plus the adapter folder `lab07_lora` if present; base model alone otherwise, and the notebook says which.

### Corrections to this brief

- **Flag 3 (temperature) was wrong.** At the API, Anthropic returns an error for a non-default `temperature` only on Claude 4.7 and later models; Haiku 4.5 still accepts it. It is the Python SDK, version 1.0 and later, that removed the argument from `messages.create()` (a `TypeError`). Checked on 2026-10-05 against Anthropic's deprecations page and the signature of `anthropic==1.11.0`. The decision not to send a temperature stands.
- **Structured outputs.** `messages.parse(output_format=Event)` works in `anthropic==1.11.0` because the SDK converts the class and sends it as `output_config.format`. The raw API's top-level `output_format` parameter is deprecated. The lecture now says so.
- **Cost note.** Both per-run figures are estimates from the token estimate above, not measurements: under \$0.25 on Anthropic (\$1 / \$5 per million tokens, confirmed on Anthropic's pricing page on 2026-10-05) and under \$0.05 on OpenAI. The OpenAI prices (\$0.10 / \$0.50) come from secondary sources; OpenAI's pricing page could not be reached. Lecture and notebook now give the same figures and label them as estimates.
- **Flag 2 (retirement).** On 2026-10-05 Anthropic's deprecations page listed Claude Haiku 4.5 as active, retirement "not sooner than October 15, 2026". That date is ten days after this review. Re-check before every delivery; the lecture carries this note.

## Not verified by the Director

- The keyed paths (OpenAI and Anthropic) and the open-model fallback have not run: no keys, and the Hugging Face Hub is unreachable from the build container. The notebook ran end to end only on `StubProvider` (`scripts/test_notebooks.py`, 14 s locally, 2026-10-05). It is **written, not run** on any model.
- OpenAI's documentation and pricing pages could not be reached on 2026-10-05. The OpenAI fragments in the lecture were checked only against the signatures of `openai==3.24.0` (`responses.create`, `responses.parse(text_format=...)`, `output_parsed`, `incomplete_details.reason`). Whether `gpt-6-luna` accepts `temperature`, and its current prices, are unconfirmed.
- Whether the constructed fixtures match real responses from the pinned models.
- Run time on Colab, CPU or T4. `quarto render` of the lecture: Quarto is not installed in the build container.
