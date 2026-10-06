---
title: "Pace sheet"
subtitle: "Minute by minute, module by module"
---

<!--
Instructor page, rendered by Quarto. Lecture rows come from each lecture's timing table.
Lab rows come from the minutes in each notebook's exercise headings, which follow the
"As built" sections of briefs/*.md. Module titles and slot lengths come from _variables.yml.
-->

## How to read this sheet

Minutes count from the start of the module's slot. The lecture runs from 0 to {{< var workshop.lecture_minutes >}} and the lab from {{< var workshop.lecture_minutes >}} to 95. Clock times are on the [schedule](schedule.qmd). "CP" is a checkpoint; Checkpoint *N* belongs to Exercise *N*. The last column says what participants should have passed by the end of that row. If more than a third of the room has not, apply the module's "behind" rule.

Lab minutes are the notebooks' own planning minutes. **No lab has been timed on Colab or on a T4.** Labs 1 to 5 were timed on a shared CPU only, and Labs 2 to 5 took far longer there than these minutes allow (see the [facilitator guide](facilitator-guide.md)). Treat every lab row as a target until you have run the notebook on the room's runtime.

Optional callouts in the lectures sit outside the 45 minutes. The stretch section of every lab sits outside the 50.

**Each morning, the opening slot (10 minutes):** 0–5 the opening lines in the facilitator guide; 5–10 the setup check (Day 1) or the recap (Days 2–4).

## Day 1, 08:00: Module 0 · {{< var modules.m00.title >}}

Optional and self-serve, {{< var modules.m00.minutes >}} minutes before the 09:00 welcome, with no lecture. Minutes count from 08:00. **Provisional:** these rows are a target for a participant who has not installed anything yet. Replace them with the step timings on the [Module 0 page](lectures/00-coding-agents.qmd) once it is drafted, and with measured times once someone has run the module on a fresh laptop. Install failures and what to do about them are in the [facilitator guide](facilitator-guide.md#module-0).

| Minutes | Segment | By the end |
|---|---|---|
| 0–15 | Install one coding agent and sign in | the agent answers a prompt in the terminal |
| 15–25 | git identity, `gh auth login`, a repository on GitHub | `gh auth status` succeeds; a first commit is pushed |
| 25–40 | App 1 with the agent: protein structure explorer (1UBQ), with its three.js page | the app's own check passes; the page renders locally |
| 40–52 | App 2 with the agent: RFM customer segmentation, with its three.js page | the app's own check passes; the page renders locally |
| 52–60 | Publish both on GitHub Pages | both Pages URLs load (allow up to 10 minutes after enabling Pages) |

**Behind at minute 25:** if the agent is still not installed, pair the participant with a neighbor and have them do the git and `gh` steps on their own machine. **At 08:55:** everyone stops; the rest is homework.

## Day 1

### Module 1 · {{< var modules.m01.title >}}

| Minutes | Segment | By the end |
|---|---|---|
| 0–5 | Lecture 1. What makes language hard | |
| 5–12 | 2. Tokens, vocabularies and Zipf's law | |
| 12–25 | 3. N-gram language models, smoothing, perplexity | |
| 25–30 | 4. Bag-of-words and TF-IDF | |
| 30–38 | 5. Naive Bayes and logistic regression | |
| 38–42 | 6. Evaluation: precision, recall, F1 | |
| 42–45 | 7. Where count-based methods stop working | |
| 45–53 | Lab: setup, data, Exercise 1 (tokens and vocabulary) | CP1 |
| 53–65 | Exercise 2 (n-gram counts, add-k) | CP2 |
| 65–73 | Exercise 3 (perplexity) | CP3 |
| 73–78 | Exercise 4 (sampling) | CP4 |
| 78–88 | Exercise 5 (TF-IDF, two classifiers) | CP5 |
| 88–95 | Exercise 6 (precision, recall, F1; read the errors); the baseline card | CP6 |

**Behind at minute 65:** run Exercise 4 as a demonstration. Exercises 3, 5 and 6 may not be cut.

### Module 2 · {{< var modules.m02.title >}}

| Minutes | Segment | By the end |
|---|---|---|
| 0–4 | Lecture 1. Where counts stop | |
| 4–9 | 2. The distributional hypothesis and dense vectors | |
| 9–15 | 3. Skip-gram with a full softmax | |
| 15–25 | 4. Negative sampling: the derivation | |
| 25–28 | 5. CBOW and GloVe in brief | |
| 28–33 | 6. PyTorch refresher | |
| 33–40 | 7. Feed-forward networks and backpropagation | |
| 40–45 | 8. Bengio's neural language model | |
| 45–49 | Lab: setup, data, three training pairs | |
| 49–61 | Exercise 1 (the SGNS loss) | CP1 |
| 61–65 | Provided: train the embeddings; predict the loss curve | first loss $= (K+1)\log 2$ |
| 65–72 | Exercise 2 (nearest neighbors) | CP2 |
| 72–77 | Provided: analogies and the PCA picture | |
| 77–83 | Exercise 3 (averaging embeddings) | CP3 |
| 83–91 | Exercise 4 (feed-forward classifier) | CP4 |
| 91–95 | The comparison with TF-IDF; results card | |

**Behind at minute 65:** the analogy step (72–77) becomes a demonstration. Skip-gram training took 695 s on the build CPU, far over its 4 minutes; start it before the Exercise 1 discussion ends.

### Module 3 · {{< var modules.m03.title >}}

| Minutes | Segment | By the end |
|---|---|---|
| 0–5 | Lecture: the recurrent neural network | |
| 5–13 | The RNN language model and teacher forcing | |
| 13–23 | Backpropagation through time and vanishing gradients | |
| 23–26 | Gradient clipping | |
| 26–36 | The LSTM | |
| 36–40 | Measuring the improvement fairly | |
| 40–45 | Generating text: sampling strategies | |
| 45–53 | Lab: setup, data, Exercise 1 (an RNN cell by hand) | CP1 |
| 53–59 | Exercise 2 (loss, perplexity, bits per character) | CP2 |
| 59–61 | Run: the n-gram baseline, measured the same way | |
| 61–65 | Run: train the RNN language model | |
| 65–74 | Exercise 3 (vanishing gradients; clipping) | CP3 |
| 74–80 | Exercise 4 (the LSTM cell) | CP4 |
| 80–87 | Run: train the LSTM; the results table | LSTM below the trigram |
| 87–95 | Exercise 5 (sampling with temperature) | CP5 |

**Behind at minute 61:** set `QUICK = True` before the training runs (a quarter of the steps; not the recorded baselines). Lecture 3 gives its timing in prose, not a table; the rows above follow it.

### Module 4 · {{< var modules.m04.title >}}

| Minutes | Segment | By the end |
|---|---|---|
| 0–5 | Lecture 1. From next-token prediction to transduction | |
| 5–13 | 2. The encoder-decoder model | |
| 13–18 | 3. The fixed-vector bottleneck | |
| 18–30 | 4. Attention | |
| 30–35 | 5. Three score functions | |
| 35–39 | 6. Attention as soft alignment | |
| 39–42 | 7. Decoding | |
| 42–45 | 8. Attention as a query-key-value lookup | |
| 45–48 | Lab: setup; read the generated dates | |
| 48–58 | Exercise 1 (teacher-forced forward pass and loss) | CP1 |
| 58–66 | Exercise 2 (exact match by input length); predict which bucket fails | CP2 |
| 66–78 | Exercise 3 (dot-product attention); accuracy with and without attention | CP3 |
| 78–88 | Exercise 4 (the additive score) | CP4 |
| 88–95 | Exercise 5 (heat-maps, alignment hit rate) | CP5 |

**Behind at minute 66:** keep Exercises 3 and 4 (the objectives need both scores); shorten Exercise 5 to viewing the heat-maps. Training runs start while participants write their next prediction.

## Day 2

### Module 5 · {{< var modules.m05.title >}}

| Minutes | Segment | By the end |
|---|---|---|
| 0–5 | Lecture 1. From attention over an encoder to self-attention | |
| 5–14 | 2. Scaled dot-product attention | |
| 14–19 | 3. The causal mask | |
| 19–25 | 4. Multi-head attention | |
| 25–30 | 5. Positional encodings | |
| 30–37 | 6. The transformer block | |
| 37–41 | 7. A decoder-only language model, and the other two variants | |
| 41–45 | 8. Cost and parallelism | |
| 45–48 | Lab: setup; Exercise 0 (read the model) | |
| 48–60 | Exercise 1 (scaled dot-product attention); predict the score spread | CP1 |
| 60–68 | Exercise 2 (the causal mask; leak test) | CP2 |
| 68–74 | Exercise 3 (positions); whole-model leak test | CP3, CP3b |
| 74–88 | Exercise 4 (train the mini-GPT and the LSTM; compare) | CP4a, CP4b |
| 88–95 | Exercise 5 (looking at the heads) | CP5 |

**Behind at minute 68:** set `QUICK = True` before Exercise 4 (on by default on a CPU runtime); shorten Exercise 5.

### Module 6 · {{< var modules.m06.title >}}

| Minutes | Segment | By the end |
|---|---|---|
| 0–4 | Lecture 1. What the from-scratch transformer leaves open | |
| 4–14 | 2. Subword tokenization: byte-pair encoding | |
| 14–18 | 3. Pretrain once, reuse many times | |
| 18–24 | 4. Causal language modeling: GPT | |
| 24–32 | 5. Masked language modeling: BERT | |
| 32–37 | 6. The Hugging Face stack | |
| 37–43 | 7. Fine-tuning an encoder for classification | |
| 43–45 | 8. What changes at scale, and what is still missing | |
| 45–48 | Lab: setup (start the downloads) | |
| 48–58 | Exercise 1 (byte-pair encoding by hand) | CP1 |
| 58–65 | Exercise 2 (train a tokenizer; tokens per word) | CP2 |
| 65–73 | Exercise 3 (causal LM perplexity) | CP3 |
| 73–83 | Exercise 4 (masked LM: select and corrupt) | CP4 |
| 83–95 | Exercise 5 (fine-tune an encoder); the results table | CP5a, CP5b, CP5c |

**Behind at minute 65:** run Exercise 2 as a demonstration; if still behind at minute 73, Exercise 3 too. Exercises 1, 4 and 5 may not be cut.

### Module 7 · {{< var modules.m07.title >}}

| Minutes | Segment | By the end |
|---|---|---|
| 0–4 | Lecture 1. A language model continues; it does not answer | |
| 4–10 | 2. Chat templates | |
| 10–16 | 3. The instruction-tuning loss | |
| 16–19 | 4. What full fine-tuning costs | |
| 19–31 | 5. LoRA: a low-rank update | |
| 31–37 | 6. LoRA in practice: where, how large, and `peft` | |
| 37–40 | 7. Decoding settings | |
| 40–45 | 8. Evaluating generated text | |
| 45–48 | Lab: setup; Step 0 (the base model continues) | |
| 48–58 | Exercise 1 (the LoRA layer) | CP1 |
| 58–63 | Exercise 2 (merging the update) | CP2 |
| 63–69 | Exercise 3 (the chat template) | CP3 |
| 69–77 | Exercise 4 (the response mask and labels) | CP4 |
| 77–87 | Exercise 5 (configure LoRA, count parameters; training starts) | CP5 |
| 87–95 | Exercise 6 (ROUGE-n; decoding settings) | CP6 |

**Behind at minute 77:** run Exercise 6's `SAMPLING` cell as a demonstration and keep `rouge_n`. The stretch is the first thing dropped on any day.

### Module 8 · {{< var modules.m08.title >}}

| Minutes | Segment | By the end |
|---|---|---|
| 0–3 | Lecture 1. From your own model to someone else's | |
| 3–10 | 2. The message list | |
| 10–14 | 3. Prompting patterns | |
| 14–22 | 4. Structured output: the schema as a contract | |
| 22–30 | 5. Tool calling: a loop you write | |
| 30–35 | 6. Cost and latency | |
| 35–39 | 7. Evaluating outputs | |
| 39–43 | 8. Failure modes and security | |
| 43–45 | 9. A fluent answer carries no confidence signal | |
| 45–48 | Lab: setup; read the provider cell | |
| 48–56 | Exercise 1 (parse a raw response) | CP1 |
| 56–68 | Exercise 2 (validate and retry) | CP2 |
| 68–76 | Exercise 3 (score the outputs); the evaluation run opens this exercise | CP3 |
| 76–88 | Exercise 4 (the tool loop) | CP4 |
| 88–93 | Exercise 5 (cost and latency) | CP5 |
| 93–95 | What the response did not tell you | |

**Behind at minute 68:** let the evaluation run finish while participants write Exercise 3, which is where the notebook places it; the checkpoints use the `FakeProvider` and do not wait for it.

## Day 3

### Module 9 · {{< var modules.m09.title >}}

| Minutes | Segment | By the end |
|---|---|---|
| 0–4 | Lecture 1. No label for "better" | |
| 4–10 | 2. The minimum reinforcement learning | |
| 10–19 | 3. The policy gradient and REINFORCE | |
| 19–26 | 4. Baselines and variance | |
| 26–31 | 5. Generation as sequential decisions | |
| 31–34 | 6. Why comparisons instead of scores | |
| 34–39 | 7. The Bradley–Terry model | |
| 39–45 | 8. Training a reward model | |
| 45–48 | Lab: setup; Exercise 0 (the toy environment, three pairs) | |
| 48–60 | Exercise 1 (returns and the REINFORCE loss) | CP1a, CP1b, CP1c |
| 60–69 | Exercise 2 (a baseline and its variance) | CP2a–CP2e |
| 69–77 | Exercise 3 (the Bradley–Terry rater) | CP3a, CP3b |
| 77–89 | Exercise 4 (reward-model loss and training) | CP4a, CP4b |
| 89–95 | Exercise 5 (normalize and save) | CP5 |

**Behind at minute 69:** run Exercise 2's five-seed training as a demonstration. Part B (69–95) needs the Lab 9 data files; see the facilitator guide if they are missing.

### Module 10 · {{< var modules.m10.title >}}

| Minutes | Segment | By the end |
|---|---|---|
| 0–3 | Lecture 1. What Module 9 left open | |
| 3–10 | 2. The InstructGPT pipeline | |
| 10–18 | 3. The KL-regularized objective | |
| 18–24 | 4. A per-token penalty and the policy gradient | |
| 24–28 | 5. PPO in outline | |
| 28–38 | 6. DPO: from the optimal policy to a classification loss | |
| 38–45 | 7. Failure modes | |
| 45–48 | Lab: setup; Step 0 (reward model and gold rule on the reference's samples) | |
| 48–56 | Exercise 1 (log-probabilities of a response) | CP1 |
| 56–64 | Exercise 2 (per-token reward with a KL penalty) | CP2 |
| 64–75 | Exercise 3 (exact KL; RLHF with the penalty) | CP3 |
| 75–83 | Exercise 4 (remove the penalty: reward hacking) | CP4 |
| 83–95 | Exercise 5 (the DPO loss; DPO training); results card | CP5 |

**Behind at minute 64:** do not cut a training run; ask the Predict questions while training runs (estimated 1.5 minutes per run on a T4, unmeasured).

### Module 11 · {{< var modules.m11.title >}}

| Minutes | Segment | By the end |
|---|---|---|
| 0–3 | Lecture 1. What Day 3 has left open | |
| 3–8 | 2. What a probability should mean | |
| 8–16 | 3. Measuring calibration: reliability diagrams and ECE | |
| 16–24 | 4. Proper scoring rules | |
| 24–29 | 5. Why models are miscalibrated | |
| 29–34 | 6. Temperature scaling | |
| 34–39 | 7. Confidence from a language model behind an API | |
| 39–45 | 8. Selective prediction and the cost of a wrong action | |
| 45–48 | Lab: setup and Exercise 0 (which classifier, which provider) | |
| 48–58 | Exercise 1 (reliability bins and ECE) | CP1 |
| 58–65 | Exercise 2 (Brier score and log loss) | CP2 |
| 65–75 | Exercise 3 (temperature scaling) | CP3 |
| 75–87 | Exercise 4 (parse a stated confidence); start `ask_all` first | CP4 |
| 87–95 | Exercise 5 (risk–coverage and the threshold), ending with the closing question: what would you let act alone? | CP5 |

**Timing:** the closing question is now part of Exercise 5's 8 minutes. If the room is late, carry it over the break and open Module 12 with it. **Behind at minute 75:** start `ask_all` before discussing Exercise 3's results.

### Module 12 · {{< var modules.m12.title >}}

| Minutes | Segment | By the end |
|---|---|---|
| 0–3 | Lecture 1. What Module 11 left open | |
| 3–9 | 2. What is public, what is reported, what is ours | |
| 9–19 | 3. Two training signals: preference and outcome | |
| 19–22 | 4. System 1 and System 2 | |
| 22–29 | 5. Jev's interface | |
| 29–33 | 6. Probabilities, not `confidence`, on a reliability diagram | |
| 33–37 | 7. Where a decision model fits: route, guard, verify | |
| 37–45 | 8. Act, ask, escalate: thresholds from costs | |
| 45–48 | Lab: setup and Exercise 0 (look at the decisions) | |
| 48–62 | Exercise 1 (accuracy reward and Brier reward; our illustration) | CP1 |
| 62–72 | Exercise 2 (typed questions, typed answers); `decide_all` starts first | CP2 |
| 72–80 | Exercise 3 (calibration arrays: $\hat{p}$ against `confidence`) | CP3 |
| 80–90 | Exercise 4 (act, ask, escalate) | CP4 |
| 90–95 | What this lab showed and what it did not | |

**Behind at minute 72:** shorten the Exercise 3 discussion. Never cut the closing cell (90–95): it carries the honesty rule.

## Day 4

### Module 13 · {{< var modules.m13.title >}}

| Minutes | Segment | By the end |
|---|---|---|
| 0–4 | Lecture 1. Why retrieve | |
| 4–8 | 2. The pipeline | |
| 8–12 | 3. Chunking | |
| 12–18 | 4. Dense retrieval | |
| 18–22 | 5. Sparse and hybrid retrieval | |
| 22–28 | 6. Reranking: cross-encoders and a decision model | |
| 28–31 | 7. Generation: sources, citations, abstention | |
| 31–36 | 8. Evaluating retrieval | |
| 36–40 | 9. Evaluating answers | |
| 40–42 | 10. Common failures | |
| 42–45 | 11. LlamaIndex and LangChain | |
| 45–49 | Lab: setup and Exercise 0 (the path banner, corpus, three questions) | |
| 49–57 | Exercise 1 (recall@k and reciprocal rank) | CP1 |
| 57–66 | Exercise 2 (build the index; choose $L$ and $k$); the sweep starts first | CP2 |
| 66–74 | Exercise 3 (the same retriever in LangChain) | CP3 |
| 74–83 | Exercise 4 (rerank the candidates) | CP4 |
| 83–93 | Exercise 5 (faithfulness); answers generated | CP5 |
| 93–95 | What this lab showed and what it did not | |

The notebook heads Setup "(4 minutes)" and Exercise 0 "(4 minutes)"; the brief counts them as one 4-minute block, as here. **Behind at minute 74:** on the keyed Jev path, rerank `test` only.

### Module 14 · {{< var modules.m14.title >}}

| Minutes | Segment | By the end |
|---|---|---|
| 0–5 | Lecture 1. What the Module 8 loop cannot do: the harness | |
| 5–9 | 2. ReAct: reasoning and acting in one loop | |
| 9–13 | 3. Tools: LangChain tools, runnables and tool design | |
| 13–21 | 4. LangGraph: state, nodes, edges, conditional routing | |
| 21–26 | 5. Checkpoints, memory, interrupts and replay | |
| 26–32 | 6. Where agents fail | |
| 32–37 | 7. A decision model in the control loop | |
| 37–40 | 8. Measuring an agent | |
| 40–45 | 9. Designing the harness: patterns, subagents and context | |
| 45–48 | Lab: setup and Exercise 0 (read the graph) | |
| 48–54 | Exercise 1 (a safe calculator) | CP1 |
| 54–66 | Exercise 2 (thresholds as edges) | CP2 |
| 66–74 | Exercise 3 (the human-in-the-loop driver) | CP3 |
| 74–81 | Exercise 4 (fork and replay); the evaluation run starts first | CP4 |
| 81–91 | Exercise 5 (measure the agent) | CP5 |
| 91–95 | What this lab showed and what it did not | |

The notebook heads Setup "(3 minutes)" and Exercise 0 "(3 minutes)"; the brief counts them as one block, as here. Lecture 14 is the densest lecture in the workshop, and its timings are checked on paper only. **Lecture behind (section 7 not started by minute 32):** give section 9 as reading, as the lecture's timing note says; that recovers its 5 minutes, and the core lab does not depend on it. **Behind at minute 74:** on a CPU runtime, cut the probability-shift test to 30 items (slice the list in the evaluation-run cell).

### Module 15 · {{< var modules.m15.title >}}

{{< var modules.m15.minutes >}} minutes across the afternoon. The capstone notebook does not exist yet; this plan follows lecture 15.

| Minutes | Segment | By the end |
|---|---|---|
| 0–10 | Part I. The brief (lecture sections 1–4) | pairs formed; path class chosen |
| 10–30 | Build: setup, self-test, write `after_verify`, baseline on `dev` then `test`, read ten traces | self-test green; both baselines run |
| 30–40 | Build: choose one component; fill in the hypothesis card | hypothesis card filled |
| 40–90 | Build: make the change; rerun the self-test after every edit; iterate on `dev` | self-test still green |
| 90–95 | Build: freeze the configuration; do not run `test` | configuration frozen |
| | Break | |
| 0–20 (after the break) | Part III. Run `test` on the frozen system; run the last cell; hand in the file | submission handed in |
| 20–55 | Two-minute shares, in menu order | |
| 55–65 | The combined table, read with the room | |
| 65–75 | Part IV. Wrap-up: section 7, four days, one line of ideas | |
| 75–83 | Section 8, an evaluation checklist for agentic systems | |
| 83–90 | Section 9, open problems | |
| 90–95 | Section 10, further study | |

**Behind at build minute 30 (module minute 40):** any pair without a `dev` baseline gets help before anyone else. **More than 15 pairs:** group the shares by component, four minutes per group.
