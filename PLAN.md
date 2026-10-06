# PLAN.md — From Traditional NLP to Modern LLMs

A 4-day intensive workshop by Genial Labs. This file is the master plan: curriculum, repository design, lab standards, and the build checklist. Agent personas for the build are in [AGENTS.md](AGENTS.md).

**Status (2026-10-05):** All 15 lectures and lab briefs drafted; all 16 notebooks written. Labs 01–05 run end to end on CPU; Labs 06–15 run only on their offline paths (stand-ins, stubs, the local toy decider), because the build container has no Hub access and no API keys. Human work outstanding: the decision-set hand items and audit, Lab 13's 80 questions and the capstone's 45 (after the corpus snapshot is frozen), and TypeSafe's own statements in lecture 12. Scaffold built (build Days 1–2); datasets fixed; lectures 1–8 drafted and each reviewed against its lab (equation-to-lab maps, quoted numbers, signatures; "As built" sections in `briefs/`); figures for lectures 1–8 done, with the Lab 2 PCA, Lab 4 alignment and Lab 5 causal-mask figures drawn from measured data. Labs 01–05 built and run end to end on a shared CPU (never on Colab or a T4). Labs 06–08 built; their pretrained-model and keyed paths are written, not run, because the build container cannot reach huggingface.co and has no API keys (offline stand-ins and a stub provider exercise the code). Open before Day 7: a Colab T4 run of Labs 02–08 for run times and the Lab 6/7/8 real paths; the Lab 6 logits file for Lab 11 and the Lab 7 Dolly subset file both need Hub access. Not rendered: Quarto is not installed in the build container. Module 0 (coding agents in the terminal; optional, self-serve, Day 1 08:00–09:00) was added on 2026-10-05: it is wired into `_variables.yml`, the generators, tests and pages; its page, four reference apps (Python and R, run here with headless renders checked) and data are in place; not yet tried on a real laptop. This plan is a living document and will be revised as the build proceeds.

---

## 1. Overview

| | |
|---|---|
| **Title** | From Traditional NLP to Modern LLMs: n-grams, attention, RLHF, RLCD and agents |
| **Length** | 4 days, 09:00–17:00, 15 modules (the capstone takes a double slot), plus an optional self-serve Module 0 on Day 1, 08:00–09:00 |
| **Module shape** | 95 minutes: about 45 min lecture, 50 min Colab lab. Module 0 is 60 minutes, self-serve, with no lecture and no notebook |
| **Audience** | ML practitioners: comfortable with Python, NumPy and basic ML, some PyTorch |
| **Site** | Quarto website, deployed to GitHub Pages |
| **Labs** | Google Colab notebooks, free-tier T4 runtime |
| **Structural model** | [`project-delphi/tensors-workshop`](https://github.com/project-delphi/tensors-workshop/) |
| **Licence** | CC BY 4.0 for teaching content, MIT for code |

### The four days

| Day | Theme | Question it answers |
|---|---|---|
| 1 | Foundations: from counts to attention | How do we turn text into something a model can learn from? |
| 2 | Transformers and LLMs | How does one architecture, pretrained at scale, become a general tool? |
| 3 | Training objectives: preference and calibration | What are we optimising these models for, and what should we be? |
| 4 | RAG, agents and capstone | How do we build reliable systems out of these models? |

### Prerequisites

- Python: functions, classes, comprehensions, reading a traceback.
- NumPy: broadcasting, indexing, matrix products.
- ML basics: train/validation/test splits, loss functions, gradient descent, logistic regression.
- PyTorch: has trained at least one model with `nn.Module` and an optimizer loop. Module 2 includes a short refresher.
- Maths: vectors, matrices, the chain rule, basic probability (conditional probability, expectation, cross-entropy).
- A Google account for Colab. API keys for OpenAI, Anthropic and TypeSafe are optional: every API lab has a free open-model path.

### Learning outcomes

By the end of the workshop a participant can:

1. Explain the line of ideas from count-based language models to transformers, and say what problem each step solved.
2. Implement the core of each model in PyTorch: skip-gram with negative sampling, an LSTM language model, seq2seq with attention, and a small GPT.
3. Use the Hugging Face stack to tokenise, load, fine-tune and evaluate pretrained models, including LoRA.
4. Use the OpenAI and Claude APIs for prompting, structured output and tool use, and compare them on the same task.
5. Explain RLHF end to end (preference data, reward model, policy optimisation, DPO), train a toy version, and name its failure modes.
6. Measure calibration (reliability diagrams, ECE, proper scoring rules), explain how RLCD's objective differs from RLHF's, and use a calibrated decision model (Jev) through its API.
7. Build and evaluate a RAG pipeline with LlamaIndex and LangChain.
8. Build a LangGraph agent with tools, state, human-in-the-loop checkpoints and confidence-gated control using Jev.
9. Combine these into one system, evaluate it, and report accuracy, abstention and cost.

### Pedagogical principles

Drawn from Stanford CS224N, CMU CS 11-747 and MIT 6.S191:

- **Derive, implement, then use the library.** Each idea is first motivated by the failure of the previous one, then derived briefly, then built, and only then used through a library (CS224N).
- **Code-first neural modelling.** Lectures show the model as code alongside the equations; every equation in a lecture maps to a named line in the lab (CMU 11-747).
- **Short lecture, immediate lab.** No lecture runs longer than 45 minutes before hands-on work (MIT 6.S191).
- **Predict → Run → Explain → Check.** The lab rhythm carried over from `tensors-workshop`: participants predict an output, run the cell, explain the result, then pass a checkpoint assertion.
- **One running thread.** The same small datasets and the same tasks reappear across modules, so improvements are measured, not asserted.
- **Retrieve and manipulate before the lab.** Each lecture opens with a recap box, closes most sections with a check-yourself question (answer folded), carries its derivations through a worked numeric example, and has at most one interactive demo. These sit outside the 45 minutes. Authoring hooks: `.recap`, `.self-check`, `.worked-example`, `.demo` (styled in `custom.scss`; `filters/pedagogy.lua` styles the per-section objective lines).
- **Honesty about what is known.** Where a method is unpublished (RLCD), the material says so and separates public facts from our own illustration.

---

## 2. Repository structure

The core pattern is copied from `tensors-workshop`; the layout is adapted from one 210-minute session to four days.

```
nlp-llms/
├── _quarto.yml              website project, output-dir: docs, explicit render: list
├── _variables.yml           single source of truth: repo URLs, colab_base, model IDs, modules
├── _includes/               GENERATED tables (schedule, notebook index, dependencies)
├── index.qmd                landing page: hero, prerequisites, resource cards
├── setup.qmd                Colab, API keys via Colab Secrets, open-model fallback
├── schedule.qmd             four-day timetable (generated table)
├── day-1.qmd … day-4.qmd    day index pages
├── lectures/                one page per module: 00-coding-agents.qmd … 15-capstone.qmd
├── notebooks.qmd            notebook index with Colab badges (generated table)
├── notebooks/               00-setup.ipynb, 01-… to 15-….ipynb (no outputs committed; Module 0 has none)
├── agents-intro/            Module 0 reference solutions (the two apps)
├── references.qmd           papers, courses, docs
├── faq.qmd
├── teach.qmd                instructor hub
├── facilitator-guide.md  instructor-pace.md  assessments.md
├── custom.scss              cosmo override; Inter body, Source Serif 4 headings (custom-dark.scss: dark theme tokens)
├── fonts/  images/  data/   vendored fonts, figures, fallback dataset copies
├── scripts/                 gen_tables.py, gen_notebooks.py, new_notebook.py, test_notebooks.py, check_links.py
├── tests/
├── pyproject.toml           uv dependency groups: notebooks, site, test, lint, execute
├── .github/workflows/       publish.yml (render + deploy), health.yml (scheduled notebook run)
├── .claude/agents/          the AGENTS.md personas as invocable subagents
├── AGENTS.md  PLAN.md  README.md  CONTRIBUTING.md  CHANGELOG.md
├── CITATION.cff  LICENSE
└── .gitignore               ignores docs/, .venv/, uv.lock
```

### Conventions copied from `tensors-workshop`

- **One ID per module.** `NN-kebab-slug` is the same for the lecture page, the notebook and the `_variables.yml` key (`m00` … `m15`). Module 0 has `notebook: false`: a lecture page but no notebook.
- **`_variables.yml` is the single source of truth.** Each module entry holds `n`, `slug`, `day`, `minutes`, `title`, `summary`, `objectives`, `stack`. Pages read it with `{{< var modules.m01.title >}}`; the generator scripts read it too. Model IDs and package pins also live here, so a version bump is a one-line change.
- **Generated files are never edited by hand.** `scripts/gen_tables.py` writes the tables in `_includes/` and the marked regions in `README.md`. `scripts/gen_notebooks.py` owns the first cell (title, Colab badge, time, objectives) and the last cell (next notebook, site link) of every notebook, strips outputs and execution counts, and is idempotent. CI fails if running the generators changes anything.
- **Notebooks are not executed at render time.** `_quarto.yml` lists pages explicitly under `render:` and ships `notebooks/*.ipynb` as `resources:`. There is no `_freeze/`.
- **Deploy from an Actions artifact.** `publish.yml` renders to `docs/`, which is gitignored, and deploys with `actions/deploy-pages`.
- **Navbar, plus a module sidebar on lecture pages.** Navbar: Home, Schedule, Days (dropdown: Module 0, then Day 1–4), Notebooks, Setup, References, Teach, FAQ. Inside `lectures/` a generated left sidebar (`_includes/sidebar.yml`) lists the modules by day, with previous/next module links at the foot of each page. (Changed 2026-10-05; `tensors-workshop` has no sidebar.)

### Deliberately left out of v1

Spanish translation of every page, 3D interactive widgets, Kahoot quizzes, the NotebookLM companion, Playwright navigation tests and revealjs slide decks. Each can be added after v1.0 without changing the structure above.

---

## 3. Timetable

| Time | Day 1: Foundations | Day 2: Transformers and LLMs | Day 3: Training objectives | Day 4: RAG, agents, capstone |
|---|---|---|---|---|
| 08:00–09:00 | 0 · Coding agents in the terminal (optional, self-serve) | | | |
| 09:00–09:10 | Welcome, setup check | Recap of Day 1 | Recap of Day 2 | Recap of Day 3 |
| 09:10–10:45 | 1 · Text as data | 5 · The transformer | 9 · Reinforcement and preference learning | 13 · Retrieval-augmented generation |
| 10:45–11:00 | Break | Break | Break | Break |
| 11:00–12:35 | 2 · Word vectors and neural nets | 6 · Pretraining and the Hugging Face stack | 10 · RLHF | 14 · Agents |
| 12:35–13:35 | Lunch | Lunch | Lunch | Lunch |
| 13:35–15:10 | 3 · Sequence models | 7 · Fine-tuning and LoRA | 11 · Calibration | 15 · Capstone: build |
| 15:10–15:25 | Break | Break | Break | Break |
| 15:25–17:00 | 4 · Seq2seq and attention | 8 · LLMs through APIs | 12 · RLCD and Jev | 15 · Capstone: evaluate and share (to 16:30), then wrap-up |

---

## 4. Curriculum

Each module lists objectives, the lecture outline, the lab, and key readings. Lab names are the notebook file names under `notebooks/`. Every lab has a core path that fits 50 minutes and one optional stretch section (see section 5).

### Day 1 — Foundations: from counts to attention

#### Module 0 · Coding agents in the terminal (optional, self-serve, 08:00–09:00)

- **Objectives:** install and drive a terminal coding agent; build and check two small data apps with it; publish them with GitHub and GitHub Pages.
- **Format:** 60 minutes before the 09:00 welcome, also usable as pre-work. No lecture and no notebook: participants follow the page on their own laptops while facilitators help with installs. Nothing later depends on it.
- **Page:** `lectures/00-coding-agents.qmd`.
- **What participants do:** install one coding agent (Claude Code, Codex or Gemini CLI); set up git and the GitHub CLI; in one language of their choice (Python or R), have the agent build (1) a protein structure explorer for ubiquitin (PDB 1UBQ) and (2) an RFM customer segmentation, each with a three.js page; check each app; publish both on GitHub Pages.
- **Reference solutions:** `agents-intro/`, with data files recorded as `datasets` entries in `_variables.yml`.
- **Stack:** a coding agent CLI, git, GitHub CLI, Python or R, three.js. npm package names and the versions checked on 2026-10-05 are in `_variables.yml` under `agents_intro`.
- **Instructor notes:** room setup, install failures per OS and the no-subscription fallback are in `facilitator-guide.md`; a provisional minute plan is in `instructor-pace.md`.

#### Module 1 · Text as data

- **Objectives:** tokenise text and justify the choices; build and evaluate an n-gram language model; train a linear text classifier and read its errors.
- **Lecture:** what makes language hard (ambiguity, sparsity, compositionality); tokenisation and normalisation; Zipf's law; n-gram language models, smoothing, perplexity; bag-of-words and TF-IDF; naive Bayes and logistic regression; evaluation (precision, recall, F1); where count-based methods stop working.
- **Lab `01-text-as-data.ipynb`:** build a tokeniser and vocabulary; implement a bigram and trigram LM with add-k smoothing and compute perplexity; sample text from it; TF-IDF + logistic regression classifier on the topic-classification set (arXiv Topics v1, see `data/README.md`); error analysis.
- **Stretch:** BM25 scoring (reused in Module 13).
- **Stack:** NumPy, scikit-learn.
- **Readings:** Jurafsky & Martin, *Speech and Language Processing* (3rd ed.), chapters on n-gram LMs and classification.

#### Module 2 · Word vectors and neural networks

- **Objectives:** explain the distributional hypothesis; derive the skip-gram objective with negative sampling; train embeddings and a feed-forward classifier in PyTorch.
- **Lecture:** one-hot vectors and their limits; distributional semantics; word2vec (skip-gram, CBOW), negative sampling, GloVe in brief; PyTorch refresher (tensors, autograd, `nn.Module`, the training loop); feed-forward networks and backpropagation; Bengio's neural LM as the bridge from n-grams.
- **Lab `02-word-vectors.ipynb`:** implement the skip-gram negative-sampling loss; train embeddings; nearest neighbours and analogies; plot embeddings; replace TF-IDF features from Lab 1 with averaged embeddings and compare.
- **Stretch:** compare with pretrained GloVe vectors.
- **Stack:** PyTorch.
- **Readings:** Mikolov et al. 2013 (word2vec); Pennington et al. 2014 (GloVe); Bengio et al. 2003.

#### Module 3 · Sequence models

- **Objectives:** implement an RNN and an LSTM language model; explain vanishing gradients and how gating addresses them; compare perplexity against the n-gram baseline.
- **Lecture:** recurrent networks and backpropagation through time; vanishing and exploding gradients, gradient clipping; LSTM and GRU gates; neural language modelling, teacher forcing; sampling strategies (greedy, temperature, top-k, nucleus).
- **Lab `03-sequence-models.ipynb`:** write an RNN cell by hand, then use `nn.LSTM`; train a character-level LM on a small corpus; measure perplexity against Lab 1's n-gram method at character level, on the same split; inspect gradient norms with and without clipping; generate text at several temperatures.
- **Stretch:** implement top-k and nucleus sampling.
- **Stack:** PyTorch.
- **Readings:** Hochreiter & Schmidhuber 1997 (LSTM); Karpathy, "The Unreasonable Effectiveness of RNNs".

#### Module 4 · Seq2seq and attention

- **Objectives:** build an encoder–decoder model; explain the fixed-vector bottleneck; implement additive and dot-product attention and read attention maps.
- **Lecture:** encoder–decoder architecture; the bottleneck problem; Bahdanau (additive) and Luong (multiplicative) attention; attention as soft alignment; beam search; attention as a general query–key–value lookup, setting up Day 2.
- **Lab `04-seq2seq-attention.ipynb`:** train a seq2seq model on a toy transduction task (human-readable dates to ISO format); observe it fail on long inputs; add attention; plot attention heat-maps; compare accuracy by input length.
- **Stretch:** beam search decoding.
- **Stack:** PyTorch.
- **Readings:** Sutskever et al. 2014; Bahdanau et al. 2015; Luong et al. 2015.

### Day 2 — Transformers and LLMs

#### Module 5 · The transformer

- **Objectives:** implement scaled dot-product and multi-head self-attention; assemble a decoder-only transformer; train a small GPT.
- **Lecture:** from attention over an encoder to self-attention; scaled dot-product attention and why the scaling; multi-head attention; positional encodings (sinusoidal, learned, rotary in brief); residual connections and layer norm; causal masking; encoder, decoder and encoder–decoder variants; cost and parallelism compared with RNNs.
- **Lab `05-transformer-from-scratch.ipynb`:** implement self-attention with a causal mask (the block scaffold and training loop are provided); train a mini-GPT on the Lab 3 corpus; compare loss and samples with the LSTM; visualise attention heads.
- **Stretch:** write the full transformer block and multi-head split yourself.
- **Stack:** PyTorch.
- **Readings:** Vaswani et al. 2017; "The Annotated Transformer".

#### Module 6 · Pretraining and the Hugging Face stack

- **Objectives:** explain subword tokenisation and the masked and causal pretraining objectives; load, inspect and run pretrained models with Hugging Face; fine-tune an encoder for classification.
- **Lecture:** subword tokenisation (BPE, WordPiece); BERT and masked LM, GPT and causal LM, T5 in brief; the transfer-learning recipe; what changes at scale (data, compute, emergent abilities); the Hugging Face ecosystem (`transformers`, `datasets`, `tokenizers`, the Hub, model cards).
- **Lab `06-pretraining-huggingface.ipynb`:** train a small BPE tokeniser and compare with a pretrained one; probe a masked LM and a causal LM; fine-tune a small encoder on the Lab 1 classification data and compare with Labs 1 and 2.
- **Stretch:** inspect attention and hidden states of the pretrained model.
- **Stack:** Hugging Face `transformers`, `datasets`, `tokenizers`; PyTorch.
- **Readings:** Devlin et al. 2019 (BERT); Radford et al. 2018 (GPT); Sennrich et al. 2016 (BPE).

#### Module 7 · Fine-tuning and LoRA

- **Objectives:** turn a pretrained causal LM into an instruction follower; apply LoRA and explain why it works; choose decoding settings; evaluate generation.
- **Lecture:** from language model to assistant: instruction tuning and chat templates; full fine-tuning versus parameter-efficient methods; LoRA: low-rank updates, rank and alpha, where to apply them; quantisation in brief; decoding for generation; evaluating generated text (exact match, overlap metrics, model-graded evaluation and its limits).
- **Lab `07-finetuning-lora.ipynb`:** implement a LoRA layer by hand on one linear module; then use `peft` to fine-tune a small causal LM on an instruction dataset; count trainable parameters; compare outputs before and after; apply a chat template.
- **Stretch:** sweep the LoRA rank and plot quality against trainable parameters.
- **Stack:** Hugging Face `transformers`, `peft`, `datasets`; PyTorch.
- **Readings:** Hu et al. 2021 (LoRA); Hugging Face PEFT documentation.

#### Module 8 · LLMs through APIs

- **Objectives:** call OpenAI and Claude models for chat, structured output and tool use; compare them on one task with one harness; reason about cost, latency and failure modes.
- **Lecture:** the commercial API surface: messages, system prompts, tokens and context windows; prompting patterns (few-shot, reasoning before answering); structured output and JSON schemas; tool/function calling; cost and latency; evaluation basics for LLM outputs; hallucination and why a fluent answer carries no confidence signal (setting up Day 3).
- **Lab `08-llm-apis.ipynb`:** a thin provider-agnostic wrapper over OpenAI, Anthropic and a local Hugging Face model; the same extraction task with schema-validated output on each; a two-tool calling loop written by hand; a small evaluation set scored automatically; a cost and latency table.
- **Stretch:** add the Lab 7 fine-tuned model as a fourth provider.
- **Stack:** `openai`, `anthropic`, Hugging Face (fallback), Pydantic.
- **Readings:** OpenAI and Anthropic API documentation (tool use, structured outputs).

### Day 3 — Training objectives: preference and calibration

#### Module 9 · Reinforcement and preference learning

- **Objectives:** frame text generation as a reinforcement-learning problem; derive and implement the policy gradient; train a reward model from pairwise preferences.
- **Lecture:** why supervised fine-tuning is not enough: no label for "better"; the minimum RL needed: policy, reward, return, the policy-gradient theorem, REINFORCE, baselines and variance; generation as sequential decisions; preference data: why comparisons instead of scores; the Bradley–Terry model; training a reward model.
- **Lab `09-preference-learning.ipynb`:** REINFORCE on a small sequence task where the optimal policy is known; add a baseline and watch variance fall; load a synthetic pairwise-preference dataset with a known hidden preference, sampled from Lab 10's reference policy (the small GPT-2 of Lab 6) and labeled through the Bradley–Terry model; train a reward model and check it recovers the hidden preference.
- **Stretch:** measure how reward-model accuracy degrades with noisy raters.
- **Stack:** PyTorch, Hugging Face.
- **Readings:** Sutton & Barto, chapter 13 (policy gradients); Christiano et al. 2017.

#### Module 10 · RLHF

- **Objectives:** describe the three-stage RLHF pipeline; optimise a small LM against a reward model with a KL constraint; apply DPO; name RLHF's failure modes and observe one.
- **Lecture:** the InstructGPT pipeline: supervised fine-tuning, reward model, policy optimisation; PPO in outline; KL regularisation toward the reference policy and why it matters; DPO as preference optimisation without an RL loop; failure modes: reward hacking, sycophancy, mode collapse, and optimising for what raters prefer rather than what is true.
- **Lab `10-rlhf.ipynb`:** fine-tune the small GPT-2 of Lab 6 (`models.causal_lm`) against the Lab 9 reward model with a KL-penalised policy-gradient step (a pre-trained reward model checkpoint is provided); measure reward gain and drift from the reference model; remove the KL penalty and observe reward hacking; train the same preference data with a DPO loss and compare.
- **Stretch:** vary the KL coefficient and plot the reward–drift trade-off.
- **Stack:** PyTorch, Hugging Face.
- **Readings:** Ouyang et al. 2022 (InstructGPT); Schulman et al. 2017 (PPO); Rafailov et al. 2023 (DPO).

#### Module 11 · Calibration

- **Objectives:** say what a probability should mean; measure calibration; explain proper scoring rules; use confidence to decide when to abstain.
- **Lecture:** calibration versus accuracy; reliability diagrams, ECE and its pitfalls, Brier score, log loss; proper scoring rules and why they reward honest probabilities; why modern neural nets and preference-tuned LLMs are often miscalibrated; post-hoc fixes (temperature scaling); verbalised confidence from LLMs; selective prediction: risk–coverage curves and the cost of a wrong action.
- **Lab `11-calibration.ipynb`:** plot a reliability diagram and compute ECE and Brier score for the Lab 6 classifier (falling back to the Lab 1 classifier, recomputed in the notebook, until the Lab 6 logits are committed); apply temperature scaling; ask an LLM for verbalised confidence on the shared decision set (`data/decisions_v1.jsonl.gz`, specified in `briefs/11-calibration.md`) and measure its calibration; draw a risk–coverage curve and choose an abstention threshold.
- **Stretch:** show numerically that the Brier score is proper and that accuracy is not.
- **Stack:** PyTorch, scikit-learn, OpenAI/Claude (fallback: local model).
- **Readings:** Guo et al. 2017 (calibration of modern neural networks); Gneiting & Raftery 2007 (proper scoring rules).

#### Module 12 · RLCD and Jev

- **Objectives:** state how RLCD's objective differs from RLHF's and what is and is not public about it; call Jev for typed decisions; use its confidence to set action thresholds.
- **Lecture:** RLHF versus RLCD: preference as the reward versus agreement with outcomes as the reward; what TypeSafe has stated publicly and what remains unpublished; System 1 (fast, typed decisions) versus System 2 (generative reasoning); Jev's interface: state plus typed questions in, typed answers (choice, score, yes/no) with confidence out; where a decision model fits in an LLM system: routing, guarding, verifying; turning confidence into policy: act, ask, escalate.
- **Lab `12-rlcd-jev.ipynb`:**
  1. Toy calibration-reward training: fine-tune a small decision model with a proper-scoring-rule reward and compare against an accuracy-only reward. **Labelled in the notebook as our illustration of the idea, not TypeSafe's method.**
  2. Call Jev on the Lab 11 decision set with typed questions.
  3. Plot Jev's reliability diagram beside the LLM's from Lab 11.
  4. Choose act / ask / escalate thresholds from a stated cost of error.
- **Stretch:** compare cost and latency of Jev against an LLM on the same decisions.
- **Stack:** PyTorch, `typesafe-sdk` (fallback: the toy model from step 1).
- **Readings:** TypeSafe's public RLCD and Jev announcement and API documentation; Kahneman on System 1 and System 2 for the framing.

### Day 4 — RAG, agents and capstone

#### Module 13 · Retrieval-augmented generation

- **Objectives:** build a RAG pipeline; choose chunking, embedding and reranking settings from measurements; evaluate retrieval and answer quality separately.
- **Lecture:** why retrieval (freshness, grounding, cost); the pipeline: load, chunk, embed, index, retrieve, rerank, generate; dense, sparse (BM25, linking back to Module 1) and hybrid retrieval; rerankers; evaluation: recall@k, MRR, faithfulness, answer relevance; common failures; LlamaIndex and LangChain: what each abstracts and where they overlap.
- **Lab `13-rag.ipynb`:** index a small document set (Workshop Lectures v1: a frozen snapshot of our own lectures, `data/workshop_lectures_v1.jsonl.gz`) with LlamaIndex; query it; vary chunk size and top-k and measure recall@k on a hand-labelled question set (`data/rag_questions_v1.jsonl`, 80 questions written and checked by people, spec in `briefs/13-rag.md`); build the same retriever as a LangChain runnable; add a reranking step (a Jev reranker written in the notebook on `typesafe-sdk`, since no official LlamaIndex integration exists; a cross-encoder is the fallback and the CI path); score faithfulness.
- **Stretch:** hybrid retrieval with BM25.
- **Stack:** LlamaIndex, LangChain, Jev, OpenAI/Claude (fallback: local embedding model and small local LLM).
- **Readings:** Lewis et al. 2020 (RAG); LlamaIndex and LangChain documentation.

#### Module 14 · Agents

- **Objectives:** build a tool-using agent as an explicit graph; add state, memory and human-in-the-loop interrupts; use a calibrated decision model for routing and tool-call approval.
- **Lecture:** from the hand-written tool loop of Module 8 to agents; the agent harness (agent = model + harness; not to be confused with an evaluation or test harness), with ARC-AGI's same-model, different-harness results as a worked example; ReAct; LangChain tools and runnables; tool design (descriptions, few tools, short results, actionable errors); LangGraph: nodes, edges, state, conditional routing, checkpoints, interrupts, resume or start fresh; where agents fail (loops, wrong tool, unsafe action, prompt injection), stopping on a final answer, and enforcement in code; using a System 1 model in the control loop: route, guard, verify, with thresholds from Module 12, plus escalation triggers and hand-offs; designing the harness: workflow patterns (Anthropic) and agentic design patterns (Ng), coordinators and subagents, context as a budget.
- **Lab `14-agents.ipynb`:** define tools (calculator, the Module 13 retriever, a mock "send email" action); build a ReAct-style LangGraph agent; add a Jev router node (`langchain-typesafe`) that picks the next step with a probability; gate the risky tool with an act / ask / escalate guard from Module 12's thresholds (a simulated human answers interrupts in unattended runs); replay from a checkpoint; test against a prompt-injection document.
- **Stretch (one section, four parts; pick one):** (A) a verification node that checks the final answer against the retrieved sources; (B) a research subagent with its own context, failures returned as results, and a coverage check at the coordinator; (C) compaction that keeps the facts word for word; (D) a hand-off record that stands alone.
- **Stack:** LangChain, LangGraph, Jev, OpenAI/Claude (fallback: local model and the Module 12 toy decision model).
- **Readings:** Yao et al. 2023 (ReAct); LangGraph documentation; Schluntz and Zhang 2024 (Building effective agents); Greshake et al. 2023; Beurer-Kellner et al. 2025.

#### Module 15 · Capstone (double slot)

- **Objectives:** combine retrieval, an agent graph and calibrated control into one system; evaluate it; explain the design choices to others.
- **Brief (10 min):** the task, the starter system, the evaluation set and the scoring.
- **Build (85 min) `15-capstone.ipynb`:** a research-assistant agent over the workshop's own reading list. The starter provides LlamaIndex retrieval, a LangGraph plan–retrieve–answer–verify loop, Claude or OpenAI for generation, and Jev for routing and for a final "is this answer supported by the sources?" check with a confidence threshold. Participants work in pairs, run the fixed evaluation set for a baseline, then improve one component of their choice (retrieval, prompts, routing, thresholds, a new tool).
- **Evaluate and share (65 min):** re-run the evaluation; report accuracy, abstention rate and cost against the baseline; each pair gives a two-minute account of what they changed and what happened.
- **Wrap-up (30 min):** the four days as one line of ideas; an evaluation checklist for agentic systems; open problems; further study.
- **Stack:** everything from Days 3 and 4.

---

## 5. Lab standards

Every notebook must meet all of these.

- **Runs cold on free Colab.** A fresh T4 (or CPU where stated) runtime, Run all, no manual steps other than adding optional API keys. Target: under 10 minutes of compute per lab.
- **A core path that fits 50 minutes.** From-scratch labs ask participants to write the core function (the loss, the attention step, the update rule), not the whole model; scaffolding, data loading and training loops are provided.
- **One optional stretch section.** Clearly marked, placed last, never required by a later lab. It gives fast participants more to do and a slow group something to skip.
- **Generated header and footer.** Cell 0 (title, Colab badge, duration, objectives) and the final cell are written by `scripts/gen_notebooks.py`. Do not edit them by hand.
- **Setup cell.** Quiet, pinned `%pip install -q package==x.y.z` for anything Colab does not preinstall. Seeds are set here.
- **Exercises and solutions in one notebook.** Each exercise is a `# TODO N` stub, followed by a folded solution cell (`#@title Solution`, form view, source hidden) and a short "why this works" note.
- **Checkpoints.** Each exercise ends with an assertion or a printed metric that tells the participant whether they got it right.
- **API keys.** Read from Colab Secrets (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and the TypeSafe key name given in its documentation). Keys are never written into a cell.
- **Open-model fallback.** A single `PROVIDER` switch at the top of each API lab. With no keys set, the lab runs on a small Hugging Face model (and, for Jev, the toy decision model from Lab 12). The fallback path is the one CI executes.
- **Cost note.** Each API lab states its approximate cost per full run.
- **No outputs committed.** The generator strips outputs and execution counts.
- **Datasets.** Small, permissively licensed, fetched by URL with a fallback copy under `data/`.

---

## 6. Risks and open items

| Item | Risk | Mitigation |
|---|---|---|
| RLCD is unpublished | No paper, reward function or reliability data from TypeSafe; teaching it as fact would be speculation | Module 11 teaches calibration theory on its own footing; Module 12 states what is public and labels the toy lab as our own illustration |
| Jev package names and API surface | Checked 2026-10-05 against the published packages and TypeSafe's and LangChain's GitHub repositories (details and sources in `briefs/jev-verification.md`). `docs.typesafe.ai` was blocked from the build container and has not been read. **SDK:** PyPI `typesafe-sdk` 0.7.2 (MIT, by TypeSafe AI), import `typesafe_sdk`; `typesafe-sdk-python` is only the repository name and is not on PyPI. `TypeSafeClient` / `AsyncTypeSafeClient`, key from `TYPESAFE_API_KEY`; `system_one(state, questions)` sends `POST /v1/systemone` with `Noul` / `Choice` / `Score` questions. `Choice` and `Score` answers carry `probabilities` and `confidence`; `Noul` carries only a probability, with no confidence field. There is no batch endpoint. **LangChain:** `langchain-typesafe` 0.0.1a3 (alpha; the class is `@beta`), `TypeSafeClassifier` confirmed; it does not depend on `typesafe-sdk`. **LlamaIndex:** there is no official integration: `llama-index-jev` does not exist, and `llama_index` main has no TypeSafe code | Use `typesafe-sdk==0.7.2` and `langchain-typesafe==0.0.1a3`. Write the Module 13 reranker in the notebook on the SDK, with a cross-encoder fallback. With no key, a local backend returns the same `SystemOneResponse` type. Re-read `docs.typesafe.ai` (confidence, limits, pricing) from a networked machine before delivery. Section 4 now names only `typesafe-sdk` and says there is no LlamaIndex integration (corrected 2026-10-05) |
| Lookalike Jev packages on PyPI | Unaffiliated packages sit on names participants may guess: `typesafe-ai` (a shim by a private individual), `jev` (no author), `typesafe-client` (a placeholder), `typesafe` (unrelated, 2010) and `llama-index-postprocessor-jev` (an individual's reranker). The names `typesafe-sdk-python` and `llama-index-jev` are unregistered and could be taken by anyone | Print only `typesafe-sdk` and `langchain-typesafe`, with exact pins; warn participants in `setup.qmd`; add a test that fails if a notebook or page installs any other TypeSafe-like name |
| Jev confidence semantics | `confidence` measures how concentrated the distribution is, not the probability of the chosen label. Jev's formula is unpublished. A recorded live response had probabilities rounded to 0.01 | Lab 12 draws reliability diagrams from `noul` and `probabilities[choice]`, never from `confidence`, and says why. Thresholds are explicit numbers derived from a stated cost of error |
| Jev experimental LangChain middleware | `AutoModeMiddleware` blocks risky tool calls at a hard-coded p ≥ 0.5 and never asks a human. It works only with `create_agent`, and omitting `criteria` silently drops its default criteria | Lab 14 writes its own act / ask / escalate guard node with `interrupt()` and explicit thresholds, and quotes the middleware's instructions only as an example |
| RLCD primary sources | The TypeSafe material read so far (SDK, `skills`, `system-one-adapter`, `WorkflowEvals`) never names RLCD. Its only training claim is "System One models are trained for calibrated decisions" (`typesafe-ai/skills`, `SKILL.md`). The RLCD details in circulation come from third-party articles | Lecture 12 cites TypeSafe's announcement directly once someone reads it from a networked machine; until then it states only the sentence above and labels everything else as our illustration |
| Jev access | Participants may not have keys | Fallback path; ask TypeSafe about workshop credits |
| Model IDs change | Hard-coded IDs go stale | IDs live only in `_variables.yml`; pinned during the build |
| Colab dependency drift | Preinstalled versions change and break labs | Pinned installs; scheduled `health.yml` run |
| Four consecutive days | Fatigue by Day 4; harder for working practitioners to attend | Day 4 afternoon is hands-on pair work; each day stands alone well enough to be offered as 2 + 2 days |
| Build size | 15 lectures and 16 notebooks | Two-week build with parallel agent workstreams (see AGENTS.md), then continued review |
| LangChain / LangGraph / LlamaIndex API churn | Tutorials age quickly | Pin versions; use only core, stable interfaces |
| Claude Haiku 4.5 retirement | Anthropic lists its retirement as "not sooner than 2026-10-15" (checked 2026-10-05); Lab 8 and lecture 8 pin it | Re-check before each delivery; change `models.anthropic` and the three dated sentences in lecture 8 |
| Capstone questions need human authors | The capstone evaluation set is Lab 13's 80 questions plus 45 new human-written ones (about 30% unanswerable, including memory-bait items) | Romeo and one instructor write and blind-check them after Lab 13's set (about 3 and 2 hours) |
| Corpus snapshot ordering | `data/workshop_lectures_v1.jsonl.gz` freezes the lectures and `references.qmd`; questions quote it verbatim | Finish `references.qmd` (Modules 1–12) and rebuild the snapshot before anyone writes questions |
| RAG questions need human authors | Model-written questions copy passage wording (inflating BM25) and model relevance labels are circular with the judge Lab 13 teaches people to check | Romeo and one instructor write and blind-check 80 questions (about 4 and 3 hours); a first batch of 40 unblocks the notebook |
| Lab 11 depends on the Lab 6 logits | `lab06_logits.npz` needs a T4 run with Hub access; without it Lab 11 analyzes the Lab 1 classifier, which is close to calibrated, so the encoder half of lecture 11's section 5 is unmeasured | Lab 11 falls back automatically; an always-on regularization sweep shows temperature scaling in both directions; commit the file after Lab 6's T4 run |
| Decision set labels need two human annotators | An agent can build the generator but cannot provide independent human labels or write the hand items (the spec forbids model-written items) | Romeo and one instructor write and label 80 hand items and audit 60 template items (estimated 2–3 hours each) |
| Labs 9 and 10 depend on GPT-2 | They need the Hub (and Lab 10 a GPU); the build container has neither | Build Lab 9's data files and run Lab 10's seed protocol on a Colab T4 |
| Module 0 tools change fast | The agent CLIs release several times a week, and their install methods and plan terms change (Claude Code's docs now recommend its native installer over npm). Only Gemini CLI states a free tier (personal Google account); Claude Code needs a paid plan or Console account; OpenAI's Codex plan pages could not be read from the build container | Versions and the check date live in `_variables.yml` `assistants` and `agents_intro`; re-check installs and plan terms before each delivery; participants without a subscription pair up or use Gemini CLI |
| Build container network | The cloud build environment blocks huggingface.co, so Labs 6–8's model paths and the Lab 6/7 data files cannot be produced there | Allow huggingface.co in the environment's network settings, or run those steps on Colab |

---

## 7. Development task list

Ten working days to a first complete version, then continued review. Lectures and labs for the same module are built in parallel by different agents, then reviewed together.

### Day 1 — Repository setup

- [x] `git init`, default branch `main`, create the GitHub repository
- [x] Add `LICENSE` (CC BY 4.0 for content, MIT for code) and `CITATION.cff`
- [x] Add `.gitignore` (`docs/`, `.venv/`, `uv.lock`, `.ipynb_checkpoints/`)
- [x] Write `pyproject.toml` with uv dependency groups: `notebooks`, `site`, `test`, `lint`, `execute`
- [x] Write a first `README.md` (what it is, who it is for, how to run locally)
- [x] Add `CONTRIBUTING.md` and `CHANGELOG.md`
- [x] Convert the four AGENTS.md personas into `.claude/agents/*.md` subagents
- [x] Draft `_variables.yml` with all 15 modules (`n`, `slug`, `day`, `minutes`, `title`, `summary`, `objectives`, `stack`)
- [x] Choose and record the datasets for the running thread (classification set, LM corpus, date transduction, instruction set, preference set, decision set, RAG documents). Decisions and licenses are in `data/README.md`: arXiv Topics v1 replaces AG News (license), Tiny Shakespeare, Dolly 15k. The repo-hosted fallback URLs work only once the repository is public and `data/` is on `main`

### Day 2 — Quarto initialisation

- [x] Write `_quarto.yml`: website project, `output-dir: docs`, explicit `render:` list, notebooks as `resources:`, navbar
- [x] Write `custom.scss` (cosmo override) and vendor the fonts
- [x] Create `index.qmd`, `setup.qmd`, `schedule.qmd`, `day-1.qmd` to `day-4.qmd`, `notebooks.qmd`, `references.qmd`, `faq.qmd`, `teach.qmd`
- [x] Create a lecture page template and stub all 15 pages under `lectures/`
- [x] Write `scripts/gen_tables.py` (schedule, notebook index, README regions)
- [x] Write `scripts/gen_notebooks.py` (header and footer cells, Colab badge, output stripping, idempotent)
- [x] Build the notebook template and `00-setup.ipynb` (runtime check, Colab Secrets, `PROVIDER` switch)
- [x] Write `scripts/test_notebooks.py` and `scripts/check_links.py`
- [x] Add `.github/workflows/publish.yml` (generate, drift gate, lint, test, render, link check, notebook run, opt-in deploy)
- [x] Confirm `quarto render` is clean
- [x] Enable GitHub Pages (set Pages to deploy from GitHub Actions) and confirm the site deploys to <https://project-delphi.github.io/nlp-llms/>

### Day 3 — Modules 1–2

- [x] Draft lecture 1: Text as data
- [x] Draft lecture 2: Word vectors and neural networks
- [x] Code `01-text-as-data.ipynb`
- [x] Code `02-word-vectors.ipynb` (run on CPU, not Colab)
- [ ] Review: each equation maps to a lab line; both labs run cold in Colab within budget (equation-to-lab half done for Modules 1 and 2; Colab half open)

### Day 4 — Modules 3–4

- [x] Draft lecture 3: Sequence models
- [x] Draft lecture 4: Seq2seq and attention
- [x] Code `03-sequence-models.ipynb` (run on CPU, not Colab)
- [x] Code `04-seq2seq-attention.ipynb` (run on CPU, not Colab)
- [x] Produce the Day 1 figures (RNN unrolling, LSTM gates, attention alignment; the alignment map is now drawn from Lab 4's measured weights)
- [ ] Review: Day 1 reads as one thread; Lab 3 perplexity is compared with Lab 1 (the Lab 3 comparison is done; the cross-module read of Day 1 is open)

### Day 5 — Modules 5–6

- [x] Draft lecture 5: The transformer
- [x] Draft lecture 6: Pretraining and the Hugging Face stack
- [x] Code `05-transformer-from-scratch.ipynb` (run on CPU, not Colab)
- [ ] Code `06-pretraining-huggingface.ipynb` (written; offline parts run; pretrained path not run: Hub blocked)
- [ ] Review: Labs 3 → 5 and 1 → 2 → 6 comparisons report consistent metrics on the same data (3 → 5 and 1 → 2 done; 6 waits on its pretrained run)

### Day 6 — Modules 7–8

- [x] Draft lecture 7: Fine-tuning and LoRA
- [x] Draft lecture 8: LLMs through APIs
- [ ] Code `07-finetuning-lora.ipynb` (written; LoRA layer and offline parts run; SmolLM2 + Dolly path not run: Hub blocked)
- [ ] Code `08-llm-apis.ipynb` with the provider wrapper and the fallback path (written; runs end to end on an offline stub; OpenAI, Claude and Qwen paths not run)
- [x] Pin OpenAI and Claude model IDs in `_variables.yml`
- [ ] Add repository secrets and make CI skip keyed paths when they are absent (labs read keys from Colab Secrets or the environment and take their no-key path when absent; `health.yml`'s keyed leg uses the secrets; Romeo to add them)
- [ ] Review: Lab 8 completes with no keys set (holds on the stub only; the Qwen fallback has not run)

### Day 7 — Modules 9–10

- [x] Draft lecture 9: Reinforcement and preference learning
- [x] Draft lecture 10: RLHF
- [ ] Code `09-preference-learning.ipynb` (written; Part A run on CPU with seed-based thresholds; Part B on an offline stand-in only, waiting for the data files)
- [ ] Code `10-rlhf.ipynb`, including the pre-trained reward model checkpoint (written, with `scripts/lab10_seed_protocol.py`; unit checks run; GPT-2/T4 path and the checkpoint wait on the Lab 9 data)
- [ ] Build the Lab 9 data files (`data/build_lab09_preferences.py`) on a machine with Hub access and record their statistics; they will exceed the 6 MB cap in `tests/test_data.py` (about 4.5–7 MB more, estimated), so raise the cap or trim the reward model's vocabulary when they land
- [ ] Review: the reward-hacking demonstration is reliable across seeds (protocol in `briefs/10-rlhf.md`; needs a Colab T4)

### Day 8 — Modules 11–12

- [ ] Verify Jev SDK, LangChain and LlamaIndex integration names and signatures against `docs.typesafe.ai`; update section 6 of this file with what was found (verified against the published packages and the TypeSafe and LangChain repositories; `docs.typesafe.ai` was blocked and is still unread, see `briefs/jev-verification.md`; left unticked until the docs are read from a networked machine)
- [x] Draft lecture 11: Calibration
- [x] Draft lecture 12: RLCD and Jev (public facts and our illustration clearly separated; TypeSafe's own statements are a TODO for Romeo until docs.typesafe.ai is read)
- [ ] Code `11-calibration.ipynb` (written; classifier exercises run on the Lab 1 fallback; LLM part run on the labelled stub only; keyed, open-model and Lab 6 paths not run)
- [ ] Code `12-rlcd-jev.ipynb` (written; local toy-decider path run on the decision set and reviewed against lecture 12; keyed Jev and the stretch not run)
- [ ] Build and label the shared decision set used by Labs 11, 12 and 14 (spec in `briefs/11-calibration.md`; template-only v1 built and committed, `data/decisions_v1.jsonl.gz`: `data/build_decisions.py` generates the template items; Romeo and one instructor write and label the 80 hand-written items and audit 60 template items)
- [ ] Review: the RLCD honesty rule (AGENTS.md) holds in both the lecture and the lab

### Day 9 — Modules 13–15

- [x] Draft lecture 13: Retrieval-augmented generation
- [x] Draft lecture 14: Agents
- [x] Draft the Module 15 capstone brief and wrap-up
- [ ] Code `13-rag.ipynb` (written; offline BM25 path run on plumbing probes; corpus snapshot v1 built, provisional until `references.qmd` is finished; neural, keyed and Jev paths not run; no retrieval numbers until the human questions exist)
- [ ] Code `14-agents.ipynb` (written; stub agent + toy router + stub guard path run, with Lab 13's BM25 retriever restated; keyed, Jev and Qwen paths not run)
- [ ] Code `15-capstone.ipynb` with its starter system and fixed evaluation set (written; stub path run on plumbing probes, self-test 14/14; scoring script, collector and manifest builder tested; keyed, open and Jev paths not run; blocked on the 125 human questions and the frozen snapshot)
- [ ] Smoke-test every keyed path (OpenAI, Claude, Jev) and every fallback path

### Day 10 — Final review and release

- [ ] Run all 16 notebooks on a fresh free-tier Colab runtime; record run time and API cost per lab
- [ ] Timing dry-run of each module against the 45 + 50 minute budget; move overflow into stretch sections
- [ ] Pedagogical review of all 15 lectures: objectives met, notation consistent, prerequisites honoured
- [x] Write `facilitator-guide.md`, `instructor-pace.md` and `assessments.md` (entry and exit checks)
- [ ] Complete `references.qmd` and check every citation (complete for all 15 modules, 135 entries; 45 checked against primary records, 85 against search summaries only because the proxy blocks arXiv, ACL Anthology and most publishers; recheck those from a networked machine)
- [ ] Link check, spelling pass, accessibility pass (alt text, heading order, contrast)
- [ ] Licence and attribution check for datasets, figures and borrowed code
- [ ] Add `health.yml` (scheduled notebook run) (written: weekly offline, Hub and manual keyed legs; validated with actionlint; not yet run on GitHub)
- [ ] Final `README.md` with badges and the generated module table
- [ ] Tag `v1.0.0`, update `CHANGELOG.md`, confirm the deployed site

### Module 0 — Coding agents in the terminal (added 2026-10-05)

- [x] Draft `lectures/00-coding-agents.qmd` (install commands checked 2026-10-05 against each tool's docs, npm package or README; its two `awk` hand checks run against the reference outputs; not rendered, not tried in a real terminal)
- [x] Reference solutions under `agents-intro/` for both apps, in each language the page offers, with checks (Python and R both run; headless renders checked)
- [x] Data files for the two apps, with `datasets` entries in `_variables.yml` (`data/1ubq.pdb` from a pinned mirror, CC0 partly verified; `data/purchases_v1.csv.gz`, synthetic; UCI Online Retail II license unverified)
- [x] Wiring: `modules.m00` (`notebook: false`), the `self_serve` slot and `days.d1.self_serve`, `agents_intro` versions; generators, tests, navbar, day, schedule, setup, index and teach pages; facilitator guide and pace sheet sections; `agents-intro` in the ruff paths 
- [ ] Verify every install command (the three agents, git, `gh`) against each tool's current documentation, and refresh `agents_intro`, before each delivery
- [ ] Run Module 0 end to end on a fresh laptop per OS (macOS, Windows with WSL 2, Linux); record the times and replace the provisional rows in `instructor-pace.md`

### Module 14 — certification, harness and ARC-AGI pass (added 2026-10-06)

- [x] Lecture 14: the harness defined in section 1, with a terminology note and an ARC-AGI worked example; tool design in section 3; resume or start fresh in section 5; stopping and enforcement in section 6; escalation triggers and hand-offs in section 7; a new section 9 (patterns, subagents, context) with an optional Claude Agent SDK and MCP mapping. Sources: the *Claude Certified Architect – Foundations* exam guide v1.0, Anthropic's engineering posts, Ng's letters in *The Batch*, ARC Prize's reports and leaderboard, all opened 2026-10-06. Three optional callouts moved out of the 45 minutes to make room. Rendered clean with Quarto 1.6.40
- [x] `references.qmd` (Module 14 and library documentation), Module 0's harness sentence, Module 15's further study, brief 14's note for the Lab Engineer
- [x] Desk timing of the rebalanced lecture 14 (2026-10-06). The measure is words of in-budget material (outside collapsed callouts, check-yourself questions and demos) per budgeted minute, the same count as for the 11 other lectures with a timing table. Per lecture, those run from 64 to 112 words a minute (median 98); per section, median 92 and 90th percentile 147. Lecture 14 carried 7,376 words, 164 a minute, about 75 minutes at the median lecture's density; its section 1 ran at 251 and section 9 at 337. Before the harness pass it carried 4,395 words, 98 a minute. The ARC-AGI worked example, tool-design habits, resume or start fresh, the stopping rule, escalation triggers and the hand-off, and the detail on subagents and context moved into collapsed optional callouts, with a short in-budget summary where a point is needed; nothing was deleted. Sections 5 and 9 are now 5 minutes each. Result, in the same units: 5,259 words, 117 a minute per lecture (about 5% above the densest other lecture), sections from 90 to 134 a minute (below the 90th percentile). The pace sheet has the new rows and a "lecture behind" rule; the cut it names is stated once, in the lecture's timing note. Rendered clean
- [ ] Spoken dry-run of lecture 14 against the 45 minutes: the desk timing is a proxy that cannot tell a table row from a sentence
- [x] Lab 14 stretch exercises for the new material: parts B (research subagent, `run_subagent` and `coverage_gaps`), C (`compact`) and D (`handoff`) beside part A (the verify node), each with a folded solution and a scripted checkpoint; spec in brief 14. The core path is unchanged
- [x] Decide whether `m14` objectives gain a fourth (designing the harness): **no** (decided 2026-10-06). The lab standards require the core path to exercise every module objective, and only the optional stretch exercises harness design; after the desk timing, most of section 9's detail is optional too. Sections 1 and 9 keep their own section objectives. Revisit if a core exercise on harness design is added, or after the pilot
- [x] Lab 14 stretch, part A: a run with no retrieved passages is delivered as `no-sources`, with no verifier call, instead of a verdict against nothing (the question left open by the review of PR #8; decided and built 2026-10-06; spec in brief 14). The checkpoint adds a calculator-only run, a search with a missing argument and a search that times out. Run with the solutions on the stub path (passes, 16.5 s) and on the open-model path with no keys (passes, 260 s on an Apple laptop CPU; the Qwen verifier scored 6 of 12 on the pairs, chance level, printed and not asserted); keyed paths not run
- [ ] Re-read the ARC-AGI figures and Anthropic's posts before each delivery: the leaderboard reprices runs, and "Building effective agents" has already been edited once since 2024

### Ongoing review (after v1.0)

- [ ] Pilot one day with a small group; record where the clock slipped and which checkpoints confused people
- [ ] Revise module objectives and stretch sections from pilot feedback
- [ ] Re-verify Jev, LangChain, LangGraph and LlamaIndex APIs and pins monthly
- [ ] Revisit Module 12 whenever TypeSafe publishes more about RLCD
- [ ] Decide on the v1 exclusions: slide decks, quizzes, Spanish translation
