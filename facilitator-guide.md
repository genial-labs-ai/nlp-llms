---
title: "Facilitator guide"
subtitle: "How to run the five days, module by module"
---

<!--
Instructor page. Rendered by Quarto (listed under render: in _quarto.yml), so the var
shortcodes below resolve on the site; on GitHub they show as written. Module titles,
minutes, model IDs, package pins and secret names come from _variables.yml. Lecture
timings come from each lecture's timing table; lab minutes and known risks come from
the "As built" sections of briefs/*.md and the notebooks' own headings.
-->

This guide is for the person running the room. It says what to set up, what to say first, where groups get stuck, what to cut when the clock slips, and what to do when Colab, an API or the Hugging Face Hub fails. The minute-by-minute plan is in the [pace sheet](instructor-pace.md). The entry and exit checks are in [assessments](assessments.md). The timetable is on the [schedule](schedule.qmd).

{{< include /_includes/module-shape.md >}}

::: {.callout-warning appearance="simple"}
**Being updated for the five-day schedule.** The day-by-day sections below, and "Each morning", still follow the earlier four-day plan, in which every module was 45 minutes of lecture and 50 of lab, Module 8 closed Day 2, and Modules 13 to 15 filled Day 4. They have not yet been rewritten for the debriefs, the retrieval and synthesis slots, or Day 5. The [schedule](schedule.qmd) is current; the notes for each module still apply.
:::

## Read this first: what has been verified

{{< include /_includes/readiness-summary.md >}}

The [readiness page](readiness.qmd) has the evidence for every module. It is generated from the run records in `runs/`. It shows which labs have run on their real path, on which machine and for how long, which have only passed the offline code check, and the blocking work that remains, such as the human-written question sets. Read it before you plan a delivery. When you time a lab, add a run record so the page shows it. Unless a line says *measured*, every run time in this guide and in the pace sheet is a planning estimate from the lab briefs.

**A CPU time is not a T4 time.** Labs 2 to 5 train their models from scratch and were designed for a T4. The readiness page shows their recorded CPU times. Some ran far over their lab plans on a shared CPU, and some ran well inside them on a GitHub runner. None of these predicts a T4. Time Labs 2 to 5 on the room's runtime before you rely on the lab plans.

**Keyed paths.** No OpenAI, Anthropic or Jev call has been made by the build. The OpenAI and Anthropic response fixtures in Lab 8 were constructed from the documented shapes, not recorded. Every cost in this guide is an estimate from token arithmetic.

**Model retirement.** Lab 8 and lecture 8 pin `{{< var models.anthropic >}}`. On 2026-10-05 Anthropic listed its retirement as "not sooner than October 15, 2026". Check both providers' deprecation pages before every delivery.

## Before the workshop

### A week before

1. Run every notebook from the [notebooks page](notebooks.qmd) on a fresh Colab runtime, with no keys set, then again with the keys you will hand out. Colab's preinstalled packages change. Record the run time per lab; the pace sheet needs it.
2. Decide on keys. Every API lab runs without keys. If you provide keys, provide them for the whole room or for no one, so that groups compare like with like.
3. Ask TypeSafe for workshop keys. A room of 30 on one key makes about 12,000 Jev calls in Lab 12, 48,000 in Lab 13's reranking and 15,000 in Lab 14. Rate limits are unknown.
4. Check the model IDs against the providers' deprecation pages: `{{< var models.openai >}}`, `{{< var models.anthropic >}}`, and the open fallback `{{< var models.fallback >}}`.
5. Check that the blocking work on the [readiness page](readiness.qmd#open-work) is done, or plan the fallback named in each module's section.
6. Send participants the [Module 0](#module-0) page as optional pre-work, decide whether you will open the room at 08:00 on Day 1 for a drop-in clinic, and tell them. Ask anyone on a managed work laptop to check now that they may install software.

### The day before

Send participants to [Setup](setup.qmd). It asks them to run `00-setup.ipynb`, switch on a T4, and add any keys to Colab Secrets.

### Runtimes, keys and downloads by lab

| Lab | Runtime | Downloads from the Hugging Face Hub | Keys it can use | Path without keys or Hub |
|---|---|---|---|---|
| 1 | CPU | none | none | runs fully |
| 2 | T4 recommended | none (stretch: 134 MB of GloVe from GitHub) | none | runs fully |
| 3 | T4 | none | none | runs fully; `QUICK = True` trains a quarter of the steps |
| 4 | T4 | none | none | runs fully; `QUICK = True` trains 500 steps and skips the accuracy checkpoints |
| 5 | T4 | none | none | runs fully; switches itself to `QUICK` on a CPU runtime |
| 6 | T4 (CPU fallback: BERT-mini on 800 training texts) | three checkpoints, several hundred MB | none | no participant fallback without the Hub |
| 7 | T4 (CPU: `FAST`) | SmolLM2-135M and the Dolly file | none | no participant fallback without the Hub |
| 8 | CPU or T4 | Qwen 0.5B on the open path | OpenAI, Anthropic | stub test double |
| 9 | CPU | none in the core path | none | Part A runs; Part B needs the Lab 9 files |
| 10 | T4 (CPU: `FAST` smoke runs only) | distilled GPT-2 | none | none: needs the Hub and the Lab 9 files |
| 11 | CPU | Qwen on the open path | OpenAI, Anthropic | Lab 1 classifier and the stub |
| 12 | CPU | none | TypeSafe (stretch: an LLM key) | local toy decider, labeled "not Jev" |
| 13 | T4 for the open path | bge-small, a cross-encoder, an NLI model, Qwen | OpenAI, Anthropic, TypeSafe | LSA and lexical stand-ins, stub generator and judge |
| 14 | CPU or T4 | Qwen for the no-key guard | OpenAI, Anthropic, TypeSafe | toy router, stub guard, stub agent |
| 15 | as Labs 13 and 14 | as Labs 13 and 14 | all three | stub path (code check only) |

### Colab Secrets and which keys matter

Participants add keys in the key icon of Colab's left sidebar and switch on **Notebook access** for each. The names are fixed:

| Secret | Labs where it changes what runs |
|---|---|
| `{{< var secrets.openai >}}` | 8, 11, 13, 14, 15; the stretch of 12 |
| `{{< var secrets.anthropic >}}` | 8, 11, 13, 14, 15; the stretch of 12 |
| `{{< var secrets.typesafe >}}` | 12, 13 (reranking), 14, 15 |

Tell the room three things. A key is never pasted into a cell. A notebook with no keys is not a lesser notebook: every checkpoint gives the same verdict on every path. A number from the no-key path describes the open model, the toy model or the test double, never OpenAI, Claude or Jev.

### The lookalike-package warning

Say this out loud on Day 4, before Lab 12, and again on Day 5. TypeSafe's SDK is published on PyPI only as `typesafe-sdk` (pinned at {{< var packages.typesafe_sdk >}}), and its LangChain integration only as `langchain-typesafe` (pinned at {{< var packages.langchain_typesafe >}}). Lab 12's stretch adds TypeSafe's emulator, `system-one-adapter` ({{< var packages.system_one_adapter >}}). Several unaffiliated packages sit on names a participant might guess, among them `typesafe-ai`, `jev` and `typesafe-client`, and a third-party LlamaIndex reranker. There is no official LlamaIndex integration. The labs install the right packages for you; nobody should install anything else under a TypeSafe-like name.

### How the notebooks behave

- **Run all always completes.** Each `# TODO` cell is followed by a folded solution, and Run all runs the solutions after the participants' cells. To test their own code, participants run the `# TODO` cell and then the checkpoint, skipping the solution.
- **Checkpoints are numbered by exercise.** Checkpoint 3 belongs to Exercise 3. Some labs split a checkpoint (3a, 3b).
- **Data files** are fetched with a hash check. `fetch` looks in a local `data/` folder first. If a URL fails in the room, download the file from the repository on one machine, upload it to the Colab **Files** panel into a folder named `data`, and rerun the cell.

## Module 0 · {{< var modules.m00.title >}} {#module-0}

Optional pre-work, before Day 1, with an optional drop-in clinic on Day 1 from 08:00 to 09:00, before the welcome. Nothing later in the workshop depends on it. The notes below describe the clinic hour.

**What the hour looks like.** There is no talk. Participants work through the [Module 0 page](lectures/00-coding-agents.qmd) on their own laptops, at their own pace. Each one installs a terminal coding agent (Claude Code, Codex or Gemini CLI), sets up git and the GitHub CLI (`gh`), uses the agent to build two small apps in one language of their choice, Python or R (a protein structure explorer for ubiquitin, PDB entry 1UBQ, and an RFM customer segmentation, each with a three.js page), and publishes both on GitHub Pages. Your job is to unblock installs and sign-ins. Expect most problems before anyone writes a prompt: installs, sign-ins and git setup.

**The 08:00 room.** Open at 07:50. Put the Wi-Fi details and the Module 0 page URL on the screen. Have power at every table: installers and agents drain batteries. Have at least one helper who has done the module on each of macOS, Windows and Linux. Keep this section open on a second screen. At 08:55 tell everyone to stop where they are and finish from the page at home.

**Common install failures.** Tool-specific facts come from each tool's README or setup page, and GitHub's Pages documentation, read on {{< var agents_intro.checked >}}; the rest is standard practice. None has been reproduced on a laptop.

- **Node.js version (all systems).** Codex and Gemini CLI install with `npm install -g`, and their npm packages require Node `{{< var agents_intro.node.codex >}}` and `{{< var agents_intro.node.gemini >}}` respectively. Claude Code's npm package requires Node `{{< var agents_intro.node.claude_code >}}`, but its docs recommend the native installer, which needs no Node at all. Run `node --version` first. A Node from an old Linux distribution's package manager is often too old; install a current LTS release from nodejs.org or a version manager such as nvm.
- **`command not found` after an install (all systems).** Open a new terminal first. The Claude Code native installer puts `claude` in `~/.local/bin`, which may not be on `PATH`. For npm installs, the global `bin` folder is under `npm prefix -g`. On macOS and Linux, an `EACCES` error from `npm install -g` means npm is writing to a system folder: do not use `sudo`; switch to a version manager or a user-owned npm prefix.
- **macOS.** The first `git` command may ask to install the Xcode Command Line Tools. Accept and wait; it takes several minutes, so start it early.
- **Windows: WSL 2 only.** Windows participants do the whole module inside WSL 2 with Ubuntu and follow the Linux steps; we never give native Windows or PowerShell instructions. Install WSL from Command Prompt as administrator (`wsl --install`, then restart); WSL 1 is not supported (convert with `wsl --set-version Ubuntu 2`). Work in the Linux home folder, not `/mnt/c/`. If `which node` inside WSL points to a path under `/mnt/c/`, it is the Windows Node: install Node inside WSL. A participant who cannot install WSL (a locked-down work laptop) pairs with someone.
- **Corporate proxies and TLS inspection.** Errors such as `SELF_SIGNED_CERT_IN_CHAIN` or `unable to get local issuer certificate` mean a proxy is re-signing traffic. Point the tools at the company's CA bundle (`NODE_EXTRA_CA_CERTS` for Node tools, `git config http.sslCAInfo` for git) and set `HTTPS_PROXY` if the network needs it. Never turn TLS verification off. If the network blocks the agent's service, tether to a phone or pair with someone.
- **`gh auth login`.** Choose GitHub.com, HTTPS and "Login with a web browser", then paste the one-time code. Check with `gh auth status`. If `git push` still asks for a password, run `gh auth setup-git`. If a commit fails with "Please tell me who you are", set `git config --global user.name` and `user.email`. Work laptops signed in to an enterprise-managed GitHub account can publish Pages only from organization repositories: use a personal account.
- **GitHub Pages returns 404.** Pages must be switched on for the repository (Settings → Pages). After that, GitHub's documentation says changes can take up to 10 minutes to publish, so a 404 in the first few minutes is normal. The entry file must be named exactly `index.html`. On a free account, Pages works only for public repositories.
- **The three.js page is blank when opened from disk.** A page that loads three.js as an ES module will not run from a `file://` URL in most browsers. Serve the folder locally (for example `python -m http.server`) or check the published Pages URL instead.

**No agent subscription.** Claude Code needs a Pro, Max, Team, Enterprise or Console account; the free Claude plan does not include it (Claude Code setup documentation, read {{< var agents_intro.checked >}}). Codex's README asks users to sign in with a ChatGPT Plus, Pro, Business, Edu or Enterprise plan, or to use an API key; we could not read OpenAI's plan pages from the build environment, so do not tell anyone Codex is free. Gemini CLI's README states a free tier for a personal Google account (60 requests a minute and 1,000 a day). So a participant with no subscription can use Gemini CLI with a personal Google account, after reading its terms, or pair with someone who has an agent; the person without the agent drives git, `gh` and Pages. Recheck these plan terms before each delivery.

## Each morning: the first ten minutes

The opening slot is 10 minutes. Use 5 to say the lines below and 5 for the check.

**Day 1.** Welcome. The workshop follows one line: each module fixes a failure of the one before it, and you build each step yourself. Every lab follows Predict, Run, Explain, Check: write your prediction before you run a cell. The same datasets come back all week (arXiv Topics, Tiny Shakespeare, later the decision set and the lecture pages), so improvements are measured, not asserted. Then: everyone runs `00-setup.ipynb` and confirms the provider line it prints.

**Day 2.** Recap the line from Day 1: counts, then embeddings, then recurrence, then attention. Module 4 ended with attention as a query-key-value lookup; Module 5 removes the recurrence. Remind the room to switch on a T4 before Lab 5. Check that last night's keys still load.

**Day 3.** Recap Day 2: one architecture, pretrained once, fine-tuned or called through an API. Day 3 asks what we optimize these models for. State the honesty rule now, in these words: "TypeSafe has not published how RLCD works. In Module 12 I will tell you what TypeSafe has said publicly, what others have said, and what is our own illustration, and I will keep them apart." Give the lookalike-package warning.

**Day 4.** Recap Day 3: preference rewards, calibration, thresholds from costs. Day 4 builds systems out of the pieces. Announce the capstone pairs now and ask each pair to decide which keys they will use; pairs are compared only with pairs on the same path.

## Day 1: Foundations

### Module 1 · {{< var modules.m01.title >}}

- **Before:** CPU runtime. No keys, no installs. Check that the arXiv Topics file loads (blocking item 1).
- **First 5 minutes:** Language is ambiguous, sparse and compositional. Today we turn text into counts and see how far counts go. Every later module is measured against the two numbers this lab prints: the best character n-gram's test perplexity and the TF-IDF classifier's test accuracy.
- **Where groups get stuck:**
  - Exercise 2: probabilities for an unseen context must sum to 1 through the add-k formula itself, not through a special case. The checkpoint tests a context never seen in training.
  - Exercise 3: the splits are consecutive, so the context before the first test token comes from the end of the validation split (the provided `history_for`). Only the start of training is padded.
  - Exercise 5: the idf must match scikit-learn's defaults (smoothed idf, then L2 normalization) for the `allclose` check to pass.
- **If the clock slips:** Exercise 4 (sampling) becomes a demonstration: run the solution. Nothing else may be cut, because Exercises 3, 5 and 6 produce the baselines. The stretch (BM25) is optional; Module 13 reteaches BM25.
- **If something fails:** only the data fetch can fail. Use the upload route above.
- **Not verified:** run time on Colab (budget 2 minutes; measured locally only) and Colab's preinstalled scikit-learn version.

### Module 2 · {{< var modules.m02.title >}}

- **Before:** T4 recommended. The GloVe stretch downloads 134 MB per participant; on shared Wi-Fi, ask only those who reach it to run it.
- **First 5 minutes:** Module 1's counts treat *dog* and *puppy* as unrelated symbols and give zero probability to anything unseen. Today words get dense vectors learned from their neighbors, and we test whether those vectors beat TF-IDF on the same split.
- **Where groups get stuck:**
  - Exercise 1: the shapes are `(B, d)`, `(B, d)` and `(B, K, d)`; the negatives enter as $\log \sigma(-u^\top e)$. The first training loss must be exactly $(K+1)\log 2$; if it is not, the sign or the sum is wrong.
  - Exercise 2: the query word must be excluded from its own neighbors.
  - Exercise 3: a document of only padding must give the zero vector, not NaN.
- **The result to prepare the room for:** averaged embeddings lose to TF-IDF (0.867 against 0.884 test accuracy in the build run). That is the expected finding. Do not let groups tune until embeddings win.
- **If the clock slips:** the analogy section becomes a demonstration first. Exercises 1, 3 and 4 and the results table carry the objectives. Then drop the stretch.
- **Not verified:** run time on Colab, CPU or T4. Its recorded CPU times (on the [readiness page](readiness.qmd)) range from well inside its slot to far over it, depending on the machine. Time it on the room's runtime.

### Module 3 · {{< var modules.m03.title >}}

- **Before:** T4 required. Do not run this lab on a CPU runtime: Run all took 72 minutes on the build CPU.
- **First 5 minutes:** A fixed window cannot see a verb's subject seven words back. A recurrent network carries a state through the whole sequence. Today we build one, watch its gradient vanish, and compare its perplexity with Module 1's n-gram on the same characters.
- **Where groups get stuck:**
  - Exercise 1: `nn.RNNCell` has two bias vectors and the lecture has one; the checkpoint handles it, but groups comparing by hand get confused.
  - Exercise 3B: clipping rescales the whole gradient vector; it must not change direction.
  - The comparison rule: never put a word-level perplexity from Module 1 beside these character-level numbers.
- **If the clock slips:** edit the setup cell to `QUICK = True` before training. It trains each model for a quarter of the steps; say that its numbers are not the recorded baselines. Then drop the stretch (top-k and nucleus sampling).
- **Not verified:** T4 run time, and therefore whether the training runs fit their 4- and 7-minute slots. A checkpoint comment says the RNN's gradient underflows "before distance 110"; the build run reports about 138. The lesson is unchanged.

### Module 4 · {{< var modules.m04.title >}}

- **Before:** T4 required. Run all took 107 minutes on the build CPU.
- **First 5 minutes:** Module 3's models predict the next token. Now the output is a different sequence: written dates to ISO format. We will watch a fixed-size vector fail as the input grows, then fix it with attention.
- **Where groups get stuck:**
  - Exercise 1: the training loop already shifts the targets; the loss is the mean over non-padding tokens.
  - Exercise 3: padded positions must get weight exactly 0. Use `masked_fill` with `-inf`, not a large negative number.
  - Exercise 5: the hit rate allows the arg-max to land up to one position after the date's span, because an encoder state summarizes the source up to its position.
- **The K = 1 variance:** without attention, exact match on one date was 0.898 in the recorded run and 0.99 on a validation set in a second run; read it as "about 0.9". The checkpoint asserts at least 0.85. If a group's run falls just below, it is run-to-run variation, not their bug. The lesson is the collapse from K = 2 onward (0.000 in the recorded run) against 1.000 with attention.
- **If the clock slips:** drop the stretch (beam search), then shorten Exercise 5 to viewing the heat-maps. If training does not fit, edit the setup cell to `QUICK = True` (500 steps instead of 4,000); it skips the accuracy checkpoints, so show the recorded table from lecture 4 instead.
- **Not verified:** T4 run time; the committed notebook at full settings (the recorded run used an earlier revision).

**End of Day 1.** Give the exit-check questions for Modules 1 to 4 as a five-minute self-check if time allows.

## Day 2: Transformers and LLMs

### Module 5 · {{< var modules.m05.title >}}

- **Before:** T4 required. The mini-GPT took 3,921 s to train on the build CPU.
- **First 5 minutes:** Module 4's attention still sat on top of a recurrence, which reads one token at a time. Today attention reads the sequence itself, all positions in parallel, and we train a small GPT on the Module 3 corpus.
- **Where groups get stuck:**
  - Exercise 1: the function must work for any leading dimensions (the heads become a batch dimension).
  - Exercise 2: the leak test must run in `eval()` mode, with `-inf` in the mask.
  - Exercise 3: the shuffle test holds for one attention layer only, not for the whole model.
- **The result to prepare the room for:** at full budget the GPT and the retrained LSTM tie (1.548 against 1.573 nats per character, one seed, within seed noise). In `QUICK` mode the LSTM wins clearly. Do not promise that the transformer wins.
- **If the clock slips:** drop the stretch (writing the multi-head layer and the block), then shorten Exercise 5. `QUICK = True` in the setup cell shortens training; it is on by default on a CPU runtime.
- **Note on objectives:** the core path has participants write the attention step, the mask and the input step; the multi-head split and the block are provided and are the stretch. Say so if someone asks why they did not write the block.
- **Not verified:** T4 run time.

### Module 6 · {{< var modules.m06.title >}}

- **Before:** T4. Three checkpoints download in the background when the setup cell runs (several hundred MB). On a CPU runtime the lab fine-tunes BERT-mini on the first 800 training texts. Unauthenticated Hub downloads print a rate-limit warning; it is harmless unless the whole room is throttled.
- **First 5 minutes:** Module 5's GPT knew only Shakespeare and 65 characters. Today: subword tokens, pretraining once on a large corpus, and fine-tuning someone else's model on our classification set.
- **Where groups get stuck:**
  - Exercise 1: ties between pairs are broken by the pair that sorts first; without that rule the merge order differs from the lecture's table.
  - Exercise 4: special and padding positions must never be selected; of the selected, 80% become `[MASK]`, 10% random, 10% unchanged.
  - Exercise 5: the untrained model's validation loss should be close to $\log 4$; predict it before running.
- **The result to prepare the room for:** expect the fine-tuned encoder to match TF-IDF or gain a little. The floor assertions (0.85 on GPU, 0.60 on CPU) are provisional.
- **If the clock slips:** Exercise 2 becomes a demonstration, then Exercise 3. Exercises 1, 4 and 5 carry the objectives.
- **If the Hub fails:** there is no participant fallback. `NLP_LLMS_OFFLINE_TINY` is a test switch whose numbers mean nothing; do not use it in the room. Exercise 1 and the TF-IDF baseline do not use the pretrained models. Use the rest of the slot to work through lecture sections 5 and 7 on paper, and rerun the lab when the Hub returns.
- **Not verified:** every pretrained-model number; Colab's preinstalled versions; `fp16` on a T4.

### Module 7 · {{< var modules.m07.title >}}

- **Before:** T4. Both the model and the Dolly file come from huggingface.co. On a CPU runtime the `FAST` flag turns on (300 training examples, 40 steps).
- **First 5 minutes:** A pretrained model continues text; ask it a question and it may repeat the question. Today we teach it to answer, while training under 1% of its parameters.
- **Where groups get stuck:**
  - Exercise 1: with $B = 0$ at initialization the output equals the base layer's. Then ask which gradient is zero: it is $A$'s, not $B$'s.
  - Exercise 3: the checkpoint is an exact string match with `apply_chat_template`, newlines included.
  - Exercise 4: the label mask covers the response and the closing `<|im_end|>`; the newline after it is not scored.
  - Exercise 5: predict 460,800 trainable parameters before running.
- **The stop-token question:** whether adapters on `q_proj` and `v_proj` alone teach this model to end a reply with `<|im_end|>` within the step budget is open. On random stand-in models they never did (0%), against 80–96% with adapters on every linear layer. If tuned replies still run to the length limit, that is this open question, not a participant's bug. The stretch's `all-linear` cell measures it.
- **If the clock slips:** drop the stretch first (the rank sweep and the `all-linear` cell may exceed their budget); then run Exercise 6's `SAMPLING` cell as a demonstration.
- **If the Hub fails:** no participant fallback. Exercises 1 and 2 (the LoRA layer on a toy `nn.Linear`) do not use the model; whether they run when the model download has failed was not checked.
- **Not verified:** every number that needs the real model, the loss margins of the after-training checkpoint, and that the tokenizer's end-of-sequence token is `<|endoftext|>` at the pinned revision.

### Module 8 · {{< var modules.m08.title >}}

- **Before:** keys optional. `PROVIDER` picks the first provider with a key, otherwise the open model; if the open model cannot download, or `NLP_LLMS_STUB=1`, the lab uses a rule-based test double labeled `stub (test double)`.
- **First 5 minutes:** Today the model is someone else's, behind an endpoint. You control the message list and little else. We build one harness that calls OpenAI, Claude and an open model the same way, and measure them on one task.
- **Where groups get stuck:**
  - Exercise 1: OpenAI returns tool arguments as a JSON string; decode it.
  - Exercise 2: "always invalid" must stop after exactly $R + 1$ calls; truncated and refused replies are not parsed.
  - Exercise 4: an unknown tool or bad arguments must return an error result to the model, not raise.
- **Discuss, do not fix:** the prompt-injection item (an announcement that tells the model to set seats to 9999). Ask what each provider did. Module 14 returns to it.
- **If the clock slips:** drop the stretch (the Lab 7 model as a fourth provider). The CPU fallback already uses 12 evaluation items and 3 tool questions.
- **If an API fails:** a 429 is retried with backoff. For an outage, set `PROVIDER = "open"` (or the stub) and rerun from the provider cell. Every checkpoint gives the same verdict.
- **Cost (estimate, not measured):** under 5 cents per full run on OpenAI, under 25 cents on Anthropic.
- **Not verified:** the keyed paths and the open-model path have never run. The OpenAI prices come from secondary sources.

## Day 3: Training objectives

### Module 9 · {{< var modules.m09.title >}}

- **Before:** CPU is enough; no downloads in the core path. Part B needs the Lab 9 data files (blocking item 2).
- **First 5 minutes:** Supervised fine-tuning can only make a given text more likely. It has no way to say "this answer is better than that one". Today: the minimum reinforcement learning, and a reward model learned from comparisons.
- **Where groups get stuck:**
  - Exercise 1: returns and the baseline must be detached; the loss is a sum over positions and a mean over the batch.
  - Exercise 2: the batch-mean baseline equals $(1 - 1/N)$ times the leave-one-out baseline; the checkpoint tests that identity exactly.
  - Exercise 3: `tau_label` divides the gold gap; `bt_prob(1, 0, 0.5)` is $\sigma(2) = 0.8808$.
- **Say it plainly:** the gold rule is a rule we wrote, known only because the data are synthetic. A real preference dataset has no gold score.
- **If the clock slips:** drop the stretch (noisier raters), then run the five-seed training of Exercise 2 as a demonstration.
- **If the Lab 9 files are missing:** Part A runs in full. Part B's data cell will fail. We have not checked whether the unit checks of Exercises 3 to 5 run after that failure; plan to take Part B as a whiteboard exercise from lecture sections 7 and 8.
- **Not verified:** Part B on real data; the Exercise 4 thresholds are provisional; run time on a T4.

### Module 10 · {{< var modules.m10.title >}}

- **Before:** T4 required. On a CPU runtime the lab switches to `FAST`: unit checkpoints run, training is a short smoke test, and the checkpoints about trained policies are skipped with a message. It needs the Hub and all three Lab 9 files.
- **First 5 minutes:** Module 9 gave us a reward model. Today we optimize a language model against it, see what happens without a leash, and then do the same job without a reward model at all (DPO).
- **Where groups get stuck:**
  - Exercise 1: logits at position $t$ predict token $t + 1$; prompt positions are excluded.
  - Exercise 2: the KL penalty uses the sampled log-ratio, detached; the drift metric uses the exact KL of Exercise 3. Do not swap them.
  - Exercise 5: the DPO loss is exactly $\log 2$ when the policy equals the reference.
- **The demonstration and its risk:** Exercise 4 removes the KL penalty and asserts the reward-hacking signature (proxy reward up, drift up, gold reward down, variety down). Whether it holds on every seed is unverified; it is the main risk of the lab. If a group's run misses one part, read the table with them; do not call it a bug.
- **If the clock slips:** drop the stretch (the $\beta$ sweep, three more training runs) before anything in the core.
- **If there is no GPU or no Lab 9 files:** with no GPU, run `FAST` and teach Exercises 4 and 5 from lecture figure 10.2 (a schematic). Without the Lab 9 files the lab cannot run; teach the KL-regularized objective and the DPO derivation on the board.
- **Not verified:** nothing involving GPT-2 has run. All times are estimates (core path 6 to 7 minutes on a T4).

### Module 11 · {{< var modules.m11.title >}}

- **Before:** CPU is enough. The classifier part runs on the Lab 1 classifier until the Lab 6 logits are committed; the notebook says which classifier `CLF` holds. The language-model part follows Lab 8's `PROVIDER`.
- **First 5 minutes:** Module 8 ended with "a fluent answer carries no confidence signal", and Module 10 with a model tuned to what raters prefer. Today: what a probability should mean, how to measure it, and how to turn it into a decision with a stated cost.
- **Where groups get stuck:**
  - Exercise 1: bins are right-closed; a confidence of exactly 1.0 lands in the last bin.
  - Exercise 3: fit the temperature on validation, never on test, "even to see".
  - Exercise 4: decide what `"85%"` and `85` mean; the solution accepts the string as 0.85 and rejects numbers above 1.
  - The binary Brier score of the language model is half the value Exercise 2's two-column `brier` returns; the notebook says which it prints.
- **Start the slow cell early:** `ask_all` runs at the top of Exercise 4, while participants write the parser.
- **Timing:** the closing question ("what would you let act alone?") sits inside Exercise 5's 8 minutes. If the room runs late, use it as the bridge into Module 12 after the break.
- **If the clock slips:** drop the stretch (properness shown numerically). On a CPU open path, the evaluation uses 40 `dev` and 80 `test` items; the noise floor on 80 items is about 0.07, so warn against reading small differences.
- **Lab 11 to Lab 12:** the export cell writes `lab11_decisions_<provider>.jsonl`. Colab runtimes do not persist, so participants who want the comparison panel in Lab 12 must download this file and upload it there.
- **Cost (estimate):** under 1 USD per full run on Anthropic, under 25 cents on OpenAI.
- **Not verified:** the open-model and keyed paths have not run; the Lab 6 encoder's calibration is unknown.

### Module 12 · {{< var modules.m12.title >}}

- **Before:** CPU is enough. With `{{< var secrets.typesafe >}}` set, `JEV_PATH` is keyed and the lab calls Jev; otherwise a local toy decider answers through the same interface, under a banner that says it is not Jev. **Replace the TODO box in lecture 12, section 2, before teaching.**
- **First 5 minutes:** Module 11 measured calibration and set a threshold from costs. Today: a reward that targets calibrated probabilities (our illustration), a decision model reached through typed questions, and three actions (act, ask, escalate) from stated costs. Then the honesty statement from the Day 3 opening, again.
- **Where groups get stuck:**
  - Exercise 2: a yes/no answer's chosen probability is `noul` for "yes" and `1 - noul` for "no", with "yes" at `noul >= 0.5`. A choice answer's chosen probability is the top entry of `probabilities`, not `confidence`.
  - Exercise 3: `confidence` (written $\kappa$) measures how concentrated the distribution is. On calibrated synthetic data its ECE is above 0.10 while the probabilities' is below 0.01.
  - Exercise 4: when asking is not cheap enough, the middle region disappears and both thresholds equal Chow's threshold.
- **The degenerate thresholds on the local path:** with the worked example's costs, the `dev`-chosen pair is $(0.000, 1.000)$. The rule acts on the 11 most confident `dev` answers, asks about the rest and never escalates; on `test` it asks about 88% of items, at 1.800 per case against 1.753 for asking about everything. This is a finding about the toy decider, explained in lecture 12, section 8: no top slice of `dev` larger than 11 answers is right 96.9% of the time, and the softmax saturates. Use it to discuss what thresholds need from a model.
- **Never cut** the closing cell, "What this lab showed and what it did not". It is part of the honesty rule.
- **If the clock slips:** drop the stretch (an LLM through TypeSafe's emulator).
- **If Jev fails:** unset the key, or rerun on the local path. The checkpoints do not change. Never quote a local-path number as Jev's.
- **Cost (estimate):** Jev under 5 cents per full run; output tokens are currently free per the SDK schema.
- **Not verified:** no live Jev call has been made; TypeSafe's announcement and pricing page have not been read (its documentation was read on 2026-10-06; see lecture 12, section 2); the emulator's key handling and its compatibility with Lab 8's pins.

**Answering questions about RLCD.** Use three sentences. What TypeSafe has stated in its own sources (lecture 12, section 2). What third parties report, which we have not checked against a TypeSafe source. What is ours: the framing of section 3 and Lab 12's Exercise 1. If someone asks "is this how Jev was trained?", the answer is "we do not know; TypeSafe has not published it". RLCR (Damani et al.) is published work by other authors; RLCR is not RLCD.

## Day 4: RAG, agents and capstone

### Module 13 · {{< var modules.m13.title >}}

- **Before:** T4 for the open path (about 200 MB of encoder, reranker and judge, plus about 1 GB for Qwen). Without the Hub the lab runs on stand-ins under a banner: an LSA encoder, a lexical reranker, a stub generator and a stub judge. **The question set is not written yet** (blocking item); until it is, the evaluation cells run on plumbing probes, sentences copied from the snapshot, which measure the code and not retrieval.
- **First 5 minutes:** Module 12's decisions came from the model's parameters. A research assistant must answer from documents it can cite, and say when it cannot. Today we index the workshop's own lecture pages, and measure retrieval and answers separately.
- **Where groups get stuck:**
  - Exercise 2: node IDs must be deterministic, and metadata must be excluded from the embedded text.
  - Exercise 3: LlamaIndex and LangChain disagree until metadata embedding is switched off; the provided cell shows the disagreement first.
  - Exercise 4: the Jev reranker must be awaited (`await reranker.apostprocess_nodes(...)`); the synchronous call fails inside Jupyter's event loop. Rank by `score`, never by `confidence`.
  - Exercise 5: citations are stripped before the judge sees a sentence; the abstention sentence must match exactly.
- **If the clock slips:** drop the stretch (hybrid retrieval with BM25). On the keyed Jev path, rerank only `test`; reranking `dev` and `test` is 1,600 calls per participant.
- **If something fails:** set `NLP_LLMS_LAB13_OFFLINE=1` to force the offline path; every number then sits under the banner.
- **Cost (estimate):** under 50 cents on Anthropic, under 5 cents on OpenAI, under 5 cents for Jev reranking.
- **Not verified:** no neural model, provider or Jev call has run; the LlamaIndex and LangChain documentation sites were not read (the code was checked against installed source).

### Module 14 · {{< var modules.m14.title >}}

- **Before:** CPU or T4. `HUMAN_MODE = "simulated"` by default, so Run all finishes with nobody at the keyboard. One optional cell answers an interrupt with `input()`. Without a TypeSafe key the router is the Lab 12 toy model and the guard is Qwen scored on " yes"; without the Hub the guard and the agent become test doubles, and the router stays the toy model.
- **First 5 minutes:** Module 8's tool loop was a `while` loop we wrote. Today the loop becomes an explicit graph with state, checkpoints and a pause for a person, and a decision model gates the one risky tool, `send_email`.
- **Where groups get stuck:**
  - Exercise 2: the tie rule acts at exactly the threshold (0.96875 acts, 0.9687 asks).
  - The guard's probability is `noul` itself, not `max(noul, 1 - noul)`; Lab 12's `chosen_answer` is right for the router and wrong for the guard.
  - Exercise 3: the decider call must not live in the node that pauses, because LangGraph reruns that node from its first line on resume.
  - Exercise 4: replay must not send an email twice; `send_email` is idempotent by design.
- **If the clock slips:** drop the stretch (parts A to D). On a CPU runtime the brief's first cut is the probability-shift test, to 30 items; there is no switch for it, so slice the list in the evaluation-run cell and say so when reporting.
- **Cost (estimate):** Jev under 5 cents; the agent model under 30 cents on Anthropic, under 5 cents on OpenAI.
- **Not verified:** keyed Jev and keyed LLM paths have not run. The open path with no keys (Qwen as agent and guard) has run end to end on a laptop CPU, not on Colab (see the [readiness page](readiness.qmd)); whether the 0.5B guard carries any signal; the pins were resolved for Python 3.12 with `uv`, not installed with `pip` on Colab.

### Module 15 · {{< var modules.m15.title >}}

**Status:** the starter notebook, `15-capstone.ipynb`, is written and passes its offline code check, but its evaluation set does not exist: neither Lab 13's 80 questions nor the 45 new ones have been written. Until they are, every run scores *plumbing probes*, which check that the system runs end to end and say nothing about how well it answers. There is no fallback that keeps the evaluation. Do not schedule Module 15 until the questions exist and an instructor has recorded the baselines (below).

**Before the day.**

1. Record the baseline: run the unmodified starter on `dev` and `test` on every path you will allow (at least keyed with Jev, open on a T4, open on a CPU subset, and the stub), and once more on each keyed path to measure run-to-run flips.
2. Decide how pairs hand in their submission file (a shared folder or an upload form). `scripts/collect_capstone.py` merges the files and marks each submission ranked, over budget, not comparable or code check only.
3. Shared keys: 15 pairs make about 14,000 Jev calls in an afternoon without reranking. Limit Jev reranking (menu option R2) to `dev` if keys are shared.

**The afternoon.** Clock times are on the [schedule](schedule.qmd); the pace sheet has the minute plan.

- **The brief (10 minutes):** the task contract (cited answer or the exact abstention sentence), the starter graph, the five numbers and the cost, the rules. It is a walk through the tables, not a derivation.
- **Build (85 minutes):** circulate. At minute 20, check that every pair has a `dev` baseline and has read traces; a pair that picks a component without looking at failures is guessing. Steer toward the failures: if Module 13's two-by-two table shows mostly retrieval failures, a prompt change is unlikely to help. Watch call counts on keyed paths. Encourage null results.
- **Evaluate and share (65 minutes):** 20 minutes to run `test` on the frozen system and hand in; 35 minutes of two-minute shares, in the order of the menu, so that pairs who changed the same component speak one after another; 10 minutes on the combined table. Group rows by path; stub submissions are not ranked. Read the three cautions aloud: many comparisons, small $N$, one domain.
- **Wrap-up (30 minutes):** lecture 15, sections 7 to 10.

**Cost (estimate):** under 2 USD per pair on Claude, under 25 cents per pair on OpenAI, under 10 cents per pair for Jev.

## When things fail

| Failure | What to do |
|---|---|
| A participant's Colab disconnects or loses its GPU | Reconnect, rerun from the top with Run all (solutions complete the notebook), then return to the exercise. If the free GPU quota is spent, use the CPU path in the next row or pair with a neighbor |
| No T4 available | Labs 1, 9, 11 and 12 are designed for CPU. For 3, 4 and 5 use `QUICK = True`. Labs 6 and 7 have CPU paths (BERT-mini; `FAST`). Lab 10 runs only its unit checkpoints |
| A data URL fails | Upload the file into a `data` folder in Colab's Files panel; `fetch` checks there first |
| The Hugging Face Hub fails | Labs 8, 11, 13 and 14 fall back to stand-ins or stubs automatically. Labs 6, 7 and 10 have no participant fallback; teach from the lecture and rerun later |
| An API is down or rate-limited | Switch `PROVIDER` to `"open"` or the stub, or unset the TypeSafe key; checkpoints do not change. Never compare numbers across paths |
| A key is pasted into a cell | Delete the cell's output and the cell, rotate the key with the provider, and remind the room to use Colab Secrets |
| A checkpoint fails with the solution | Note the lab, the path and the message. Several thresholds are provisional (listed per module above); report it to the maintainers rather than editing the threshold in the room |

## Feedback for the next delivery

After each day, record: where the clock slipped (by module and exercise), which checkpoints confused people, the measured run time per lab on the room's runtimes, and the measured cost per API lab. These replace the estimates in this guide and in the pace sheet.
