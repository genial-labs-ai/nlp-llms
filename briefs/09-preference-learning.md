# Lab brief: `notebooks/09-preference-learning.ipynb`

**Status (2026-10-06).** `notebooks/09-preference-learning.ipynb` exists and passes its offline code check in CI. The data files this brief specifies and the seed-0 reward model are built and committed, after one change to the prompts. Part B has run on them, on an Apple M1 Pro CPU only. See "As built" below. The "Not verified" section at the end records the state when this brief was written. For what has run since, and where, see the [readiness page](../readiness.qmd).

From the Academic Director to the Neural Lab Engineer. Lecture: `lectures/09-preference-learning.qmd` (same symbols, equation labels and function names). Lab standards: `PLAN.md` section 5. Data contract: `data/README.md`. Lab 10's needs: `briefs/10-rlhf.md`. This file is not rendered by Quarto.

**Revision 2 (2026-10-05).** Romeo kept `PLAN.md`'s design: Lab 10 fine-tunes a small GPT-2, not the Lab 5 mini-GPT. Part A is unchanged. Part B (the preference data, the gold rule, the reward model and the interface with Lab 10) is rewritten for GPT-2 samples. The interface section below is identical to the one in `briefs/10-rlhf.md`.

## As built (2026-10-06, Neural Lab Engineer)

Measured on an Apple M1 Pro laptop (16 GB, CPU; Python 3.12.13, `torch` 2.14.1, `transformers` 5.18.0), with `distilbert/distilgpt2` at commit `2290a62682d06624634c1f46a6ad5be0f47f38aa`. **Not on Colab and not on a T4.**

**What changed, and why.** The first `--stats-only` run on real GPT-2 samples failed two of this brief's acceptance criteria: tie share 0.7560 (criterion at most 0.6) and $\mathrm{Acc}^\star$ at $\tau_{\text{label}} = 0.5$ of 0.5963 (criterion at least 0.6). The proposed frames ended one word before the evaluation ("... and said", "... and saw", "... found the"), and GPT-2 mostly continued them with narrative, so 86% of responses held no list word. **The fix: every frame now ends at the evaluative slot ("... and felt so", "... feeling so"), each keeping its original scene.** I chose the wording by a readability rule ("and felt so" after a completed action, "feeling so" after motion or duration), not by screening rank. The gold rule, `GOLD`, `tau_label`, the lengths, the pair counts and the Lab 9 → Lab 10 interface are unchanged. So are both lectures' descriptions of the rule; only lecture 9's example prompt was updated. The two "ordinary reply" preview items were rewritten to continue the new prompts, with the same descriptions and the same gold scores (1 and −2), and so was Lab 9's `MY_REPLY`.

**Why not the other remedies** (measured, not assumed):

| Attempt | Settings | Samples | $g \ne 0$ | Tie share | $\mathrm{Acc}^\star$ at 0.25 / 0.5 / 1 | Cap / crowding active | Verdict |
|---|---|---|---|---|---|---|---|
| 0 | Proposal as written (the build script at seed 0) | 2,000 | 0.1435 | **0.7560** | 0.6182 / **0.5963** / 0.5616 | 0 / 0 | fails tie share and $\mathrm{Acc}^\star$ |
| 1 | Lists broadened: +25 positive and +23 negative evaluative words seen in attempt 0's samples (45 and 43 words). Re-scored on the same samples, so optimistic | 2,000 (same) | 0.2385 | **0.6290** | 0.6801 / 0.6490 / 0.5977 | 0 / 0 | fails tie share even in-sample, with the lists more than doubled; each further clear word adds at most 10 of 1,713 zero samples |
| 2 | Responses of 32 tokens instead of 24 (an interface constant; `T_max` and Lab 10's sampling cost would change) | 2,000 | 0.1820 | **0.7000** | 0.6454 / 0.6188 / 0.5763 | 0 / 0 | fails tie share |
| 3 | **Frames ending at the evaluative slot** (build script, seed 0) | 2,000 | 0.3270 | 0.5070 | 0.7391 / 0.6965 / 0.6278 | 0 / 0.0015 | passes all four |
| 3, repeat | Same, `--seed 1` | 2,000 | 0.3305 | 0.4980 | 0.7435 / 0.7014 / 0.6323 | 0.0005 / 0.0010 | passes all four |
| Full build | Same, seed 0, all pairs | 18,000 | 0.3381 | 0.5019 | 0.7414 / 0.6984 / 0.6290 | 0.0001 / 0.0022 | passes all four; files written |

Raising the train pairs to 12,000 (this brief's second remedy) cannot change the tie share, and at 0.756 it would carry about 2,900 informative labels, short of the 3,200 the criterion protects. A lower $\tau_{\text{label}}$ cannot change the tie share either. The frames were screened before the final measurement: 128 samples each, for 22 candidates, on a separate screening seed (12345). The acceptance numbers above come from the build script's own seeds, not from the screening. Other measured facts from the full build: positive words twice or more 0.0104 (expected under 5%); mean complete words 17.37 (expected 14 to 18); mean gold 0.1144; every list word occurs (fewest "ugly", 4; most "good", 2,016). The sampling took 410 s on the M1 Pro CPU, against this brief's estimate of 15 to 25 minutes.

**Files.** `data/lab09_prompts.json` (49,071 bytes, SHA-256 `966971fd2f9f8533711db4cd2f8eaf36e839bd0fb9ed1c1c52efc970f6a25ed6`), `data/lab09_preferences.jsonl.gz` (1,837,490 bytes, `c00b775865dabf491542719b223ab6fb539f3728ca0348c8ca50ecb51b752326`) and `data/lab09_reward_model.pt` (4,105,277 bytes, `61c13537e4d8d0fd82e20006033f256937da01746885816c2522d607e6fad183`). They are registered in `_variables.yml` `datasets` (`lab09_prompts`, `lab09_preferences`, `lab09_reward_model`) and in `data/README.md`. `models.causal_lm_revision` is pinned, and Lab 10 now loads that commit. `tests/test_data.py`'s cap on `data/` was raised from 6 MB to 12 MB (`data/` is now 11.40 MB).

**Reward model, Exercise 4** (full settings, CPU, seeds 0 to 2, `NLP_LLMS_LAB09_MEASURE=1`):

| Seed | Held-out accuracy | $\mathrm{Acc}^\star$ | Gap | Gold-order accuracy, untied | Spearman | Epoch kept | Training |
|---|---|---|---|---|---|---|---|
| 0 (committed) | 0.6890 | 0.7127 | 0.0237 | 0.9381 | 0.7848 | 1 of 6 | 15 s |
| 1 | 0.6980 | 0.7127 | 0.0147 | 0.9343 | 0.7781 | 1 of 6 | 14 s |
| 2 | 0.6710 | 0.7127 | 0.0417 | 0.9250 | 0.7454 | 1 of 6 | 14 s |

Held-out tie share 0.467. `n_rows` 12,820; the checkpoint is 4,105,277 bytes, under the 5 MB limit for trimming the vocabulary. The whole notebook ran in 90 to 98 s per seed. Three seed-0 runs gave bit-identical weights on this CPU. The preview shows the overrating that Lab 10 relies on. With seed 0, "great" repeated (g = −6.5) scores above 83% of held-out reference responses, and the comma-separated list of nine positive words (g = −4.5) above 99.9%. The ordinary reply with one positive word (g = 1) scores above 74%. Seed 2's preview is weaker, with the list above 69%. **One observation, not acted on:** every seed keeps epoch 1. Validation-slice accuracy falls in each later epoch while the training loss goes to about 0.006, so the provided schedule (6 epochs at learning rate 2e-3) overfits these data. Keeping the best epoch handles it. A shorter schedule or a lower learning rate would save about 12 s and might raise accuracy, but that would change the reward model Lab 10 hacks, so I left it for a separate decision.

**Exercise 4 thresholds**, set by the rule above (worst seed, margin of half the spread, rounded outward):

- `ACC_MARGIN = 0.06`, from 0.0417 + 0.0270 / 2 = 0.0552.
- `GOLD_ORDER_FLOOR = 0.91`, from 0.9250 − 0.0131 / 2 = 0.9184.
- `SPEARMAN_FLOOR = 0.72`, from 0.7454 − 0.0394 / 2 = 0.7257.

Seed 0, re-run without `NLP_LLMS_LAB09_MEASURE`, passes Checkpoint 4b. These come from CPU training on one machine; a T4 trains on CUDA and was not measured. Per-seed values are recorded in `data/baselines.json` (`lab09.reward_model.seed0` to `seed2`; split `test` is the held-out pairs) and in `data/README.md`.

**Still not verified.** Nothing here ran on Colab or a T4; every number above is from an Apple M1 Pro (CPU/MPS). Lab 10's reward-hacking signature depends on these files. Lab 10 ran on its real path only in `FAST` mode on the CPU, where the trained-policy checkpoints are skipped, plus one exploratory full-settings run on MPS at seed 0. That run passed Checkpoints 3 to 5 at the provisional thresholds, with a narrow gold gap of 0.152; see `briefs/10-rlhf.md`, "As built". The seed protocol on a T4 has not run.

**Objectives exercised** (from `_variables.yml`, `m09`): frame text generation as a reinforcement-learning problem (Exercises 1 and 2, on a toy sequence task); derive and implement the policy gradient (Exercises 1 and 2); train a reward model from pairwise preferences (Exercises 3 to 5).

**Status of the numbers.** Values marked *checked* come from my own throwaway prototypes, run on 2026-10-05 in the build container (Linux x86_64, 4 vCPU, CPU only, Python 3.12.3, `torch` 2.14.1, `scipy` 1.18.1), not on Colab and not from the notebook. They show that the design is feasible; they are not the lab's results. Everything else is a target or an estimate for you to replace with a measurement. **Nothing in Part B has been measured on GPT-2 samples:** huggingface.co is blocked from the build container, so the two data files cannot be built there.

## The lab in one paragraph

Two halves. **Part A** (Exercises 1 and 2) is REINFORCE on a toy sequence task whose optimal policy and exact gradient are known, so every estimator can be checked against the truth and the effect of a baseline can be measured. **Part B** (Exercises 3 to 5) trains a reward model on synthetic pairwise preferences: two continuations sampled from **Lab 10's reference policy, the pretrained GPT-2 named by `models.causal_lm`**, labeled through Bradley–Terry by a known gold rule. It checks that the reward model recovers the gold rule on held-out samples and saves it for Lab 10. The pairs file stores GPT-2's token ids and the decoded text, and the reward model is small and trained from scratch, so **the core path downloads no model and runs on a CPU runtime.**

## Design choices for Part B (decided here, with reasons)

| Question | Decision | Reason |
|---|---|---|
| Which GPT-2 | `models.causal_lm`, the existing key (Lab 6's distilled GPT-2, 81.9M parameters, Apache 2.0), plus a new key `models.causal_lm_revision` for the commit hash | Participants met it in Lab 6, so it continues the running thread; no new license to check; with 6 blocks instead of 12 it is roughly half the cost per token of `openai-community/gpt2` (124M), which matters for Lab 10's three training runs. Quality of 24-token continuations is not what the lab measures. Fallback if its samples prove unreadable on the first run: a new key for the 124M model, and re-measure Lab 10's budget |
| Where prompts come from | A fixed list **we write**: 16 subjects × 16 frames, split by frame. No download; licensed with the rest of the repository | arXiv Topics titles (CC0, already in `data/`) were considered and rejected. GPT-2 continues a paper title with technical prose, which rarely contains evaluative words, so most pairs would be ties; titles vary in length, which brings padding into every exercise; and "positive tone in an abstract" is a muddier hidden rule than "a warm reply". Everyday openings that end where an evaluation naturally follows ("... and said") make the gold rule's words frequent in the reference's own samples |
| Fixed lengths | Prompts of exactly 8 GPT-2 tokens, responses of exactly 24, the end-of-text token never sampled | No padding, no attention masks, no position-id shifts in Labs 9 or 10; the score sits at the last column of `token_rewards` as in lecture 10. Eight common words with a leading space are almost always eight GPT-2 tokens; the build script asserts it |
| Gold rule | Positive and negative word lists for modern English; the cap kept; the **non-word term replaced by a crowding term** | See "Gold rule" below |
| Reward model | A small transformer trained from scratch on GPT-2 token ids, with a compact embedding table (ids seen at least twice in the train pairs, all others sharing row 0) | Runs on a CPU in Lab 9 with no download; the file is small enough to commit (estimate 2 to 4 MB); its blind spots are predictable, because it can only learn from tokens that occur in reference samples. A GPT-2-backbone scorer (82M parameters with a scalar head, frozen or LoRA-adapted) was considered and rejected: it needs the Hub and, at Lab 9's data size, a GPU (a CPU forward pass over the 18,000 responses of the pairs alone is an estimated 10 to 20 minutes); its file is either too large to commit or, as a LoRA adapter, adds `peft` and a second GPT-2 copy to Lab 10; and its pretrained features generalize "positivity" to words outside the lists, which blurs the gap between proxy and gold that Lab 10 must show reliably. What we lose: the practice of section 8, where the reward model starts from the SFT model. The lecture now says so |

## Dependency and build order (read first)

1. **`data/build_lab09_preferences.py`** (suggested name, in the pattern of `data/build_arxiv_topics.py`). Runs once, on Colab or any machine with Hub access; a T4 makes it about a minute of sampling (estimate), a CPU about 15 to 25 minutes (estimate). It writes `data/lab09_prompts.json` and `data/lab09_preferences.jsonl.gz`, and prints the reference-sample statistics below. It cannot run in the build container (huggingface.co is blocked), so until someone runs it elsewhere, Part B is **written, not run**.
2. **`data/lab09_reward_model.pt`**: the reward model from this notebook's own solution code at full settings and seed 0. This step needs no Hub access and no GPU, so it can run in the build container once step 1's files are committed.
3. Lab 10 loads all three files.

There is no longer a reference checkpoint to build: the reference is the pretrained GPT-2, so `data/lab10_reference_gpt.pt` and `data/build_lab10_reference.py` from revision 1 are dropped.

## Part A: the toy task (provided, in the notebook, no data)

Unchanged from revision 1.

| | |
|---|---|
| Vocabulary | 8 symbols; a start symbol `BOS` (index 8) for the first step |
| Episode | $T = 6$ actions; state $s_t$ = (position $t$, previous symbol) |
| Policy | tabular "bigram policy": logits `theta` of shape `(T, 9, 8)`, indexed by position and previous symbol; initialized to zero, so training starts at the uniform policy |
| Target | `y_star`, 6 symbols drawn with `torch.Generator().manual_seed(0)` (*checked*: `[4, 7, 5, 0, 3, 3]` with `torch.randint(0, 8, (6,), generator=...)`) |
| Rewards | sequence-level: zero except $r_T = R = \frac{1}{T}\sum_t \mathbf{1}[y_t = y^\star_t]$; token-level: $r_t = \mathbf{1}[y_t = y^\star_t] / T$. Same total |
| Known optimum | $\pi(y^\star) = 1$, $J = 1$. Uniform policy: $J = 1/8$ (*checked*: exact enumeration gives 0.1250) |
| Exact gradient | `exact_gradient(theta)`: enumerate all $8^6 = 262{,}144$ responses, compute $J$ (lecture eq. `objective`) and backpropagate. *Checked:* 0.30 s on CPU |
| Cheap exact $J$ | `expected_return(theta)`: forward marginals over the previous symbol, $O(T \cdot 9 \cdot 8)$; use it to monitor training (calling `exact_gradient` every step is too slow: *checked*, it made 3 seeds × 400 steps take 43 s) |

**Provided:** `sample_toy(theta, N, generator) -> (logits (N, T, 8), actions (N, T))`; `token_logprobs(logits, actions) -> (N, T)` (lecture eq. `seq-logprob`, summed over `t` for the sequence); `toy_rewards(actions, level)`; `exact_gradient`; `expected_return`; `batch_mean_baseline(returns)`; `train_toy(baseline_fn, seed, steps, lr, N)`; `estimate_gradients(theta, K, N, kind)` for the variance table.

## Part B: preference data (provided as committed files)

**Prompts.** Subject (2 tokens) + frame (6 tokens), every combination, 256 prompts. Splitting by frame keeps the frames of held-out pairs and of Lab 10's evaluation prompts out of the training pairs. Adjust any phrase that does not tokenize to the stated length, and keep every list word out of the prompts (the build script asserts both). **As built (2026-10-06)**; the proposal's frames ended one word earlier and failed the acceptance criteria (see "As built" above):

- Subjects (16): My mother, My father, My sister, My brother, My friend, My boss, Our neighbor, The teacher, The doctor, The manager, The waiter, The student, The driver, The coach, The nurse, The chef.
- Train frames (10, giving 160 prompts): looked at the results feeling so; opened the letter and felt so; walked into the room feeling so; tasted the soup and felt so; read the review and felt so; came home late and felt so; listened to the song feeling so; looked out the window feeling so; heard the news and felt so; finished the long day feeling so.
- Held-out frames (2, giving 32 prompts): watched the game and felt so; visited the old house feeling so.
- Evaluation frames for Lab 10 (4, giving 64 prompts): left the new restaurant feeling so; read the email and felt so; saw the bill and felt so; spent the whole weekend feeling so.

Prompts have no end-of-text token in front. `text == tokenizer.decode(ids)` must hold exactly; assert it.

**Responses.** Sampled from the reference policy exactly as Lab 10 samples (see the interface): float32, `eval()`, temperature 1, no truncation, end-of-text removed, 24 tokens. Record the seed, the GPU or CPU type, and the `torch` and `transformers` versions in the JSON files: sampling on different hardware is not bit-for-bit reproducible, which is why the files are committed and hashed.

**Pairs.** For each pair, a prompt (cycling through the split's prompts in a fixed shuffled order) and two independent responses. Starting sizes: **8,000 train pairs** (50 per train prompt) and **1,000 held-out pairs** (about 31 per held-out prompt). Label each pair with probability `bt_prob(g_a, g_b, tau_label)` from a fixed generator and store the result as `chosen`/`rejected`. Store both gold scores, so that the stretch can relabel. File size estimate: 4 to 6 MB uncompressed, 1.5 to 2 MB gzipped.

**Gold rule.** The function is fixed in the interface section below. What changed from revision 1, and why:

1. **Lists for modern English.** Twenty positive words (counted once each: `n_pos` counts *distinct* words) and twenty negative words (every occurrence counts), lowercased, exact forms. Words are runs of letters with internal apostrophes (`don't`), so a quotation mark is no longer part of a word.
2. **Complete words only**, as in revision 1: the prompt ends with a complete word, a response that starts with a letter continues it (the two are joined), and the last word is dropped when it touches the end of the response, since the 24-token limit may have cut it.
3. **The cap stays** at three distinct positive words: no credit for a fourth.
4. **The non-word term is replaced by a crowding term**, $-10 \max(0, f_{\text{list}} - 0.25)$, where $f_{\text{list}}$ is the share of complete words that are on either list. Two reasons. First, a non-word term needs an English word list, and no license-clean one is in `data/`: the vocabularies of Tiny Shakespeare and arXiv Topics miss everyday words and names, so the term would fire on ordinary GPT-2 text, the reward model would learn it from the pairs, and it would stop being a blind spot. Second, GPT-2's byte-level vocabulary rarely produces non-words at temperature 1, unlike the 1.55-nats-per-character mini-GPT; an unpenalized GPT-2 policy hacks a word-based reward by stuffing real words, so the hidden rule has to object to stuffing. The crowding term does, whether the stuffing repeats one word or lists many: on hand-written cases (*checked* with the interface code, prompt "My sister opened the old letter and felt"), an ordinary reply with one positive word scores 1.00; one positive word repeated to fill the response, −6.50; nine distinct positive words, −3.50; "I love it!" repeated, 0.38; a natural reply with four distinct positive words in 14 complete words, 2.64 (cap plus a small crowding penalty).

The structure is what Lab 10 needs: on the reference's own samples the cap and the crowding term should almost never be active, so the reward model learns "positive words up, negative words down" and nothing about the limits. An unpenalized policy that stuffs positive words is then overrated by $r_\phi$ and penalized by $g$.

**Measure on 2,000 reference samples before fixing anything** (the build script prints these; send them to me and to the Lab 10 author):

| Statistic | Acceptance criterion | If it fails |
|---|---|---|
| Share of samples with $g \ne 0$; tie share of random same-prompt pairs | tie share at most 0.6, so that 8,000 pairs carry at least 3,200 informative labels | add frequent evaluative words seen in the samples to the lists (first choice), or raise the train pairs to 12,000 |
| Cap active ($n_{\text{pos}} \ge 4$) | under 1% of samples | tell me: the cap is named in lecture 10, section 7 |
| Crowding active ($f_{\text{list}} > 0.25$) | under 1% of samples | raise `crowd_threshold` in steps of 0.05 and re-measure |
| Some positive word occurring twice or more | report; expected under 5% | none; it is the "distinct" blind spot, report it |
| Mean number of complete words per response | report (expected 14 to 18) | none |
| Count of each list word | report | drop a word only if it occurs in prompts or is ambiguous in the samples |
| $\mathrm{Acc}^\star$ (lecture eq. `oracle`) at $\tau_{\text{label}}$ = 0.25, 0.5, 1 | report | keep 0.5 unless $\mathrm{Acc}^\star$ on all pairs is below 0.6 |

Change only the lists, `crowd_threshold`, the pair count or `tau_label`; do not change the structure of the rule without telling me, because both lectures describe it in words. Record the final values in both JSON files, in the reward-model checkpoint, in `_variables.yml` and in `data/README.md`.

## Participant writes / provided

Participants write six short functions: `returns_to_go`, `reinforce_loss`, `loo_baseline`, `bt_prob`, `bt_loss`, `fit_normalizer`. Everything else is provided: the toy environment and exact-gradient code above; `complete_words`, `gold_reward` and `GOLD`; the loaders for the two data files; the construction of `vocab_map`; `oracle_accuracy(g_a, g_b, tau_label)` (lecture eq. `oracle`); the `RewardModel` class, the training loop and the evaluation cell; plotting; the save and reload cell.

## Core path (50 minutes)

Format per exercise: Predict, Run, Explain, Check; `# TODO N` stub, folded solution, short "why this works" note.

| # | Participant writes | Equation | Checkpoint | Metric tested | Min |
|---|---|---|---|---|---|
| 0 | Nothing: run setup; read the toy environment; print `y_star`, $J$ of the uniform policy (0.125) and three preference pairs (prompt, both responses as text, gold scores, which was chosen) | – | none | none | 3 |
| 1 | `returns_to_go(rewards)` `(N, T) -> (N, T)` and `reinforce_loss(logp, returns, baseline=0.0)` (lecture code, section 5) | `return`, `pg-loss`, `reinforce` | (a) hand cases: `[[1, 0, 2]] -> [[3, 2, 2]]`; loss on a 2 × 3 example; no gradient reaches `returns` or `baseline`. (b) At the uniform policy, the mean of $K = 500$ estimates (batches of $N = 16$, sequence-level reward, no baseline) against `exact_gradient`: cosine ≥ 0.95 (*checked*: 0.984, relative error 0.18). (c) Provided training, Adam, learning rate 0.05, 300 steps, $N = 16$, no baseline: $J$ above a threshold set from five seeds (*checked*: all 5 seeds reached $J \ge 0.9$, at steps 148 to 295; final $J$ 0.91 to 0.98) | Cosine similarity; $J$ | 12 |
| 2 | `loo_baseline(returns)` (eq. `loo`) | `score-zero`, `baseline`, `loo`, `reward-to-go` | (a) hand case. (b) **Exact identity**, deterministic: on the same batch, the batch-mean gradient equals $(1 - 1/N)$ times the leave-one-out gradient (`allclose`). (c) Unbiasedness: mean of $K = 500$ estimates with your baseline against `exact_gradient`, cosine ≥ 0.95 (*checked*: 0.991). (d) Variance table, predicted first: total variance (sum of per-coordinate variances) of $\hat g$ at the uniform policy for no baseline, leave-one-out, and token-level rewards through `returns_to_go` with leave-one-out (*checked*: 0.0110, 0.0062, 0.0037; ratios 0.56 and 0.34). Assert only the ordering, if it holds on all seeds. (e) Five seeds trained with and without the baseline at the same learning rate: steps to $J \ge 0.9$ (*checked* at 0.05: with the baseline 90 to 107 steps, without 148 to 295) | Variance ratio; steps to $J \ge 0.9$ | 9 |
| 3 | `bt_prob(g_a, g_b, tau_label)` (eq. `bt-rater`) | `bt`, `bt-rater`, `oracle` | (a) `bt_prob(1, 0, 0.5) == sigmoid(2)` (0.8808); `bt_prob(a, b) + bt_prob(b, a) == 1`; works on tensors. (b) Against the file: binned by $\lvert g_a - g_b \rvert$, the share of training pairs where the higher-gold response is `chosen` is within a tolerance of your `bt_prob` (set from the file's size). (c) Provided: tie share and $\mathrm{Acc}^\star$ on the held-out pairs, printed: the ceiling for Exercise 4 | Agreement of predicted and observed win rates | 8 |
| 4 | `bt_loss(reward_w, reward_l)` (eq. `rm-loss`) | `rm-loss` | (a) `bt_loss(0, 0) == log 2`; matches `F.binary_cross_entropy_with_logits(reward_w - reward_l, ones)`; unchanged when a constant is added to both inputs. (b) Provided training of `RewardModel` on the GPT-2 ids (CPU). On held-out pairs: pairwise accuracy ≥ $\mathrm{Acc}^\star$ − margin; accuracy against the gold ordering on untied pairs above a threshold; Spearman correlation between $r_\phi$ and $g$ on the held-out responses above a threshold. All thresholds from at least three seeds. (c) Provided "preview" cell, printed only: $r_\phi$ beside $g$ for the `preview` items of `lab09_prompts.json` (one positive word repeated to fill 24 tokens; a comma-separated list of distinct positive words; "I love it!" repeated; an ordinary reply with one positive word; an ordinary reply with two negative words). An optional cell lets participants score their own reply of up to 24 tokens if the GPT-2 tokenizer can be downloaded; it is skipped otherwise | Held-out accuracy against $\mathrm{Acc}^\star$; rank correlation with $g$ | 12 |
| 5 | `fit_normalizer(raw_scores) -> (mean, std)` (eq. `normalize`), population standard deviation | `normalize` | Normalized scores of the held-out responses have mean 0 and standard deviation 1 (atol 1e-4); the provided cell saves `lab09_reward_model.pt`, reloads it into a fresh `RewardModel`, and asserts identical outputs on a batch; `forward` returns shape `(N,)` for `(N, 8)` prompt and `(N, 24)` response inputs | Exact reload; normalization | 6 |

Minutes: 3 + 12 + 9 + 8 + 12 + 6 = 50.

**Prototype result for the preference half, as a feasibility check only** (*checked*, revision 1, a different design: real 20-word Shakespeare spans, a distinct-count rule without the cap or a penalty term, and a word-level linear reward model, 5,000 train and 1,000 held-out pairs; not the lab's numbers and not GPT-2 text): held-out accuracy 0.673 against $\mathrm{Acc}^\star$ 0.682, accuracy against the gold ordering on untied pairs 0.934, Spearman 0.71, all 40 list words learned with the right sign; one cheerful word repeated ten times scored 23.6 against 8.2 for a line with five distinct cheerful words (gold 1 against 5). That is the overrating Lab 10 relies on. At the lowest label temperature, held-out accuracy exceeded $\mathrm{Acc}^\star$ by 0.01 on 1,000 pairs, within sampling error (standard error about 0.014). Set the Exercise 4 margin with that in mind.

## Stretch (one section, last, optional; not needed by Lab 10)

Relabel the **training** pairs from their stored gold scores at $\tau_{\text{label}} \in \{0.25, 0.5, 1, 2, 4\}$ (fixed seed), retrain the reward model with a shorter budget for each, and plot held-out accuracy against the gold ordering, beside $\mathrm{Acc}^\star$ for each temperature. Held-out labels stay at the file's temperature. Optional second cell: flip a share $\varepsilon$ of labels at random instead. Assert nothing beyond shapes.

## Lab 9 → Lab 10 interface

*This section is identical, character for character, in `briefs/09-preference-learning.md` and `briefs/10-rlhf.md`. Change both or neither.* Notebooks cannot import each other, so Lab 10 restates every constant, function and class below verbatim from Lab 9. Changing a name, signature, shape, constant or file field means changing both labs.

**Reference policy and sampling.** $\pi_{\text{ref}}$ is the causal LM named by `models.causal_lm` in `_variables.yml` (the distilled GPT-2 of Lab 6: 6 blocks, 81.9M parameters), at the revision to be pinned as `models.causal_lm_revision` (a proposed key), with its own tokenizer. It is loaded in float32 and always used in `eval()` mode, since GPT-2 configurations set dropout to 0.1. Responses are sampled at temperature 1 with no top-$k$ or nucleus truncation, from the next-token distribution **without the end-of-text token**: GPT-2's end-of-text id is 50256, the last row of its 50,257-entry vocabulary, so the policy's logits are `logits[..., :EOS_ID]` (50,256 entries) and every response has exactly `RESPONSE_LEN` tokens. No padding anywhere: every prompt is `PROMPT_LEN` tokens and the prompts carry no end-of-text token at the start.

```python
PROMPT_LEN = 8      # tokens in every prompt; the build script asserts it
RESPONSE_LEN = 24   # tokens in every response; end-of-text is never sampled
EOS_ID = 50256      # == tokenizer.eos_token_id == len(tokenizer) - 1; asserted on load
```

**Files.** All three are loaded through the `data/README.md` contract (canonical repository URL, jsDelivr fallback, SHA-256) and get entries in `_variables.yml` `datasets`.

| File | Built by | Contents |
|---|---|---|
| `data/lab09_prompts.json` | `data/build_lab09_preferences.py`, run once on a machine with Hub access | `{"format": "nlp-llms/lab09-prompts", "version": 1, "model": str, "revision": str, "subjects": [16 str], "frames": {"train": [10 str], "heldout": [2 str], "eval": [4 str]}, "prompts": {"train": [160], "heldout": [32], "eval": [64]}, "preview": [...]}`. A prompt is subject + frame, `{"text": str, "ids": [8 int]}`, with `text == tokenizer.decode(ids)`. A preview item is `{"name": str, "response": str, "response_ids": [int]}`, scored after `prompts["heldout"][0]` |
| `data/lab09_preferences.jsonl.gz` | the same script | One JSON object per line: `prompt` (str), `prompt_ids` (8 int), `chosen`, `rejected` (str), `chosen_ids`, `rejected_ids` (24 int each), `gold_chosen`, `gold_rejected` (float), `split` (`"train"` or `"heldout"`). Train pairs use train prompts and held-out pairs held-out prompts. Response text is `tokenizer.decode(ids, clean_up_tokenization_spaces=False)` |
| `data/lab09_reward_model.pt` | Lab 9's own solution code at full settings, seed 0; CPU, no Hub access needed | The checkpoint below |

Lab 10's evaluation prompts are `prompts["eval"]`; their frames occur in no preference pair.

**Functions** (lecture 9, sections 5, 7 and 8):

```python
def returns_to_go(rewards):                       # (N, T) -> (N, T); G_t = r_t + ... + r_T
def reinforce_loss(logp, returns, baseline=0.0):  # logp, returns: (N, T); baseline: (N, T), (1, T) or scalar
                                                   # -((returns - baseline).detach() * logp).sum(-1).mean()
def loo_baseline(returns):                        # (N, T) -> (N, T); per position, mean of the other samples
def batch_mean_baseline(returns):                 # (N, T) -> (1, T); per position, mean over the batch
def bt_loss(reward_w, reward_l):                  # (n,), (n,) -> scalar: -F.logsigmoid(reward_w - reward_l).mean()
def bt_prob(g_a, g_b, tau_label):                 # sigmoid((g_a - g_b) / tau_label); floats or tensors
```

Returns are not normalized in either lab. Lab 10 calls `reinforce_loss(logp, returns, batch_mean_baseline(returns))`; the batch-mean baseline is $(1 - 1/N)$ times the leave-one-out estimate (lecture 9, section 4). Lab 10's `dpo_loss` calls `bt_loss` with the implicit rewards $\beta(\log\pi_\theta - \log\pi_{\text{ref}})$.

**Gold rule.** Provided in both labs, and called what it is: a rule we wrote, known only because the data are synthetic. The values in `GOLD` are starting values; the build script's measurements may change the lists, `crowd_threshold`, `crowd_weight` and `tau_label` (see Lab 9's brief), and the final values are recorded in both data files and in the reward-model checkpoint.

```python
WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)*")
GOLD = {
    "positive": ["good", "great", "happy", "love", "loved", "wonderful", "beautiful", "nice",
                 "best", "amazing", "perfect", "excellent", "glad", "fun", "enjoyed",
                 "delicious", "proud", "lovely", "fantastic", "excited"],
    "negative": ["bad", "sad", "terrible", "awful", "hate", "angry", "worst", "horrible",
                 "poor", "wrong", "sick", "afraid", "worried", "upset", "disappointed",
                 "pain", "ugly", "cried", "scared", "dead"],
    "cap": 3, "crowd_threshold": 0.25, "crowd_weight": 10.0, "tau_label": 0.5,
}

def complete_words(prompt, response):
    """Lowercased words that end inside the response. A first word that continues the
    prompt's last word is joined to it; a word that touches the end may have been cut
    by the length limit, so it is dropped."""
    text = prompt + response
    return [m.group().lower() for m in WORD.finditer(text) if len(prompt) < m.end() < len(text)]

def gold_reward(prompt, response, gold=GOLD):
    words = complete_words(prompt, response)
    if not words:
        return 0.0
    pos, neg = set(gold["positive"]), set(gold["negative"])
    n_pos = len({w for w in words if w in pos})             # distinct positive words
    n_neg = sum(w in neg for w in words)                    # every negative occurrence
    f_list = sum(w in pos or w in neg for w in words) / len(words)
    return float(min(n_pos, gold["cap"]) - n_neg
                 - gold["crowd_weight"] * max(0.0, f_list - gold["crowd_threshold"]))
```

In symbols, $g(x, y) = \min(n_{\text{pos}}, 3) - n_{\text{neg}} - 10 \max(0, f_{\text{list}} - 0.25)$.

**Reward model class** (keep the attribute names: the `state_dict` keys depend on them):

```python
class RewardModel(nn.Module):
    """Small transformer trained from scratch on GPT-2 token ids: Lab 5's pre-norm block
    with a causal mask; a scalar head reads the last position of prompt + response."""
    def __init__(self, n_rows, d=64, n_h=4, n_l=2, T_max=32, d_ff=256): ...
        # buffers: vocab_map (50257,) int64, GPT-2 id -> embedding row (row 0 = any other token);
        #          score_mean = 0.0, score_std = 1.0 (set by Lab 9, Exercise 5)
        # modules: tok_emb (n_rows, d), pos_emb (T_max, d), blocks (Lab 5 Block, dropout 0.0),
        #          ln_f, head = nn.Linear(d, 1)
    def raw(self, prompt_ids, response_ids):      # (N, P), (N, L) int64 GPT-2 ids -> (N,) raw r_phi
    def forward(self, prompt_ids, response_ids):  # (N,) normalized: (raw - score_mean) / score_std
```

- `vocab_map` is built once in Lab 9 from the **train** pairs (prompt, chosen and rejected ids): each GPT-2 id that occurs at least twice gets its own row, numbered from 1 in increasing id order; every other id maps to row 0. `n_rows` is that count plus 1. Row 0 is trained like any other row, on the rare tokens that map to it.
- Input: GPT-2 ids taken straight from the policy's samples; Lab 10 decodes nothing to score a response. `P + L <= T_max`; in both labs `P = 8` and `L = 24`.
- Output: `(N,)` float32, normalized so that the held-out reference responses (both responses of every held-out pair) have mean 0 and population standard deviation 1. Lab 10's reward-gain thresholds are in these units.
- In Lab 10 it is frozen: `eval()`, `requires_grad_(False)`. The policy gradient never differentiates it.

**Checkpoint `data/lab09_reward_model.pt`**, a dictionary that loads with `torch.load(path, weights_only=True)`:

```python
{"format": "nlp-llms/lab09-reward-model", "version": 2,
 "config": {"n_rows": ..., "d": 64, "n_h": 4, "n_l": 2, "T_max": 32, "d_ff": 256},
 "policy": {"model": ..., "revision": ..., "eos_id": 50256, "vocab_size": 50257,
            "prompt_len": 8, "response_len": 24},
 "state_dict": model.state_dict(),      # includes vocab_map, score_mean and score_std
 "gold": GOLD,                          # the final lists, cap, crowd_threshold, crowd_weight, tau_label
 "metrics": {"heldout_acc": ..., "oracle_acc": ..., "acc_gold_order_untied": ..., "spearman_gold": ...},
 "data_sha256": {"prompts": ..., "preferences": ...}}
```

On load, Lab 10 asserts that `policy` agrees with its constants and its tokenizer, and that `gold` equals its restated `GOLD`. Size estimate: 2 to 4 MB (the embedding is `n_rows` × 64 float32, with `n_rows` expected between 8,000 and 15,000; `vocab_map` adds 0.4 MB; the rest is about 0.1M parameters). If it exceeds 5 MB, raise the minimum count from 2 to 3 and record it. The committed copy is written by the notebook's own solution code (for example, when an environment variable names the output path), so the class in the notebook and the committed weights cannot drift; a test loads the committed file into the class as restated in Lab 10.

## Compute budget (CPU runtime or free Colab T4; whole notebook under 10 minutes)

| Step | Budget (estimates unless *checked*) |
|---|---|
| Setup, load the two data files (about 2 MB), Exercise 0 | under 0.5 min |
| Part A: checkpoints, five seeds × two baselines × 300 steps, variance table | under 0.5 min on CPU (*checked* in the prototype: about 3 to 6 s per five seeds, 2.4 s per variance table) |
| Exercise 4: reward-model training, 8,000 pairs of 32-token sequences, $d = 64$, 2 blocks | 1 to 2 min on CPU, less on a T4 |
| Evaluation, preview, save and reload | under 0.5 min |
| Stretch: five shorter retrainings | under 3 min |

No GPU is needed and no model is downloaded in the core path. A `FAST` flag (on without a GPU or when `NLP_LLMS_QUICK` is set, as in Labs 3 and 5) may shorten reward-model training for CI; set its thresholds from its own seeds. The build script is not part of the 10 minutes.

## Reliability: how the numbers get verified

- **Part A** needs nothing from the Hub. Set its thresholds from five seeds on CPU, as revision 1 said; this can be done in the build container.
- **Part B** needs the two data files, which need the Hub. Until someone runs `data/build_lab09_preferences.py` on Colab or another machine with Hub access and commits the files, Part B is **written, not run**. For code testing before then, a test-only synthetic stand-in file (random GPT-2-range ids, texts assembled from list words and filler) may exercise the cells; its numbers are not results and must not set thresholds.
- Once the files exist: train the reward model at seeds 0, 1 and 2 (CPU is enough). Each threshold of Exercise 4 is set from the worst of the three seeds, with a margin of about half the spread across seeds; no threshold is set from seed 0 alone. Record per-seed values in `data/baselines.json` under `lab09`, with the date, hardware and versions. Commit the seed-0 checkpoint. Also save the seed-1 and seed-2 checkpoints outside the repository, for Lab 10's secondary robustness check.

## Flags for the Lab Engineer

1. **Learning rate in Part A.** *Checked:* at Adam 0.1 without a baseline, 2 of 5 seeds stall near $J = 0.83$ (one position locked on a wrong symbol); at 0.2, all 5 stall; with the leave-one-out baseline every seed converges at 0.05, 0.1 and 0.2. The core path uses 0.05 so that Exercise 1 converges without a baseline. An optional cell in Exercise 2 may rerun at 0.1 to show stalling, which the lecture mentions without numbers; assert nothing on it.
2. **Unbiasedness checks run at the uniform policy.** *Checked:* at a mid-training policy ($J \approx 0.67$) the exact gradient is small and the Monte Carlo mean of 500 batches was far from it (relative error 0.4 to 2.0), so cosine checks there are not reliable.
3. **Use the lecture's names:** `theta`, `y_star`, `logp`, `rewards`, `returns`, `baseline`, `g_a`, `g_b`, `tau_label`, `reward_w`, `reward_l`, `gold`. Do not name anything `V` (the vocabulary) or `r` alone. No `kappa`, no `beta` (Module 10's KL coefficient).
4. **The gold rule is ours.** The notebook must call it what it is: a hidden rule we wrote so that we can check the reward model, known only because the data are synthetic. A real preference dataset has no gold score.
5. **Exercise 4's claim is in-distribution only.** The checkpoint says the reward model recovers the gold rule on held-out *reference* samples, nothing more. The preview cell prints and never asserts.
6. **The reward model is from scratch on purpose.** Say in the notebook, in one or two sentences, that real reward models start from a pretrained model (lecture section 8) and why this one does not: CPU budget, and a reward model whose blind spots we can predict.
7. **Build script checks.** Assert: every prompt is 8 tokens and decodes to its text; no list word occurs in a prompt; no response contains the end-of-text id; every response is 24 tokens; stored gold scores equal `gold_reward(prompt, decode(ids))` recomputed with the interface code. Print the statistics table above.
8. **No RLCD or TypeSafe content.** Nothing in this lab concerns Module 12.
9. **Save the gradient estimates** of Exercise 2 (one coordinate, both estimators, at the uniform policy) to `images/09-baseline-variance.json`, so that the lecture's data figure can be drawn from measured values.
10. **Send me, measured:** the reference-sample statistics, Part A numbers for five seeds, the reward model's held-out accuracy, $\mathrm{Acc}^\star$, gold-order accuracy and Spearman for three seeds, the preview table, the size of the checkpoint and `n_rows`, the stretch curve, and run times on CPU and, if possible, a T4.

## Coordination and open questions

1. **Policy model** (Romeo, decided 2026-10-05): the small GPT-2 of `PLAN.md`, by its `_variables.yml` key `models.causal_lm`.
2. **Notation agreed with lecture 10** (unchanged): $s_t = (x, y_{<t})$, actions are tokens, $G_t$ the return from step $t$, $b_t$ the baseline, $\succ$, $\mathcal{D}_x$, $\mathcal{D}_{\text{pref}}$, $(x, y_w, y_l)$, $r_\phi$, $v_\psi$ for a value function, $\tau$ never a trajectory. Lecture 9 writes the response length $T$, as Modules 1 to 7 do; lecture 10 writes $\lvert y \rvert$. Harmless; left as is.
3. **Label temperature:** $\tau_{\text{label}}$, dividing the gold gap. Starting value 0.5, to be confirmed from the reference-sample statistics.
4. **`_variables.yml`** (not edited; for the Architect or Romeo): add `models.causal_lm_revision`; add `datasets` entries for `lab09_prompts.json`, `lab09_preferences.jsonl.gz` and `lab09_reward_model.pt` with SHA-256, byte counts, the repository and jsDelivr URLs, license (ours, CC BY 4.0; the responses are samples from an Apache 2.0 model), `modules: [9, 10]`, the seed and $\tau_{\text{label}}$. `modules.m09.stack` stays `[PyTorch, Hugging Face]`: the data are GPT-2 token ids and the optional preview cell uses GPT-2's tokenizer. This withdraws revision 1's proposal of `[PyTorch]`.
5. **`PLAN.md`** (not edited): section 4, Lab 9: replace "build a synthetic pairwise-preference dataset with a known hidden preference" with "load a synthetic pairwise-preference dataset with a known hidden preference, sampled from Lab 10's reference policy (the small GPT-2 of Lab 6), and check its labels against the Bradley–Terry model". Section 6: add the Hub dependency of the two data files (see the Lab 10 brief for the exact row). Section 7, Day 7: add "Build `data/lab09_prompts.json` and `data/lab09_preferences.jsonl.gz` on a machine with Hub access, and send the reference-sample statistics".
6. **`data/README.md`** (Architect): the "Pairwise preferences" row now reads "Generated in the notebook. No file"; it becomes: "Synthetic: GPT-2 continuations of 256 prompts we wrote, labeled by a known rule. Built by `data/build_lab09_preferences.py`; copies committed".

## Not verified by the Director

- No notebook exists. Nothing above was run as the lab; the *checked* numbers come from prototypes with simplified code, and the preference prototype used a different design (word-level, Shakespeare text).
- **Everything about GPT-2 samples**: the statistics of the gold rule, the tie share, how often the cap and the crowding term fire, $\mathrm{Acc}^\star$. None could be measured: huggingface.co is blocked in the build container. The gold function itself was checked only on nine hand-written replies.
- That every proposed prompt is exactly 8 GPT-2 tokens and every frame 6. I could not load GPT-2's tokenizer (huggingface.co, openaipublic.blob.core.windows.net and cdn.jsdelivr.net were all refused by the proxy on 2026-10-05); the build script must assert it.
- That a from-scratch reward model of the suggested size reaches $\mathrm{Acc}^\star$ − margin on 8,000 GPT-2 pairs within the budget, and the checkpoint size (2 to 4 MB is an estimate from an assumed `n_rows`).
- That the cap and the crowding term produce a reliable reward-hacking gap in Lab 10. That is Lab 10's main risk; this brief preserves the structure it needs.
- Run times on a T4 and of the build script.
- Citations in the lecture: arxiv.org, ar5iv, Springer, JSTOR and the Sutton and Barto site were blocked by the build container's proxy on 2026-10-05. Gao et al. (ICML 2023, PMLR 202) and Ahmadian et al. (ACL 2024) were confirmed through search results only; the InstructGPT reward-model description (6B GPT-3, scalar head) through secondary sources only; the rest is from memory. No link was opened.
- `quarto render` was not run (Quarto is not installed in the build environment).
