# Lab brief: `notebooks/10-rlhf.ipynb`

From the Academic Director to the Neural Lab Engineer. Lecture: `lectures/10-rlhf.qmd` (same symbols and equation names). Lab standards: `PLAN.md` section 5. Data contract: `data/README.md`. This file is not rendered by Quarto.

**Objectives exercised** (from `_variables.yml`, `m10`): describe the three-stage RLHF pipeline; optimize a small LM against a reward model with a KL constraint; apply DPO and compare; name RLHF's failure modes and observe one.

**Every time, size and threshold below is an estimate or a target, not a measurement,** unless marked *checked*. Replace each with the value you measure, and tell me if a target cannot be met. *Checked* means I ran it on 2026-10-05 in the build container (CPU, 4 threads, `torch` 2.14.1), not on Colab.

## Decision needed from Romeo: the policy model

`PLAN.md` (section 4, Lab 10) says "fine-tune a small GPT-2". **I recommend the character-level mini-GPT of Lab 5 instead**, trained on Tiny Shakespeare and shipped as a checkpoint. The lecture is written for that choice; if GPT-2 is kept, the lines to change are listed under "If GPT-2 is kept" below.

| | Lab 5 mini-GPT (recommended) | Small GPT-2 (`distilbert/distilgpt2`, 82M, or `gpt2`, 124M) |
|---|---|---|
| Hub access | none | model and tokenizer; and Lab 9's reward model would need GPT-2's tokenizer too |
| Can be built and checked in the build container | yes (huggingface.co is blocked there) | no: written, not run, like Labs 6–8 |
| PLAN.md's "reliable across seeds" review | can be measured over several seeds before Day 7 sign-off | cannot be measured until someone has a T4 with Hub access |
| CI (CPU, no keys) | runs the real lab in a `FAST` mode | would need a random stand-in, so CI would not test the demonstration |
| Compute on a T4 | small: 826,433 parameters (*checked* for the Lab 5 shape) | 100 to 150 times more parameters; two RL runs plus DPO plausibly fit 10 minutes, the stretch sweep does not (estimate) |
| Exact KL over the vocabulary | 65 characters per position: trivial | 50,257 tokens per position: affordable, heavier |
| Running thread (PLAN.md section 1) | participants align the model they built in Lab 5 | a new model |
| Readability of samples | pseudo-Shakespeare; degeneration (repeated words, non-words) is easy to see | real English |
| Exercises the Hugging Face stack | no | yes (`transformers` loading and generation) |

Why I recommend the mini-GPT: the one claim PLAN.md requires us to verify for this lab, that the reward-hacking demonstration is reliable across seeds, can only be verified with a model the build container can run, and that same model is what CI will run. Readable English is the main loss; for a demonstration whose point is degenerate text, pseudo-Shakespeare is enough. The Hugging Face stack is exercised in Labs 6, 7 and 8.

If accepted, two changes for Romeo or the Architect (I have not made them): in `PLAN.md` section 4, replace "fine-tune a small GPT-2" with "fine-tune the Lab 5 mini-GPT (a checkpoint is provided)"; in `_variables.yml`, `modules.m10.stack` becomes `[PyTorch]`, since nothing from Hugging Face is used.

## Model and data (provided)

**Reference policy $\pi_{\text{ref}}$.** The Lab 5 architecture and configuration, unchanged: `CONFIG = {"d": 128, "n_h": 4, "n_l": 4, "T_max": 128, "d_ff": 512, "dropout": 0.1}`, the 65-character vocabulary of the training split. Lab 5 does not save weights, so a build script (suggested `data/build_lab10_reference.py`, in the pattern of `data/build_arxiv_topics.py`) retrains it with Lab 5's full settings and seed 0 and writes `data/lab10_reference_gpt.pt` (float32 `state_dict`, about 3.3 MB). Record its test nats per character beside Lab 5's measured 1.5481 (`data/baselines.json`, `lab05.mini_gpt`). Lab 10 restates the model classes (notebooks cannot import each other) with Lab 5's reference attention in place of the participant's, and loads the file through the `data/README.md` contract (URL, fallback, SHA-256).

**Dropout off.** Build the policy and the reference with `dropout=0.0` for this lab, or keep both in `eval()` mode whenever log-probabilities are computed. Otherwise the log-ratio of Exercise 1's checkpoint is not exactly zero and the KL penalty measures dropout noise.

**Prompts and responses.** A prompt $x$ is a 32-character window of Tiny Shakespeare starting at a line start; a response $y$ is the next 48 characters sampled from the policy at temperature 1 (no top-$k$: the KL algebra assumes samples from $\pi_\theta$ itself). Training prompts come from the `train` split; a fixed set of 128 evaluation prompts from `test`. Prompt plus response is 80 characters, inside `T_max = 128`.

**Reward model $r_\phi$ and preference pairs: from Lab 9.** Provided as checkpoints; see "What Lab 10 needs from Lab 9".

**Gold reward.** Lab 9's hidden preference rule, restated as a provided function `gold_reward(text)`. Only a synthetic setup has one; the notebook must say that a real RLHF run does not.

**Diversity.** Provided `distinct_n(texts, n=2)`: distinct character bigrams over total bigrams, pooled across 4 samples per evaluation prompt.

## Core path (50 minutes)

Format per exercise: Predict, Run, Explain, Check; `# TODO N` stub, folded solution, short "why this works" note.

| # | Participant writes | Equation | Checkpoint | Metric tested | Min |
|---|---|---|---|---|---|
| 0 | Nothing: load $\pi_{\text{ref}}$, $r_\phi$ and the pairs; sample 4 responses for 3 prompts; print reward-model score and gold reward for each. Predict whether the two agree | – | none | none | 3 |
| 1 | `response_logprobs(model, prompt_ids, response_ids)` returning `(N, L)` log-probs of the response tokens | `seq-logprob` | Matches a provided hand computation on a 2 × 5 example (atol 1e-5); prompt positions excluded (shape `(N, L)`); log-ratio policy vs reference `== 0` exactly before training | Max absolute error; exact zero | 8 |
| 2 | `token_rewards(score, logp, logp_ref, beta)` returning `(N, L)` | `token-reward` | With `beta = 0` only the last column is nonzero and equals `score`; row sums equal `score - beta * (logp - logp_ref).sum(1)` on a hand-made batch; `rewards.requires_grad` is `False` when `logp` requires grad | Exact equality on small integers; allclose | 8 |
| 3 | `exact_kl(logits, logits_ref)` returning `(N,)`: per-position KL over the vocabulary, summed over response positions. Then run the provided `rlhf_train(beta=BETA)` | `kl`, `kl-chain`; `rlhf-objective` | Unit: `>= 0`; `0` for identical logits; agrees with `F.kl_div(..., log_target=True)` summed (atol 1e-5). After training, on the 128 evaluation prompts: mean $r_\phi$ up by at least a margin over $\pi_{\text{ref}}$; mean drift below a bound; gold reward printed | Reward gain (reward-model score units) and drift (nats per response) | 11 |
| 4 | Nothing new: set `BETA_HACK = 0.0`, predict, and rerun `rlhf_train` with the same seed, steps and learning rate | section 7 | **The reward-hacking signature**, asserted against Exercise 3's run (see below); 4 samples from each policy for the same prompt printed side by side | Proxy, gold, drift, distinct-2 | 8 |
| 5 | `dpo_loss(logp_w, logp_l, logp_ref_w, logp_ref_l, beta)` returning `(loss, reward_w, reward_l)`. Then run the provided `dpo_train` from a fresh copy of $\pi_{\text{ref}}$, same `BETA` as Exercise 3 | `dpo`, `implicit-reward` | Unit: loss `== log 2` (atol 1e-6) for identical policy and reference; matches a hand computation; one SGD step raises `logp_w` and lowers `logp_l` on a fixed pair. After training: held-out preference accuracy of the implicit reward above a threshold, printed beside $r_\phi$'s accuracy on the same pairs. Final table (below) | Held-out preference accuracy; the table | 12 |

Minutes: 3 + 8 + 8 + 11 + 8 + 12 = 50. Exercises 3 and 4 include waiting for training (about a minute each on a T4, estimated); the Predict questions are meant to be answered during it.

**Provided scaffolding:** the model classes and checkpoint loading; `sample_responses(policy, prompts, L, generator)`; `returns_to_go(rewards)`; `reinforce_loss(logp, returns)` with the per-position batch-mean baseline $b_t$ (lecture eq. `pg-kl`, restated from Lab 9 with Lab 9's names); `rlhf_train` and `dpo_train`; `evaluate(policy)` returning mean $r_\phi$, mean gold, mean exact KL and distinct-2 on the 128 evaluation prompts with a fixed sampling seed; plotting. `rlhf_train` logs, every few steps, the batch's mean $r_\phi$, mean gold and mean exact KL, so Exercise 4 can plot proxy and gold against drift (lecture figure 10.2). `dpo_train` logs the mean of `logp_w` and `logp_l` (lecture section 6, caution 3).

### Final comparison table (end of Exercise 5)

Rows: $\pi_{\text{ref}}$; RLHF with `BETA`; RLHF with `beta = 0`; DPO with `BETA`. Columns: mean $r_\phi$, mean gold reward, mean drift (exact KL, nats per response), distinct-2, held-out preference accuracy (implicit reward for DPO, $r_\phi$ for the reference row), training time. No assertion on DPO against RLHF in either direction: the comparison depends on budget and data, and the lecture says so.

## The reward-hacking demonstration: how it is made reliable, and what it asserts

`PLAN.md` (Day 7 review) requires this to be reliable across seeds. Five design choices make it so; the first is the one that matters most.

1. **The gold rule has structure the preference data barely shows.** The pairs are two samples from $\pi_{\text{ref}}$, so they cover only the text the reference writes. If the hidden rule has properties that are almost never active on such text (a cap, a penalty for non-words; see the proposal under Lab 9), the reward model cannot learn them, and an unpenalized policy that leaves the reference's distribution will find text that $r_\phi$ overrates. The gap between proxy and gold then exists by construction, which is the honest miniature of Goodhart's law the lecture describes. Say this in the notebook.
2. **A controlled comparison.** The runs of Exercises 3 and 4 differ only in $\beta$: same seed, initialization, prompts, learning rate, steps and evaluation sampling seed.
3. **Large effects, measured without noise.** Drift uses the exact per-position KL (eq. `kl-chain`), not the sampled log-ratio; all metrics are averaged over 128 evaluation prompts with 4 samples each.
4. **Hyperparameters chosen across seeds.** Choose the learning rate, steps and `BETA` so that the signature holds for **every** seed in 0–4 in both the full and the `FAST` configuration, not for seed 0 alone.
5. **Thresholds from the worst seed.** Record per-seed values in `data/baselines.json` under `lab10` and set each threshold at about half the smallest gap observed across the five seeds.

**What Checkpoint 4 asserts** (`beta = 0` run against the `BETA` run, same seed):

- (a) mean $r_\phi$ is at least as high, within a tolerance set from the seeds;
- (b) drift is at least 3 times larger and above an absolute floor (both set from the seeds);
- (c) mean gold reward is lower by a margin;
- (d) distinct-2 is lower by a margin (mode collapse).

Also print, without asserting unless it holds on all seeds, the gold reward at its peak during the `beta = 0` run against its final value: the "rises, peaks, falls" shape of lecture section 7.

**If any of the five seeds fails (c) or (d)**, change the design (steps, learning rate, the gold rule's cap or penalty) and re-run all seeds; do not weaken a threshold until it passes by default. If it still cannot be made reliable, keep (b) as the only hard assertion, print (a), (c), (d), and tell me: the lecture's Exercise 4 callout states all four.

## What Lab 10 needs from Lab 9

Lab 9 (lecture `lectures/09-preference-learning.qmd`, brief `briefs/09-preference-learning.md`) is being drafted in parallel. Lab 10 works only if Lab 9's preference task lives in Lab 10's policy's output space. Concretely:

1. **The pairs are samples from `data/lab10_reference_gpt.pt`.** For each training prompt (the 32-character windows above), two 48-character continuations sampled at temperature 1 from the reference policy. A build script generates them once and commits them, suggested `data/lab09_preferences.jsonl.gz` with fields `prompt`, `chosen`, `rejected`, `gold_chosen`, `gold_rejected`, `split`. Suggested size: 4,000 train and 500 held-out pairs (about 0.5 MB uncompressed; estimate). So the reference checkpoint must be built before Lab 9's data.
2. **Labels from a hidden rule through Bradley–Terry.** $P(a \succ b) = \sigma\big((g(a) - g(b)) / \tau_{\text{label}}\big)$ with the gold rule $g$ and a stated label temperature; Lab 9's noisy-rater stretch can vary $\tau_{\text{label}}$.
3. **A gold rule that the reward model can mostly but not fully learn** (point 1 of the reliability section). Proposal, for the Lab 9 author to accept or replace: $g(y) = \min(n_{\text{pos}}(y), 3) - n_{\text{neg}}(y) - 2\, f_{\text{non}}(y)$, where $n_{\text{pos}}$ counts *distinct* words from a fixed list of about 20 cheerful Shakespearean words (`love`, `sweet`, `fair`, `joy`, `gentle`, `merry`, `dear`, ...), $n_{\text{neg}}$ counts words from a list of grim ones (`death`, `blood`, `kill`, `grief`, `woe`, ...), and $f_{\text{non}}$ is the share of whitespace-separated words not in the training split's word list. On reference samples the cap and the non-word term are rarely active, so the reward model learns "more cheerful words is better"; an unpenalized policy that repeats cheerful words or invents fragments of them raises $r_\phi$ and lowers $g$. Lab 9 must check that its reward model recovers the hidden preference **on held-out reference samples**, which is all its checkpoint should claim.
4. **The reward model checkpoint.** A small character-level network with a scalar head reading prompt and response together (for example Lab 5's block with $d = 64$, 2 layers, the score read at the last position), saved as `data/lab09_reward_model.pt` from Lab 9's solution run with a fixed seed, with its class definition stated so Lab 10 can restate it verbatim. It must accept a batch of `(prompt_ids, response_ids)` of the lengths above and return `(N,)` scores. Report its held-out pair accuracy; Lab 10 prints it beside DPO's.
5. **Names.** `reinforce_loss`, `returns_to_go` and the baseline as written in Lab 9, so Lab 10 restates them unchanged; the Bradley–Terry loss written so that DPO's code reads as the same function of a different reward.

All three files (`lab10_reference_gpt.pt`, `lab09_preferences.jsonl.gz`, `lab09_reward_model.pt`) need entries in `_variables.yml` `datasets` and `data/README.md` (owner: Architect), with SHA-256 and the repository-hosted fallback URL. Tiny Shakespeare is public-domain text, so the weights and pairs can be committed under the repository's MIT/CC BY terms.

## Stretch (one section, last, optional; not needed by any later lab)

Sweep `beta` over about five values from 0 to well above `BETA` (for example 0, `BETA/4`, `BETA`, `4·BETA`, `16·BETA`; set from measurement), same seed and steps, starting from $\pi_{\text{ref}}$ each time. Plot mean $r_\phi$ and mean gold against drift, one point per `beta`, with DPO's point added. Assert only that drift falls as `beta` rises (monotone across the sweep, if it holds on all seeds). Send me the measured points: lecture figure 10.2 is drawn as a schematic until they exist.

## Compute budget (free Colab T4; whole notebook under 10 minutes)

*Checked* on the build container's CPU, Lab 5 shape with random weights (attention via `nn.MultiheadAttention`, not the lab's code), no KV cache: sampling 48 characters after a 32-character prompt plus one update with the reference forward took 1.0–1.5 s + 0.12–0.19 s per step at batch 32; 0.5 s + 0.07 s at batch 16; 2.7–2.9 s + 0.33 s at batch 64 with 64-character responses. Sampling dominates. Reward-model scoring is not included.

| Step | T4 budget (estimate) | CPU `FAST` budget (estimate from the timings above) |
|---|---|---|
| Setup, downloads, Exercises 0–2 | under 1 min | under 1 min |
| Exercise 3: `rlhf_train`, about 150 steps at batch 32 | about 1 min | 80 steps at batch 16: about 1 min |
| Exercise 4: the same with `beta = 0` | about 1 min | about 1 min |
| Exercise 5: DPO, one pass over 4,000 pairs at batch 64 | under 0.5 min | under 1 min |
| Evaluation (4 policies × 128 prompts × 4 samples) | under 0.5 min | about 1.5 min (reduce to 64 prompts in `FAST`) |
| Stretch: 5 runs of the Exercise 3 length | about 3 to 4 min | 3 runs only, or skipped |

If sampling is too slow, add a key–value cache to the provided sampler (scaffolding, not an exercise). A `FAST` flag (on by default without a GPU, forced by `NLP_LLMS_QUICK` as in Labs 3 and 5) is the path CI runs; its thresholds are set separately from its own seeds.

## What Lab 10 saves

Nothing that a later lab needs. Lab 11 uses the Lab 6 classifier; Lab 12 trains its own toy decision model.

## If GPT-2 is kept

The lecture names the Lab 5 model in section 2's mapping table ("skipped: the reference is the small GPT you trained on Tiny Shakespeare in Lab 5"), in section 3's lab note ("$\lvert V \rvert = 65$ characters") and in section 7's lab note and failure table ("distinct character bigrams"). With GPT-2 those become "the pretrained GPT-2 (`{{< var models.causal_lm >}}` or a new pin)", "50,257 tokens" and "distinct token bigrams". The exercises, equations and checkpoints are unchanged. Lab 9's pairs, reward model and gold rule would then be built on GPT-2 samples and its tokenizer, and every Hub-dependent step (weights, tokenizer, pair generation, reward-model training) would be written, not run, in the build container.

## Flags for the Lab Engineer

1. Use the lecture's names: `pi_ref` or `ref`, `policy`, `beta`/`BETA`, `logp`, `logp_ref`, `score` for $r_\phi(x, y)$, `rewards` for $r_t$, `returns` for $G_t$, `gold` for the hidden rule's score. Do not name anything `V` (the vocabulary) or `r` alone.
2. The KL penalty in `token_rewards` uses the sampled log-ratio, detached; the drift metric uses `exact_kl`. Do not swap them, and do not add the exact KL as a differentiable loss term: the lecture derives the per-token reward form.
3. Sample at temperature 1 with no top-$k$ or nucleus truncation during training and evaluation; state it in the notebook.
4. `reinforce_loss` sums over response positions and averages over the batch; normalize returns only if Lab 9 does, and say so.
5. Exercise 1's zero-log-ratio check needs dropout off (see above) and the same device and dtype for both models.
6. DPO's reference log-probabilities can be computed once before training; do so, and say why it is allowed (the reference is frozen).
7. The notebook text for Exercise 4 must call the gold reward what it is: a hidden rule we wrote, known only because the data are synthetic. Do not describe it as human preference.
8. No TypeSafe or RLCD content belongs in this lab. The lecture's last paragraph points to Module 12 and nothing more.
9. Send me, measured: per-seed values for Checkpoints 3 and 4 (full and `FAST`), the final table for seed 0, the DPO log-probability curves, the stretch points, run time per section on a T4 and on a CPU runtime, and four side-by-side samples from each policy for one prompt.

## Open questions

1. **Policy model:** mini-GPT (recommended) or GPT-2. Romeo decides; see the top of this brief.
2. **Lab 9's gold rule and reward-model architecture** are proposals here, to be reconciled with `briefs/09-preference-learning.md`.
3. **Notation to reconcile with lecture 9:** see the list in the Director's report (states $s_t$, return $G_t$ and baseline $b_t$, $\succ$ for preference, $\mathcal{D}_{\text{pref}}$, $v_\psi$ in lowercase).
4. **`BETA` and the learning rate** are not set here; the lecture names no values.

## Not verified by the Director

- No notebook exists; nothing in this brief has been run except the CPU timing above, on random weights.
- That the reward-hacking signature holds on all five seeds with the proposed gold rule. This is the main risk of the lab.
- That the Lab 5 configuration retrained with seed 0 reproduces Lab 5's test nats.
- All T4 run times.
- Citations in the lecture were written from memory: arxiv.org and other paper hosts were blocked from the build container on 2026-10-05.
- `quarto render` was not run (Quarto is not installed in the build environment).
