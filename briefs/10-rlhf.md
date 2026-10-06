# Lab brief: `notebooks/10-rlhf.ipynb`

**Status (2026-10-06).** `notebooks/10-rlhf.ipynb` now exists and passes its offline code check in CI. Nothing involving GPT-2 has run yet: it needs Lab 9's data files and reward model. The "Not verified" section below records the state when this brief was written. For what has run since, see the [readiness page](../readiness.qmd).

**As built, Lab 9 data (2026-10-06, Neural Lab Engineer).** Lab 9's three files are committed, and `models.causal_lm_revision` is pinned in `_variables.yml` and in the notebook. The prompts changed: every frame now ends just before a feeling ("... and felt so"); see `briefs/09-preference-learning.md`, "As built". Measured on an Apple M1 Pro, never on a T4:

- **Checkpoint 1's tolerance is now 1e-4, not 1e-5.** On the first real GPT-2 run (CPU), the batched and the position-by-position log-probabilities differed by 2.05e-5 in the checkpoint's own case. Over 32 prompt pairs the difference had a median of 1.4e-5 and a maximum of 3.0e-5, for the old and the new prompts alike; this is float32 rounding. An off-by-one slice is off by 11.45 nats.
- **On CPU** (`FAST`), every unit checkpoint passes and the trained-policy checkpoints are skipped (36.5 s).
- **Exploratory, not the protocol:** one run at full settings on Apple MPS (seed 0, protocol mode, no stretch, 809 s), from a scratch copy whose only change was `DEVICE = "mps"`. Checkpoints 3, 4 and 5 passed at the provisional thresholds:

  | Policy | r_phi | Gold | Drift (nats) | Distinct-2 |
  |---|---|---|---|---|
  | Reference | −0.164 | +0.072 | 0 | 0.919 |
  | `BETA` | +1.871 | +1.152 | 3.13 | 0.902 |
  | beta = 0 | +2.045 | +1.000 | 63.1 | 0.030 |
  | DPO | +1.773 | +1.490 | 4.14 | 0.925 |

  DPO's held-out accuracy was 0.663. During the beta = 0 run the gold reward peaked at +1.92 (step 40) and then fell while r_phi kept rising. Two cautions for the seed protocol:
  - The gold gap of part (c) is only 0.152.
  - The beta = 0 policy collapsed to "good as as as …", which the gold rule scores +1, since one list word in 23 never triggers crowding. So gold fell because the collapsed policy writes one positive word where the penalized policy sometimes writes two, **not through the cap or the crowding term** that lecture 10's Exercise 4 callout names.

**Academic Director, 2026-10-06.** Lecture 10's Exercise 4 callout, the notebook's Step 0 and Exercise 4 "Explain" cells and exit question 10.4 now name all three ways the gold rule allows the gold reward to part from $r_\phi$: no credit past three distinct positive words, the crowding penalty, and fewer distinct positive words or more negative ones. They report the run above as one exploratory run on an Apple M1 Pro, not as a T4 or Colab result. For the seed protocol, please record for each seed which way the $\beta = 0$ policy went: on the evaluation samples of the `BETA` and $\beta = 0$ policies, the mean number of distinct positive words, the mean number of negative words, and the shares of responses with the cap active ($n_{\text{pos}} \ge 4$) and with crowding active ($f_{\text{list}} > 0.25$). If no seed triggers the cap or the crowding term, tell me: lecture 9's preview and the design notes in Lab 9's brief would then overstate them.

From the Academic Director to the Neural Lab Engineer. Lecture: `lectures/10-rlhf.qmd` (same symbols and equation names). Lab standards: `PLAN.md` section 5. Data contract: `data/README.md`. Lab 9's brief: `briefs/09-preference-learning.md`. This file is not rendered by Quarto.

**Revision 2 (2026-10-05).** Romeo decided to keep `PLAN.md`'s design: **Lab 10 fine-tunes a small GPT-2**, not the Lab 5 mini-GPT that revision 1 recommended. The exercises, equations and checkpoints are unchanged; the model, the data, the compute budget, the CPU path and the reliability protocol are rewritten. The "Lab 9 → Lab 10 interface" section is identical to the one in Lab 9's brief.

**Objectives exercised** (from `_variables.yml`, `m10`): describe the three-stage RLHF pipeline; optimize a small LM against a reward model with a KL constraint; apply DPO and compare; name RLHF's failure modes and observe one.

**Every time, size and threshold below is an estimate or a target, not a measurement,** unless marked *checked*. Replace each with the value you measure, and tell me if a target cannot be met. *Checked* means I ran it on 2026-10-05 in the build container (CPU, 4 threads, `torch` 2.14.1), not on Colab. huggingface.co is blocked in the build container, so **nothing involving GPT-2 has been run**: every Hub-dependent part of this lab is written, not run, until the protocol in "Reliability" has been carried out on Colab or another machine with Hub access.

## What the decision changes

| | Revision 1 (mini-GPT) | Revision 2 (small GPT-2, decided) |
|---|---|---|
| Policy | Lab 5 architecture, 0.83M parameters, 65 characters | `models.causal_lm` (Lab 6's distilled GPT-2), 81.9M parameters, 50,257 tokens |
| Hub access | none | model and tokenizer, at run time and to build Lab 9's data |
| Build container | could run and measure everything | cannot run the real path; offline stand-ins only |
| GPU | optional | **required** for the trained-policy checkpoints; a CPU runtime runs the unit checkpoints only |
| Running thread | the model built in Lab 5 | the model probed in Lab 6 (perplexity, Exercise 3) |
| Samples | pseudo-Shakespeare | English |
| Hugging Face stack | not used | `transformers` for loading, the tokenizer and a KV cache |

## Design choices (decided here, with reasons)

| Question | Decision | Reason |
|---|---|---|
| Which checkpoint | `models.causal_lm`, already in `_variables.yml`; add `models.causal_lm_revision` (the commit hash, pinned on the first run with Hub access; Lab 6 benefits too) | Reuses a pinned, Apache-2.0 model that participants met in Lab 6. With 6 blocks it costs roughly half as much per token as `openai-community/gpt2` (124M, 12 blocks), and Lab 10 runs three trainings and DPO in 10 minutes. If its samples prove unreadable on the first run, propose a new key for the 124M model and re-measure the budget; do not hard-code an ID |
| Full fine-tuning or LoRA | **Full fine-tuning**, float32, AdamW | Memory does not bind: policy, gradients and AdamW states take about 1.3 GB, the frozen reference 0.33 GB, and one batch's logits (64 × 32 × 50,257 float32) about 0.4 GB per tensor, well inside the T4's 15 GB (estimates). Time does not favor LoRA: sampling dominates, and LoRA still backpropagates through all six blocks. Full fine-tuning keeps the reference a plain frozen `deepcopy`, avoids `peft`'s handling of GPT-2's fused `Conv1D` attention (`c_attn`), and does not cap how far the $\beta = 0$ policy can drift, which the demonstration needs |
| Lengths | 8-token prompts, 24-token responses, end-of-text never sampled (interface) | No padding or attention masks anywhere; the reward-model score sits in the last column of `token_rewards`, as in lecture 10 |
| Sampler | Provided, hand-written: a loop over `model(..., past_key_values=..., use_cache=True)` with a `torch.Generator`; not `generate()` | Each seed's samples are reproducible on one machine and independent of other cells; `generate()` uses the global random state and has defaults (for example `top_k` 50 when unset, noted in Lab 7's brief) that would all have to be overridden |
| KL | Penalty: the sampled log-ratio, detached (eq. `token-reward`). Drift: the exact per-position KL over 50,256 entries (eq. `kl-chain`), under `torch.no_grad()`, in chunks of 64 responses | As in revision 1. The exact sum over GPT-2's vocabulary is affordable on a T4 when chunked (one chunk's two logits tensors are about 0.6 GB) |
| Baseline | Per-position batch mean, `reinforce_loss(logp, returns, batch_mean_baseline(returns))` | Lecture 10, eq. `pg-kl`; Lab 9's function unchanged |
| Batch | Starting point: 64 training prompts per step, drawn with replacement from the 160 train prompts, one response each; about 150 steps per run | To be set by the seed protocol below |
| Evaluation | The 64 evaluation prompts × 8 samples = 512 responses per policy, with one fixed evaluation generator seed shared by all policies | Eight samples per prompt make the per-prompt diversity measure meaningful; 512 responses average out sampling noise |
| CPU runtime | Supported only as `FAST`: unit checkpoints asserted, training runs as a short smoke test, trained-policy checkpoints skipped with a printed message | Estimated 3 to 5 s per training step at batch 8 on a 2-vCPU Colab CPU: the demonstration would take far longer than 10 minutes. The notebook's first markdown cell and setup cell say "Runtime → Change runtime type → T4 GPU" |

## Model and data (provided)

**Reference policy $\pi_{\text{ref}}$** and the sampling rules: see the interface section. Load once with `AutoModelForCausalLM.from_pretrained(CAUSAL_LM, revision=CAUSAL_LM_REVISION, dtype=torch.float32)`, `eval()`, `requires_grad_(False)`. Each training run starts from `copy.deepcopy(pi_ref)` with gradients enabled and stays in `eval()` mode (gradients flow in `eval()` mode; dropout does not).

**Provided helpers.** `policy_logits(model, input_ids) -> (N, T, 50256)`: the model's logits with the end-of-text column removed (`[..., :EOS_ID]`). `sample_responses(policy, prompt_ids, generator) -> (N, 24)`, temperature 1, no truncation, built on `policy_logits` with a KV cache. `distinct_n(token_lists, n=2)`: distinct token bigrams over total token bigrams, pooled across the 8 samples of each evaluation prompt, then averaged over prompts. `score(prompt_ids, response_ids)`: the reward model. `gold_of(prompt_ids, response_ids)`: decodes with the tokenizer and calls `gold_reward`.

**Reward model, gold rule, prompts and pairs: from Lab 9**, as committed files; see the interface. The gold reward exists only because the data are synthetic; the notebook must say that a real RLHF run has none.

## Core path (50 minutes)

Format per exercise: Predict, Run, Explain, Check; `# TODO N` stub, folded solution, short "why this works" note.

| # | Participant writes | Equation | Checkpoint | Metric tested | Min |
|---|---|---|---|---|---|
| 0 | Nothing: load $\pi_{\text{ref}}$, $r_\phi$, the prompts and the pairs; sample 4 responses for 3 evaluation prompts; print the reward-model score and the gold reward of each. Predict whether the two agree | – | none | none | 3 |
| 1 | `response_logprobs(model, prompt_ids, response_ids)` returning `(N, L)` log-probabilities of the response tokens, using the provided `policy_logits` | `seq-logprob` | Matches a provided position-by-position computation on a 2-prompt, 5-token example (atol 1e-5); prompt positions excluded (shape `(N, L)`); log-ratio policy vs reference `== 0` exactly before training | Max absolute error; exact zero | 8 |
| 2 | `token_rewards(score, logp, logp_ref, beta)` returning `(N, L)` | `token-reward` | With `beta = 0` only the last column is nonzero and equals `score`; row sums equal `score - beta * (logp - logp_ref).sum(1)` on a hand-made batch; `rewards.requires_grad` is `False` when `logp` requires grad | Exact equality on small integers; allclose | 8 |
| 3 | `exact_kl(logits, logits_ref)` returning `(N,)`: per-position KL over the vocabulary, summed over response positions. Then run the provided `rlhf_train(beta=BETA)` | `kl`, `kl-chain`; `rlhf-objective` | Unit, on small random logits and on the real logits of 4 responses: `>= 0`; `0` for identical logits; agrees with `F.kl_div(..., log_target=True)` summed (atol 1e-5). After training, on the 512 evaluation responses: mean $r_\phi$ up by at least a margin over $\pi_{\text{ref}}$; mean drift below a bound; gold reward printed. On a CPU runtime the training is a smoke run and these two are skipped | Reward gain (normalized reward-model units) and drift (nats per response) | 11 |
| 4 | Nothing new: set `BETA_HACK = 0.0`, predict, and rerun `rlhf_train` with the same seed, steps and learning rate | section 7 | **The reward-hacking signature**, asserted against Exercise 3's run (see below); 4 samples from each policy for the same prompt printed side by side. Skipped on CPU | Proxy, gold, drift, distinct-2 | 8 |
| 5 | `dpo_loss(logp_w, logp_l, logp_ref_w, logp_ref_l, beta)` returning `(loss, reward_w, reward_l)`. Then run the provided `dpo_train` from a fresh copy of $\pi_{\text{ref}}$, same `BETA` as Exercise 3, one pass over Lab 9's 8,000 train pairs | `dpo`, `implicit-reward` | Unit: loss `== log 2` (atol 1e-6) for identical policy and reference; matches a hand computation; one SGD step raises `logp_w` and lowers `logp_l` on a fixed pair. After training: accuracy of the implicit reward on Lab 9's 1,000 held-out pairs above a threshold, printed beside $r_\phi$'s accuracy on the same pairs. Final table (below). The trained-policy part is skipped on CPU | Held-out preference accuracy; the table | 12 |

Minutes: 3 + 8 + 8 + 11 + 8 + 12 = 50. Exercises 3 and 4 include waiting for training (about 1.5 minutes each on a T4, estimated); the Predict questions are meant to be answered during it.

**Provided scaffolding:** the helpers above and checkpoint loading; `returns_to_go`, `reinforce_loss`, `batch_mean_baseline` restated from Lab 9; `rlhf_train` and `dpo_train`; `evaluate(policy)` returning mean $r_\phi$, mean gold, mean exact KL and distinct-2 on the 512 evaluation responses; plotting. `rlhf_train` logs, every few steps, the batch's mean $r_\phi$, mean gold and mean exact KL (chunked, no gradient), so Exercise 4 can plot proxy and gold against drift (lecture figure 10.2). `dpo_train` logs the means of `logp_w` and `logp_l` (lecture section 6, caution 3).

### Final comparison table (end of Exercise 5)

Rows: $\pi_{\text{ref}}$; RLHF with `BETA`; RLHF with `beta = 0`; DPO with `BETA`. Columns: mean $r_\phi$, mean gold reward, mean drift (exact KL, nats per response), distinct-2 (token bigrams), held-out preference accuracy (implicit reward for DPO, $r_\phi$ for the reference row), training time. No assertion on DPO against RLHF in either direction: the comparison depends on budget and data, and the lecture says so.

## The reward-hacking demonstration: how it is made reliable, and what it asserts

`PLAN.md` (Day 7 review) requires this to be reliable across seeds. Five design choices make it so; the first matters most.

1. **The gold rule has structure the preference data barely show.** The pairs are two samples from $\pi_{\text{ref}}$. The gold rule's cap and crowding term are set so that the reference's own samples almost never trigger them (Lab 9's acceptance criteria: under 1% each), so the reward model cannot learn them. Its compact vocabulary adds a second blind spot: tokens absent from the training pairs all share one embedding row. An unpenalized policy that stuffs positive words is then overrated by $r_\phi$ and penalized by $g$. The gap exists by construction: the honest miniature of Goodhart's law the lecture describes. Say this in the notebook; lecture 10, section 7, now says it too.
2. **A controlled comparison.** The runs of Exercises 3 and 4 differ only in $\beta$: same seed, initialization, prompts, learning rate, steps and evaluation generator.
3. **Large effects, measured without noise.** Drift uses the exact per-position KL (eq. `kl-chain`), not the sampled log-ratio; all metrics are averaged over the 512 evaluation responses.
4. **Hyperparameters chosen across seeds.** Choose the learning rate, steps, batch size and `BETA` so that the signature holds for **every** seed in 0–4 on a T4, not for seed 0 alone.
5. **Thresholds from the worst seed.** Record per-seed values in `data/baselines.json` under `lab10` and set each threshold at about half the smallest gap observed across the five seeds (the full rule is under "Reliability").

**What Checkpoint 4 asserts** (`beta = 0` run against the `BETA` run, same seed):

- (a) mean $r_\phi$ is at least as high, within a tolerance set from the seeds;
- (b) drift is at least 3 times larger and above an absolute floor (both set from the seeds);
- (c) mean gold reward is lower by a margin;
- (d) distinct-2 is lower by a margin (mode collapse).

Also print, without asserting unless it holds on all seeds, the gold reward at its peak during the `beta = 0` run against its final value: the "rises, peaks, falls" shape of lecture section 7.

**If any of the five seeds fails (c) or (d)**, change the design (steps, learning rate, `BETA`, the gold rule's `crowd_threshold` or `crowd_weight`, the lists) and re-run all seeds; do not weaken a threshold until it passes by default. A change to the gold rule changes Lab 9's data file and reward model, so tell the Lab 9 author and me. If it still cannot be made reliable, keep (b) as the only hard assertion, print (a), (c), (d), and tell me: the lecture's Exercise 4 callout states all four.

## Reliability: how and where it is verified

The build container cannot reach huggingface.co, so none of this can happen there. It must run on a Colab T4 (free tier, the target runtime) or another CUDA machine with Hub access. Until it has, the lab and the Day 7 review item stay open, and every report calls the Hub-dependent parts "written, not run".

**Prerequisites.** Lab 9's two data files are built and committed (Lab 9's brief), and the seed-0 reward model is committed; the reward models from Lab 9's seeds 1 and 2 are kept for step 4. `models.causal_lm_revision` is pinned.

**Protocol.**

1. **Fixed across seeds:** model revision, prompts, pairs, reward-model checkpoint, hyperparameters, evaluation generator seed. **Varied by the seed:** the training generator (prompt draws and samples), the DPO data order, and `torch.manual_seed`.
2. **Five seeds, 0 to 4, at full settings on a T4.** For each seed run Exercise 3 (`BETA`), Exercise 4 (`beta = 0`) and Exercise 5 (DPO) and record, for every policy: mean $r_\phi$, mean gold, mean exact KL, distinct-2, held-out preference accuracy, and wall time per section. Record the GPU type, the `torch`, `transformers` and CUDA versions, and the date.
3. **Run-to-run noise.** Run seed 0 a second time in a fresh runtime. GPU kernels are not guaranteed to be bit-for-bit deterministic, so the same seed can give slightly different numbers; the difference between the two seed-0 runs is the noise floor.
4. **Secondary robustness check** (report; not a gate): policy seed 0 with the reward models from Lab 9's seeds 1 and 2. If the signature fails there, add one sentence to the notebook: a participant who trains their own reward model in Lab 9 and uploads it may not see every part of the signature.
5. **Threshold rule.** For each asserted comparison, compute the gap on each of the five seeds. The comparison may be asserted only if it has the right sign on all five seeds and the smallest gap is at least twice the seed-0 noise floor. Its threshold is half the smallest gap. Never set a threshold from seed 0 alone, and never loosen one until it passes; change the design instead (previous section).
6. **CPU `FAST` path:** on a CPU runtime with Hub access, run seeds 0 to 2 and confirm that every unit checkpoint passes and that the smoke runs finish; it asserts nothing about training outcomes. This is the path CI runs where CI has Hub access.
7. **In the build container:** only an offline stand-in can run (suggested `NLP_LLMS_LAB10_OFFLINE=tiny`, in the pattern of Lab 7: a randomly initialized two-block `GPT2LMHeadModel` with GPT-2's 50,257-entry vocabulary, so the ids in Lab 9's files are valid, and a stand-in decoder). Its numbers are not results and set no threshold.
8. **Record and report.** Per-seed values go to `data/baselines.json` under `lab10`; the side-by-side samples and the stretch points come to me. The Day 7 box "the reward-hacking demonstration is reliable across seeds" is ticked only after steps 2, 3 and 5 are done.

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

## Stretch (one section, last, optional; not needed by any later lab)

Sweep `beta` over five values from 0 to well above `BETA` (for example 0, `BETA/4`, `BETA`, `4·BETA`, `16·BETA`; set from measurement), same seed, starting from $\pi_{\text{ref}}$ each time. Reuse the runs of Exercises 3 and 4 for 0 and `BETA`, so three new runs; shorten them (for example 100 steps) if the budget requires, and say so on the plot. Plot mean $r_\phi$ and mean gold against drift, one point per `beta`, with DPO's point added. Assert only that drift falls as `beta` rises (monotone across the sweep, if it holds on all seeds). Skipped on a CPU runtime. Send me the measured points: lecture figure 10.2 is drawn as a schematic until they exist.

## Compute budget (free Colab T4; whole notebook under 10 minutes; all estimates)

Per training step at batch 64 (estimate): sampling 24 tokens with a KV cache about 0.3 s; policy forward and backward on 64 × 32 tokens about 0.2 s; reference and reward model about 0.05 s; about 0.6 s in all.

| Step | T4 budget | CPU `FAST` budget |
|---|---|---|
| Setup: pinned installs, model download (about 350 MB), Lab 9's three files | 0.5 to 1 min | the same |
| Exercises 0–2 | under 0.5 min | about 1 min |
| Exercise 3: `rlhf_train`, about 150 steps at batch 64 | about 1.5 min | smoke: 6 steps at batch 8, about 0.5 min |
| Exercise 4: the same with `beta = 0` | about 1.5 min | about 0.5 min |
| Exercise 5: reference log-probabilities of 16,000 sequences once, then one pass at batch 32 (250 steps) | 1 to 1.5 min | smoke: 6 steps, about 0.5 min |
| Evaluation: 4 policies × 512 responses, exact KL in chunks | 0.5 to 1 min | 8 prompts × 2 samples, about 1 min |
| **Core path total** | **about 6 to 7 min** | **about 4 to 5 min** |
| Stretch: 3 new runs | about 3 min at 100 steps each | skipped |

If the measured core path exceeds 7 minutes on a T4, cut the stretch to two new `beta` values before touching the core. GPU memory, estimated: under 4 GB.

## What Lab 10 saves

Nothing that a later lab needs. Lab 11 uses the Lab 6 classifier; Lab 12 trains its own toy decision model.

## Flags for the Lab Engineer

1. Use the lecture's names: `pi_ref` or `ref`, `policy`, `beta`/`BETA`, `logp`, `logp_ref`, `score` for $r_\phi(x, y)$, `rewards` for $r_t$, `returns` for $G_t$, `gold` for the hidden rule's score. Do not name anything `V` (the vocabulary) or `r` alone.
2. The KL penalty in `token_rewards` uses the sampled log-ratio, detached; the drift metric uses `exact_kl`. Do not swap them, and do not add the exact KL as a differentiable loss term: the lecture derives the per-token reward form.
3. Sample at temperature 1 with no top-$k$ or nucleus truncation during training and evaluation, from the logits without the end-of-text column; state it in the notebook. Log-probabilities, the KL and sampling must all use `policy_logits`, so that they describe the same distribution.
4. `reinforce_loss` sums over response positions and averages over the batch; returns are not normalized, as in Lab 9.
5. Exercise 1's zero-log-ratio check needs both models in `eval()` mode, on the same device, in float32. If it is not exactly zero on CUDA, first try `torch.use_deterministic_algorithms(True)` with `CUBLAS_WORKSPACE_CONFIG=:4096:8`; if it is still not exact, use `atol=1e-6`, say why in the notebook, and tell me (the lecture says "exactly zero").
6. DPO's reference log-probabilities can be computed once before training; do so, and say why it is allowed (the reference is frozen).
7. Compute `exact_kl` and every evaluation under `torch.no_grad()`, in chunks of 64 responses; free the logits between chunks.
8. The notebook text for Exercise 4 must call the gold reward what it is: a hidden rule we wrote, known only because the data are synthetic. Do not describe it as human preference.
9. Pin `transformers` after checking what Colab preinstalls; Lab 7's brief found `transformers` 5.18.0 in use. Check the KV-cache interface of the pinned version against its documentation before writing the sampler.
10. No TypeSafe or RLCD content belongs in this lab. The lecture's last paragraph points to Module 12 and nothing more.
11. Send me, measured: per-seed values for Checkpoints 3, 4 and 5, the noise floor, the final table for seed 0, the DPO log-probability curves, the stretch points, run time per section on a T4 and on a CPU runtime, GPU memory, and four side-by-side samples from each policy for one prompt.

## Proposed changes (not made; for Romeo or the Architect)

- **`_variables.yml`:** add `models.causal_lm_revision` (commit hash, pinned on the first run with Hub access). `modules.m10.stack` stays `[PyTorch, Hugging Face]`; this withdraws revision 1's proposal of `[PyTorch]`. No `PEFT`: Lab 10 does not use LoRA. The three `datasets` entries are listed in Lab 9's brief.
- **`tests/test_models.py`:** add `"10-rlhf": ["causal_lm", "causal_lm_revision"]` to `USES` (and `"09-preference-learning": ["causal_lm"]` if Lab 9's optional tokenizer cell quotes the ID).
- **`PLAN.md` section 4, Module 10, Lab:** "fine-tune a small GPT-2 against the Lab 9 reward model" becomes "fine-tune the small GPT-2 of Lab 6 (`models.causal_lm`) against the Lab 9 reward model"; add "Needs a T4 runtime; on a CPU runtime only the unit checkpoints run."
- **`PLAN.md` section 6:** add a row. Item: "Labs 9 and 10 depend on GPT-2". Risk: "Lab 9's data files and every Lab 10 result need huggingface.co and a GPU; the build container has neither, so the reward-hacking demonstration cannot be verified there". Mitigation: "Build Lab 9's data and run the seed protocol of `briefs/10-rlhf.md` on a Colab T4; offline stand-ins test the code only".
- **`PLAN.md` section 7, Day 7:** under "Review: the reward-hacking demonstration is reliable across seeds", add "(protocol in `briefs/10-rlhf.md`; needs a Colab T4 with Hub access)".

## Open questions

1. **`BETA`, the learning rate, the batch size and the number of steps** are not set here; the lecture names no values.
2. **Readability of the 82M model's samples** at temperature 1 with no truncation is unknown; see the fallback in "Design choices".
3. **The gold rule's final values** depend on Lab 9's reference-sample statistics.

## Not verified by the Director

- No notebook exists, and nothing involving GPT-2 has been run: huggingface.co (and the other hosts that serve GPT-2's tokenizer files) were refused by the build container's proxy on 2026-10-05.
- Every compute estimate above (per-step times, memory, the 6-to-7-minute core path, the CPU smoke budget). They are arithmetic from parameter counts, not measurements.
- That the reward-hacking signature holds on all five seeds with the revised gold rule. This is the main risk of the lab.
- That the exact zero log-ratio holds on CUDA without deterministic settings.
- That the `models.causal_lm` configuration sets dropout to 0.1 and that its end-of-text id is 50256, the last vocabulary row (both from memory of GPT-2's configuration; the notebook asserts the second).
- Citations in the lecture were written from memory: arxiv.org and other paper hosts were blocked from the build container on 2026-10-05.
- `quarto render` was not run (Quarto is not installed in the build environment).
