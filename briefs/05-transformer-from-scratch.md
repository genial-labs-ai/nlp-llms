# Lab brief: `notebooks/05-transformer-from-scratch.ipynb`

From the Academic Director to the Neural Lab Engineer. Lecture: `lectures/05-transformer-from-scratch.qmd` (same symbols and equation names). Lab standards: `PLAN.md` section 5. Data contract: `data/README.md`. This file is not rendered by Quarto.

**Objectives exercised** (from `_variables.yml`, `m05`): implement scaled dot-product and multi-head self-attention; assemble a decoder-only transformer; train a small GPT and compare it with the LSTM.

**No number below has been measured.** Every size, time and threshold is a starting point or a budget. Replace each with the value you measure on a free T4, and tell me if a target cannot be met.

## Data and the fair-comparison contract (same as Lab 3)

- `load_lm_corpus()` pasted unchanged from `data/README.md`: Tiny Shakespeare, train `[0, 1,000,000)`, val `[1,000,000, 1,055,000)`, test `[1,055,000, 1,115,394)`. Character-level. Vocabulary: the 65 characters of `train`. Do not re-split or subsample the evaluation data.
- Loss and metric: Lab 3's `lm_loss_and_ppl`, restated unchanged (provided, not an exercise here): mean NLL in nats per character, `PPL = exp(mean NLL)`, bits per character beside it. Tune on `val`; report on `test` once per model.
- **Same scored positions as Lab 3:** every test character that Lab 3's evaluation routine scores, and no others. Restate that routine's indexing; assert the number of scored characters equals Lab 3's.
- **Context at evaluation.** The LSTM carries its state across chunks, so it never scores with a cold state. The GPT has a window of `T_max`. Scoring non-overlapping chunks would give the first characters of each chunk almost no context and understate the GPT. Use overlapping windows: stride `T_max / 2`, score only the second half of each window (all of the first), so every scored character after the first window has at least `T_max / 2` characters of context. Say in the notebook that the two models are scored on the same characters but with different context, and why.
- **Baselines.** Notebooks cannot import from each other on Colab.
  - Character n-gram (Lab 1's method, as restated in Lab 3): recompute here if it stays inside the budget, otherwise quote Lab 3's measured numbers.
  - LSTM: Lab 3 is being built now; its numbers do not exist yet. Either retrain Lab 3's LSTM here with Lab 3's exact configuration, or quote Lab 3's measured test PPL, BPC, training time and one fixed-seed sample as constants labeled "measured in Lab 3, configuration X". Choose by the compute budget and tell me which. Never print an unmeasured number, and never a word-level perplexity.

## Model (provided)

Pre-norm decoder-only transformer, lecture eq. `gpt`: token embedding plus learned position table, `n_l` blocks (eq. `block`), final layer norm, output layer. Starting point, to be tuned: `d = 128`, `n_h = 4` (`d_k = 32`), `n_l = 4`, `T_max = 128`, `d_ff = 4d`, dropout 0.1, AdamW, batch 64, a fixed step count. The provided `MultiHeadAttention` calls the participant's `scaled_dot_product_attention` with tensors of shape `(B, n_h, T, d_k)` and returns the weights `A` for plotting. The training loop, the block, sampling (Lab 3's temperature sampler restated) and the plotting helper are scaffolding.

## Core path (50 minutes)

Format per exercise: Predict, Run, Explain, Check; `# TODO N` stub, folded solution, short "why this works" note.

| # | Participant writes | Checkpoint | Metric tested | Min |
|---|---|---|---|---|
| 0 | Nothing: load the corpus, read the provided block and find the call to their function | none | none | 3 |
| 1 | `scaled_dot_product_attention(Q, K, V, mask=None)` returning `(V_bar, A)`, working with any leading dimensions (eqs. `score`, `sdpa`) | Shapes for 3-D and 4-D inputs; rows of `A` sum to 1; matches a reference on fixed tensors (atol 1e-5). Then a provided cell prints the std of scores with and without `1/sqrt(d_k)` for `d_k` in {16, 64, 256} on unit-variance random `Q`, `K`; participant predicts first (eq. `variance`) | Numerical agreement with the reference; measured score std against `sqrt(d_k)` and 1 | 12 |
| 2 | `causal_mask(T)` (eq. `mask`), passed to Exercise 1's function | **Leak test:** all weights above the diagonal are exactly 0; change the input at one position `t0` and assert the outputs at positions `< t0` are unchanged (`allclose`) and the output at some position `>= t0` differs. Then the same test on the whole untrained model in `eval()` mode, on logits | Max absolute change in earlier outputs (must be 0 within float tolerance) | 8 |
| 3 | The model's input step: `h0 = tok_emb(idx) + pos_emb(arange(T))` (eq. `pos`) | First, provided, on **one** attention layer with no positions: shuffling the tokens before the last position leaves the last position's output unchanged. Then with their input step: shape assert, and the same shuffle now changes the output | Max absolute difference at the last position, without and with positions | 6 |
| 4 | Nothing new: start training (provided loop), predict the ranking of n-gram, LSTM and GPT while it runs, then call the provided evaluation | Printed table: test PPL, BPC and training time for the character n-gram, the LSTM and the mini-GPT; the untrained GPT's loss is within a tolerance of `ln 65`; samples at the temperatures Lab 3 used, beside the LSTM's | **Test PPL in nats on the full test split** | 12 |
| 5 | `mean_attention_distance(A)`: for one head, the mean over `t` of `sum_i alpha[t, i] * (t - i)`; then plot all heads of one layer on a short passage (helper provided) | Asserts on hand-built matrices: identity gives 0, a pure previous-token matrix gives 1 (ignoring row 1). Printed table of distance per head and layer; every plotted `A` is lower triangular with rows summing to 1 | Mean attention distance in characters, per head | 7 |

Total 48 minutes, 2 of slack.

## Stretch (one section, last, optional; not needed by any later lab)

Write `MultiHeadAttention.forward` with the split (`view`, `transpose`, concatenate, project; lecture eq. `mha`) and `Block.forward` in pre-norm form (eq. `block`). Checkpoints: with the provided reference's weights loaded, outputs match (`allclose`); `n_h = 1` reproduces Exercise 1 followed by the projection; the leak test of Exercise 2 passes on their block; parameter count equals the reference's and does not change with `n_h`.

## Compute budget (free Colab T4; whole notebook under 10 minutes)

| Step | Budget |
|---|---|
| Setup, data, checkpoints of Exercises 1 to 3 | under 1 min |
| Character n-gram baseline (if recomputed) | about 1 min |
| Train the mini-GPT | at most 5 min |
| LSTM (only if retrained here) | whatever remains; otherwise quoted from Lab 3 |
| Evaluation with overlapping windows, samples, attention plots, stretch checks | under 1.5 min |

For orientation only, not our measurement: the nanoGPT README (read 2026-10-04) reports a validation loss of 1.88 nats for a 4-layer, 4-head, 128-wide model with context 64 after about 3 minutes on a CPU, and 1.47 for a 6-layer, 384-wide model with context 256 after about 3 minutes on an A100. Its split differs from ours, so these are not targets.

## Flags for the Lab Engineer

1. **Do not promise that the GPT beats the LSTM.** A small transformer on 1 MB of text within a few minutes may or may not. The table prints the result with training time beside it; there is no assertion on the direction of GPT against LSTM. The lecture says the same. Set an assertion only for GPT test PPL below the character trigram's, with a margin from at least three seeds, and only if the measurements support it. All comparison thresholds are yours to set from measurements. If the GPT loses to the LSTM, report the numbers and the configuration; do not tune on `test` and do not weaken the LSTM.
2. State in the notebook each model's parameter count, training steps and time, because the comparison is only meaningful at a stated budget.
3. The leak test must run in `eval()` mode (dropout off) and on a model with no batch-dependent layer. Use `float("-inf")` with `masked_fill`, not a large negative number, so the zeros are exact.
4. Exercise 3's shuffle test holds exactly for a **single** attention layer only. In a causal stack, deeper layers can infer position from the mask. Do not run it on the full model.
5. Reference for Exercise 1: an explicit loop or `einsum` written in the notebook is enough. `torch.nn.functional.scaled_dot_product_attention` is an option, but I could not read its documentation page today, so check its boolean-mask convention (I believe `True` means "may attend") and that it returns only the output, not the weights, against the pinned PyTorch version.
6. Use the lecture's names in code and comments: `H`, `Q`, `K`, `V`, `A`, `V_bar`, `d_k`, `n_h`, `n_l`, `T_max`, `W_Q`, `W_K`, `W_V`, `proj`, `tau`. Do not name anything `c` or `h_bar` (taken by Labs 3 and 4).
7. Exercise 5 makes no claim about what a head "means". The text should repeat the lecture's caution about reading attention maps. No threshold on the distances unless you measure a stable one.
8. Send me, measured: the results table (three seeds if time allows), run time per section on a T4, the score standard deviations of Exercise 1, and one attention map per head for a fixed passage, so that the illustrative figure `images/05-causal-mask.svg` can be replaced with real weights.
9. Module 6 loads pretrained transformers through Hugging Face and does not reuse this model. Nothing here needs to be saved for a later lab.
