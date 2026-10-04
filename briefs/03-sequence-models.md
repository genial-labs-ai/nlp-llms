# Lab brief: `notebooks/03-sequence-models.ipynb`

**From:** Academic Director. **To:** Neural Lab Engineer. **Lecture:** `lectures/03-sequence-models.qmd`.
**Status:** brief only. No number below has been measured; every figure marked "estimate" must be replaced by a measured one.

## Objectives exercised

1. Implement an RNN and an LSTM language model.
2. Explain vanishing gradients and how gating addresses them.
3. Compare perplexity against the n-gram baseline.

## Data and the fair-comparison contract

- Corpus: the Shakespeare corpus from Lab 1, with the same fixed train/validation/test split by character offset (defined once with the dataset, not in this lab). Fetched by URL with the fallback copy under `data/`.
- Tokens are characters. Vocabulary $V$ = the characters of the train split; assert that validation and test contain no character outside it.
- **The comparison with Module 1 is character-level on both sides.** Lab 1's n-gram model may be word-level, and perplexities at different token units are not comparable. This lab therefore includes, as provided scaffolding, a character bigram and trigram model built with Lab 1's method (counts with add-$k$ smoothing), trained on the same train split. Notebooks cannot import from each other on Colab, so the code is restated here in about 20 lines and labeled "Lab 1's method, applied to characters".
- Rules the notebook must enforce: same split; same $V$; the n-gram, RNN and LSTM are scored on exactly the same test characters; $\mathrm{PPL}=\exp(\text{mean NLL in nats})$; bits per character ($\log_2 \mathrm{PPL}$) printed beside it; add-$k$ and all neural settings chosen on validation, never on test. Neural models are evaluated over the test stream in contiguous chunks with the hidden state carried across chunks, so no position is scored with a cold state except the first.
- Never print a word-level perplexity from Lab 1 beside these numbers.

## Core path (50 minutes)

Participants write only the functions named below. Data loading, batching, the model classes, both training loops, the n-gram baseline and all plots are provided. Each exercise follows Predict → Run → Explain → Check, as a `# TODO N` stub with a folded solution.

| # | Min | Participant writes | Lecture equation | Checkpoint (what it tests) |
|---|---|---|---|---|
| 1 | 8 | `rnn_cell_step(x_t, h_prev, W, U, b)` | RNN step | `torch.allclose` with `nn.RNNCell` given the same weights (atol 1e-5), output shape `(B, d_h)`. Tests correctness of the recurrence. |
| 2 | 6 | `lm_loss_and_ppl(logits, targets)` returning mean NLL (nats), PPL, BPC | loss, perplexity | Uniform logits give PPL equal to $|V|$ and loss equal to $\ln|V|$; agrees with `F.cross_entropy`. Tests the metric used in every later comparison. |
| – | 5 | Nothing. Run the provided n-gram baseline and train the RNN LM built on their cell | – | Printed: validation PPL of bigram, trigram and RNN. Predict first: which is lowest? |
| 3 | 9 | Part A: nothing; predict, run and explain the provided plot of $\lVert\partial\mathcal{L}_T/\partial h_s\rVert$ against distance $T-s$ (log scale). Part B: `clip_gradients(params, kappa)` | Jacobian product and bound; clipping | A: assert the norm at the largest distance is below the norm at distance 1 by a measured factor. B: global norm after clipping is at most $\kappa$; direction unchanged (cosine 1); a gradient already below $\kappa$ is untouched; agrees with `torch.nn.utils.clip_grad_norm_`. Then a provided short run at a deliberately high learning rate, with and without clipping, plots gradient norm and loss per step. |
| 4 | 6 | The last two lines of `lstm_cell_step`: cell update and hidden state (the four gate lines are given) | LSTM equations | `torch.allclose` with `nn.LSTMCell` given the same weights, for both $h_t$ and $c_t$. Tests the gated additive update. |
| – | 6 | Nothing. Train the provided `nn.LSTM` LM; repeat the gradient-distance plot for the LSTM on the same axes; print the results table | direct-path Jacobian | Test PPL and BPC for bigram, trigram, RNN, LSTM. Assert LSTM test PPL is below the trigram's, with a margin set from measured runs. Tests objective 3. |
| 5 | 8 | `sample_with_temperature(logits, tau, generator)` | temperature sampling | On a fixed logit vector, empirical frequencies over 20,000 draws match $\mathrm{softmax}(z/\tau)$ (total variation below a measured tolerance) at $\tau\in\{0.5, 1, 2\}$; at $\tau=0.01$ every draw is the arg max. Then generate 300 characters at $\tau\in\{0.5, 1.0, 1.5\}$ and at greedy; participants describe the difference. |

Total 48 minutes, 2 minutes of slack.

## Stretch (optional, last, not needed by any later lab)

Implement `top_k_filter(logits, k)` and `nucleus_filter(logits, p)`, each returning logits with excluded tokens set to $-\infty$, composable with Exercise 5's sampler. Checkpoints: exactly $k$ finite entries; for nucleus, the kept set is the smallest prefix of the sorted distribution with mass at least $p$ (test on a hand-built distribution, including the case where the top token alone exceeds $p$); $k=|V|$ and $p=1$ leave the logits unchanged.

## Compute budget (free Colab T4)

Target: under 10 minutes of compute for Run all with solutions, stretch included. Proposed starting point, to be tuned: $d=64$, $d_h=256$, one layer, chunk length $T=128$, batch 64, Adam, clipping at $\kappa=1$, a fixed step count per model. Estimates, not measurements: about 1 minute for the n-gram baseline, 2 to 3 minutes each for the RNN and the LSTM, under 1 minute for everything else. The hand-written cells are used for the checkpoints and for a short demonstration only; if a Python-loop RNN is too slow to train within budget, train an `nn.RNN` with identical equations and say so in the notebook. The gradient-distance plot needs one backward pass per model, not training.

## Notes for the engineer

- Use the lecture's symbols as variable names (`W`, `U`, `b`, `h_prev`, `c_prev`, `tau`, `kappa`). The count function and the cell state must not share the name `c`: use `counts` for the baseline.
- `nn.RNNCell` and `nn.LSTMCell` each have two bias vectors (`bias_ih`, `bias_hh`); the lecture has one. In the checkpoints, pass their sum or zero one of them. `nn.LSTMCell` documents its gates in the order input, forget, cell, output (checked against the PyTorch 2.14 documentation); slice `weight_ih` and `weight_hh` in that order, and let the Exercise 4 checkpoint confirm it.
- Set seeds in the setup cell. Sampling checkpoints take an explicit `torch.Generator`.
- Report measured values for: test PPL and BPC of all four models, the decay factor in Exercise 3A, the tolerance in Exercise 5, and run time per section. Set the assertion thresholds from at least three seeds.
- If the LSTM does not beat the character trigram within the budget, or if the RNN's gradient-distance curve does not decay visibly, report it to the Academic Director. Do not weaken the comparison or the assertion to make the notebook pass.
- Lab 5 trains a mini-GPT on this corpus and compares with this LSTM. Keep the split, the vocabulary, `lm_loss_and_ppl` and the evaluation routine in a form Lab 5 can restate unchanged.
