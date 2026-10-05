# Lab brief: `notebooks/04-seq2seq-attention.ipynb`

From the Academic Director to the Neural Lab Engineer. Lecture: `lectures/04-seq2seq-attention.qmd` (same symbols and equation names). Lab standards: `PLAN.md` section 5. This file is not rendered by Quarto.

**Objectives exercised** (from `_variables.yml`, `m04`): build an encoder-decoder model; explain the fixed-vector bottleneck; implement additive and dot-product attention and read attention maps.

**Every number below is a target or a budget, not a measurement.** Replace each with the value you measure on a free T4, and tell me if a target cannot be met.

## Task and data (provided, generated in the notebook, no download)

- Character-level transduction from written dates to ISO format, e.g. `3 March 2021` to `2021-03-03`.
- **Input length is varied by the number of dates, K = 1 to 4.** A source holds K dates joined by short connectives (`from … to …`, `and`, `;`); the target is the K ISO dates in the same order, separated by a space. The amount the decoder must remember then grows with the input, which is what a fixed vector cannot absorb. Distractor padding around a single date is *not* the primary design: a GRU can learn to skip filler, so the baseline may not fail.
- Several surface formats, all unambiguous (month written as a word, or year-first numerals; no `03/04/21`). Fixed seed. A held-out test set with an equal number of examples per K (suggested 500 each).
- Buckets for every metric: K = 1, 2, 3, 4. Also print the source length in characters per bucket.

## Models (provided)

GRU encoder (unidirectional) and GRU decoder with the same hidden size `d_h`, so that the dot-product score applies. Embedding layers, the combination layer (lecture eq. `attentional`), greedy decoding, the training loop and the plotting helper are scaffolding. Both models use the same `d_h`, the same number of steps and the same optimizer, and the notebook says so, because the comparison is only fair under an equal budget.

## Core path (50 minutes)

| # | Participant writes | Checkpoint | Metric tested | Min |
|---|---|---|---|---|
| 0 | Nothing: read generated examples | none | none | 3 |
| 1 | Teacher-forced forward pass and loss of the plain model: `s_0 = h_S`, targets shifted right, cross-entropy ignoring padding (eqs. `decoder`, `output`, `loss`) | Logit shape assert; loss of the untrained model within a tolerance of `ln |V|` | Mean per-token cross-entropy | 10 |
| 2 | `exact_match(pred, gold)`; predict which bucket fails before running the evaluation | Assert on a hand-made batch; then print the table of accuracy by K for the no-attention model | **Exact-match accuracy by bucket** (whole output string correct) | 8 |
| 3 | `dot_product_attention(q, keys, mask)` returning weights and context (eqs. `dot`, `weights`, `context`) | Shapes; each row of weights sums to 1; padded positions have weight exactly 0; matches reference values on fixed tensors. After training: accuracy by K for both models side by side | Exact-match accuracy by bucket, attention against no attention | 12 |
| 4 | `additive_score(q, keys)` (eq. `additive`), plugged into the Exercise 3 step | Shape; matches reference values for seeded `W_q`, `W_k`, `u`; resulting weights sum to 1 | Numerical agreement with the reference | 10 |
| 5 | Plot heat-maps for 3 to 4 test inputs (helper provided); complete `alignment_hit_rate` | Printed hit rate above a threshold you set from measurement | Share of output digits whose arg-max source position lies inside the span of the correct date | 7 |

Format per exercise: Predict, Run, Explain, Check; `# TODO N` stub, folded solution, short "why this works" note. Training runs start while participants write their prediction for the next step.

## Stretch (one section, last, optional): beam search

Participant writes `beam_search(model, src, B)` with length normalization. Checkpoints: `B = 1` reproduces greedy decoding exactly; print exact match and mean log-probability for `B = 1, 3, 5`. Do not assert that a larger beam scores higher: that is not guaranteed. Not required by any later lab.

## Compute budget (free Colab T4; whole notebook under 10 minutes)

| Step | Budget |
|---|---|
| Setup and data generation | under 0.5 min |
| Train the no-attention model | 3 min |
| Train the dot-product attention model | 3.5 min |
| Evaluation, heat-maps, additive checkpoint | under 1 min |
| Stretch: beam search on a test subset | under 1.5 min |

No third full training run. Train the additive model only if your measured total leaves room, and then for a short run behind a flag.

## Flags for the Lab Engineer

1. **The failure must be measured, not assumed.** The lab's argument needs the no-attention model to be good on K = 1 and poor on K = 4, and the attention model to be good on both, under the same budget. Working targets: no attention at least 0.90 on K = 1 and at most 0.50 on K = 4; attention at least 0.90 on K = 4. Report the measured table. If the baseline does not fail, change the task in this order: lower `d_h`, raise the maximum K, add more surface formats. Do not weaken the baseline in a way the attention model does not share. If the targets still cannot be met, stop and tell me; the lecture's "In the lab" notes will need to change.
2. State in the notebook that the baseline could improve with more capacity or training, and that the claim is about equal budgets.
3. Dot-product attention on raw GRU states sometimes trains slowly. If it does, report it; do not switch the main run to another score without telling me, because the lecture names the dot product for Exercise 3.
4. The Exercise 5 threshold and whether the arg-max lands on or just after the expected characters (an encoder state summarizes the source up to its position) need to be measured. Allow a tolerance of one position if needed and say so in the text.
5. Use the lecture's names in code and comments: `h` (encoder states), `s` (decoder state), `alpha`, `h_bar` (context; not `c`, which is the LSTM cell state in Lab 3), `W_q`, `W_k`, `u`.
6. Send me one measured heat-map for `3 March 2021` so the figure `images/04-attention-alignment.svg` can show real weights.

## As built (Director review, 2026-10-05)

The notebook departs from this brief in these ways. All are accepted; the lecture now matches the notebook.

- Exercise 1: two functions, `forward_plain(model, src, src_len, tgt_in)` and `seq2seq_loss(logits, tgt_out)`. The training loop does the shift (`tgt_in = tgt[:, :-1]`, `tgt_out = tgt[:, 1:]`). The loss is the mean over non-padding target tokens, not the sum of lecture eq. `loss`; the lecture now says so.
- Exercise 3: an extra function, `masked_attention(scores, keys, mask)`, holds the mask, the softmax and the average; `dot_product_attention(q, keys, mask)` is one line that calls it and returns `(alpha, h_bar)`. Exercise 4 reuses `masked_attention`.
- Exercise 4: `additive_score(q, keys, W_q, W_k, u)`, with the parameters passed in (they live on the model), not `additive_score(q, keys)`. The additive model is trained only behind `TRAIN_ADDITIVE = False`, as the brief allowed.
- Exercise 5: `alignment_hit_rate(alphas, examples, tol=1)` is scored on attention from a teacher-forced pass over the gold target, so each row lines up with a target character; the heat-maps use greedy decoding. The threshold is 0.95, set from the measured 0.996. The tolerance counts an arg-max up to `tol` positions after the end of the date's span.
- Stretch: `beam_search(model, source, B, max_len=MAX_OUT)` takes one source string and returns `(output, log-probability)`.
- A `QUICK` switch (environment variable `NLP_LLMS_QUICK=1`) trains each model for 500 instead of 4,000 steps and skips the accuracy checkpoints. It exists because the full run does not fit the CPU budget.
- The Exercise 2-3 accuracy checkpoint asserts no-attention exact match of at least 0.85 on $K = 1$, below the brief's 0.90 target, because the recorded run measured 0.898.

**Measured** (recorded run: 4-core CPU shared with other jobs, PyTorch 2.14.1, 4,000 steps per model, seed 0; see `data/README.md`). Exact match on the test set (500 sources per $K$), greedy decoding:

| $K$ | No attention | Dot-product attention |
|---|---|---|
| 1 | 0.898 | 1.000 |
| 2 | 0.000 | 1.000 |
| 3 | 0.000 | 1.000 |
| 4 | 0.000 | 1.000 |

Without attention the first output date is right for 0.90 to 0.93 of the sources in every bucket and every later date for none. Alignment hit rate 0.996 at `tol = 1` (0.992 to 1.000 per bucket at `tol = 0`). Beam search on 200 test sources: $B = 1, 3, 5$ all give exact match 1.000 and mean log-probability $-0.0007$. The baseline targets of flag 1 are met except no-attention $K = 1$, which is 0.002 short of 0.90; the failure appears from $K = 2$, earlier than the brief anticipated. The measured heat-map for `3 March 2021` (flag 6) is in `data/lab04_attention_example.json` and is now the lecture figure, drawn by `scripts/make_figures_04.py`.

## Not verified by the Director

- Run time on a Colab T4. The only full run was on a shared CPU and took 107 minutes (81 and 24 minutes of training), far over the 10-minute budget. Whether a T4 meets the budget is unmeasured.
- The committed notebook at full settings. The recorded run used an earlier revision; since then the code changed only in the run switch (`QUICK` replaces `LAB04_STEPS`), the `K = 1` threshold (0.90 to 0.85; the earlier revision's assertion failed at 0.898) and the `TRAIN_ADDITIVE` flag. The Director ran the committed revision with `QUICK = True` only (`NLP_LLMS_QUICK=1 python scripts/test_notebooks.py 04-seq2seq-attention`): it passed in 372.5 s on the shared CPU, with the accuracy checkpoints skipped by design.
- Run-to-run variation: a second CPU run with the same settings scored 0.99 on $K = 1$ of a validation set (`data/README.md`). The no-attention $K = 1$ number should be read as "about 0.9", not as 0.898.
