# Lab brief: `notebooks/02-word-vectors.ipynb`

**From:** Academic Director. **To:** Neural Lab Engineer. **Lecture:** `lectures/02-word-vectors.qmd` (equation labels below refer to it).
**Status:** brief only. Every number here is a target or an estimate, not a measurement. Nothing has been run.

## Purpose

Participants write the skip-gram negative-sampling (SGNS) loss, train embeddings, inspect them, and then **replace Lab 1's TF-IDF features with averaged embeddings on the same classification split, reporting the same metrics (accuracy and macro-F1 on the test split)**. The lab must show the two rows side by side and report the result whichever way it falls. Do not tune until embeddings win; the lecture tells participants this is an empirical question.

## Data

- **Classification:** the news-topic classification set from Lab 1, with the identical train/validation/test split, tokenizer and vocabulary rule. Import or copy Lab 1's loader and seed; do not re-split. Refer to the dataset generically in prose so a license-driven swap does not force a rewrite.
- **Embedding corpus:** the text of that set's **training split only**, labels ignored. Reasons: it is in-domain for the classifier, it needs no second download or license, and it keeps the test split unseen. If Lab 1 subsamples the training split for its classifier, embeddings may still use the full training text, but the classifier must use Lab 1's subset.
- **Not** the LM corpus from Labs 1 and 3: at roughly 0.2M words in an archaic register it is too small for sensible neighbors and is out of domain.
- **Risk to measure (unverified):** I estimate the training text at a few million tokens, which should give sensible neighbors for frequent words (countries, sports, companies) and mostly wrong analogies. Measure token count, vocabulary size, pairs per epoch and seconds per epoch on a T4 before fixing hyperparameters. If neighbors are poor within the budget, report it; do not hide it with a cherry-picked word list. Fallback options to raise with the coordinator: more epochs with a smaller vocabulary, or a larger license-clean corpus.

## Provided (participants do not write)

Setup and seeds; data loading; `make_pairs` (windowing, with frequent-word subsampling); `noise_distribution` (@eq-noise); negative sampler; `SkipGram` module (`E`, `U` as in the lecture's refresher code); both training loops; `analogy` (@eq-analogy); the PCA plot; the TF-IDF + logistic regression baseline, recomputed in a cell so the notebook runs cold without Lab 1's outputs; the results table.

Suggested starting hyperparameters, to be confirmed by measurement: $d = 100$, window half-width $m = 5$, $K = 5$, vocabulary capped near 20,000 by minimum count, large batches (several thousand pairs), Adam.

## Core path (50 minutes)

Each exercise follows Predict, Run, Explain, Check, as a `# TODO N` stub with a folded solution.

| # | Participant writes | Equation | Checkpoint (assertion) and metric | Min |
|---|---|---|---|---|
| – | Setup, load data, read three sample (center, context) pairs | | Printed corpus statistics | 4 |
| 1 | `sgns_loss(e_center, u_context, u_negatives)` with shapes `(B, d)`, `(B, d)`, `(B, K, d)`, returning the batch mean | @eq-sgns | (a) all-zero inputs give exactly $(K+1)\log 2$; (b) matches a reference value on a fixed seeded batch; (c) autograd gradient with respect to `e_center` equals @eq-sgns-grad. **Metric:** training loss, which must fall below its initial value of about $(K+1)\log 2$ | 12 |
| – | Run the provided training loop; predict what the loss curve looks like | | Loss curve; assertion that final loss is below the initial loss | 4 |
| 2 | `nearest_neighbors(word, E, k)` by cosine similarity, excluding the query word | @eq-cosine | Toy 5-word matrix with a known answer; similarities sorted descending; query excluded. **Metric:** printed neighbors for a fixed probe list, plus the share of probe words whose top-10 contains a listed expected neighbor (printed, not asserted, until the engineer has measured its variance across seeds) | 7 |
| – | Run `analogy` on a provided list; view the PCA plot; explain one failure | @eq-analogy | Printed analogy accuracy on the provided list. Expected to be low; say so | 5 |
| 3 | `average_embeddings(token_ids, mask, E)` returning `(N, d)` | @eq-avg | Padding does not change the result (same document with and without padding); a document of only padding gives the zero vector, not NaN; output shape | 6 |
| 4 | `FeedForwardClassifier.forward` (one hidden ReLU layer, returns logits) | @eq-ffn | Logits shape `(B, C)`; provided cell checks autograd's $\partial\mathcal{L}/\partial z$ against $\hat{y} - \mathrm{onehot}(y)$ (@eq-backprop). **Metric:** test accuracy and macro-F1 after the provided training loop, with an assertion that accuracy clears a floor the engineer sets from measured runs (well above the majority-class rate, with margin for seed variance) | 8 |
| – | Results table: TF-IDF + logistic regression against averaged SGNS embeddings + feed-forward network, same test split, accuracy and macro-F1; one-sentence explanation of the gap | | Table printed; both rows computed in this notebook | 4 |

Embeddings are **frozen** in the classifier on the core path, so the comparison isolates the features. Do not add a fine-tuning variant to the core path.

## Stretch (optional, last, not required by any later lab)

**Pretrained GloVe comparison.** Load a small pretrained GloVe set (50 or 100 dimensions), restrict it to the lab vocabulary, and rerun three things with it: neighbors for the probe list, the analogy list, and the Exercise 3 and 4 classifier. Add a third row to the results table and report vocabulary coverage. Point to draw out: the same method trained on billions of tokens gives much better analogies, which is a statement about data, not about the objective. To confirm before building: download size and time on Colab, the license of the vectors, and a pinned source with a fallback.

## Run time

Target, not measured: under 10 minutes of compute cold on a free Colab T4 for the core path, of which about 3 minutes for SGNS training and under 1 minute for the classifier. The engineer reports the measured figures, and the measured CPU time if the T4 is unavailable.

## Coordination and open questions

1. Lab 1 must expose its split, tokenizer and vocabulary in a form Lab 2 can reuse exactly; Lab 6 reuses the same split and metrics again.
2. The Exercise 1 function signature above is the one shown in the lecture's refresher code. If it changes, tell the Academic Director so the lecture changes too.
3. The lecture's equation-to-lab table uses the function names in this brief: `sgns_loss`, `nearest_neighbors`, `average_embeddings`, `FeedForwardClassifier`, `noise_distribution`, `analogy`.
4. The lecture has a placeholder for a PCA figure to be produced from this lab's trained embeddings; save the plotted words and coordinates.
5. If 50 minutes proves too tight in a dry run, cut the analogy step to a demonstration cell first; Exercises 1, 3 and 4 and the results table are the part the module objectives require.
