# Lab brief: `notebooks/01-text-as-data.ipynb`

From the Academic Director to the Neural Lab Engineer. Lecture: `lectures/01-text-as-data.qmd`. Standards: `PLAN.md` section 5. Stack: NumPy, scikit-learn (no PyTorch, no GPU).

## Objectives exercised

1. Tokenize text and justify the choices (Exercise 1).
2. Build and evaluate an n-gram language model (Exercises 2 to 4).
3. Train a linear text classifier and read its errors (Exercises 5 and 6).

## Data and splits (fixed; later labs depend on them)

| | Language-model corpus | Classification set |
|---|---|---|
| Dataset | The Shakespeare corpus (proposed: Tiny Shakespeare) | The news-topic set, 4 classes (proposed: AG News subset) |
| Train | First 90% of the characters | 10,000 examples, stratified sample of the official train split, seed 0 |
| Validation | Next 5% | 2,000 examples, stratified, disjoint from train, same seed |
| Test | Final 5% | The official test split, complete |
| Preprocessing | None for the character level: case, punctuation and newlines kept | Title and body joined with one space |

Splits are contiguous for the LM corpus (no shuffling) and are computed on character offsets, so Labs 3 and 5 can reproduce them exactly. The sizes for the classification set are my proposal; if the dataset is swapped or the sizes change, change them here first, because Labs 2, 6 and 11 must use the same three splits. Put the split code in one provided cell that later labs copy verbatim.

## Core path (50 minutes)

Each exercise follows Predict → Run → Explain → Check: a `# TODO N` stub, a folded solution, a "why this works" note, and the checkpoint below. Participants write only the functions named; everything else is scaffold.

| # | Min | Participant writes | Lecture equation | Checkpoint (what it tests) |
|---|---|---|---|---|
| 1 | 8 | `tokenize(text)` (word level, lowercased, punctuation as separate tokens) and `build_vocab(tokens, min_count)` with `<unk>` and `<s>` | Section 2 | Assert exact token list for two fixed strings; assert `<unk>` and `<s>` are in the vocabulary and IDs are contiguous. Printed metric: vocabulary size and **token coverage of the validation split** (fraction of tokens not mapped to `<unk>`) at `min_count` 1 and 2. Scaffold plots rank against frequency on log–log axes (Zipf). |
| 2 | 12 | `ngram_counts(ids, n)` and `prob(w, context, counts, k, vocab_size)` | MLE and add-k (@eq-mle, @eq-addk) | On a five-token toy corpus, assert two hand-computed probabilities (one seen, one unseen n-gram). Assert that for three sampled contexts, including one never seen, the probabilities **sum to 1 over the vocabulary** within 1e-9. |
| 3 | 8 | `perplexity(ids, n, counts, k, vocab_size)` | Perplexity (@eq-ppl) | Assert that with empty counts (uniform model) PPL equals the vocabulary size within 1e-6. A scaffold cell shows that the unsmoothed model (`k = 0`, with the 0/0 case for unseen contexts guarded) has infinite validation perplexity. Scaffold then sweeps `k` on the validation split and prints the perplexity table described under "Baselines". |
| 4 | 5 | `sample(context, length, ...)`: the one line that draws the next token from the smoothed distribution | Section 3, sampling | Assert the output has the requested length, all IDs are in the vocabulary, and a fixed seed reproduces the same sample. Predict first: how will n = 2 and n = 4 samples differ? Scaffold prints samples for n = 1 to 4 at word and character level. |
| 5 | 10 | `tfidf(counts)`: idf from a document-term count matrix (from `CountVectorizer` with the Exercise 1 tokenizer), multiply, L2-normalize | TF-IDF (@eq-tfidf-sklearn) | Assert `allclose` with `TfidfTransformer()` on the training matrix. Scaffold fits `LogisticRegression` (regularization chosen on validation from a short grid) and `MultinomialNB`, and prints **validation accuracy and macro-F1** for both. A provided cell named "softmax by hand" recomputes `predict_proba` from `coef_` and `intercept_` (@eq-softmax) and asserts agreement. |
| 6 | 7 | `prf_from_confusion(cm)`: per-class precision, recall, F1 and macro-F1 from a confusion matrix | Precision, recall, F1 (@eq-prf) | Assert `allclose` with `sklearn.metrics.precision_recall_fscore_support` on the test predictions. Scaffold shows the confusion matrix, the 10 highest-weight words per class, and the 10 most confident errors. Participant writes two sentences: which class pair is most confused, and one error category they found. |

Design notes:

- Write the n-gram code over sequences of integer IDs, independent of the tokenization, so the same functions serve the word-level run (for intuition and samples) and the character-level run (the baseline).
- Pad with `n - 1` start tokens; treat each split as one stream. Perplexity is averaged over the `T` tokens of the split (start tokens are context only, never predicted). Natural log throughout.
- Unknown contexts must return the uniform distribution through the add-k formula, not through a special case.
- If the room is behind, Exercise 4 becomes a demonstration (run the solution). Nothing else may be cut: Exercises 3, 5 and 6 produce the baselines.
- The lab does not ask participants to implement the logistic-regression gradient; the lecture says that happens in Lab 2.

## Baselines this lab reports (the contract with Labs 2, 3, 5 and 6)

The last core cell prints one "baseline card" with exactly these numbers and nothing estimated:

**Language model (compared in Lab 3, and through it Lab 5).**

- Metric: perplexity per **character**, natural log, on the **test split** of the Shakespeare corpus (final 5% of characters). Also print the cross-entropy in nats per character (`log PPL`), since that is the loss the LSTM prints.
- Vocabulary: every distinct character in the full corpus; no `<unk>` at character level.
- Report: bigram (n = 2), trigram (n = 3), and the best order n ≤ 5, each with add-k and with `k` chosen on the **validation split** from a fixed grid (suggested: 1, 0.1, 0.01, 0.001). State the chosen `k` for each.
- The headline number for Lab 3 is "best character n-gram, n ≤ 5, add-k, test split".
- Word-level perplexities (n = 2, 3; `min_count = 2`) are printed for the sparsity discussion and are labeled **not comparable** with any later lab.

**Classifier (compared in Labs 2 and 6; its probabilities are examined in Lab 11).**

- Metrics: **accuracy and macro-F1 on the test split** of the news-topic set, plus per-class precision, recall and F1.
- Model: TF-IDF (unigrams, the Exercise 1 tokenizer, scikit-learn default idf and L2 normalization) with multinomial logistic regression; `C` chosen on the validation split by macro-F1; trained on the 10,000-example training split only.
- Also report naive Bayes on the same features and split, as a second reference.
- Test metrics are computed once, after `C` is fixed.

Because a Colab runtime does not persist, later labs cannot read these numbers from this notebook. After executing the notebook, record the measured values, the dataset version and the seed in a small file under `data/` (proposed: `data/baselines.json`) that Labs 2, 3 and 6 load. Agree the location with the Quarto/Colab Architect.

## Stretch (optional, last, not required by any later lab): BM25

Participant writes `bm25_scores(query_tokens, doc_term_counts, doc_lengths, k1=1.5, b=0.75)` using the formula in the lecture's optional box (non-negative idf). Treat the training documents of the classification set as the collection. Checkpoints: assert scores are non-negative; assert that doubling a document's count of a query term raises its score by less than a factor of two (saturation); assert that with `b = 0` the score does not depend on document length. Printed comparison: top-5 documents for three fixed queries under TF-IDF cosine and under BM25. Module 13 re-teaches BM25 through a library, so nothing downstream depends on this code.

## Run time and environment

- CPU runtime; no GPU and no API keys. Only pinned installs that Colab lacks (expected: none beyond the dataset fetch).
- Target: under 2 minutes of compute for Run all with solutions. This is a budget, not a measurement; the character 5-gram sweep and the `C` grid are the two cells to time. Report the measured figure.
- Seeds set in the setup cell; sampling and the stratified subsample must be reproducible.
- Datasets fetched by URL with the fallback copy under `data/`.

## Not verified by the Director

- No code in this brief or in the lecture was executed. The baseline values are unknown until the notebook runs.
- That scikit-learn's `LogisticRegression.predict_proba` equals softmax of `decision_function` for the pinned version with more than two classes (true for the multinomial formulation; confirm for the pin, since `multi_class` handling changed across recent releases).
- That `TfidfTransformer` defaults (`smooth_idf=True`, `norm="l2"`, `sublinear_tf=False`) are unchanged in the pinned version.
- Dataset licenses, sizes and the official test-split size.
