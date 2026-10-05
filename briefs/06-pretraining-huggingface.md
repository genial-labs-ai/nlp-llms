# Lab brief: `notebooks/06-pretraining-huggingface.ipynb`

From the Academic Director to the Neural Lab Engineer. Lecture: `lectures/06-pretraining-huggingface.qmd` (same symbols, equation labels and function names). Standards: `PLAN.md` section 5. Data contract: `data/README.md`. This file is not rendered by Quarto.

**Objectives exercised** (`_variables.yml`, `m06`): explain subword tokenization and the masked and causal pretraining objectives (Exercises 1 to 4); load, inspect and run pretrained models with Hugging Face (Exercises 2 to 5); fine-tune an encoder for classification (Exercise 5).

**Status of the numbers.** Values marked *measured* were run by me on 2026-10-04 on macOS (arm64, Python 3.12) with `transformers` 5.18.0, `tokenizers` 0.23.2, `datasets` 5.0.1, `torch` 2.14.1. Nothing was run on Colab or on a T4. Everything marked *estimate* is for you to replace with a measurement.

## Models (propose pinning in `_variables.yml`; I may not edit it)

| Role | Model ID | Size (measured) | Why |
|---|---|---|---|
| Encoder, GPU path | `distilbert/distilbert-base-uncased` | 66.4M parameters, 6 layers, d = 768, 512 positions; Apache 2.0 | BERT-family, so the lecture's WordPiece, `[CLS]`, `[MASK]` and 80/10/10 apply unchanged. The checkpoint includes the masked-LM head, so one download serves Exercises 4 and 5 and the stretch. Half the layers of BERT-base. It is the model in the official text-classification guide |
| Encoder, CPU path | `google/bert_uncased_L-4_H-256_A-4` (BERT-mini) | 11.2M parameters, 4 layers, d = 256; Apache 2.0 | Same vocabulary as the GPU encoder (*measured*: the two vocabularies are identical, 30,522 entries), so tokenization cells do not change. Its model card suggests learning rates of 3e-5 to 3e-4 and 4 epochs |
| Causal LM | `distilbert/distilgpt2` | 81.9M parameters, byte-level BPE, 50,257 entries; Apache 2.0 | Smallest GPT-2 checkpoint. Inference only in this lab |

A single `DEVICE`-dependent switch picks the encoder: DistilBERT when a GPU is present, BERT-mini otherwise. Print which one ran next to every reported number.

## Data (fixed; do not change)

arXiv Topics v1 through `load_topics()` from `data/README.md`, pasted unchanged: 4 classes, train 4,800 / val 600 / test 1,600. Never re-split, subsample or filter val or test. Tune on val; report on test once. Input is `title + "\n" + abstract`; truncate in the tokenizer call, not in the data.

**Truncation: a finding you need to act on.** *Measured* with the DistilBERT tokenizer on the training split: median 258 tokens, 95th percentile 370, maximum 508; **51% of training texts are longer than 256 tokens**. The 183-word median in `data/README.md` is words, and the ratio is about 1.4 tokens per word. So `max_length=256` cuts the end of about half the abstracts. That is acceptable for topic classification (the title and opening sentences carry the topic) and it keeps the compute in budget, and the lecture says so. Please measure test accuracy at 256 and at 384 once; if 384 is clearly better and still fits the budget, tell me and I will change the lecture. Keep 256 as the default until then.

## Provided (participants do not write)

Setup cell with pinned installs and seeds; the loading cell; `train_bpe` and `bpe_encode` (lecture Section 2, built on the Exercise 1 functions); all model and tokenizer loading; the fill-in-the-blank probe; the masked-LM loss cell; the `Trainer` configuration; the TF-IDF + logistic regression baseline, recomputed in a cell so the notebook runs cold (0.7 s per `data/README.md`); the results table; the hand-off cell for Lab 11.

## Core path (50 minutes)

Predict, Run, Explain, Check for each exercise: a `# TODO N` stub, a folded solution, a "why this works" note.

| # | Min | Participant writes | Lecture | Checkpoint (assertion) and metric |
|---|---|---|---|---|
| 0 | 3 | Nothing. Setup, load data, load the three checkpoints (start the downloads first, they run while Exercise 1 is done) | | Printed split sizes and device |
| 1 | 10 | `pair_counts(words)` and `merge_pair(words, pair)` on the dictionary `{tuple of symbols: count}` | Section 2, steps 3a and 3c | On the toy corpus `low` 5, `lower` 2, `newest` 6, `widest` 3 with end-of-word `_` and ties broken by the pair that sorts first: assert the count of `("e","s")` is 9; assert the first five merges are `e+s`, `es+t`, `est+_`, `l+o`, `lo+w`; assert `bpe_encode("lowest")` is `["low", "est_"]`; assert the vocabulary size is 11 + 5 (@eq-bpe-size). All *measured* with the reference code in the lecture. **Metric:** exact match with the lecture's table |
| 2 | 7 | `tokens_per_word(encode_fn, texts)`: total tokens divided by total whitespace-separated words | Sections 2 and 6 | Assert the function returns 1.0 for a whitespace tokenizer on a fixed string. Scaffold trains BPE (`vocab_size=4000`, `Whitespace` pre-tokenizer) on the training texts, then prints a three-row table on the **val** texts for our BPE, the DistilBERT tokenizer and the GPT-2 tokenizer (special tokens excluded), and the three tokenizations of one fixed sentence. Assert our tokenizer's vocabulary size equals the requested size. **Metric:** tokens per word. Reference, *measured*: 1.55 (ours, 4,000), 1.40 (WordPiece), 1.38 (GPT-2); training took 0.5 s. Predict first: will a 4,000-entry vocabulary give more or fewer tokens per word than a 30,000-entry one? |
| 3 | 8 | `lm_perplexity(text)`: logits from `AutoModelForCausalLM`, shift by one position, mean negative log-probability of tokens 2 to T, exponentiate | @eq-clm | Assert agreement within 1e-3 relative with `exp(model(ids, labels=ids).loss)`. Assert a fixed sentence has lower perplexity than the same words shuffled with a fixed seed. Reference, *measured* with distilgpt2: 107.8 for "The model is trained on a large corpus of text." against 2,632 shuffled. Scaffold prints the top-5 next tokens for a prompt and one greedy continuation. **Metric:** perplexity per BPE token, labeled **not comparable** with the character-level numbers of Labs 1, 3 and 5; a provided line converts the total to nats per character to show how one would compare |
| 4 | 10 | `mask_tokens(input_ids, special_mask, mask_id, vocab_size, p=0.15)` returning `(corrupted, labels)` with `-100` outside the selected set | Section 5, steps 1 and 2 | On a padded batch of 256 training texts with a fixed seed: no special or padding position is selected; `labels` equals the original ID at selected positions and `-100` elsewhere; selected share in [0.13, 0.17]; of the selected, share equal to `[MASK]` in [0.77, 0.83] and share unchanged in [0.08, 0.125]. *Measured* with my reference: 0.148, 0.798, 0.100 over 8,989 selected tokens; widen the tolerances if your seed needs it. Scaffold then computes @eq-mlm with `AutoModelForMaskedLM(..., labels=labels)` on arXiv text (*measured* on 16 texts: 2.58 nats) and runs the fill-in-the-blank probe from the lecture. **Metric:** the three shares; the masked-LM loss is printed, not asserted |
| 5 | 12 | `tokenize(batch)` (truncation at 256, no padding) and `compute_metrics(eval_pred)` (accuracy and macro-F1) | Section 7, @eq-head | Assert `compute_metrics` on a hand-made `(logits, labels)` pair matches known values. Predict the starting loss, then assert the untrained model's val loss is within 0.25 of `log 4`. Run the provided `Trainer`. A provided cell recomputes the logits of one batch from `h_[CLS]` and the head's parameters in eval mode and asserts agreement with `model(...).logits`; for DistilBERT the head is `pre_classifier` (768 to 768), ReLU, `classifier` (768 to 4), which I confirmed from the parameter names. **Metric:** **test accuracy and macro-F1**, in the results table below; assert accuracy clears a floor you set from measured runs across at least three seeds |

Time: 50 minutes with no slack. If the room is behind, Exercise 2 becomes a demonstration (run the solution); then Exercise 3. Exercises 1, 4 and 5 are the ones the objectives require.

### The results table (the contract with Labs 1, 2 and 11)

One table, same test split (1,600), same two metrics, computed with the same scikit-learn calls as Lab 1:

| Row | Source |
|---|---|
| TF-IDF + logistic regression | recomputed in this notebook, with Lab 1's settings |
| Averaged SGNS embeddings + feed-forward network | read from the recorded baseline file agreed in the Lab 1 brief (`data/baselines.json` was proposed); print "not recorded yet" if absent. Do not retrain it here |
| Fine-tuned encoder (name the model, `max_length`, epochs, seed) | this lab |

**Do not promise a gain, in prose or in assertions.** The dataset builder measured TF-IDF + logistic regression at 0.892 test accuracy (`data/README.md`). My one reference run per model (*measured*, single seed 0, Apple-silicon MPS, not a T4, the `Trainer` settings in the lecture):

| Model | Settings | Val accuracy | Test accuracy | Test macro-F1 | Training time (MPS) |
|---|---|---|---|---|---|
| DistilBERT | 2 epochs, lr 5e-5, batch 32, `max_length=256` | 0.897 | 0.899 | 0.898 | 332 s |
| BERT-mini | 3 epochs, lr 1e-4, batch 32, `max_length=256` | 0.892 | 0.886 | 0.886 | 79 s |

So DistilBERT is about 0.7 points above the baseline (11 fewer errors in 1,600), which one seed cannot distinguish from zero, and BERT-mini is slightly below it. The lecture tells participants to expect a match or a small margin. Print the difference whichever way it falls, with the count of test errors for each row, and do not tune until the encoder wins. Run three seeds and report the spread. The floor assertion in Exercise 5 must sit below the worst seed of the model that ran, and must not be "beats TF-IDF": on the CPU path it probably does not.

## Hand-off to Lab 11 (calibration)

Lab 11 draws reliability diagrams for this classifier and fits temperature scaling. It needs **logits and labels on val (to fit the temperature) and on test (to evaluate)**. A Colab runtime does not persist, so:

1. **In this lab,** the final core cell saves `lab06_logits.npz` with `val_logits` (600, 4), `val_labels`, `test_logits` (1600, 4), `test_labels`, as float32 and int64, plus the model ID, `max_length`, seed and library versions in a small JSON string. It also calls `trainer.save_model("lab06-classifier")` so a participant who keeps the runtime can reuse the model. Raw logits, not probabilities: temperature scaling divides logits.
2. **For Lab 11,** commit the `.npz` from your measured GPU run under `data/` (about 35 kB) and register it in `_variables.yml` with its hash, like the other data files. That needs the coordinator and the Quarto/Colab Architect, who own those files; raise it with them. Lab 11 loads this file by default through `fetch`, so it starts in seconds and everyone analyzes the same model.
3. **To recreate,** keep the whole fine-tuning step in one function, `finetune(model_name, max_length, epochs, lr, seed)`, that returns the four arrays. Lab 11 can paste it for participants who want their own model.

Two constraints that follow: no label smoothing, mixup or other training change that alters calibration without saying so (Lab 11 wants the plain cross-entropy model); and fix the seed, because the committed logits must match the numbers this notebook prints.

## Stretch (optional, last, not required by any later lab): attention and hidden states

Load `AutoModel` with `attn_implementation="eager"` and call it with `output_attentions=True, output_hidden_states=True`. The default attention implementation does not return weights: *measured*, without `eager` the `attentions` tuple came back empty with a warning. With it, DistilBERT returns 6 attention tensors of shape `(1, 12, T, T)` and 7 hidden-state tensors of shape `(1, T, 768)` (embeddings plus 6 layers).

- Participant writes `attention_entropy(attn)`: the mean entropy of the rows of one head's weights. Checkpoints: every row of every head sums to 1 within 1e-5; the entropy of a uniform row over T tokens equals `log T`.
- Scaffold plots the heat-map of two heads for one arXiv sentence (the helper from Lab 5 if it exists) and the cosine similarity between the layer-0 and layer-6 vectors of the same word in two different contexts (*bank*, or a domain word such as *model*).
- Question to answer in one sentence: which layer separates the two senses? Carry over Module 4's caution: weights show where the model read, not why.

## Run time and environment

- Pinned installs in the setup cell: `transformers==5.18.0`, `tokenizers==0.23.2`, `datasets==5.0.1` (current on PyPI on 2026-10-04), plus `accelerate` if Colab lacks it (`Trainer` needs it; 1.15.0 worked here). `transformers` 5.18.0 requires `tokenizers>=0.23.1,<0.24.0`. Check what Colab preinstalls before pinning; I could not.
- `report_to="none"` and `save_strategy="no"` so nothing prompts for a login or fills the disk. No Hub token is needed; unauthenticated downloads print a rate-limit warning that the notebook should mention.
- Downloads: three checkpoints, roughly 270 MB, 350 MB and 45 MB (*estimate* from parameter counts).
- **GPU path, *estimate*, to be measured on a free T4:** DistilBERT, 2 epochs, batch 32, `max_length=256`, `fp16=True`, is 300 optimizer steps; I expect 2 to 4 minutes of training (it took 332 s on a laptop MPS device without `fp16`, *measured*; a T4 with `fp16` should be faster, but that is a guess) and under 30 seconds for val and test prediction, and under 7 minutes for Run all including downloads. Budget: 10 minutes.
- **CPU path, *estimate*:** BERT-mini has about one sixth of DistilBERT's parameters and a third of its width. It trained in 79 s on a laptop MPS device (*measured*), about a quarter of DistilBERT's time. I expect full fine-tuning (3 epochs, learning rate 1e-4) to take several minutes on Colab's two CPU cores, possibly more than the budget allows. If it exceeds the budget, train on a stated subset of the training split (allowed by the data rules, and say so beside the number); never shrink val or test.
- Seeds set in the setup cell and in `TrainingArguments(seed=...)`.

## API facts I checked, and against what

Checked on 2026-10-04 against the Transformers documentation `main` branch (the stable docs were v5.17.0; PyPI had 5.18.0) and by running the calls on 5.18.0:

- `Trainer(...)` takes `processing_class`, not `tokenizer`. The lecture passes neither, only `data_collator=DataCollatorWithPadding(tokenizer=tokenizer)`.
- `TrainingArguments`: `eval_strategy` (the old `evaluation_strategy` is gone); `warmup_ratio` is not in the documented signature, and `warmup_steps` accepts a float below 1 as a ratio; `report_to` defaults to `"none"`; `output_dir` defaults to `"trainer_output"`.
- `AutoTokenizer`, `AutoModel`, `AutoModelForMaskedLM`, `AutoModelForCausalLM`, `AutoModelForSequenceClassification.from_pretrained(name, num_labels=4)`: ran as written in the lecture. The DistilBERT tokenizer returned `token_type_ids` as well as `input_ids` and `attention_mask`; the collator and model accepted them.
- `datasets.Dataset.from_dict(...).map(fn, batched=True)`: ran.
- `tokenizers`: `Tokenizer(BPE(unk_token=...))`, `Whitespace`, `BpeTrainer(vocab_size=..., special_tokens=...)`, `train_from_iterator`, `encode(...).tokens/.ids`, `encode_batch`: ran on 0.23.2 and match the quicktour.
- `DataCollatorForLanguageModeling` exists with `mlm_probability=0.15`, `mask_replace_prob=0.8`, `random_replace_prob=0.1`. Use it only as a cross-reference in the "why this works" note; participants write `mask_tokens` themselves.

## As built (Director review, 2026-10-05)

The notebook departs from this brief in these ways. All are accepted; the lecture now matches the notebook.

- Exercise 3: `lm_perplexity(text, model, tokenizer)`, not `lm_perplexity(text)`. The shuffled-sentence assertion is skipped only in the offline test mode.
- Exercise 5: `finetune(model_name, max_length, epochs, lr, seed, train_rows=None)` returns a dict (`val_logits`, `val_labels`, `test_logits`, `test_labels`, `trainer`, `start_val_loss`, `train_seconds`), not four arrays. `train_rows` is the CPU subset.
- CPU fallback: BERT-mini (`models.encoder_cpu`) fine-tuned on the first **800** training texts (`CPU_TRAIN_ROWS`), 3 epochs, lr 1e-4, with per-epoch evaluation off; validation and test are complete. The results table prints the model and the number of training rows beside the number. The accuracy floors (0.85 on GPU, 0.60 on CPU) are provisional until three seeds are measured on Colab.
- The masked LM is always DistilBERT (`models.encoder`), on CPU as on GPU: Exercises 2 and 4 and the stretch use it; only Exercise 5 switches encoders.
- Stretch: loads `AutoModel` with `attn_implementation="eager"`; participants write `attention_entropy(attn)`; the scaffold plots the entropy per layer and head and the two extreme heads, and compares *bank* across three sentences layer by layer.
- Setup is labeled 3 minutes in the notebook, so Setup plus Exercises 1 to 5 sum to 50 (3 + 10 + 7 + 8 + 10 + 12).
- A test-only switch, `NLP_LLMS_OFFLINE_TINY=1`, replaces the three checkpoints with tiny random models so the repository's offline machine can execute every cell. It is off on Colab.

Fixed in this review (notebook): the Lab 2 row was looked up by the prefix `lab02.` and the first `topics` entry, which happened to be `lab02.sgns_avg_ffn` but would have picked the GloVe stretch row (`lab02.glove_avg_ffn`, 0.805) if the file's order changed. It now reads the exact id `lab02.sgns_avg_ffn` (0.8669 accuracy, 0.8670 macro-F1). The comment in Checkpoint 5c no longer presents the 0.899 and 0.886 reference accuracies as verified.

**Measured in the repository build (2026-10-05, offline, Linux CPU).** Hand BPE first merges `e+s`, `es+t`, `est+_`, `l+o`, `lo+w`. Our BPE (4,000 entries, 4,800 training texts): 1.548 tokens per word on val. TF-IDF + logistic regression recomputed: 0.8838 accuracy, 0.8841 macro-F1, equal to Lab 1. Masking shares from the reference `mask_tokens`: 0.148 selected; of those 0.799 `[MASK]`, 0.099 unchanged, 0.102 random. These shares were measured on the offline stand-in WordPiece tokenizer, not DistilBERT's, so the real token counts differ. Median training text: 183 words. Every model cell ran only on tiny random stand-ins; their numbers mean nothing.

**Not reproduced in the repository build, so treated as unverified.** Every pretrained-model number above marked *measured* (the 258-token median and 51% truncation share, 1.40 and 1.38 tokens per word, 107.8 and 2,632 perplexities, 0.798/0.100 shares with DistilBERT, the 2.58-nat masked-LM loss, the 0.897/0.899 and 0.892/0.886 accuracies, the training times) and the fill-in-the-blank output once quoted in the lecture. The build machine cannot reach the Hugging Face Hub. The lecture no longer quotes any of them as measured; replace them from a Colab run.

## Not verified

- **Anything on Colab or a T4**: run time, memory, preinstalled versions, `fp16` behavior.
- **The fine-tuned accuracy beyond one seed.** The two reference rows above are single runs on a laptop GPU. Seed variance, the 384-token setting, `fp16` and other learning rates were not tried.
- **The `tokenizers` documentation is in flux.** The live quicktour and the trainers reference carry notices that a release candidate ("rc0 bindings") does not yet expose building or training tokenizers. Training works on 0.23.2, which is what `transformers` 5.18.0 requires. Keep the pin, and re-check before any upgrade.
- ~~**The Lab 2 row** of the results table: Lab 2 is not built, and the location of the recorded baselines is a proposal.~~ Resolved: Lab 2 is built and `data/baselines.json` holds `lab02.sgns_avg_ffn`.
- **Lab 5's notation and plotting helper**: the lecture was drafted in parallel with Lecture 5.
