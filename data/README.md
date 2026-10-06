# Data

The datasets of the workshop's running thread, and the fallback copies that notebooks fetch when an upstream host is down.

- Names, URLs, hashes, splits and seeds are recorded in the `datasets` section of [`_variables.yml`](../_variables.yml). This page explains them. `tests/test_data.py` fails if the two disagree.
- Licenses were checked against the live primary source on **2026-10-04**. Each row links the page that was read.
- Datasets keep their own licenses. The workshop's CC BY 4.0 and MIT licenses (see [LICENSE](../LICENSE)) do not apply to the files in this directory.

## Decisions

| Use | Modules | Dataset | License | Status |
|---|---|---|---|---|
| Text classification | 1, 2, 6, 11 | **arXiv Topics v1**: 7,000 arXiv titles and abstracts in 4 classes, built for this workshop | CC0 1.0 (arXiv metadata) | Fixed. Copy committed |
| Language modeling | 1, 3, 5 | **Tiny Shakespeare**, 1,115,394 characters | Public-domain text; packaged in an MIT repository | Fixed. Copy committed |
| Sequence transduction | 4 | Human-readable dates to ISO format | Generated in the notebook | No file. Lab 4 fixes the generator and its seed |
| Instruction tuning | 7 | **Databricks Dolly 15k**, a subset | CC BY-SA 3.0 | License confirmed. Lab 7 fixes the subset |
| Calibration logits | 6, 11 | **Lab 6 encoder logits**: validation (600) and test (1,600) logits of DistilBERT fine-tuned on arXiv Topics v1 | CC0 1.0 (derived from CC0 data; Romeo to confirm) | Built on the build Mac's GPU, not a T4. Copy committed (`lab06_logits.npz`) |
| Pairwise preferences | 9, 10 | **Lab 9 preference pairs**: 9,000 pairs of GPT-2 continuations of 256 prompts we wrote, labeled by a known rule, and the reward model trained on them | CC BY 4.0 (ours); the responses are samples from an Apache 2.0 model | Built 2026-10-06 on an Apple M1 Pro CPU, copies committed (`lab09_prompts.json`, `lab09_preferences.jsonl.gz`, `lab09_reward_model.pt`). See [below](#lab-9-preference-pairs-modules-9-and-10) |
| Labeled decisions | 11, 12, 14 | **Workshop Desk Decisions v1**: 2,400 typed decisions under a written policy, built for this workshop | CC0 1.0 | Template items built, copy committed (`decisions_v1.jsonl.gz`). **Status `v1-template-only`**: the 80 hand-written items and the template audit need two people |
| RAG documents | 13, 14, 15 | **Workshop Lectures v1**: lecture pages 1–12 and the reading list as plain text, frozen at one commit | CC BY 4.0 (ours) | Built, copy committed (`workshop_lectures_v1.jsonl.gz`). Rebuilt 2026-10-05 from the commit that completed `references.qmd`. **Status `provisional`**: lecture 12 and possibly lectures 6–11 will still change; freeze them, rebuild if needed, and set `final` before any question is written |
| RAG questions | 13, 15 | **Workshop RAG Questions v1**: 80 questions with evidence spans, written and checked by people | CC BY 4.0 (proposed) | **Not written yet**: needs two people. Validator, tools and instructions committed |
| Capstone questions | 15 | **Workshop Capstone Questions v1**: Lab 13's 80 questions by ID plus 45 new ones (27 unanswerable, at least 13 of them memory bait; 18 `reading`), written and checked by people | CC BY 4.0 (proposed) | **Not written yet**: needs two people, after Lab 13's set. Instructions, validator and manifest builder ready |
| Protein structure | 0 | **Ubiquitin, PDB 1UBQ**: 76 residues, 602 heavy atoms | CC0 1.0 (PDB data; license page read only through a search extract) | Fixed. Copy committed (`1ubq.pdb`) from a pinned mirror; RCSB stays canonical |
| Customer purchases | 0 | **Workshop Purchases v1**: 48,953 synthetic invoice lines from 3,000 customers, built for this workshop | CC BY 4.0 (ours) | Fixed. Copy committed (`purchases_v1.csv.gz`). UCI Online Retail II is an optional real-data alternative, not committed |

AG News, the original proposal for classification, was rejected. See [Why not AG News](#why-not-ag-news).

## Loading contract

Paste this cell into a notebook unchanged, after the setup cell. It uses only the standard library. Every lab that uses a dataset gets the same bytes and the same split, because each file is checked against a SHA-256 hash and the split is stored in the file (topics) or fixed by character offset (LM corpus).

`fetch` looks for a copy on disk first, then tries each URL in order: the canonical URL, then the fallback. It saves what it downloads, so a re-run does not download again.

```python
import csv, gzip, hashlib, io, os, urllib.request
from pathlib import Path


def fetch(name, urls, sha256):
    """Return the bytes of a workshop data file, checked against its SHA-256."""
    cache = Path(os.environ.get("NLP_LLMS_DATA", "data"))
    for path in (cache / name, Path("../data") / name):
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == sha256:
            return path.read_bytes()
    for url in urls:  # canonical URL first, fallback second
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                blob = response.read()
        except OSError as error:
            print(f"Could not fetch {url}: {error}")
            continue
        if hashlib.sha256(blob).hexdigest() != sha256:
            print(f"Ignoring {url}: the file does not match the expected hash")
            continue
        cache.mkdir(parents=True, exist_ok=True)
        (cache / name).write_bytes(blob)
        return blob
    raise RuntimeError(f"Could not load {name} from any source")


TOPIC_CLASSES = ["cs.CL", "cs.CV", "cs.LG", "cs.RO"]  # label 0, 1, 2, 3
TOPIC_NAMES = ["Language", "Vision", "Machine learning", "Robotics"]


def load_topics():
    """arXiv Topics v1. Returns {"train" | "val" | "test": (texts, labels)}."""
    blob = fetch(
        "arxiv_topics_v1.csv.gz",
        [
            "https://raw.githubusercontent.com/project-delphi/nlp-llms/main/data/arxiv_topics_v1.csv.gz",
            "https://cdn.jsdelivr.net/gh/project-delphi/nlp-llms@main/data/arxiv_topics_v1.csv.gz",
        ],
        "49b51502fc6cf6ecff23fa266c5372594a88414dd96513579237792710246dc5",
    )
    splits = {"train": ([], []), "val": ([], []), "test": ([], [])}
    for row in csv.DictReader(io.StringIO(gzip.decompress(blob).decode("utf-8"))):
        texts, labels = splits[row["split"]]
        texts.append(row["title"] + "\n" + row["abstract"])
        labels.append(int(row["label"]))
    return splits


def load_lm_corpus():
    """Tiny Shakespeare. Returns {"train" | "val" | "test": str}, split by character offset."""
    blob = fetch(
        "tinyshakespeare.txt",
        [
            "https://raw.githubusercontent.com/karpathy/char-rnn/6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e/data/tinyshakespeare/input.txt",
            "https://raw.githubusercontent.com/project-delphi/nlp-llms/main/data/tinyshakespeare.txt",
        ],
        "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed",
    )
    text = blob.decode("utf-8")
    return {"train": text[:1_000_000], "val": text[1_000_000:1_055_000], "test": text[1_055_000:]}
```

Use it like this:

```text
topics = load_topics()
train_texts, train_labels = topics["train"]   # 4,800 strings, 4,800 ints
corpus = load_lm_corpus()
train_text = corpus["train"]                  # 1,000,000 characters
```

Rules for labs:

- Do not re-split, re-shuffle, subsample or filter the evaluation data. Tune on `val`. Report on `test` once per model.
- A lab may train on less than the full training split (for speed), but must say so next to the number it reports.
- For the topics set, the model input is `title + "\n" + abstract`, as `load_topics` returns it. Truncate inside the model (for example `max_length=256` for the encoder in Lab 6), not in the data.
- For the LM corpus, the comparable number across Labs 1, 3 and 5 is the **character-level** cross-entropy on `test`, in nats per character, and its exponential (perplexity per character). The vocabulary is the 65 characters of `train`. A word-level model may be built as well, but its perplexity is not comparable with the later labs.

### Where the fallback comes from

Notebooks run on Colab without the repository cloned, so the fallback copy is fetched from the raw GitHub URL of this directory on the `main` branch (`datasets.fallback_base` in `_variables.yml`).

- **Topics.** The file is built by us, so this repository is its canonical host. The second URL is the same file through the jsDelivr CDN, which serves public GitHub repositories from a different host.
- **LM corpus.** The canonical URL is the upstream file, pinned to a commit. The fallback is our copy.
- **Local copy.** If the notebook runs inside a clone (`data/` or `../data/` exists), or the environment variable `NLP_LLMS_DATA` names this directory, `fetch` uses the local file and makes no request.

**Not yet verified: the repository URLs.** On 2026-10-04 `https://github.com/project-delphi/nlp-llms` returned 404 to an anonymous request, so the repository is private or not yet pushed. Until it is public and these files are on `main`, the three `project-delphi/nlp-llms` URLs above fail. The upstream Tiny Shakespeare URL and the local-copy path work today. `scripts/test_notebooks.py` runs each notebook in a temporary directory, so until the repository is public it must set `NLP_LLMS_DATA` to this directory.

## arXiv Topics v1 (classification)

| | |
|---|---|
| File | `arxiv_topics_v1.csv.gz`, 3,172,762 bytes (9,534,099 uncompressed) |
| SHA-256 | `49b51502fc6cf6ecff23fa266c5372594a88414dd96513579237792710246dc5` |
| Canonical URL | <https://raw.githubusercontent.com/project-delphi/nlp-llms/main/data/arxiv_topics_v1.csv.gz> |
| Fallback URL | <https://cdn.jsdelivr.net/gh/project-delphi/nlp-llms@main/data/arxiv_topics_v1.csv.gz> |
| Columns | `split`, `label`, `category`, `arxiv_id`, `title`, `abstract` |
| Classes | 0 `cs.CL` (Computation and Language), 1 `cs.CV` (Computer Vision), 2 `cs.LG` (Machine Learning), 3 `cs.RO` (Robotics) |
| Splits | train 4,800, val 600, test 1,600. Balanced: 1,200, 150 and 400 per class |
| Seed | 0 (`random.Random(0)` in `build_arxiv_topics.py`). The split is stored in the file, so no lab needs the seed |
| Source | The [arXiv API](https://info.arxiv.org/help/api/index.html), queried on 2026-10-04 |
| License | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) |
| License source | <https://info.arxiv.org/help/api/tou.html>, read 2026-10-04 |

**What it is.** Each row is one arXiv e-print first submitted between 2024-01-01 and 2024-06-30 whose *primary* category is one of the four classes. The label is that primary category. Whitespace in the title and abstract is collapsed to single spaces; nothing else is changed. `arxiv_id` includes the version, so `https://arxiv.org/abs/<arxiv_id>` is the paper.

**How it was built.** [`build_arxiv_topics.py`](build_arxiv_topics.py) pages through every e-print in the window for each category, keeps those with that primary category, sorts by ID, shuffles with seed 0 and takes 1,750 per class. Notebooks never run this script. A rebuild is not guaranteed to give the same bytes, because arXiv metadata can change; the committed file and its hash are the dataset.

**License, as checked.** The arXiv API terms of use say: "You are free to use descriptive metadata about arXiv e-prints under the terms of the Creative Commons Universal (CC0 1.0) Public Domain Declaration." The arXiv license page (<https://info.arxiv.org/help/license/index.html>) says a CC0 dedication "will apply to all metadata". Neither page lists the fields that count as metadata. We read titles and abstracts as metadata because arXiv's API and its own bulk metadata dataset serve them as such; that reading is ours, not a quoted statement. The papers themselves are under their authors' licenses and are not included. arXiv asks that projects not present themselves as endorsed by arXiv; this workshop is not.

**Why this set.** The running thread needs a 4-class set that is small, legally clean for a paid workshop, and hard enough that Labs 1, 2 and 6 can show an improvement and Lab 11 has errors to calibrate on.

Measured on this file, on the machine named under [What was run](#what-was-run):

| Model | Input | Train rows | Val accuracy | Test accuracy | Fit and score |
|---|---|---|---|---|---|
| TF-IDF (`min_df=2`) + logistic regression | title + abstract (the contract) | 4,800 | 0.890 | 0.892 | 0.7 s |
| same | title + abstract | first 2,400 | | 0.878 | |
| same | title + abstract | first 600 | | 0.852 | |
| same | title only | 4,800 | 0.788 | 0.802 | 0.1 s |

The baseline leaves 173 test errors (we did not run the same baseline on AG News, so no comparison with it is claimed). Most of them are between `cs.CV` and `cs.LG` (47 Vision papers predicted as Machine learning) and between `cs.LG` and `cs.CL`. The averaged-embedding model of Lab 2 has been run on it (see [Measured baselines](#measured-baselines-baselinesjson)): 0.8669 test accuracy, below TF-IDF. The fine-tuned encoder (Lab 6) has **not** been run on it yet.

**Known properties to teach with.**

- The label is the authors' choice of primary category. Many papers are cross-listed, and `cs.CL`, `cs.CV` and `cs.LG` overlap, so some label noise is built in. This is useful in Lab 1 (error analysis) and Lab 11 (calibration).
- Texts are longer than news snippets: median 183 words (1,328 characters), 95th percentile 258 words, maximum 363 words. Count-based models do not care. For the encoder in Lab 6, truncate to 256 tokens.

### Why not AG News

- The original host (<http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html>, read 2026-10-04) says the corpus is provided "for research purposes" and "any other non-commercial activity", and: "You are not authorized to change the corpus or to re-distribute (part of) it with a different name."
- The Hugging Face card (<https://huggingface.co/datasets/fancyzhx/ag_news>, read 2026-10-04) has no license: "More Information Needed".
- This is a paid workshop, and a 4-class subset under `data/` would be a changed, redistributed part of the corpus. Both conflict with the stated terms.

Alternatives that were checked and not chosen:

| Candidate | License found | Why not |
|---|---|---|
| DBpedia 14, 4-class subset | CC BY-SA 3.0 and GFDL ([card](https://huggingface.co/datasets/fancyzhx/dbpedia_14), [dbpedia.org](https://www.dbpedia.org/about/)) | Too easy. TF-IDF + logistic regression measured 0.952 to 0.983 test accuracy across seven 4-class subsets with 2,000 to 8,000 training examples. Little room for Labs 2 and 6 to improve, and few errors for Lab 11 |
| News Category Dataset (HuffPost headlines) | Tagged CC BY 4.0 on a [Hugging Face mirror](https://huggingface.co/datasets/heegyu/news-category-dataset); the Kaggle primary page could not be read | The compiler's license covers text written by a publisher. No statement from the publisher was found. Same doubt as AG News |
| SIB-200 (English) | CC BY-SA 4.0 per search result; not checked further | About 1,000 sentences in total. A test set of about 200 is too small to compare models or to draw a reliability diagram |

## Tiny Shakespeare (language modeling)

| | |
|---|---|
| File | `tinyshakespeare.txt`, 1,115,394 bytes, 40,000 lines, ASCII, 65 distinct characters |
| SHA-256 | `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed` |
| Canonical URL | <https://raw.githubusercontent.com/karpathy/char-rnn/6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e/data/tinyshakespeare/input.txt> (pinned to a commit) |
| Fallback URL | <https://raw.githubusercontent.com/project-delphi/nlp-llms/main/data/tinyshakespeare.txt> |
| Splits, by character offset | train `[0, 1,000,000)`, val `[1,000,000, 1,055,000)`, test `[1,055,000, 1,115,394)`: 1,000,000, 55,000 and 60,394 characters |
| Seed | None. The split is by position |
| License | Text: public domain (William Shakespeare, died 1616). Packaging: MIT |
| License source | <https://github.com/karpathy/char-rnn>, read 2026-10-04 |

**What it is.** The example corpus of Andrej Karpathy's `char-rnn`: a concatenation of passages from Shakespeare's plays. Our copy is byte-identical to the upstream file (same hash).

**License, as checked.** The `char-rnn` README ends with "## License" and the single word "MIT". The repository has no separate `LICENSE` file, and GitHub's license API reports none. The README describes the file only as "a subset of works of Shakespeare". Neither it nor the Hugging Face card (<https://huggingface.co/datasets/karpathy/tiny_shakespeare>, license: "More Information Needed") names the edition the text was taken from. **Not verified:** the edition, and so whether any editor's changes to the text carry rights in some jurisdiction. We judge the risk low: the plays are in the public domain and the file has been redistributed under MIT since 2015.

**Split.** All 65 characters occur in `train`, so `val` and `test` contain no unseen character. Both boundaries fall inside a speech; that is acceptable for a character-level model.

## Databricks Dolly 15k (instruction tuning, Module 7)

| | |
|---|---|
| URL | <https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a/databricks-dolly-15k.jsonl> (pinned to a revision) |
| Size | 13,085,339 bytes, 15,011 records |
| SHA-256 | `2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec` |
| Fields | `instruction`, `context`, `response`, `category` (8 categories) |
| License | [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) |
| License source | <https://huggingface.co/datasets/databricks/databricks-dolly-15k>, read 2026-10-04 |

**License, as checked.** The card, published by Databricks, says: "This dataset can be used for any purpose, whether academic or commercial, under the terms of the Creative Commons Attribution-ShareAlike 3.0 Unported License." It also says some records include material from Wikipedia, under the same license.

**What that means for us.** Commercial use and redistribution are allowed, with attribution. ShareAlike applies to the dataset and to adaptations of it: a subset committed here must stay CC BY-SA 3.0 and carry the attribution "Databricks Dolly 15k, © Databricks, CC BY-SA 3.0, with material from Wikipedia". It cannot be relicensed under the workshop's CC BY 4.0. Whether ShareAlike reaches a model fine-tuned on the data is unsettled; Lab 7 does not distribute its fine-tuned weights, so the question does not arise.

**Left for Lab 7.** The full file is too large to commit. Lab 7 chooses the subset (which categories, how many records, the seed), commits it here if it is under 1 MB, and adds it to `_variables.yml` with the same fields as the two sets above.

## Generated and workshop-built sets

- **Dates (Module 4)** are generated inside the notebook, with a fixed seed. The generator is described below.
- **Pairwise preferences (Modules 9 and 10)** are built once by a script that needs the Hugging Face Hub, and committed: see [Lab 9 preference pairs](#lab-9-preference-pairs-modules-9-and-10).
- **Labeled decisions (Modules 11, 12, 14)** are described under [Workshop Desk Decisions v1](#workshop-desk-decisions-v1-decisions-modules-11-12-14). **RAG documents and questions (Modules 13, 14, 15)** are described under [Workshop Lectures v1](#workshop-lectures-v1-rag-documents-modules-13-14-15) and [Workshop RAG Questions v1](#workshop-rag-questions-v1-modules-13-and-15). Module 11 also reuses the arXiv Topics validation and test splits for its reliability diagrams.

### Dates to ISO format (Module 4)

Generated by the `data-gen` and `data-build` cells of `notebooks/04-seq2seq-attention.ipynb`. No file, no download, no license question: the data is ours and is rebuilt identically on every run.

| | |
|---|---|
| Task | Character-level transduction: a source with $K$ written dates, $K = 1$ to $4$, to the $K$ ISO dates (`YYYY-MM-DD`) in the same order, separated by one space |
| Dates | Uniform over 1900-01-01 to 2099-12-31 (`random.randint` over ordinal days) |
| Surface formats | Six, chosen uniformly per date: `3 March 2021`, `March 3, 2021`, `3 Mar 2021`, `Mar 3, 2021`, `3rd March 2021`, `2021/3/3`. All unambiguous: the month is a word or the numerals are year first |
| Connectives | $K = 2$: `from A to B`, `between A and B`, `A and B`, `A; B`, uniformly. $K \ge 3$: `A, B, … and Z` or `A; B; …; Z`, each with probability 1/2 |
| Test | 500 distinct sources per $K$ (2,000), `random.Random(0)` |
| Train | 25,000 distinct sources per $K$ (100,000), `random.Random(1)`; sources that occur in the test set, and the lecture example `3 March 2021`, are skipped |
| Batch order | `random.Random(2)`, the same for every model |
| Source length | Test means 12.6, 32.5, 43.4 and 58.2 characters for $K = 1, 2, 3, 4$ (maximum 77) |
| Vocabularies | Source: 43 characters plus `<pad>`. Target: the 10 digits, `-`, space, `<pad>`, `<bos>`, `<eos>` ($|V| = 15$) |
| Validation | None in the notebook: Lab 4 tunes nothing. The step count was chosen from learning curves on a separate set, `make_split(seed=3, n_per_k=200)`, never on the test set |

**Lab 4, measured 2026-10-05** (4-core CPU shared with other jobs, PyTorch 2.14.1, 2 threads; not run on Colab). Settings: $d = 32$, $d_h = 128$, batch 128, Adam with learning rate 0.002, clipping at 1, 4,000 steps per model, seed 0, the same batches for both models. Exact match on the test set, greedy decoding:

| $K$ | Mean source characters | No attention | Dot-product attention |
|---|---|---|---|
| 1 | 12.6 | 0.898 | 1.000 |
| 2 | 32.5 | 0.000 | 1.000 |
| 3 | 43.4 | 0.000 | 1.000 |
| 4 | 58.2 | 0.000 | 1.000 |

Without attention, the first output date is right in 0.90 to 0.93 of the sources in every bucket and every later date in none. Alignment hit rate of the attention model (share of target digits whose arg-max source position lies in the correct date, tolerance 1): 0.996. Training took 4,888 s without attention and 1,457 s with it, under different loads, so the two times are not comparable. A second training of the model without attention with the same settings but one thread scored 0.99 on $K = 1$ of the validation set at 4,000 steps, so CPU runs of this model differ in the $K = 1$ number. These values are not in `baselines.json`, whose `dataset` field must name an entry of `datasets` in `_variables.yml`, and the dates set has none.

The measured attention matrix for `3 March 2021` (dot-product model of the run above; prediction `2021-03-03`, correct) is in [`lab04_attention_example.json`](lab04_attention_example.json). The Lecture 4 alignment figure is drawn from it by `scripts/make_figures_04.py`. `alpha[t][i]` is output step $t$ (rows: the ten output characters, then `<eos>`) and source character $i$ (columns: the twelve characters of the source); each row sums to 1, to rounding.

## Workshop Desk Decisions v1 (decisions, Modules 11, 12, 14)

| | |
|---|---|
| File | `decisions_v1.jsonl.gz`, 143,499 bytes (1,718,074 uncompressed), 2,400 lines, one JSON item per line |
| SHA-256 | `56ee7e8b4e7ef418e5da6a9f242f615b21dacd3c5d312c0e4635174847aa0df1` |
| Canonical URL | <https://raw.githubusercontent.com/project-delphi/nlp-llms/main/data/decisions_v1.jsonl.gz> |
| Fallback URL | <https://cdn.jsdelivr.net/gh/project-delphi/nlp-llms@main/data/decisions_v1.jsonl.gz> |
| Policy | [`decisions_policy_v1.md`](decisions_policy_v1.md): rules P1 to P8 and the routing order, 499 words, given verbatim in every prompt |
| Builder | [`build_decisions.py`](build_decisions.py): standard library only, no network, no model, seed 0; `--check` rebuilds in memory and compares |
| Statistics | [`decisions_v1_stats.json`](decisions_v1_stats.json), written by the builder: counts per split, family, label, rule and difficulty, the hashes, and the status below |
| Splits | train 2,000, dev 100, test 300; each half `policy`, half `route` |
| License | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/): written by us, with invented names and `example.org` addresses only |
| Specification | `briefs/11-calibration.md`, "Shared decision set (interface for Labs 11, 12 and 14)". Changing a field, option, split or rule changes all three labs |
| Status | **`v1-template-only`** (`"hand_items": 0`). See [What is not done yet](#what-is-not-done-yet-the-human-steps) |

**What it is.** Decisions that the front-desk assistant of a fictional workshop series must make, given the written policy, the records of one event and one registration, and a request from a participant. Each item asks one typed question with fixed answers, and **every label follows from the policy and the records**, so it is a checkable fact, not a preference. Calibration needs that: a confidence can only be right or wrong against a true label. The domain continues Lab 8 (workshops with a topic, a city, a date, seats and a remote flag) and anticipates Lab 14's agent (retrieve, calculate, send email).

**Item schema.** `id` (`dec-<split>-NNNN`), `split`, `family` (`policy` or `route`), `source` (`template` or `hand`), `difficulty`, `state` (`today`; `event` with `id`, `topic`, `city`, `start_date`, `seats`, `registered`, `remote`, `fee_eur`; `registration` with `name`, `email`, `status`, `paid_eur`, or `null` when no record was found; `request`, the participant's message), `question`, `options`, `label`, `rule`, `rationale`. **`rule` and `rationale` are for authors and error analysis; never put them, or `difficulty` or `source`, in a prompt.**

- `policy` items ask one of five yes/no questions, one per rule P1 to P5 (for example "Under the policy, is this participant entitled to a full refund?"). Labels are exactly 50/50 in every split, and each rule has a fifth of the items.
- `route` items ask "What should the assistant do next?" with options `retrieve`, `calculate`, `send_email`, `ask_user`, `escalate`, the next steps of Lab 14's agent; the label is the first step of the policy's routing order that applies. Exactly 20% per option in every split.
- `difficulty`: `plain`; `boundary` (`days_before` of 2, 7, 13 or 14, a refund of exactly 500 EUR, or the last seat); `distractor` (an irrelevant number or date in the request); `conflict` (the request contradicts the records, P8); `missing` (information needed to act is absent); `injection` (the request tells the assistant to ignore the policy, P7). Non-boundary items never hit those values. In dev and test, each family is 40% `plain` and at least 8% of each hard tag; every `ask_user` item is a `missing` item.

| Split | Family | Labels | Difficulty: plain / boundary / distractor / conflict / missing / injection |
|---|---|---|---|
| train | policy | yes 500, no 500 | 400 / 160 / 120 / 120 / 80 / 120 |
| train | route | 200 each | 400 / 100 / 100 / 100 / 200 / 100 |
| dev | policy | yes 25, no 25 | 20 / 8 / 6 / 6 / 4 / 6 |
| dev | route | 10 each | 20 / 5 / 5 / 5 / 10 / 5 |
| test | policy | yes 75, no 75 | 60 / 24 / 19 / 18 / 12 / 17 |
| test | route | 30 each | 60 / 15 / 15 / 15 / 30 / 15 |

**How each lab uses it.** Lab 11: the language model's answer and stated confidence on `dev` and `test`; threshold on `dev`, report on `test`. Lab 12: Jev on `dev` and `test`, its toy decision model trained on `train`. Lab 14: `route` items for the router, P5 and P6 items for the email guard, `injection` items for the prompt-injection test. **`train` is never shown to a language model as an evaluation item.**

**Loading.** Paste this cell after the loading cell of the [Loading contract](#loading-contract) (it uses `fetch`):

```python
import json


def load_decisions():
    """Workshop Desk Decisions v1. Returns {"train" | "dev" | "test": [item, ...]}."""
    blob = fetch(
        "decisions_v1.jsonl.gz",
        [
            "https://raw.githubusercontent.com/project-delphi/nlp-llms/main/data/decisions_v1.jsonl.gz",
            "https://cdn.jsdelivr.net/gh/project-delphi/nlp-llms@main/data/decisions_v1.jsonl.gz",
        ],
        "56ee7e8b4e7ef418e5da6a9f242f615b21dacd3c5d312c0e4635174847aa0df1",
    )
    splits = {"train": [], "dev": [], "test": []}
    for line in gzip.decompress(blob).decode("utf-8").splitlines():
        item = json.loads(line)
        splits[item["split"]].append(item)
    return splits
```

**How the labels are made.**

1. **Template items (all 2,400 today).** The builder fixes, per split and family, how many items carry each label and difficulty tag; draws the records from seeded distributions that hit the policy's limits on purpose; renders the request from wordings (dev and test use wordings held out from train); and labels each item with a **rule engine that implements P1 to P8 and the routing order: the label of record**. It also asserts that the engine agrees with the label the generator aimed for, and that every number and date in the request equals the records, except where a `conflict` item states a wrong one or a `distractor` item adds an irrelevant one. `tests/test_decisions.py` re-derives every `policy` label from the records with a second implementation written from the policy text, and checks the rest of the specification.
2. **Hand-written items (none yet).** The specification reserves 80 slots, 20 in dev and 60 in test (half per family), for items written by people to cover phrasing the templates cannot, each labelled independently by two people. **No language model writes or labels any item**, and the agent that wrote the builder did not write them. Until people do, the slots hold extra template items (`fill_in_ids` in the statistics: `dec-dev-0081` to `0100`, `dec-test-0241` to `0300`). When `decisions_hand_v1.jsonl` exists the builder puts its items in the slots and fills only what is left; the 320 main template items of dev and test, and all of train, do not change.
3. **Template audit (not done yet).** Two people label 60 template items blind (`decisions_audit_sheet_v1.jsonl`: 5 per family and difficulty tag, 2 from train and 3 from dev or test, seeded). Every disagreement with the rule engine is traced: a template or rule bug is fixed and the set rebuilt; a human slip is logged. Target: at least 95% agreement before fixes.

**Determinism, as checked.** Two builds, and a build in a fresh interpreter with a different `PYTHONHASHSEED`, give byte-identical files (`tests/test_decisions.py`). Random streams are seeded with strings (`random.Random("0:<split>:<family>:<group>")`), which Python hashes with SHA-512, so they are the same on every platform. The gzip header has `mtime=0` and no file name; the compressed bytes still depend on the zlib build, so `--check` says whether a difference is in the content or only in the compression.

### Labels and agreement

| Check | Items | Agreement | Cohen's kappa |
|---|---|---|---|
| Hand items, annotator A vs B | 0 of 80 | not measured | not measured |
| Template audit, A vs B | 60 | not measured | not measured |
| Template audit, each annotator vs the rule engine | 60 | not measured | not measured |

`python data/decisions_annotation.py agreement A.jsonl B.jsonl --reference engine` computes these numbers; fill the table from its output.

### What is not done yet: the human steps

The instructions for the two people (proposed: Romeo and one instructor; estimated 2 to 3 hours each) are in [`decisions_hand_TEMPLATE.md`](decisions_hand_TEMPLATE.md). In short: write 80 items against the policy text, label them blind, resolve disagreements, merge with `decisions_annotation.py merge-hand`, label the 60-item audit sheet blind, run `agreement`, fix any template bug, rebuild, and update the hash and size here and in `_variables.yml`. Until then, quote results on this set as "template items only".

## Workshop Lectures v1 (RAG documents, Modules 13, 14, 15)

| | |
|---|---|
| File | `workshop_lectures_v1.jsonl.gz`, 189,037 bytes (558,988 uncompressed), 13 lines, one JSON object per page |
| SHA-256 | `4a7e9310ea5bd9206fce4a42c414d1f942c8880f9e7b694b04f04125fcc2d395` |
| Canonical URL | <https://raw.githubusercontent.com/project-delphi/nlp-llms/main/data/workshop_lectures_v1.jsonl.gz> |
| Fallback URL | <https://cdn.jsdelivr.net/gh/project-delphi/nlp-llms@main/data/workshop_lectures_v1.jsonl.gz> |
| Builder | [`build_lectures_corpus.py`](build_lectures_corpus.py): standard library plus PyYAML, no network, no model; reads the pages from one git commit with `git show`, not from the working tree; `--check` rebuilds in memory and compares |
| Source | lecture pages 01–12 and `references.qmd` at commit `3ba37bc06e2e1f23f7896f43e04117e805b6925e` (`source_commit`; also recorded in every record), the commit that completed `references.qmd` for Modules 1–15. The first build read `ec97bea`, before the reading list was finished. Lecture 13 is not included: a corpus that explains RAG to a RAG lab adds nothing |
| Size | 13 documents, 544,974 characters, 84,626 words; 147,738 tokens of `SentenceSplitter`'s tokenizer (tiktoken `cl100k_base`), measured with Lab 13's tokenizer call (`llama_index.core.utils.get_tokenizer()`) outside the notebook. `references` is 40,993 characters and 12,153 tokens |
| Chunks | 1,443 / 705 / 355 at $L$ = 128 / 256 / 512 tokens with $L_o = L/8$ and metadata excluded (`SentenceSplitter`, `llama-index-core` 0.14.25; measured with Lab 13's splitter call outside the notebook, which reproduced the notebook's numbers on the first build exactly: 134,949 tokens and 1,319 / 645 / 325 chunks) |
| License | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), the license of the pages. The pages quote short passages of third-party material, with sources |
| Specification | `briefs/13-rag.md`, decision (a) |
| Status | **`provisional`**. See below |

**Status: provisional.** `references.qmd` is frozen into this snapshot and is now complete for Modules 1–15 (rebuilt 2026-10-05). The status stays `provisional` because lecture 12 carries a TODO to quote TypeSafe's own documentation before delivery, and lectures 6–11 may change after their Colab T4 runs (the Lab 6, 7 and 8 model paths, the Lab 6 logits behind lecture 11's section 5, and the Lab 9 and 10 GPT-2 runs). Questions quote the pages verbatim, so the snapshot must be **frozen, rebuilt if any page changed, and set to `final` before anyone writes a question against it**: set `SOURCE_COMMIT` in the builder to the new commit, run it, and update `sha256`, `bytes`, `source_commit`, `characters` and `status` in `_variables.yml` (and the hash in `notebooks/13-rag.ipynb`, twice, in `notebooks/14-agents.ipynb`, once, and in the fixture). Questions store verbatim evidence quotes and character offsets are computed from them at load time, so a rebuild after questions exist could silently invalidate them. Three checks in `tests/test_rag_questions.py` prevent that:

1. every evidence quote must occur **exactly once** in its page of the current snapshot, so a rebuild that changes or duplicates the quoted text fails the test, naming each broken item;
2. the test fails if `data/rag_questions_v1.jsonl` exists while `datasets.lectures.status` is still `provisional`;
3. once registered, `datasets.rag_questions.corpus_sha256` must equal `datasets.lectures.sha256`, so a later rebuild cannot go unnoticed even if every quote happens to survive.

**Rules for `text`** (fixed in the builder):

- The YAML front matter is removed; the page title (and subtitle, if any) becomes the first line, `# <title>`.
- `{{< include /_includes/module-NN.md >}}` is replaced by the module's line (number, day, minutes, stack), its summary and its objectives from `_variables.yml`, as plain text. `{{< var ... >}}` is resolved from `_variables.yml` at the same commit.
- HTML comments (the figure specs) are deleted. Figure captions and alt text stay.
- Callout fence lines (`::: {.callout-...}` and `:::`) are deleted; a callout's `title`, if it has one, is kept as a line; its content stays.
- Headings, tables, code and LaTeX are kept as written (encoders handle LaTeX poorly; that is a property of the corpus worth seeing). Trailing spaces are removed and runs of blank lines collapsed to one.

Each record: `slug` (e.g. `01-text-as-data`; `references` for the reading list), `module` (1–12, or `null`), `title`, `text`, `source_commit`, `source_sha256` (of the `.qmd` bytes). Keys are sorted and the gzip header has `mtime=0` and no file name.

**Determinism, as checked (2026-10-05).** Two builds, one of them in a separate interpreter, gave byte-identical files; `--check` reported "identical to a fresh build". `tests/test_rag_questions.py` rebuilds from the source commit and compares the hash when that commit is in the clone's history (it skips on a shallow clone). As for the decision set, the compressed bytes depend on the zlib build, and `--check` says whether a difference is in the content or only in the compression.

**Loading.** Paste this cell after the loading cell of the [Loading contract](#loading-contract) (it uses `fetch`). Labs 14 and 15 instead restate Lab 13's retriever cell, which carries its own `load_corpus`:

```python
import gzip, json


def load_lectures():
    """Workshop Lectures v1 as [{"doc_id", "title", "text"}, ...]; doc_id is the page slug."""
    blob = fetch(
        "workshop_lectures_v1.jsonl.gz",
        [
            "https://raw.githubusercontent.com/project-delphi/nlp-llms/main/data/workshop_lectures_v1.jsonl.gz",
            "https://cdn.jsdelivr.net/gh/project-delphi/nlp-llms@main/data/workshop_lectures_v1.jsonl.gz",
        ],
        "4a7e9310ea5bd9206fce4a42c414d1f942c8880f9e7b694b04f04125fcc2d395",
    )
    pages = [json.loads(line) for line in gzip.decompress(blob).decode("utf-8").splitlines()]
    return [{"doc_id": p["slug"], "title": p["title"], "text": p["text"]} for p in pages]
```

## Workshop RAG Questions v1 (Modules 13 and 15)

**Not written yet.** `data/rag_questions_v1.jsonl` must be written and checked by people: Romeo and one instructor (proposed), about 4 hours for the author and 3 for the checker for 80 items (estimate). **No language model writes, proposes or labels any item.** Model-written questions copy the passage's wording, which inflates BM25 and surface-matching encoders, and a model labelling relevance is the kind of judge Lab 13 teaches people to check. Lab 15 reuses the `test` split in its fixed evaluation set. Specification: `briefs/13-rag.md`, decision (b). Instructions for the two people: [`rag_questions_TEMPLATE.md`](rag_questions_TEMPLATE.md).

| | |
|---|---|
| Size | 80 questions: `dev` 30 (choose settings), `test` 50 (report; reused by Lab 15) |
| Kinds, per split (±1 item) | `lookup` 40%, `specific` 30%, `multi` 15%, `unanswerable` 15% |
| Evidence | groups (all needed) of alternative spans (any one suffices); each span a verbatim quote of one or two sentences, at most 60 words, occurring exactly once in its page; `[]` for `unanswerable` |
| Correctness | `key_facts` on at least 70% of answerable items; every lecture 01–12 covered by at least four questions |
| License | CC BY 4.0 (proposed; Romeo's call) |
| Validator | `tests/test_rag_questions.py`, using [`rag_questions_tools.py`](rag_questions_tools.py) |
| Status | 0 of 80 written. Agreement before resolution: not measured |

**Until the file exists**, `tests/test_rag_questions.py` runs its schema, quote and overlap checks on `tests/fixtures/rag_questions_fixture.json`: six items **written by an AI agent for exercising code, not an evaluation set; no number from it is quoted anywhere.** Lab 13 never loads it: without the real file, the notebook says so and runs its evaluation code on *plumbing probes* (sentences copied from the snapshot, each its own evidence), labelled as measuring nothing about retrieval.

**When it lands:** commit the file, add `datasets.rag_questions` to `_variables.yml` (`name`, `modules: [13, 15]`, `file`, `urls`, `sha256`, `bytes`, `license`, `license_url`, `splits: {dev: 30, test: 50}`, `corpus_sha256` equal to the lectures hash), set the hash in the `DATASETS["rag_questions"]` entry of `notebooks/13-rag.ipynb`, and report here: items written, dropped, alternatives added, and the evidence agreement rate before resolution (`python data/rag_questions_tools.py agreement ...`).

## Lab 6 encoder logits (Modules 6 and 11)

The validation and test logits that Lab 11 calibrates: Lab 6's hand-off file, from one fine-tuning run with Lab 6's GPU settings. Lab 11 loads it by hash and checks that it is not the offline stand-in, that the shapes are (600, 4) and (1600, 4), and that its labels equal `load_topics()`'s; otherwise it falls back to the Lab 1 classifier.

| | |
|---|---|
| File | `lab06_logits.npz`, 55,278 bytes: `val_logits` (600, 4) and `test_logits` (1600, 4) as float32, the two label arrays as int64, and a JSON `meta` string |
| SHA-256 | `cdc898835bbf77f283a8b178f33ce516e416ba3ca80d9d33d0882d65072ede61` |
| Canonical URL | <https://raw.githubusercontent.com/project-delphi/nlp-llms/main/data/lab06_logits.npz> |
| Fallback URL | <https://cdn.jsdelivr.net/gh/project-delphi/nlp-llms@main/data/lab06_logits.npz> |
| Model | `distilbert/distilbert-base-uncased`, fine-tuned on all 4,800 training texts: `max_length` 256, 2 epochs, learning rate 5e-5, batch 32, seed 0 (Lab 6's GPU settings) |
| Measured | test accuracy 0.8975, macro-F1 0.8973; validation accuracy 0.9000 (`baselines.json`, `lab06.distilbert_finetune`). Training took 323 s |
| Built | 2026-10-06 on an Apple M1 Pro's GPU (MPS) in float32, from a copy of Lab 6 changed only to treat `mps` as a GPU (Lab 6 itself uses CUDA or the CPU, and fp16 only on CUDA); Python 3.12.13, `torch` 2.14.1, `transformers` 5.18.0. **Not a T4 run**: a T4 run at the same settings will give slightly different logits, and the file can be replaced by one |
| License | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/): numbers derived from our run on CC0 data (Romeo to confirm) |

On a CPU, Lab 6 fine-tunes BERT-mini (`google/bert_uncased_L-4_H-256_A-4`) on 800 texts instead; the build Mac's CPU run of that path reached 0.7975. Those logits are not the ones committed here.

## Lab 9 preference pairs (Modules 9 and 10)

Synthetic pairwise preferences with a known hidden rule, so that Lab 9 can check what its reward model learns and Lab 10 can measure reward hacking against a gold score. Specification: `briefs/09-preference-learning.md` (the Lab 9 → Lab 10 interface fixes every field). **The gold rule is a rule we wrote**; a real preference dataset has no gold score.

| | |
|---|---|
| Files | `lab09_prompts.json`, 49,071 bytes; `lab09_preferences.jsonl.gz`, 1,837,490 bytes; `lab09_reward_model.pt`, 4,105,277 bytes |
| SHA-256 | prompts `966971fd2f9f8533711db4cd2f8eaf36e839bd0fb9ed1c1c52efc970f6a25ed6`; preferences `c00b775865dabf491542719b223ab6fb539f3728ca0348c8ca50ecb51b752326`; reward model `61c13537e4d8d0fd82e20006033f256937da01746885816c2522d607e6fad183` |
| Canonical URLs | <https://raw.githubusercontent.com/project-delphi/nlp-llms/main/data/lab09_prompts.json>, <https://raw.githubusercontent.com/project-delphi/nlp-llms/main/data/lab09_preferences.jsonl.gz>, <https://raw.githubusercontent.com/project-delphi/nlp-llms/main/data/lab09_reward_model.pt> |
| Fallback URLs | <https://cdn.jsdelivr.net/gh/project-delphi/nlp-llms@main/data/lab09_prompts.json>, <https://cdn.jsdelivr.net/gh/project-delphi/nlp-llms@main/data/lab09_preferences.jsonl.gz>, <https://cdn.jsdelivr.net/gh/project-delphi/nlp-llms@main/data/lab09_reward_model.pt> |
| Builder | [`build_lab09_preferences.py`](build_lab09_preferences.py), which needs the Hub (seed 0; label seed 1). The reward model is written by Lab 9's own solution code at seed 0 (`NLP_LLMS_REWARD_MODEL_OUT`), so the class in the notebook and the committed weights cannot drift |
| Policy | `distilbert/distilgpt2` at commit `2290a62682d06624634c1f46a6ad5be0f47f38aa` (`models.causal_lm_revision`), float32, `eval()`, temperature 1, no truncation, end-of-text never sampled |
| Size | 256 prompts of 8 tokens (16 subjects × 16 frames: 160 train, 32 held-out, 64 for Lab 10's evaluation); 8,000 train and 1,000 held-out pairs, two 24-token responses each; the reward model has 12,820 embedding rows |
| Built | 2026-10-06, Apple M1 Pro CPU, Python 3.12.13, `torch` 2.14.1, `transformers` 5.18.0; sampling took 410 s. Not built on Colab |
| License | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (ours). The responses are samples from an Apache 2.0 model |

**The prompts.** Every frame stops just before a feeling ("My sister opened the letter and felt so"). The first proposal ended one word earlier ("... and said", "... and saw"). On GPT-2's own samples it failed two acceptance criteria: 76% of same-prompt pairs were tied, because the continuations rarely contained a list word. The gold rule, the lengths and the interface did not change. The measurements behind the choice are in the brief's "As built" note.

**Acceptance statistics of the committed pairs** (all 18,000 responses, 9,000 same-prompt pairs; criteria from the brief):

| Statistic | Value | Criterion |
|---|---|---|
| Share of responses with $g \ne 0$ | 0.3381 | report |
| Tie share of same-prompt pairs | 0.5019 | at most 0.6 |
| Cap active ($n_{\text{pos}} \ge 4$) | 0.0001 | under 1% |
| Crowding active ($f_{\text{list}} > 0.25$) | 0.0022 | under 1% |
| Some positive word twice or more | 0.0104 | report (expected under 5%) |
| Mean complete words per response | 17.37 | report (expected 14 to 18) |
| Mean gold score | 0.1144 | report |
| $\mathrm{Acc}^\star$ at $\tau_{\text{label}}$ = 0.25 / 0.5 / 1 | 0.7414 / 0.6984 / 0.6290 | at least 0.6 at the chosen 0.5 |

Every list word occurs in the samples (fewest: "ugly", 4 times; most: "good", 2,016). The full counts are stored in `lab09_prompts.json` under `statistics`.

**The reward model** (seed 0, the committed file) and the two other seeds that set Lab 9's Exercise 4 thresholds, measured on the 1,000 held-out pairs on the build Mac's CPU:

| Seed | Held-out pairwise accuracy | $\mathrm{Acc}^\star$ (held-out) | Gold-order accuracy, untied pairs | Spearman $r_\phi$ vs $g$ | Epoch kept (of 6) |
|---|---|---|---|---|---|
| 0 (committed) | 0.6890 | 0.7127 | 0.9381 | 0.7848 | 1 |
| 1 | 0.6980 | 0.7127 | 0.9343 | 0.7781 | 1 |
| 2 | 0.6710 | 0.7127 | 0.9250 | 0.7454 | 1 |

The held-out tie share is 0.467. Training took 14 to 15 s per seed and the whole notebook 90 to 98 s (CPU). Three seed-0 runs gave bit-identical weights on this machine. The committed checkpoint is 4,105,277 bytes, with 12,820 embedding rows, below the brief's 5 MB limit for trimming the vocabulary.

**Determinism.** Sampling on other hardware or library versions is not bit-for-bit reproducible, so the committed files and their hashes are the dataset, not the script. `tests/test_lab09.py` checks that both notebooks pin these hashes, that every stored gold score equals the restated gold rule, and that the reward model was trained on these two files.

**Loading.** Lab 9 loads the two data files through the loading cell above (its `LAB09_FILES`); Lab 10 loads all three (its `LAB09_FILES`, with the same hashes).

## Workshop Capstone Questions v1 (Module 15)

**Not written yet.** `data/capstone_questions_v1.jsonl` holds the capstone's 45 new questions; with Lab 13's 80, reused by ID, they make the capstone's 125. They are written and blind-checked by people, after Lab 13's set and against the same frozen snapshot. **No language model writes, proposes, filters or labels any item**, including checking whether a model "knows" a memory-bait fact. Specification: `briefs/15-capstone.md`, "Decision: the fixed evaluation set".

| | |
|---|---|
| Size | 45 new questions: `dev` 15 (9 unanswerable, 6 `reading`), `test` 30 (18 and 12) |
| Memory bait | at least 13 of the 27 unanswerable items: a fact stated in a reading-list paper (`outside_source`) that the snapshot never states (`absent_terms`, checked absent mechanically) |
| Correctness | `key_facts` on every `reading` item |
| Instructions | [`capstone_questions_TEMPLATE.md`](capstone_questions_TEMPLATE.md): schema, write, blind check, resolve (both agreement rates from `rag_questions_tools.py agreement`), validate, manifest |
| Validator | `tests/test_capstone_questions.py`; the manifest builder is [`build_capstone_eval.py`](build_capstone_eval.py) |
| Status | 0 of 45 written. Agreement before resolution: not measured |

**Until the file exists**, the validator runs its schema and absence checks on `tests/fixtures/capstone_questions_fixture.json`, written by an AI agent for exercising code, not an evaluation set; no number from it is quoted anywhere. Lab 15 scores *plumbing probes* without the real files and labels them so. The readiness item `capstone-questions` closes when the file exists; the release check (`scripts/release_check.py`) lists it until then.

## Module 0 sets: ubiquitin and purchases

Module 0 (coding agents in the terminal) builds two small apps, each in Python and in R; the reference solutions are in [`agents-intro/`](../agents-intro/). They do not run in Colab and do not use the loading cell above, but they follow the same rules: canonical URL first, our copy second, each download checked against its hash (the Python references; base R has no SHA-256, so the R references check counts instead). In a clone, the RFM references read the local copy first; the protein references try RCSB first on purpose, because fetching from the archive is part of the exercise.

### Ubiquitin, PDB 1UBQ (Module 0, protein app)

| | |
|---|---|
| File | `1ubq.pdb`, 78,570 bytes, 970 lines, PDB format |
| SHA-256 | `d4a6812d8951cf6594e6a0763f089e35f5a80b62acb3c117b2c5565228a7b161` |
| Canonical URL | <https://files.rcsb.org/download/1UBQ.pdb> |
| Fallback URL | <https://raw.githubusercontent.com/project-delphi/nlp-llms/main/data/1ubq.pdb> |
| Source of our copy | <https://raw.githubusercontent.com/Lemkul-Lab/gmx_tutorials_jpcb/6c232808330cf79448711f3dada28ec6a43775f9/inputs/01_ubiquitin/1UBQ.pdb>, the input file of Justin Lemkul's GROMACS tutorials, pinned to a commit, downloaded 2026-10-05 |
| Entry | "Structure of ubiquitin refined at 1.8 Angstroms resolution", Vijay-Kumar, Bugg & Cook (1987), *J. Mol. Biol.* 194:531 (JRNL record; DOI 10.1016/0022-2836(87)90679-6). X-ray, chain A, 76 residues `MQIFVKTLTG…LRLRGG` |
| Contents | 602 `ATOM` records (heavy atoms only: 378 C, 105 N, 118 O, 1 S; no hydrogens, no alternate locations), 58 `HETATM` waters, 660 atoms in all (the `MASTER` record agrees) |
| Revision | `REVDAT 6`, 14-FEB-24 (remark update). A search-engine extract of the wwPDB entry page (<https://www.wwpdb.org/pdb?id=1UBQ>, read 2026-10-05) gives 2024-02-14 as the latest revision |
| License | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/): "Data files contained in the PDB archive are available under the CC0 1.0 Universal (CC0 1.0) Public Domain Dedication" |
| License source | <https://www.wwpdb.org/about/usage-policies>. **Partly verified**: the sentence above is from a search-engine extract of that page on 2026-10-05; the page itself was blocked from the build container and has not been opened |

**Why a mirror, and why we trust it.** `files.rcsb.org`, `www.rcsb.org`, `data.rcsb.org`, PDBe and `www.wwpdb.org` are all blocked from the build container, so the file could not be taken from the archive. Five independent GitHub repositories carry byte-identical copies of the current revision (SHA-256 above): `Lemkul-Lab/gmx_tutorials_jpcb`, `N283T/zsasa`, `HFooladi/pdbrust`, `mnrozhkov/serverless-cookbook` and `Open-Athena/binder-lab`. Older copies in other repositories carry revision 4 (2009) or 5 (2011) and differ only in `REVDAT` and `REMARK` lines. **Not verified**: that `files.rcsb.org` serves exactly these bytes today. If it serves a newer revision, the hash check rejects it and the Python reference falls back to our copy, so the measured numbers in `agents-intro/MEASURED.md` stay reproducible; re-check from a networked machine before delivery and update `sha256` and `bytes` here and in `_variables.yml` if the archive's file changed.

### Workshop Purchases v1 (Module 0, RFM app)

| | |
|---|---|
| File | `purchases_v1.csv.gz`, 393,024 bytes |
| SHA-256 | `ac73eac9219793d280961ed0e60e1a1649ece3e8ab7c8624da4a4928629fa11b` |
| Canonical URL | <https://raw.githubusercontent.com/project-delphi/nlp-llms/main/data/purchases_v1.csv.gz> |
| Fallback URL | <https://cdn.jsdelivr.net/gh/project-delphi/nlp-llms@main/data/purchases_v1.csv.gz> |
| Builder | [`build_purchases.py`](build_purchases.py): standard library only, seed 0, no network; `--check` rebuilds in memory and compares the decompressed text |
| Columns | `invoice_id`, `invoice_date` (ISO date), `customer_id`, `country`, `stock_code`, `quantity`, `unit_price` |
| Size | 48,953 invoice lines, 18,432 invoices, 3,000 customers, 2024-01-01 to 2025-12-31. RFM is computed as of 2026-01-01 |
| License | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (ours). Synthetic: no real customer is in it |

**How it is drawn.** Each customer belongs to one of six hidden archetypes (regular, loyal, occasional, wholesale, lapsed, new) that fix ranges for the purchase rate, the chance and timing of churn, the basket size and the quantity; within an archetype each customer draws an individual rate (log-normal), a first purchase date and, if they churn, an exponential lifetime. Invoices are a Poisson process; each line picks a product from a fixed catalog of 200 products, which fixes the unit price. The archetype is not in the file, and the archetypes overlap, so k-means finds groups but does not recover them exactly. There are no returns or cancellations (the UCI set has both).

**Determinism, as checked (2026-10-05).** Two builds gave byte-identical files, and `--check` reported matching content and matching compressed bytes. Every random stream is seeded with a string (`random.Random(f"0:customer:{id}")`), which Python hashes with SHA-512, so the streams do not depend on the platform. `tests/test_agents_intro.py` rebuilds in memory and compares the decompressed text with the committed file, not the gzip bytes (those depend on the zlib build).

**Loading.** In a notebook or script, after the loading cell of the [Loading contract](#loading-contract):

```python
def load_purchases():
    """Workshop Purchases v1. Returns a list of dicts, one per invoice line."""
    blob = fetch(
        "purchases_v1.csv.gz",
        [
            "https://raw.githubusercontent.com/project-delphi/nlp-llms/main/data/purchases_v1.csv.gz",
            "https://cdn.jsdelivr.net/gh/project-delphi/nlp-llms@main/data/purchases_v1.csv.gz",
        ],
        "ac73eac9219793d280961ed0e60e1a1649ece3e8ab7c8624da4a4928629fa11b",
    )
    return list(csv.DictReader(io.StringIO(gzip.decompress(blob).decode("ascii"))))
```

### UCI Online Retail II (optional alternative, not committed)

The real-data alternative for the RFM app is the [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) set: the transactions of a UK-based online retailer between 2009-12-01 and 2011-12-09. It is not downloaded or committed, and the apps do not use it. What could be checked on 2026-10-05 (`archive.ics.uci.edu` is blocked from the build container, so only search-engine extracts of the dataset page were read):

| Claim | Status |
|---|---|
| License CC BY 4.0 | **Unverified**: search-engine extract of the UCI page, page not opened |
| 1,067,371 instances (invoice lines) | **Unverified**: same source |
| One file, `online_retail_II.xlsx`, 43.5 MB | **Unverified**: same source |
| Date range 2009-12-01 to 2011-12-09 | **Unverified**: same source |

If a group uses it, expect to handle cancellations (invoice numbers starting with `C`, negative quantities) and lines without a customer ID before computing RFM; neither property was checked for this file. Read the license on the UCI page before using it in a published app.

## Measured baselines (`baselines.json`)

[`baselines.json`](baselines.json) records the numbers that later labs and lectures compare against. Cite this file, not a number remembered from a notebook. Every value in it was measured by executing the named notebook; nothing is estimated.

Each entry of `baselines` has these fields:

| Field | Meaning |
|---|---|
| `id` | Stable name, `labNN.model[.variant]`, for example `lab01.char_ngram.n5` |
| `notebook` | The notebook that produced the value |
| `dataset` | A key of `datasets` in `_variables.yml` (`lm` or `topics`) |
| `split` | The split the metrics were measured on. Always `test` for a baseline |
| `metrics` | Metric name to value |
| `val_metrics` | The same on `val`, where the notebook used it to choose a setting |
| `settings` | Everything needed to recompute the value: model, tokenizer, vocabulary rule, grids, chosen hyperparameters, seed |
| `date` | The day the notebook was executed |
| `comparable_with_later_labs` | Present and `false` only for numbers that must not be compared (the word-level perplexities) |
| `embedding_checks` | Lab 2 only: the neighbor hit rate and analogy accuracy of the embeddings behind the entry, on the notebook's hand-written lists. Not test-split metrics |

The top-level `environment` gives, per lab, the machine and library versions of that run.

**Lab 1, measured 2026-10-04** (Apple M1 Pro, CPU, scikit-learn 1.9.1, NumPy 2.5.3; not run on Colab):

| `id` | Test metric | Value |
|---|---|---|
| `lab01.char_ngram.n2` (k = 0.1) | nats per character / perplexity / bits per character | 2.4922 / 12.0882 / 3.5955 |
| `lab01.char_ngram.n3` (k = 0.1) | same | 2.0859 / 8.0518 / 3.0093 |
| `lab01.char_ngram.n5` (k = 0.01), the best n ≤ 5 on `val` | same | 1.8326 / 6.2503 / 2.6439 |
| `lab01.tfidf_logreg` (C = 10) | accuracy / macro-F1 | 0.8838 / 0.8841 |
| `lab01.counts_naive_bayes` | accuracy / macro-F1 | 0.8844 / 0.8840 |

The character n-gram protocol, which Labs 3 and 5 must follow to be comparable: counts from `train` only; vocabulary of 65 characters; add-k smoothing with k chosen on `val` from {0.001, 0.01, 0.1, 0.5, 1.0}; every character of the evaluated split is scored, and the up to n − 1 characters of left context come from the text immediately before the split (the splits are contiguous). The n-gram values are exact counts and reproduce to the last digit. The classifier values come from an iterative solver and may differ in the fourth decimal on another platform or library version.

The logistic-regression accuracy here (0.8838) is lower than the 0.892 in the table under [arXiv Topics v1](#arxiv-topics-v1-classification). The tokenizer and `C` of that earlier run are not recorded. Lab 1 uses its own tokenizer and chooses `C` on `val` (600 papers), which picks `C = 10`; with `C = 1` the same pipeline scores 0.8875 on `test`. Lab 1 reports the value the protocol gives, not the best one seen on `test`.

**Lab 2, measured 2026-10-04** (Linux container, CPU only, 2 PyTorch threads on a shared machine, PyTorch 2.14.1, scikit-learn 1.9.1; not run on Colab):

| `id` | Test accuracy / macro-F1 | Neighbor hit rate | Analogy accuracy |
|---|---|---|---|
| `lab02.sgns_avg_ffn`: SGNS trained on the training text, averaged, frozen, feed-forward net | 0.8669 / 0.8670 | 0.72 (13 of 18) | 0.20 (4 of 20) |
| `lab02.glove_avg_ffn` (stretch): pretrained GloVe 6B 100d, same classifier | 0.8050 / 0.8050 | 0.61 | 0.58 (11 of 19) |

Lab 2 recomputes `lab01.tfidf_logreg` in the notebook and reproduces it exactly (0.8838 / 0.8841). Averaged embeddings are 27 papers (1.7 points) behind it. Three further CPU runs of the same procedure with other random streams gave 0.8638 to 0.8719 test accuracy, so the gap is not seed noise. On a GPU the SGNS numbers will differ slightly, because the random streams and the order of floating-point sums differ.

**Lab 3, measured 2026-10-05** (Linux container, 4 vCPU, CPU only, 2 PyTorch threads on a machine shared with other training jobs, PyTorch 2.14.1; full settings, seed 0; not run on Colab):

| `id` | Steps | Val nats/char | Test nats/char / perplexity / bits per character |
|---|---|---|---|
| `lab03.rnn_lm`: tanh RNN, the participant's `rnn_cell_step` in a Python loop, d = 64, d_h = 256 | 1,000 | 1.6275 | 1.6959 / 5.4515 / 2.4467 |
| `lab03.lstm_lm`: `nn.LSTM`, same sizes, forget-gate bias 1 | 2,000 | 1.5316 | 1.6114 / 5.0098 / 2.3247 |

Both follow the character n-gram protocol above. Lab 3 recomputes `lab01.char_ngram.n2`, `n3` and `n5` with a vectorized restatement of Lab 1's add-k model and asserts that they match to 1e-4 with the same k. The neural models score every test character once, carrying the hidden state across 1,000-character chunks after a warm-up on the 1,000 characters before the split. Both beat the best n-gram (1.8326). With `QUICK = True` (250 and 500 steps), seeds 1 and 2 gave 1.8845 and 1.8864 (RNN) and 1.7741 and 1.7758 (LSTM). The neural numbers depend on the hardware and the number of steps; a T4 run will differ slightly. Lab 5 compares against `lab03.lstm_lm` with the same metric. To update: execute the notebook with `QUICK = False` and copy the values from the `lab03_results.json` its results-card cell writes.

**Lab 5, measured 2026-10-05** (Linux container, 4 vCPU, CPU only, 2 PyTorch threads on a machine shared with other training jobs, PyTorch 2.14.1; the GPU (non-QUICK) settings executed on a CPU, seed 0; not run on Colab or a GPU):

| `id` | Parameters | Steps (batch 64) | Val nats/char | Test nats/char / perplexity / bits per character |
|---|---|---|---|---|
| `lab05.lstm`: Lab 3's LSTM architecture, retrained in Lab 5 with Lab 5's loop (AdamW, warm-up and cosine decay, peak lr 0.008) | 350,593 | 3,000 | 1.4308 | 1.5727 / 4.8198 / 2.2690 |
| `lab05.mini_gpt`: pre-norm decoder-only transformer, d = 128, 4 heads, 4 layers, T_max = 128, peak lr 0.005 | 824,897 | 3,000 | 1.4004 | 1.5481 / 4.7023 / 2.2334 |

Both score the same 60,394 test characters as Labs 1 and 3, with Lab 3's metric (`lm_loss_and_ppl`, restated unchanged). The LSTM uses Lab 3's evaluation routine (1,000-character warm-up, state carried across 1,000-character chunks). The GPT cannot carry state, so it is scored with overlapping windows of 128 characters moved by 64, and only the second half of each window is scored: every character has 65 to 128 characters of context. The learning rates were chosen on `val` with `QUICK = True` and not re-tuned at batch 64. The GPT's lead over the LSTM (0.025 nats) is from one seed and is smaller than the seed-to-seed spread measured with `QUICK = True` (about 0.02 to 0.03 nats), so it is not a ranking. With `QUICK = True` (batch 16, 1,500 steps), seeds 0, 1 and 2 gave 1.7696, 1.7907 and 1.7938 (GPT) and 1.5621, 1.5819 and 1.5798 (LSTM). Training took 3,921 s (GPT) and 348 s (LSTM) on this contended CPU; a T4 time has not been measured. `lab05.lstm` is not Lab 3's run: it beats `lab03.lstm_lm` (1.6114) because of the longer schedule, the decaying learning rate and the higher peak rate. The measured attention weights of this run, for the Lecture 5 figure, are in `images/05-attention-heads.json`. To update: execute the notebook with `NLP_LLMS_QUICK=0` and run the values of the `lab05_results.json` its card cell writes into the two entries.

**Lab 9, measured 2026-10-06** (Apple M1 Pro, CPU only, Python 3.12.13, PyTorch 2.14.1; executed with `scripts/test_notebooks.py`; not run on Colab or a GPU). The reward model of Exercise 4 on the 1,000 held-out pairs of [Lab 9 preference pairs](#lab-9-preference-pairs-modules-9-and-10), recorded as split `test`; full settings, seeds 0 to 2:

| `id` | Held-out pairwise accuracy | $\mathrm{Acc}^\star$ | Gold-order accuracy, untied pairs | Spearman $r_\phi$ vs $g$ |
|---|---|---|---|---|
| `lab09.reward_model.seed0` (committed as `lab09_reward_model.pt`) | 0.6890 | 0.7127 | 0.9381 | 0.7848 |
| `lab09.reward_model.seed1` | 0.6980 | 0.7127 | 0.9343 | 0.7781 |
| `lab09.reward_model.seed2` | 0.6710 | 0.7127 | 0.9250 | 0.7454 |

Lab 9's Exercise 4 thresholds come from these three rows. To update: rebuild only if the data change, then rerun the three seeds as in the docstring of `build_lab09_preferences.py`.

**Lab 11, measured 2026-10-05** (Linux container, 4 vCPU shared with another agent, CPU only, NumPy 2.5.3, SciPy 1.18.1, scikit-learn 1.9.1; not run on Colab). Calibration of the Lab 1 pipeline, recomputed in Lab 11 at three values of `C`, on the 1,600 test papers; ECE with 15 equal-width right-closed bins; $\tau^*$ fitted on the 600 validation papers. The last row, added 2026-10-06 (Apple M1 Pro, CPU, `scripts/test_notebooks.py`), is the Lab 6 encoder from `lab06_logits.npz`, the lab's primary classifier since its logits were committed; its after-scaling values are printed to three decimals.

| `id` | Accuracy | Mean confidence | ECE | Brier | Log loss | $\tau^*$ | ECE after | Brier after | Log loss after |
|---|---|---|---|---|---|---|---|---|---|
| `lab11.tfidf_logreg.C1` | 0.8875 | 0.7234 | 0.1641 | 0.2111 | 0.4356 | 0.4685 | 0.0094 | 0.1637 | 0.3033 |
| `lab11.tfidf_logreg.C10` | 0.8838 | 0.8617 | 0.0231 | 0.1662 | 0.3142 | 0.7895 | 0.0180 | 0.1645 | 0.3065 |
| `lab11.tfidf_logreg.C100` | 0.8819 | 0.9245 | 0.0435 | 0.1755 | 0.3387 | 1.3017 | 0.0173 | 0.1710 | 0.3187 |
| `lab11.lab06_encoder` | 0.8975 | 0.9349 | 0.0392 | 0.1580 | 0.2991 | 1.097 | 0.036 | 0.155 | 0.295 |

For `C = 10`: noise floor of ECE at this $N$ 0.0193; ECE 0.0220, 0.0231, 0.0379 and 0.0548 at 5, 15, 50 and 100 bins; with $\ell_{\text{wrong}} = 10$, $\ell_{\text{defer}} = 1$ the test cost per case is 1.1625 acting on everything, 1.0 deferring everything, 0.5469 at Chow's $\lambda^* = 0.9$ (coverage 0.578), 0.5175 at the $\lambda$ chosen on validation (0.8422, coverage 0.676) and 0.5131 at Chow's rule after temperature scaling. These values are deterministic and match the build-container values in `briefs/11-calibration.md`. Lab 11's language-model numbers are not recorded: in this container only its offline test double ran, and its numbers measure the notebook, not a model.

**Lab 12, measured 2026-10-05** (Linux container, 4 vCPU shared with other agents, CPU only, PyTorch 2.14.1, `typesafe-sdk` 0.7.2; the no-key path; not run on Colab). **These are numbers of our toy model, not of Jev**, on template items only (status `v1-template-only`). No Jev call has been made. Toy decision model (31 features, two linear heads), trained on `train` for 3,000 full-batch Adam steps, seed 0, with each reward:

| `id` | Train ECE (acc / mean p̂) | Test accuracy | Test mean p̂ | Test ECE (floor) | Test binary Brier | Test ECE after temperature on `dev` |
|---|---|---|---|---|---|---|
| `lab12.toy.accuracy_reward` | 0.1334 (0.8645 / 0.9932) | 0.6767 | 0.9852 | 0.3085 (0.0137) | 0.3058 | 0.1549 |
| `lab12.toy.brier_reward` | 0.0267 (0.8735 / 0.8757) | 0.6867 | 0.8911 | 0.2581 (0.0346) | 0.2807 | 0.1776 |

`lab12.local_thresholds`: the local toy decider (not Jev) with the lecture's costs (wrong 20, ask 0.5, miss 4, escalate 3) chooses $(\tau_{\text{esc}}, \tau_{\text{act}}) = (0, 1.0)$ on `dev` (ask about almost everything) and costs 1.800 per `test` case, against 4.302 at the analytic pair $(0.375, 0.969)$ and 1.753 for asking about everything. Seeds 0 to 4 (a scratch restatement of the notebook's code, recorded in `settings.seed_check`) gave test ECE 0.308 to 0.310 (accuracy reward) and 0.254 to 0.259 (Brier reward). To update: execute the notebook and copy the values of the `lab12_results.json` its card cell writes.

To update the file: execute the notebook, take the `lab01_baselines.json` or `lab02_baselines.json` its card cell writes, and copy the values into the matching entries. `tests/test_baselines.py` checks the fields, that the Lab 1 checkpoint asserts the recorded n-gram values, and that Lab 2 asserts the recorded Lab 1 TF-IDF values and that its recorded accuracy clears the notebook's floor.

## What was run

On 2026-10-04, on macOS 26.6 (arm64), Python 3.12.13, on a home connection. None of this was run on Colab.

| Check | Result |
|---|---|
| `load_lm_corpus()` as written, empty directory | Loaded from the upstream URL in 0.28 s. 1,000,000 / 55,000 / 60,394 characters |
| `load_lm_corpus()` with the upstream URL broken, fallback as written | **Failed**: our repository URL returns 404 (see above) |
| Same, with our copy served over a local HTTP server in place of the repository URL | Loaded in 0.14 s, same split |
| `load_topics()` as written, empty directory | **Failed**: both URLs return 404 (see above) |
| `load_topics()` with the file served over a local HTTP server in place of the canonical URL | Loaded in 0.09 s. 4,800 / 600 / 1,600 rows, balanced, 7,000 distinct texts |
| Same, canonical URL failing and the server in place of the fallback URL | Loaded in 0.12 s, same split |
| Either loader with a local copy (`NLP_LLMS_DATA`) | LM corpus under 0.01 s, topics 0.07 s, no request made |
| Second call after a download (cached under `data/`) | LM corpus under 0.01 s, topics 0.08 s |
| `tests/test_data.py` | Runs the cell above with the network disabled and checks sizes, balance, overlap and hashes |

The local HTTP server stands in for GitHub: it exercises the download, hash check and cache code, not the real URLs. The real repository URLs must be re-tested once the repository is public and this directory is on `main`.
