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
| Pairwise preferences | 9, 10 | Synthetic, with a known hidden preference | Generated in the notebook | No file. Lab 9 fixes the generator and its seed |
| Labeled decisions | 11, 12, 14 | Built for the workshop | Ours | Built on build Day 8 |
| RAG documents | 13, 15 | The workshop's own lecture pages and reading list | Ours | Built on build Day 9 |

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

The baseline leaves 173 test errors (we did not run the same baseline on AG News, so no comparison with it is claimed). Most of them are between `cs.CV` and `cs.LG` (47 Vision papers predicted as Machine learning) and between `cs.LG` and `cs.CL`. The averaged-embedding model (Lab 2) and the fine-tuned encoder (Lab 6) have **not** been run on it yet; whether they fit the 10-minute T4 budget is checked when those labs are built. With 4,800 training texts of about 183 words, we expect them to, but that is an expectation, not a measurement.

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

- **Dates (Module 4)** and **pairwise preferences (Modules 9 and 10)** are generated inside the notebook. Each lab fixes its generator's seed and records the sizes here when it is written.
- **Labeled decisions (Modules 11, 12, 14)** and **RAG documents (Modules 13, 15)** are ours, built on build Days 8 and 9. Module 11 also reuses the arXiv Topics test split for its reliability diagrams.

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
