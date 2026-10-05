# Lab brief: `notebooks/13-rag.ipynb`

From the Academic Director to the **Agentic Systems Engineer**, who owns Lab 13 (`AGENTS.md`, "Who owns what"). Lecture: `lectures/13-rag.qmd` (same symbols and equation names: `budget`, `dense`, `topk`, `contrastive`, `bm25-recall`, `rrf`, `cross`, `expected-level`, `recall`, `mrr`, `faith`). Lab standards: `PLAN.md` section 5. Verified Jev API surface: `briefs/jev-verification.md` (cited below as JV §n). Provider wrapper: `notebooks/08-llm-apis.ipynb` and `briefs/08-llm-apis.md`. This file is not rendered by Quarto.

**Objectives exercised** (from `_variables.yml`, `m13`): build a RAG pipeline; choose chunking, embedding and reranking settings from measurements; evaluate retrieval and answer quality separately.

**Every time, size and cost below is an estimate or a target, not a measurement,** unless marked *checked*. *Checked* means I ran it on 2026-10-05 in the build container (CPU, Python 3.12, a scratch environment with `llama-index-core` 0.14.25, `llama-index-retrievers-bm25` 0.8.0, `langchain-core` 1.6.6, `langchain-text-splitters` 1.1.3, `typesafe-sdk` 0.7.2, scikit-learn 1.9.1, all installed from PyPI), not on Colab, with no Hub access, no API keys and no network access to the LlamaIndex or LangChain documentation. No neural encoder, cross-encoder or language model was run.

## The lab in one paragraph

Participants index a committed snapshot of the workshop's own lecture pages with LlamaIndex, then measure retrieval against a human-labelled question set: they write recall@k and MRR, build the index, and sweep chunk size and k, choosing a setting on `dev` at a fixed context budget and reporting it on `test`. They rebuild the same retriever as a LangChain runnable and check that the two frameworks return the same chunks, which fails until LlamaIndex's default of embedding metadata is switched off. They add a reranking step over 20 candidates: Jev through a postprocessor written in the notebook on `typesafe-sdk` when a `TYPESAFE_API_KEY` is set, otherwise a cross-encoder, and measure what it changes. Finally they generate cited answers through Lab 8's `PROVIDER` switch, score faithfulness with a judge, check correctness against recorded answer forms, and fill the lecture's two-by-two table that separates retrieval failures from generation failures. The stretch adds BM25 and reciprocal rank fusion. Every checkpoint tests the participant's function on hand-made inputs or fakes, so it gives the same verdict on every path. No retrieval or answer metric of any model is asserted.

## Design decisions (resolved here; for Romeo to confirm)

### (a) The document set: Workshop Lectures v1

**Recommendation: a committed, versioned plain-text snapshot of lecture pages 01–12 plus `references.qmd`.** One corpus serves Labs 13, 14 and 15. Reasons:

1. **It is ours.** The pages are CC BY 4.0 workshop content (`LICENSE`). No third-party license, no Hub, no download from anywhere but our repository. The pages quote short passages of third-party material (for example TypeSafe's MIT-licensed `SKILL.md`), as quotations with sources; that does not change the corpus license.
2. **It is the capstone's subject.** Module 15 is "a research-assistant agent over the workshop's own reading list". The lecture pages *are* the annotated reading list: every page ends with readings and says why each matters, and `references.qmd` collects them. A participant can ask the capstone "which paper introduced in-batch negatives, and which lab uses the idea?" and the answer is in this corpus.
3. **It contains facts no model knows.** Measured numbers (Lab 2's 0.8669 test accuracy), our notation ($\kappa$ for Jev's `confidence`), our design choices (Lab 12's thresholds 0.375 and 0.969). Questions about these separate grounded answers from parametric memory, which questions about general ML do not.
4. **It is the right size.** About 495,000 characters (*checked*, current pages 01–12); 1,400 / 660 / 320 chunks at 128 / 256 / 512 tokens with overlap $L/8$ (*checked*, `SentenceSplitter`). Small enough to embed three times on a CPU, large enough that retrieval is not trivial.
5. **It contains a natural injection test.** Lecture 8, section 8 quotes a prompt-injection string. A question about prompt injection retrieves it (lecture 13, section 7), and Lab 14 builds on that.

**Not recommended for v1: arXiv abstracts.** The `data/arxiv_topics_v1.csv.gz` abstracts (CC0) are 7,000 papers from January–June 2024 and contain none of the reading-list papers (*checked*: no title match for "Attention Is All You Need", "Dense Passage", "Lost in the Middle", "RAGAS"). They would serve only as distractors. Abstracts of the reading-list papers themselves would need the arXiv API, which the build container cannot reach (*checked*: no response from `export.arxiv.org`), and would invite questions a frontier model answers from memory. **Proposal for Module 15's brief, not Lab 13:** an optional "reading-list abstracts" file built on a networked machine with the pattern of `data/build_arxiv_topics.py`, and an optional distractor experiment with the 1,600 arXiv Topics test abstracts.

**The file.**

| File | Contents |
|---|---|
| `data/build_lectures_corpus.py` | Standard library plus PyYAML, no network. Reads `lectures/01-*.qmd` … `lectures/12-*.qmd` and `references.qmd` at the current commit and writes the snapshot deterministically (sorted keys, gzip `mtime=0`), so the hash is reproducible |
| `data/workshop_lectures_v1.jsonl.gz` | One JSON object per page: `slug` (e.g. `01-text-as-data`; `references` for the reading list), `module` (1–12, or `null`), `title`, `text`, `source_commit`, `source_sha256` (of the `.qmd`). About 500 KB uncompressed, an estimated 150 KB compressed |

Rules for `text`, fixed in the builder and documented in `data/README.md`:

- Resolve `{{< var ... >}}` from `_variables.yml` (titles would otherwise vanish). Replace `{{< include /_includes/module-NN.md >}}` with the module's summary and objectives from `_variables.yml`, as plain text.
- Delete HTML comments (the figure specs) and the YAML front matter. Keep figure captions and alt text.
- Delete callout fence lines (`:::` and `::: {.callout-...}`) but keep their content and titles.
- Keep headings (`## 4. Dense retrieval`), tables and LaTeX as written. Encoders handle LaTeX poorly; that is a property of the corpus worth seeing, not something to clean away.
- `tests/test_data.py` gains the snapshot's hash and size; `_variables.yml` gains a `datasets.lectures` entry (proposed below).

**The snapshot is frozen.** Lectures will keep changing; the snapshot does not, so the question set's evidence quotes stay valid. A `v2` is built only with a re-check of every quote (the validator below fails otherwise). Lecture 13 itself is not in v1: it was not written when the questions were, and a corpus that explains RAG to a RAG lab adds nothing.

### (b) The question set: Workshop RAG Questions v1

**The decision-set rule applies: people write and label every item; no language model writes, proposes or labels any.** Reasons specific to retrieval:

1. **Model-written questions copy the passage.** A model asked for questions about a passage reuses its words. That inflates BM25 and any encoder that matches surface form, and it would bias exactly the dense-against-BM25 comparison the stretch makes.
2. **The labels are the yardstick.** Recall@k is only as good as the evidence labels. A model that labels relevance is the kind of judge lecture section 9 asks participants to check, not to trust; using one to build the reference would be circular.
3. **Lab 15 reuses the `test` split as part of its fixed evaluation set.** Its numbers would inherit any bias.
4. **License.** Items written by us are ours; nothing needs a provenance check.

**What an agent may do:** write the builder, the schema validator and the test below; write a development fixture of at most 10 items under `tests/fixtures/`, headed "agent-written fixture for exercising code; not an evaluation set; no number from it is quoted anywhere". The fixture must never be loaded by the notebook as evaluation data.

**Who:** Romeo and one instructor, the same pair as the decision set (proposed). **Estimated effort:** about 4 hours for the author and 3 for the checker for 80 items (estimate; about 3 and 2 minutes per item). A first batch of 40 (15 `dev`, 25 `test`) unblocks the notebook build; the remaining 40 must land before delivery.

**Size and splits:** 80 questions, `dev` 30 and `test` 50. **Why:** Lab 15 needs a held-out `test`; settings are chosen on `dev`. At $N = 50$ and recall near 0.8 the standard error is about 0.06, so differences are read pairwise (gained and lost counts), which the lab prints. More items would cost the annotators time and the keyed path money without changing that reading rule.

**Kinds** (in each split, within ±1 item):

| Kind | Share | What it is | Example of the shape (not an item) |
|---|---|---|---|
| `lookup` | 40% | a fact stated in one place, asked in different words | "Why do dot-product scores need scaling as the key dimension grows?" |
| `specific` | 30% | a fact that only exists in this corpus: a measured number, our notation, a lab's design choice | "What test accuracy did the averaged-embedding classifier reach on arXiv Topics?" |
| `multi` | 15% | needs two pieces of evidence, usually from two modules | "Which earlier loss does the contrastive retrieval loss generalize, and in which lab was that loss implemented?" |
| `unanswerable` | 15% | plausible, on a topic the corpus covers, but the fact is not in it | "What learning rate did Lab 7 use for LoRA?" (only true if the lecture does not state it) |

At least 70% of answerable items must have `key_facts` (below) so correctness can be checked by a program. Cover every lecture 01–12 with at least four questions across both splits.

**File:** `data/rag_questions_v1.jsonl`, uncompressed so diffs are readable. One object per line:

```json
{"id": "q001", "split": "dev", "kind": "lookup",
 "question": "...",
 "evidence": [[{"slug": "05-transformer-from-scratch", "quote": "..."},
               {"slug": "...", "quote": "..."}]],
 "answer": "a one- or two-sentence reference answer",
 "key_facts": ["0.8669"],
 "author": "RA", "checker": "IN", "notes": ""}
```

- `evidence` is a list of **groups** (all needed); each group is a list of **alternative** spans (any one suffices); `[]` for `unanswerable`. Lecture section 8 defines recall over exactly this structure.
- `quote` is copied verbatim from the snapshot's `text`, one or two sentences, at most 60 words. Character offsets are computed at load time by exact search, so a quote must occur **exactly once** in its page.
- `key_facts`: case-insensitive strings, any one of which a correct answer must contain (alternatives such as `["κ", "kappa"]`). Empty when the answer is not short.
- `author` and `checker` are initials.

**Labelling protocol.**

1. **Write.** The author reads a page, writes the question **in words different from the passage**, then records the evidence quote(s), the reference answer and the key facts. Unanswerable items: the author searches the snapshot for the fact and records where it *would* have been.
2. **Check, blind.** The checker sees only the question. With any search except a language model (reading, `grep`, the lab's own BM25), they record the evidence they find and their answer, or "not in the corpus".
3. **Resolve.** Evidence agrees when the checker's quote overlaps one of the author's spans. If the checker found a valid different passage, add it as an alternative in the same group. If they disagree on the answer, rewrite or drop the item. An `unanswerable` item survives only if the checker also could not find it. No item stays while disputed.
4. **Report** in `data/README.md`: items written, dropped, alternatives added, and the evidence agreement rate before resolution.

**Validator** (`tests/test_rag_questions.py`, an agent may write it): the schema; split sizes and kind shares; every quote found exactly once in its page of the snapshot; **lexical overlap flag**: report every question that shares a run of four or more consecutive non-stopword tokens with one of its quotes (the authors rewrite those). The test runs on the fixture until the real file lands.

**License: CC BY 4.0**, the license of the pages the quotes come from (we own both, so CC0 would also be clean; Romeo's call, as for the decision set).

### (c) Models, and the path the notebook takes without the Hub

Proposed `_variables.yml` keys (below). None was downloaded or run in the build container (the Hub is blocked); licenses and limits come from model cards read through web search on 2026-10-05, **not from the Hub**.

| Role | Key | Proposed ID | Why | To check on the Hub |
|---|---|---|---|---|
| Encoder | `models.embedding` | `BAAI/bge-small-en-v1.5` | 33M parameters, 384-d, 512-token input (reported); MIT (reported). The 512-token limit covers the largest chunk in the sweep; 256-token encoders would truncate it | revision; license; whether its sentence-transformers config defines a query prompt (`encode_query` applies it if so) |
| Reranker | `models.reranker` | `cross-encoder/ms-marco-MiniLM-L6-v2` | 6 layers, fast on CPU; Apache 2.0 (reported) | the exact ID (`...-L6-v2` and `...-L-6-v2` both appear in search results); revision; whether MS MARCO's terms matter for teaching use |
| Faithfulness judge (open) | `models.nli` | `cross-encoder/nli-deberta-v3-xsmall` | small NLI cross-encoder; Apache 2.0 (reported); label order contradiction, entailment, neutral (reported; read it from the model config, do not hard-code) | revision; label mapping in `config.json` |
| Generator (open) | `models.fallback` | `Qwen/Qwen2.5-0.5B-Instruct` (existing) | Lab 8's open model, through Lab 8's wrapper | as Lab 8 |

Embeddings are **always** computed locally, on every path. No corpus text is sent to an embedding API: it would add a key, a cost and a third-party data flow for no teaching gain.

**Four paths, chosen once in the setup cell and printed:**

| Path | When | Encoder | Reranker | Generator | Judge | Label on every result |
|---|---|---|---|---|---|---|
| **Keyed** | an OpenAI or Anthropic key | bge-small | Jev if `TYPESAFE_API_KEY`, else cross-encoder | `PROVIDER` (Lab 8) | the NLI judge, plus an LLM judge from the *other* provider if both keys are set; with one key, the same provider, labelled "self-graded" | provider and model ID |
| **Open** | no keys, Hub reachable (Colab) | bge-small | cross-encoder | Qwen, Lab 8's `LocalProvider` | NLI judge | `open: Qwen2.5-0.5B-Instruct` |
| **Offline (CI)** | no Hub, or `NLP_LLMS_STUB=1` | **LSA stand-in**: TF-IDF fitted on the chunks, then `TruncatedSVD(256)`, L2-normalized | **lexical stand-in**: BM25 over the $k_0$ candidates only | Lab 8's `StubProvider`, extended (below) | **stub judge**: a sentence is "supported" if at least 80% of its content words appear in the sources | `stand-in (not a neural model)` / `stub (test double)` |
| **Jev** | `TYPESAFE_API_KEY`, on top of either of the first two | – | `JevRerank` | – | – | `Jev (<resp.model>)` |

Rules for the offline path, as in Labs 8 and 11:

- A banner above every result table: "Offline stand-ins: these numbers measure the notebook's code, not any model. Do not quote them."
- LSA (Deerwester et al., 1990) is a real, if old, dense retriever, which is why it is the stand-in: the code path through `BaseEmbedding` and `Embeddings` is identical. It is never called "embeddings" in a printed label.
- `StubProvider.rag_answer(question, sources)`: abstains (the exact abstention sentence) if no source shares a content word of five or more letters with the question; otherwise returns the first sentence of source [1] followed by ` [1]`. It must not read the gold evidence or the answer. Rule stated in the notebook.
- Never mix: a table either has all neural rows or is under the banner.

### (d) Versions to pin, and how to verify them

| Package | Pin | Verified how (2026-10-05) |
|---|---|---|
| `llama-index-core` | `0.14.25` | installed from PyPI; source read; the checks in "Checked in the build container" run against it |
| `llama-index-retrievers-bm25` | `0.8.0` (requires `bm25s`, `PyStemmer`) | installed; source read; scores checked against Lab 1's formula |
| `langchain-core` | `1.6.6` | installed; source read; `InMemoryVectorStore`, `Embeddings`, `BaseRetriever`, `RunnableLambda` run |
| `langchain-text-splitters` | `1.1.3` | installed; used only for the units comparison |
| `typesafe-sdk` | `0.7.2` (`packages.typesafe_sdk`) | installed; `Score`, `AsyncTypeSafeClient.system_one`, `SystemOneResponse.from_http_response` run offline |
| `sentence-transformers` | `6.1.0` (latest on PyPI) | wheel source read only: `CrossEncoder(model, max_length=..., revision=...)`, `.predict(pairs, batch_size=...)`, `.rank(query, documents, top_k=...)`, `SentenceTransformer.encode_query` / `encode_document`. **Not run** (needs the Hub). Check what Colab preinstalls; pin only if it differs or breaks |
| `langchain-community` | **not used** | its last release is 2026-05-22; nothing in the core path needs it |
| `llama-index-embeddings-huggingface`, `llama-index-llms-*`, `llama-index-postprocessor-sbert-rerank` | **not used** | the lab writes 10-line adapters instead (below); fewer pins, nothing hidden |

**The engineer verifies at build time:** the documentation sites `docs.llamaindex.ai` and `docs.langchain.com` are blocked from the build container, so read the **installed source** of each pinned package for every class used, and record the file and line in the notebook's setup comment. Re-check the pins against PyPI on the build day: these packages release weekly.

**Install with pip (as Colab does), not with uv's default hard-link mode, when testing locally.** *Checked:* under `uv pip install` with hard links, `SentenceSplitter` failed: NLTK 3.10.3 refused to open the stopword file bundled in `llama-index-core` because it had two hard links (`PermissionError: ... refusing multiply-linked file`). `--link-mode=copy` fixed it. CI must install the same way.

### (e) Faithfulness: how, with what judge, and its limits

- **Claims are sentences.** `split_claims(answer)` (provided): strip the citations `[n]`, split on sentence ends, drop empty strings; the abstention sentence yields no claims. Lecture section 9 states this simplification.
- **Judge interface:** `judge(sentence, sources) -> float`, the probability that the sources support the sentence.
  - **NLI judge** (open path, and always computed when the Hub is reachable): for each source, `CrossEncoder.predict([(source_text, sentence)])` with softmax over the three labels; $J$ is the largest entailment probability over the $k$ sources. Read the label index from the model config. Pairs longer than the model's limit are truncated by the library; chunks of 256 tokens plus a sentence fit in 512.
  - **LLM judge** (keyed): Lab 8's `extract` with `class Verdict(BaseModel): supported: Literal["yes", "no"]`, one call per sentence, a pinned prompt that says what "supported" means ("every factual statement in the sentence follows from the sources; background knowledge does not count"). $J$ = 1 or 0. Print the agreement between the NLI and LLM judges on the same sentences.
  - Jev as the judge ("is this answer supported by the sources?", a `Noul`) belongs to **Lab 15**, not here: it keeps Lab 13 inside 50 minutes.
- **Answer relevance** (provided, printed, keyed path only): one judged call per answer with a three-level verdict (answers / partly / no). On the open and offline paths print "not computed".
- **Correctness** (provided, every path): an answer is correct if it contains one of the item's `key_facts` (case-insensitive, after Unicode normalization); items without `key_facts` are "not auto-checked" and excluded from the denominator, which is printed.
- **Limits, printed in the notebook beside the faithfulness table:** sentence is not claim; the NLI model was trained on short everyday sentences and sees LaTeX and notation here; an LLM judge may favor its own provider's output; faithfulness is not correctness; **the judge has not been compared with human labels** until the audit set below exists.
- **Judge audit set (proposed, for Romeo):** `data/rag_judge_audit_v1.jsonl`. An instructor runs the keyed path once on `dev`, exports every (answer sentence, sources) pair (about 100, estimate), and the same two people label each sentence supported / not supported, blind to the judges. The notebook then prints each judge's agreement and Cohen's kappa with the human labels. Until it exists, a banner: "Faithfulness judge not validated against human labels."

## The retriever interface Labs 14 and 15 reuse

**Adopted from Lab 14's request** (`briefs/14-agents.md`, constraint (c)): `build_retriever(docs)` over **any** list of `{"doc_id", "title", "text"}` dicts, `retrieve(query, k)` with `k` chosen at call time, and `Passage` carrying at least `doc_id`, `text` and `score`, plus a BM25 path that needs no download so CI can run it. Lab 13 adds fields and keyword arguments with defaults, so a call written to Lab 14's minimal form works unchanged. Lab 13's own corpus goes through the same function: `load_corpus()` returns the snapshot as that list of dicts (`doc_id` is the page slug, `title` the module title).

Colab notebooks share no runtime, so Labs 14 and 15 **restate** this cell verbatim and rebuild the index (seconds for a small corpus; under a minute for the lecture snapshot with a neural encoder; estimates), as they restate Lab 12's `LocalDecider`. Mark the cell "provided; reused by Labs 14 and 15". Lab 15's settings come from Lab 13's recorded run (`data/baselines.json` `lab13.chosen`, proposed), restated as constants.

```python
@dataclass(frozen=True)
class Passage:
    doc_id: str          # the document's ID, e.g. "05-transformer-from-scratch" or Lab 14's "faq-poisoned"
    text: str            # the chunk's text
    score: float         # the score of the last stage that ranked it (cosine, BM25, RRF, cross-encoder or Jev level)
    title: str = ""      # the document's title
    chunk_id: str = ""   # deterministic: f"{doc_id}:{i:04d}" (SentenceSplitter id_func)
    section: str = ""    # the nearest preceding "## " heading, "" if none
    start: int = 0       # character offsets of the chunk in the document's text
    end: int = 0

def load_corpus(path: str | None = None) -> list[dict]: ...
    # Workshop Lectures v1 as [{"doc_id", "title", "text"}, ...]; tries NLP_LLMS_DATA,
    # then the URLs in DATASETS["lectures"]; checks sha256

def build_retriever(docs: list[dict], *,
                    method: str = "auto",          # "dense" | "bm25" | "hybrid" | "auto"
                    chunk_size: int = 256,         # tokens; replace with lab13.chosen after the recorded run
                    chunk_overlap: int = 32,
                    rerank: str | None = None,     # None | "cross" | "jev"
                    candidates: int = 20) -> "Retriever": ...
    # "auto": "dense" with bge-small if the encoder loads, else "bm25" (prints which, and why)
    # "bm25": llama-index-retrievers-bm25 only; no model download, no network: the CI and stub path
    # "hybrid": RRF of dense and BM25 (lecture eq-rrf), r0 = 60
    # Documents shorter than chunk_size stay one chunk each, so a small corpus of short pages
    # (Lab 14's desk corpus) is retrieved document by document.

class Retriever:
    describe: dict   # method, encoder, reranker, chunk_size, overlap, candidates, n_docs, n_chunks
    def retrieve(self, query: str, k: int = 3) -> list[Passage]: ...
    async def aretrieve(self, query: str, k: int = 3) -> list[Passage]: ...   # needed for rerank="jev" under Jupyter's loop
    def as_langchain(self, k: int = 3) -> langchain_core.runnables.Runnable:   # str -> list[Document]
        ...                                                                    # metadata carries every Passage field

def format_sources(passages: list[Passage]) -> str: ...
    # "[1] (05-transformer-from-scratch › 4. Dense retrieval)\n<text>\n\n[2] ..."
ABSTAIN = "I cannot answer from the provided sources."
```

Rules that hold on every method:

- **Metadata never enters the embedding or the BM25 index** (Flag 3); `title` is returned, not scored. Lab 14's poisoned document is therefore retrieved, or not, on its text alone, which is what its test needs.
- **Deterministic**: same docs, same method, same query → same list, same scores. No randomness, no UUIDs.
- **`k` larger than the number of chunks** returns all chunks, sorted (`BM25Retriever` warns and caps; *checked* in its source).
- **`rerank="jev"`** requires a key and is only reachable through `aretrieve`; with no key it raises at build time with a message. `rerank="cross"` needs the Hub; with `method="auto"` and no Hub, `build_retriever` drops the reranker and says so.
- **No dependency on the stretch.** The cell carries its own copy of the solution `rrf` for `method="hybrid"`, so Labs 14 and 15 restate the core-path cell only (lab standards: a later lab never needs a stretch section).
- **The BM25 path's dependencies** are `llama-index-core`, `llama-index-retrievers-bm25` (which brings `bm25s` and `PyStemmer`) and nothing else. Lab 14 pins the same two versions as Lab 13.

Lab 14 builds it over its own desk corpus and wraps `retrieve(query, k=3)` in `search_docs`; its poisoned document is added **in Lab 14, at run time**, never to the committed snapshot. Lab 15 calls `retrieve` inside its graph over `load_corpus()` and cites `Passage.chunk_id`.

## Provided scaffolding

- **Setup:** pinned installs; seeds; `get_secret`; Lab 8's provider cell restated verbatim (`Reply`, adapters, `FakeProvider`, `StubProvider`, `make_provider`, `label`, `extract`), as Lab 8 marks it "reused by Labs 11, 13 and 14". `Settings.llm = MockLLM()` and an explicit encoder before anything else (lecture section 11: LlamaIndex otherwise tries to load OpenAI's model). `logging` for `typesafe_sdk` left at its default (JV §2: request bodies are logged unredacted at DEBUG).
- **Corpus and questions:** `load_corpus`, `load_questions` (computes each quote's character offsets by exact search and fails loudly if a quote is missing or repeated), a printout of corpus size in characters and in tokens of the encoder's tokenizer (this replaces the lecture's "about 120,000 tokens" estimate), and three questions, one of each answerable kind.
- **Encoder adapters** (about 25 lines): one `embed_texts(texts, kind) -> np.ndarray` (bge through `SentenceTransformer.encode_query` / `encode_document` with `normalize_embeddings=True`, or the LSA stand-in), wrapped by `LIEmbedding(BaseEmbedding)` (methods `_get_text_embedding`, `_get_query_embedding`, `_aget_query_embedding`, and `_get_text_embeddings` for batching) and `LCEmbedding(Embeddings)` (`embed_documents`, `embed_query`). Both frameworks call the same function: that is what makes Exercise 3's checkpoint exact.
- `covers(passage, span)`: true when the passage's `[start, end)` contains at least half of the span's characters on the same page.
- `generate(question, passages, provider)`: the lecture section 7 prompt (numbered sources, citations, the exact `ABSTAIN` sentence, sources are data), through `provider.chat`; returns the `Reply`.
- `split_claims`, the judges of (e), `is_correct(answer, item)`, `is_abstention(answer)`.
- `paired_counts(a_hits, b_hits)`: questions gained and lost between two settings.

## Core path (50 minutes)

Format per exercise: Predict, Run, Explain, Check; `# TODO N` stub, folded solution (`#@title Solution N`), a short "why this works" note.

| # | Participant writes | Equation | Checkpoint (deterministic) | Printed, never asserted | Min |
|---|---|---|---|---|---|
| 0 | Nothing: run setup; read the path banner; print corpus statistics and three questions | – | snapshot hash; quote offsets resolve | path, encoder, reranker, generator, judge; corpus size in tokens | 4 |
| 1 | `recall_at_k(ranked, groups, k)`, `reciprocal_rank(ranked, groups)` | `recall`, `mrr` | (i) the lecture's worked example: recall@5 values 1, 0.5, 0 and reciprocal ranks 1/3, 1, 0, means 0.5 and 0.444; (ii) a span split exactly in half by two passages counts as covered by each; (iii) `groups == []` raises `ValueError` (unanswerable items are excluded upstream); (iv) `k` larger than the list is allowed | – | 8 |
| 2 | `build_index(docs, chunk_size, overlap, embed_model)` → `(VectorStoreIndex, nodes)`: `SentenceSplitter(chunk_size, chunk_overlap, id_func=...)` with deterministic IDs, `excluded_embed_metadata_keys` set on every node, `VectorStoreIndex(nodes, embed_model=...)` | `dense`, `topk`, `budget` | (i) node IDs equal on two calls; (ii) no node embeds its metadata (`node.get_content(metadata_mode=MetadataMode.EMBED) == node.get_content()`); (iii) every node's text equals `doc.text[start:end]` (*checked* that LlamaIndex 0.14.25 sets these offsets); (iv) a query made of one node's own first sentence retrieves that node at rank 1 | Provided sweep: $L \in \{128, 256, 512\}$, $L_o = L/8$, $k = 1..10$ on `dev`; plot recall@k and MRR against $n_{\text{ctx}} = kL$; table of recall at $n_{\text{ctx}} \le 1{,}024$ tokens per $L$; the participant picks $(L, k)$ on `dev`; then the pick and the runner-up on `test`, with $N$ and `paired_counts` | 9 |
| 3 | `make_lc_retriever(nodes, embed_fn, k)`: `InMemoryVectorStore(embedding=LCEmbedding(...))`, documents added with `ids=[n.node_id ...]` and the passage fields in metadata, returned as `.as_retriever(search_kwargs={"k": k})`; plus `lc_chain = retriever | RunnableLambda(to_passages)` | `dense`, `topk` | (i) for every `dev` question the LangChain and LlamaIndex top-$k$ ID lists are equal, except where adjacent scores tie within 1e-6; (ii) `lc_chain.invoke(q)` and `lc_chain.batch([q1, q2])` return lists of `Passage` | Provided cell **before** the TODO: the same comparison with LlamaIndex's default metadata embedding, printing on how many questions the two disagree (*checked* on six test queries with the LSA stand-in: 3 of 6 agree with metadata embedded, 6 of 6 with it excluded, identical scores) | 8 |
| 4 | `relevance_question(query)` → `typesafe_sdk.Score` with the four-level rubric of lecture section 6, in increasing order, and `instructions` that contain the full question (question names are not sent to the model, JV §2); `rerank_by_scores(nodes, scores, top_n)` → `list[NodeWithScore]` sorted by score, ties kept in retrieval order, scores replaced | `cross`, `expected-level` | (i) the `Score` validates, has 4 criteria, contains the query; (ii) hand case: scores `[0.1, 2.6, 0.4, 2.6, 0.0]`, `top_n=3` gives node order `[1, 3, 2]`; (iii) `JevRerank` with a fake async client returning real `SystemOneResponse` objects (built with `from_http_response` and an `x-typesafe-request-id` header, JV §8) reranks as expected; (iv) one simulated `TypeSafeError` → the first `top_n` in retrieval order, with a printed message (fail open). (iii)–(iv) *checked* in the build container against `typesafe-sdk` 0.7.2 and `llama-index-core` 0.14.25 | Retrieve $k_0 = 20$ with the chosen $(L, k)$, rerank to $k$: recall@k and MRR before and after on `dev` and `test`, `paired_counts`, median and 95th-percentile reranking latency per query; with a key, `resp.model`, total Jev input tokens and measured USD; with both a key and the Hub, Jev and the cross-encoder side by side on the same candidates | 9 |
| 5 | `faithfulness(answer, sources, judge, threshold=0.5)` → `(score or None, verdicts)` | `faith` | (i) fake judge returning `[0.9, 0.2, 0.7]` for three sentences gives 2/3 and verdicts `[True, False, True]`; (ii) citations `[1]`, `[2, 3]` are removed before the judge sees the sentence; (iii) the `ABSTAIN` sentence gives `(None, [])`; (iv) `threshold` is respected at exactly 0.5 (counts as supported) | Provided: generate for every `test` question (CPU open path: a fixed 12-item subset, as Lab 8) with the chosen retriever and reranker; per kind: answered, correct (of auto-checkable), faithfulness, abstention on `unanswerable`, false abstention on answerable; the lecture's two-by-two table (evidence retrieved × correct); the judge banner; cost | 10 |
| – | Nothing: read and answer the closing cell | – | none | – | 2 |

Minutes: 4 + 8 + 9 + 8 + 9 + 10 + 2 = 50.

**Where the slow cells go.** The three-size sweep embeds about 2,400 chunks (estimate from the *checked* counts); start it at the top of Exercise 2 with the solution `build_index`, so it runs while participants write their own. On the keyed Jev path, `JevRerank` makes $k_0 = 20$ calls per question: 1,600 for `dev` and `test`. Run `test` only if time is short, and say so.

**`JevRerank` itself is provided**, about 35 lines (JV §10). The design was *checked* offline in the build container: a `BaseNodePostprocessor` subclass with the client in a Pydantic `PrivateAttr`, `_apostprocess_nodes` scoring all candidates with `asyncio.gather` under `asyncio.Semaphore(8)`, sorting by `answer.score` (the expected level, lecture @eq-expected-level), storing `answer.confidence` in node metadata as `jev_confidence` without using it, and returning the first `top_n` in retrieval order on any `TypeSafeError`. **Call it with `await reranker.apostprocess_nodes(nodes, query_bundle=QueryBundle(q))`**: the synchronous `postprocess_nodes` would call `asyncio.run` inside Jupyter's running event loop and fail. State: `{"question": q, "passage": node.get_content()}`. Default `RetryPolicy`. Never construct `AsyncTypeSafeClient` without a key: it raises (*checked*).

**Closing cell: "What this lab showed and what it did not"** (markdown, then two questions):

- Which setting you chose, on `dev`, at what context budget, and what it scored on `test`, with $N$.
- Whether reranking changed recall@k and MRR by more than the pairwise noise; on the keyed path, what Jev cost. Nothing here says anything about Jev beyond these 50 questions on this corpus.
- If you ran offline, every number is the stand-ins' and measures the notebook's code.
- Faithfulness was scored by a judge that has (or has not) been compared with human labels; say which.
- Questions: (1) In your two-by-two table, which cell held most of the failures, and which component would you change first in Lab 15? (2) Lab 14 lets an agent call this retriever and then send an email. What could a passage like lecture 8's quoted injection string do there, and what would stop it?

## What is asserted on each path

| Path | Asserted | Printed, never asserted |
|---|---|---|
| **Keyed** | every unit checkpoint; harness: every question yields a record; every retrieved list has length `min(k, M)`; Jev probabilities per answer sum to 1 within 0.03 (rounded to 0.01, JV §2) | recall@k, MRR, reranking gains, faithfulness, relevance, correctness, abstention, latency, tokens, USD. **No retrieval or answer metric is asserted, for any model** |
| **Open** | the same | the same, labelled with the encoder, reranker, generator and judge IDs |
| **Offline (CI)** | the same unit and harness checks; after the engineer's recorded run, the stand-ins' deterministic recall@k and MRR at the chosen setting against `data/baselines.json` `lab13.offline` to 1e-6, as Labs 1, 2 and 11 do with their fallbacks | the same tables under the offline banner |

## Stretch (one section, last, optional; not required by any later lab)

**Hybrid retrieval with BM25.**

1. Provided: `BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=k0)` over the same nodes, metadata excluded. Its scores equal lecture @eq-bm25-recall divided by $k_1 + 1$, with $k_1 = 1.5$, $b = 0.75$, English stopwords removed and stemming on by default. *Checked* on a five-document corpus with `skip_stemming=True` and bm25s's English stopword list: maximum absolute difference $2.8 \times 10^{-8}$ against Lab 1's formula divided by 2.5. The provided checkpoint restates Lab 1's solution `bm25_scores` and repeats that check on three chunks of the snapshot.
2. Participant writes `rrf(rankings, r0=60)`: `rankings` is a list of ranked ID lists; returns IDs sorted by @eq-rrf, missing IDs contributing nothing, ties broken by first appearance. Checkpoint: the lecture's worked example (the chunk ranked 3rd and 3rd beats the one ranked 1st and 10th); one input list returns that list unchanged.
3. Printed: dense, BM25 and hybrid recall@k and MRR on `test` at the chosen $k$, **by question kind**. Predict before running: on which kind should BM25 help most? (Our expectation is `specific`, with its numbers and identifiers; not asserted.)
4. Do not use LlamaIndex's `QueryFusionRetriever` for this: it resolves `Settings.llm` at construction even with `num_queries=1`, and without a configured model it tries to import OpenAI's and raises (*checked*: `ImportError: llama-index-llms-openai package not found`). With `llm=MockLLM()` it runs. Mention it as the library route after participants have written `rrf`.

## Compute and cost budget

| Part | Offline (CPU, CI) | Open (Colab) | Keyed |
|---|---|---|---|
| Installs; corpus and questions | under 1 min (estimate) | under 1 min; plus model downloads, about 200 MB for bge-small, the cross-encoder and the NLI model (estimate) | the same |
| Exercise 2 sweep: about 2,400 chunks embedded | seconds (*checked*: LSA split plus index under 2 s per size) | T4: under 0.5 min; CPU: 1–3 min (estimates) | the same |
| Exercise 3: second index | seconds | under 0.5 min | the same |
| Exercise 4: rerank 80 questions × 20 candidates | seconds | cross-encoder, 1,600 pairs: T4 under 0.5 min, CPU about 1 min (estimates) | Jev, 1,600 calls at 8 concurrent: 1–3 min (estimate; latency unmeasured, third parties report 70–500 ms) |
| Exercise 5: 50 answers and their judging | seconds | Qwen on T4: about 3 min; CPU 12-item subset: about 3 min (estimates); NLI judging under 0.5 min | about 1 min for answers plus about 150 judge calls (estimate) |
| **Core path total** | **under 2 min** | **under 8 min** on T4 (estimate) | **under 8 min** (estimate) |

**Cost note** (state in the notebook; estimates from token arithmetic, **not measured invoices**; replace with figures measured from `usage` on a real run):

- **Generator:** 50 questions × about 1,500 input tokens (five 256-token chunks plus prompt) and 150 output tokens ≈ 75k in, 7.5k out. **Judge (LLM, keyed):** about 3 sentences per answer, 150 calls × about 1,500 input tokens ≈ 225k in. Anthropic `models.anthropic` at \$1 / \$5 per million (Anthropic's pricing page, read 2026-10-05 for Lab 8): about \$0.30 + \$0.05: state **"under 50 cents"**. OpenAI `models.openai` at \$0.10 / \$0.50 per million (secondary sources, as Lab 8): state **"under 5 cents"**.
- **Jev reranking:** 1,600 calls × about 400 input tokens (question, a 256-token passage, the rubric) ≈ 0.64M tokens; at \$0.042 per million input tokens and output "currently free" (`WorkflowEvals` price table and the SDK schema, JV §2, §5; TypeSafe's pricing page unread): about **\$0.03: state "under 5 cents"**.
- **Open and offline paths:** free.
- A room of 30 on one TypeSafe key is about 48,000 calls in a few minutes. Rate limits are unknown (JV §9); run Jev reranking on `test` only in a live session, and ask TypeSafe about workshop keys (PLAN section 6, "Jev access").

## Flags for the Lab Engineer

1. **Names.** Use the lecture's: `recall_at_k`, `reciprocal_rank`, `chunk_size` ($L$), `overlap` ($L_o$), `k`, `k0` (candidates), `n_ctx`, `groups`, `judge`. Do not call the reranked list "confidence"-sorted.
2. **Never sort by `confidence`.** `JevRerank` sorts by `score`; `confidence` is stored only.
3. **Metadata out of the embedding and out of BM25.** Set `excluded_embed_metadata_keys` to every metadata key on every node; `BM25Retriever` also indexes `MetadataMode.EMBED` text (*checked* in its source).
4. **Deterministic node IDs** through `SentenceSplitter(id_func=lambda i, doc: f"{doc.id_}:{i:04d}")`; the default is a random UUID (*checked*, `default_id_func`).
5. **`Settings.llm = MockLLM()`** in setup. Nothing in this lab may let LlamaIndex choose a language model.
6. **Fit on `dev`, report on `test`**, for $(L, k)$ and for the reranker decision. The `test` numbers are printed once per configuration, not used to choose.
7. **Unanswerable items** are excluded from recall and MRR (no gold groups) and used only for abstention.
8. **The CI path never touches the network** beyond the repository's own data URLs: no Hub, no `api.typesafe.ai`, no provider APIs.
9. **Package names.** Install only `typesafe-sdk==0.7.2` for Jev. The unaffiliated LlamaIndex reranker package on PyPI (JV §4, §6) is not mentioned in the notebook; `tests/test_package_names.py` fails on it in an install line.
10. **No personal data in Jev state**: the corpus has none; keep it that way if participants add documents.
11. **Report back:** measured corpus size in tokens; chunk counts per size; the sweep table on `dev` and `test` for each path you ran; Exercise 3's agreement count with and without metadata on the real encoder; reranking gains with $N$ and gained/lost; Jev `resp.model`, tokens, cost and latency if a key was available; Exercise 5's table; run time per section on CPU and on a T4; anything in the lecture the notebook contradicts (in particular the 495,000-character and 120,000-token figures, and the units and metadata claims of lecture section 3).

## Proposed changes (not made; for Romeo or the Architect)

- **`_variables.yml` `models`:**
  ```yaml
  # Lab 13 (and 14, 15): local encoder, reranker and faithfulness judge. Licenses and
  # limits from model cards via web search, 2026-10-05; Hub unreachable, revisions not pinned.
  embedding: "BAAI/bge-small-en-v1.5"
  reranker: "cross-encoder/ms-marco-MiniLM-L6-v2"   # confirm exact ID on the Hub
  nli: "cross-encoder/nli-deberta-v3-xsmall"
  ```
- **`_variables.yml` `packages`:** `llama_index_core: "0.14.25"`, `llama_index_retrievers_bm25: "0.8.0"`, `langchain_core: "1.6.6"`, `langchain_text_splitters: "1.1.3"`, `sentence_transformers: "6.1.0"` (only if Colab's preinstalled version differs), with `checked: "2026-10-05"`. Lab 14 should share the `langchain_core` pin (`langchain-typesafe` 0.0.1a3 needs `>=1.6.2,<2`, JV §3; 1.6.6 satisfies it).
- **`_variables.yml` `datasets`:** `lectures` ("Workshop Lectures v1", modules `[13, 14, 15]`, `file: workshop_lectures_v1.jsonl.gz`, `license: "CC BY 4.0"`, `sha256`, `bytes`, `source_commit`) and `rag_questions` ("Workshop RAG Questions v1", modules `[13, 15]`, `file: rag_questions_v1.jsonl`, `license: "CC BY 4.0"`, `splits: {dev: 30, test: 50}`); later `rag_judge_audit`.
- **`_variables.yml` `modules.m13.stack`:** add "Hugging Face" (sentence-transformers is the open path), giving `[LlamaIndex, LangChain, Hugging Face, Jev]`.
- **`data/`:** `build_lectures_corpus.py` and the snapshot (an agent may build both now); `rag_questions_v1.jsonl` (people, decision (b)); `README.md` sections for both, replacing the "Built on build Day 9" row; `baselines.json` `lab13.chosen` and `lab13.offline` after the recorded run.
- **`tests/`:** `test_rag_questions.py` (validator above, on the fixture until the real file lands); snapshot hash and size in `test_data.py`.
- **`PLAN.md` section 4, Module 13:** Stack → "LlamaIndex, LangChain, sentence-transformers (open encoder, cross-encoder and NLI judge), Jev, OpenAI/Claude (fallback: Qwen through Lab 8's wrapper)". Readings → "Lewis et al. 2020 (RAG); Karpukhin et al. 2020 (DPR); Liu et al. 2024 (Lost in the Middle); Es et al. 2024 (RAGAs); Thakur et al. 2021 (BEIR); the installed source of the pinned LlamaIndex and LangChain packages". Lab → add "on a committed snapshot of the workshop's lecture pages (`data/workshop_lectures_v1.jsonl.gz`) with a human-written question set (`data/rag_questions_v1.jsonl`)".
- **`PLAN.md` section 6, new row.** Item: "RAG question set needs two human authors". Risk: "an agent can build the corpus snapshot and the validator but must not write or label questions; without them Lab 13 has no recall numbers and Lab 15 no fixed evaluation set". Mitigation: "Romeo and one instructor write and check 80 questions (estimated 4 and 3 hours), 40 first to unblock the build; the faithfulness judge audit is a further 1–2 hours each after one keyed run".
- **`PLAN.md` section 6, new row.** Item: "Lab 13 open models need the Hub". Risk: "bge-small, the cross-encoder and the NLI model cannot be downloaded in the build container; their paths will be written, not run". Mitigation: "offline LSA and lexical stand-ins exercise the code under a banner; run the open path on a Colab T4 and pin revisions".
- **`PLAN.md` section 7, Day 9:** tick "Draft lecture 13: Retrieval-augmented generation", with the note "(not rendered; lab brief in `briefs/13-rag.md`)"; under "Code `13-rag.ipynb`" add "(blocked on `data/rag_questions_v1.jsonl`; build against the fixture meanwhile)".
- **`references.qmd`:** under Module 13, the five readings and the "also cited" list of the lecture.
- **Lecture 1's lab** says "Module 13 uses BM25 again through a library": true as designed (the stretch uses `llama-index-retrievers-bm25` and checks it against Lab 1's function). No change needed.

## Checked in the build container (2026-10-05)

Scratch environment as listed at the top; scripts not committed.

- **LlamaIndex interfaces:** `BaseNodePostprocessor` has one abstract method, `_postprocess_nodes`, and an async `apostprocess_nodes`; `BaseEmbedding` needs `_get_text_embedding`, `_get_query_embedding`, `_aget_query_embedding`; `SentenceSplitter(chunk_size=1024, chunk_overlap=200, ..., id_func=...)` defaults; `TextNode.start_char_idx` / `end_char_idx` index the document text exactly; `default_id_func` returns `uuid4`; the default QA prompt text quoted in lecture section 11.
- **LangChain interfaces:** `Embeddings` needs `embed_documents`, `embed_query`; `BaseRetriever` needs `_get_relevant_documents`; `InMemoryVectorStore.add_documents(docs, ids=...)`, `.as_retriever(search_kwargs={"k": k})`, `.similarity_search_with_score`; an LCEL chain of a retriever and `RunnableLambda` runs.
- **Corpus:** lecture pages 01–12, 494,904 characters after removing front matter, shortcodes and HTML comments; 1,400 / 659 / 324 nodes at $L$ = 128 / 256 / 512 with $L_o = L/8$; median 59 / 129 / 274 words per chunk.
- **Same retriever in two frameworks:** LSA stand-in encoder (TF-IDF, `min_df=2`, sublinear tf, SVD 256), six test queries, top 5: LlamaIndex and LangChain agree on 3 of 6 with LlamaIndex's default metadata embedding and 6 of 6 with metadata excluded, with scores equal to six decimals.
- **BM25 library against Lab 1's formula:** as in the stretch.
- **`QueryFusionRetriever`** raises without a configured model, runs with `llm=MockLLM()`.
- **`JevRerank`** design against `typesafe-sdk` 0.7.2 with a fake async client: reranks, keeps ties in retrieval order, fails open on `TypeSafeError`; `AsyncTypeSafeClient()` without a key raises.
- **NLTK hard-link failure** under uv's default link mode, fixed by `--link-mode=copy`.

## Not verified by the Director

- **Nothing in this lab has been built or run.** No notebook, corpus snapshot or question set exists. The question set needs two people.
- **No neural model was run.** bge-small, the cross-encoder, the NLI model and Qwen were not downloaded (Hub blocked). Their IDs, licenses, input limits and label orders come from model cards seen through web search, not from the Hub. `sentence-transformers` 6.1.0 was read, not run.
- **Documentation sites unread:** `docs.llamaindex.ai` and `docs.langchain.com` (blocked); everything framework-related rests on installed source.
- **No live Jev call**, as in Module 12: Jev's relevance judgments, latency, rate limits and the current model behind `jev-latest` are unknown. Whether Jev helps or hurts reranking on this corpus is an open question the lab answers on the keyed path.
- **All times and costs are estimates.** The 120,000-token corpus figure assumes four characters per token.
- **Whether Colab preinstalls `sentence-transformers`**, and which version, is unverified.
- **Citations:** Lewis et al. 2020, Liu et al. 2024, Es et al. 2024 and Cormack et al. 2009 checked by web search (authors, venue), not against the papers; Karpukhin et al. 2020, Thakur et al. 2021, Nogueira and Cho 2019, Greshake et al. 2023 and Deerwester et al. 1990 cited from memory.
- `quarto render` was not run: Quarto is not installed in the build container.
