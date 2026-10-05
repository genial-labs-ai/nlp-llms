# Writing Workshop RAG Questions v1: instructions for the two people

For the author (proposed: Romeo) and the checker (proposed: one instructor). Specification: `briefs/13-rag.md`, decision (b). Estimated effort, not measured: about 4 hours for the author and 3 for the checker, for 80 items (about 3 and 2 minutes per item). A first batch of 40 (15 `dev`, 25 `test`) unblocks the Lab 13 build; the remaining 40 must land before delivery. Lab 15 reuses all 80 by ID.

**Hard rule: no language model writes, proposes, rewrites or labels any item.** Not for a draft, not for "inspiration", not to check your evidence. The checker may search with anything except a language model: reading, `grep`, `python data/rag_questions_tools.py find`, or Lab 13's own BM25.

## Before you start

1. **The snapshot must be final.** `datasets.lectures.status` in `_variables.yml` must not be `provisional`. Today it is: `references.qmd` is still being completed. Once it is, the snapshot is rebuilt (`data/README.md`, "Status: provisional") and only then do you write questions. A question written against the provisional snapshot may break when it is rebuilt, and the validator will fail on it.
2. Read the pages from the snapshot, not from the site: the snapshot is what the retriever sees (shortcodes resolved, figure specs removed, LaTeX as written).

   ```bash
   python -c "import gzip,json; [print(json.loads(l)['text']) for l in gzip.open('data/workshop_lectures_v1.jsonl.gz','rt') if json.loads(l)['slug']=='05-transformer-from-scratch']" | less
   ```

## The file

`data/rag_questions_v1.jsonl`, uncompressed, one object per line, IDs `q001` to `q080`:

```json
{"id": "q001", "split": "dev", "kind": "lookup", "question": "...", "evidence": [[{"slug": "05-transformer-from-scratch", "quote": "..."}]], "answer": "one or two sentences", "key_facts": ["..."], "author": "RA", "checker": "IN", "notes": ""}
```

| Field | Rule |
|---|---|
| `split` | `dev` 30, `test` 50 |
| `kind` | per split, within one item: `lookup` 40% (a fact stated once, asked in other words), `specific` 30% (a fact only this corpus has: a measured number, our notation, a lab's design choice), `multi` 15% (needs two pieces of evidence, usually from two modules), `unanswerable` 15% (plausible, on a topic the corpus covers, but the fact is not in it) |
| `question` | **in words different from the passage**. The validator flags any question that shares four or more consecutive non-stopword words with one of its quotes; rewrite those |
| `evidence` | a list of **groups**, all needed; each group a list of **alternative** spans, any one sufficient. `[]` for `unanswerable`. A `multi` item has at least two groups |
| `quote` | copied **verbatim** from the snapshot's text (copy from the snapshot, not the rendered site), one or two sentences, at most 60 words, and it must occur **exactly once** in its page: lengthen it if `find` reports more than one occurrence |
| `answer` | a one- or two-sentence reference answer |
| `key_facts` | case-insensitive strings, any one of which a correct answer must contain (alternatives such as `["κ", "kappa"]`); `[]` when the answer is not short. At least 70% of answerable items need them |
| `would_be_in` | optional, `unanswerable` items only: the slug(s) of the page(s) where the fact would have been |
| `author`, `checker` | initials, two different people |

Coverage: every lecture 01–12 must appear in the evidence (or `would_be_in`) of at least four questions across both splits.

## Protocol

1. **Write (author).** Read a page; write the question in your own words; then record the evidence quote(s), the answer and the key facts. For an `unanswerable` item, search the snapshot for the fact and record where it would have been. Check quotes as you go:

   ```bash
   python data/rag_questions_tools.py find "So the unscaled score has standard deviation"
   ```

2. **Check, blind (checker).** Make the checker's sheet, which holds IDs and questions only:

   ```bash
   python data/rag_questions_tools.py checker-sheet data/rag_questions_v1.jsonl checker_sheet.jsonl
   ```

   For each question, fill `evidence` with the passage(s) you found (same format, `[[{"slug", "quote"}]]`) and `answer` with your answer, or `"not in the corpus"`. Do not look at the author's file until you have finished.
3. **Resolve (together).** Evidence agrees when one of the checker's quotes overlaps one of the author's spans. If the checker found a different valid passage, add it as an alternative span in the same group. If you disagree on the answer, rewrite or drop the item. An `unanswerable` item survives only if the checker also could not find the fact. No item stays while disputed. Before resolving, record the agreement rate:

   ```bash
   python data/rag_questions_tools.py agreement data/rag_questions_v1.jsonl checker_sheet.jsonl
   ```

4. **Validate.**

   ```bash
   python data/rag_questions_tools.py validate data/rag_questions_v1.jsonl
   python -m unittest tests.test_rag_questions -v
   ```

5. **Register and report.** Follow "When it lands" in `data/README.md`: the `_variables.yml` entry (with `corpus_sha256`), the hash in `notebooks/13-rag.ipynb`, and the report (items written, dropped, alternatives added, agreement before resolution). Do not commit `checker_sheet.jsonl`.

## Shapes of questions (examples of the shape, not items)

| Kind | Shape |
|---|---|
| `lookup` | "Why do dot-product scores need scaling as the key dimension grows?" |
| `specific` | "What test accuracy did the averaged-embedding classifier reach on arXiv Topics?" |
| `multi` | "Which earlier loss does the contrastive retrieval loss generalize, and in which lab was that loss implemented?" |
| `unanswerable` | "What learning rate did Lab 7 use for LoRA?" (only if the lecture does not state it) |
