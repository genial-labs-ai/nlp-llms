# Writing Workshop Capstone Questions v1: instructions for the two people

For the author (proposed: Romeo) and the checker (proposed: one instructor). Specification: `briefs/15-capstone.md`, "Decision: the fixed evaluation set". Estimated effort, not measured: about 3 hours for the author and 2 for the checker, for 45 items (about 4 and 3 minutes per item; unanswerable items take longer, because absence has to be searched for).

The capstone's evaluation set is **Lab 13's 80 questions, reused by ID, plus these 45 new ones**. Write these only after Lab 13's 80 exist (`data/rag_questions_TEMPLATE.md`): the new IDs must not collide with Lab 13's, and the two sets are written against the same frozen snapshot.

**Hard rule: no language model writes, proposes, rewrites, filters or labels any item.** Not for a draft, not to check whether a model "knows" a memory-bait fact, not to keep the items a model gets wrong. Filtering by a model would make the set adversarial to that one model and bias every comparison involving it. The checker may search with anything except a language model: reading, `grep`, `python data/rag_questions_tools.py find`, or Lab 13's own BM25.

## Before you start

1. **The snapshot must be final.** `datasets.lectures.status` in `_variables.yml` must not be `provisional`. The validator refuses items written against a provisional snapshot.
2. **Lab 13's 80 questions exist** (`data/rag_questions_v1.jsonl`) and validate.
3. Read the pages from the snapshot, not from the site, as in Lab 13's instructions.

## The file

`data/capstone_questions_v1.jsonl`, uncompressed, one object per line, IDs `c001` to `c045`:

| Split | `unanswerable` | `reading` | Total |
|---|---|---|---|
| `dev` | 9 | 6 | 15 |
| `test` | 18 | 12 | 30 |

At least **13 of the 27 unanswerable items are memory bait**.

```json
{"id": "c001", "split": "test", "kind": "unanswerable",
 "question": "...",
 "evidence": [], "answer": "", "key_facts": [],
 "would_be": {"slug": "05-transformer-from-scratch", "where": "section 6, where training details would appear"},
 "absent_terms": ["4.5 million sentence pairs", "4.5M sentence pairs"],
 "outside_source": "Vaswani et al. 2017, section 5.1",
 "memory_bait": true,
 "author": "RA", "checker": "IN", "notes": ""}
```

| Field | Rule |
|---|---|
| `kind` | `unanswerable`: plausible, on a topic the corpus covers, but the fact is not in it. `reading`: answerable; it joins a paper on the reading list to the module and lab that use it (the research-assistant task) |
| `question` | in words different from the passage; the validator flags a question that shares four or more consecutive non-stopword words with one of its quotes |
| `evidence` | as Lab 13: a list of groups, all needed, each a list of alternative spans. `[]` for `unanswerable` |
| `key_facts` | **mandatory on every `reading` item** (Lab 13 needs them on 70%; here on 100%, so every new answerable item is scored by program). `[]` for `unanswerable` |
| `would_be` | `unanswerable` only: the page where the fact would have been, and where on it |
| `absent_terms` | `unanswerable` only: strings a memory-based answer would contain. The validator checks that none occurs anywhere in the snapshot, case-insensitively: a mechanical check that the fact is absent |
| `outside_source` | memory-bait items only: where the fact **is** stated (a reading-list paper, with section), read by the author in the paper itself, never taken from a model's output |
| `memory_bait` | `true` for a fact stated in a reading-list paper that a model may well know but the snapshot never states; `false` or absent otherwise |
| `author`, `checker` | initials, two different people |

## Protocol

1. **Write (author).** For a `reading` item: pick a paper on a lecture's reading list, find where the lecture and lab use it, and write the question in your own words; record the evidence quote(s), the answer and the key facts. For an `unanswerable` item: write the question, then search the snapshot for the fact and every way it could be phrased; record `would_be` and `absent_terms`. For memory bait, open the paper and record `outside_source`.

   ```bash
   python data/rag_questions_tools.py find "4.5 million sentence pairs"
   ```

   Search for each absent term exactly as you will list it. A short term can match inside a longer one: "4.5 million" occurs in lecture 7's "134.5 million", so it could not be an absent term.

2. **Check, blind (checker).** Make the checker's sheet, which holds IDs and questions only:

   ```bash
   python data/rag_questions_tools.py checker-sheet data/capstone_questions_v1.jsonl capstone_checker_sheet.jsonl
   ```

   For each question, fill `evidence` with the passage(s) you found and `answer` with your answer, or exactly `"not in the corpus"`. Do not look at the author's file until you have finished.

3. **Resolve (together).** Record the agreement first:

   ```bash
   python data/rag_questions_tools.py agreement data/capstone_questions_v1.jsonl capstone_checker_sheet.jsonl
   ```

   It prints the evidence agreement on `reading` items and how many `unanswerable` items the checker also found nothing for. A `reading` item whose evidence the checker found elsewhere gains that passage as an alternative span. **An `unanswerable` item survives only if the checker also found nothing and its `absent_terms` pass the validator**; otherwise rewrite it or drop it. No item stays while disputed.

4. **Validate.**

   ```bash
   python -m unittest tests.test_capstone_questions -v
   ```

   With the real file present, this runs the full specification: split sizes, kind counts, memory bait, ID collisions with Lab 13, quotes found exactly once, `absent_terms` absent, key facts on every `reading` item.

5. **Build the manifest and register.** Once both question files validate:

   ```bash
   python data/build_capstone_eval.py           # writes data/capstone_eval_v1.json
   python data/build_capstone_eval.py --check   # rebuilds in memory and compares
   ```

   The manifest pins the hashes of both question files and the snapshot, and the stratified CPU subsets. Then: the `datasets` entries in `_variables.yml` (brief 15, "What changes elsewhere"), the readiness item `capstone-questions` closes by itself (`scripts/readiness.py`), and the report (items written, dropped, alternatives added, both agreement rates before resolution). Do not commit `capstone_checker_sheet.jsonl`.

6. **After the manifest: the recorded baselines.** An instructor runs the unmodified starter on `dev` and `test` on every path available and records `lab15.baseline.*` in `data/baselines.json` (brief 15, "Instructor-recorded baselines"). Only then can CI compare the stub path's numbers against `lab15.baseline.stub` (brief 15's stub regression check).

## Shapes of questions (examples of the shape, not items)

| Kind | Shape |
|---|---|
| `reading` | "Which reading-list paper introduced the loss that Lab 9's reward model minimizes, and which exercise implements it?" |
| `unanswerable`, memory bait | "How many sentence pairs was the original transformer trained on for English–German?" (only if no lecture page states it) |
| `unanswerable`, not bait | "Which learning-rate schedule does Lab 12's toy decision model train with?" (only if no lecture page states it) |
