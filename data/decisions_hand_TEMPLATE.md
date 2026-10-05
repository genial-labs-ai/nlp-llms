# Workshop Desk Decisions v1: instructions for the human annotators

Two jobs in the decision set must be done by people, not by a language model or an agent (specification: `briefs/11-calibration.md`, "How the labels are made"):

1. **Write and label 80 hand-written items** (20 in `dev`, 60 in `test`) that cover phrasing the templates cannot.
2. **Audit 60 template items** by labelling them blind and comparing with the rule engine.

Until both are done the set's status is `v1-template-only` (`data/decisions_v1_stats.json`), and the 80 slots hold extra template items. Estimated effort: 2 to 3 hours per person.

**Ground rules for everyone.**

- **Do not use a language model** (ChatGPT, Claude, a local model, an autocomplete that writes sentences) to write, rephrase or label any item. The workshop evaluates language models on this set; text or labels from a model would contaminate it and muddy its CC0 license.
- Work from the policy text in [`decisions_policy_v1.md`](decisions_policy_v1.md), between the markers, and from nothing else. If the text does not settle an item, the item (or the text) is wrong: say so in `notes`; do not guess.
- **No personal data.** Use invented names (not your own, a colleague's or a public figure's). Every email address must end in `@example.org`; the builder rejects any other. Cities may be real; nothing else about them is used.
- American English. Amounts as `300 EUR`, in the request and the records alike. Dates in the records as `YYYY-MM-DD`; in the request, any way a person would write them.

## Who labels what

The specification asks for each hand item to be labelled by **two people who did not write it and do not see the author's intended answer**. With three people that is easy: the author writes, the other two label. **If only two people are available** (the proposal is Romeo and one instructor), the specification cannot be met as written; choose one of these and record the choice in `data/README.md`:

- (a) Each person writes 40 items; the other labels them blind; the author labels their own items only after at least a week, from the blind sheet, without their drafts open. Report agreement as "author (delayed) vs. second annotator".
- (b) Bring in a third person for labelling only.

## Job 1: the 80 hand-written items

### Quotas

The builder keeps the set's balance only if the hand items follow these quotas. "Label" means the answer you intend; the final label is whatever the annotators settle on.

| Split | Family | Items | Labels |
|---|---|---|---|
| `dev` | `policy` | 10 | 5 `yes`, 5 `no`; two per rule P1 to P5 |
| `dev` | `route` | 10 | 2 per option: `retrieve`, `calculate`, `send_email`, `ask_user`, `escalate` |
| `test` | `policy` | 30 | 15 `yes`, 15 `no`; six per rule P1 to P5 |
| `test` | `route` | 30 | 6 per option |

Within each group, aim for about 40% `plain` and the rest spread over `boundary`, `distractor`, `conflict`, `missing` and `injection` (definitions in `data/README.md`). Tag each item with the one tag that fits best.

### What to write

What the templates cannot do, for example: indirect requests ("my manager says I can't go after all"); two requests in one message (the route label is still the first step of the routing order that applies to the message as a whole); polite pressure or urgency; a long preamble before the request; a request that quotes the policy wrongly; a reply to an earlier message; an injection hidden in a quoted email. Keep each request under about 120 words.

**Questions, verbatim.** A `policy` item uses one of these five questions exactly; a `route` item uses the route question exactly.

- P1: `Under the policy, should the registration be accepted?`
- P2: `Under the policy, is this participant entitled to a full refund?`
- P3: `Under the policy, is the transfer allowed?`
- P4: `Under the policy, can the catering request be met?`
- P5: `Under the policy, can the assistant process this refund without a person's approval?`
- route: `What should the assistant do next?`

### Step 1: the authors' drafts

Each author writes one JSON object per line in a file of their own (for example `drafts_romeo.jsonl`; concatenate them into `drafts.jsonl` for the next steps). Do not commit drafts with intended labels until both annotators have labelled. The shape of one line, shown with placeholders (this is a format illustration, not an item):

```json
{"hand_id": "h-romeo-001", "author": "<your name>", "split": "dev", "family": "policy",
 "difficulty": "plain",
 "state": {"today": "YYYY-MM-DD",
           "event": {"id": "E<n>", "topic": "<Language | Vision | Machine learning | Robotics>",
                     "city": "<city>", "start_date": "YYYY-MM-DD", "seats": <int>,
                     "registered": <int, at most seats>, "remote": <true | false>, "fee_eur": <int>},
           "registration": {"name": "<invented name>", "email": "<first.last>@example.org",
                            "status": "<confirmed | waitlisted>", "paid_eur": <int>},
           "request": "<the participant's message, written by you>"},
 "question": "<one of the six questions above, verbatim>",
 "options": ["yes", "no"],
 "intended_label": "<your answer>", "rule": "<P1 to P8, or R1 to R5 for a route item>",
 "rationale": "<one sentence: which rule and which record decide it>"}
```

- `registration` is `null` when no record is found for the person writing (always the case for a new registration).
- `options` is `["yes", "no"]` for `policy` and `["retrieve", "calculate", "send_email", "ask_user", "escalate"]` for `route`, in that order.
- `rule` and `rationale` never reach a model; they are for error analysis.

### Step 2: blind labelling

```bash
python data/decisions_annotation.py hand-sheet drafts.jsonl --out hand_sheet.jsonl
```

The sheet holds, for each item, only `id`, `records`, `request`, `question` and `options`, in a shuffled order, with `"label": null` and `"notes": ""`. Each annotator copies it (`hand_A.jsonl`, `hand_B.jsonl`), fills `label` on every line with one of the options, and adds `notes` where useful. Do not discuss items until both copies are complete.

### Step 3: agreement

```bash
python data/decisions_annotation.py agreement hand_A.jsonl hand_B.jsonl --reference drafts.jsonl
```

It prints the raw agreement and Cohen's kappa of A against B, and of each against the author, overall and per family, and lists every disagreement.

### Step 4: resolve every disagreement

For each item where A and B disagree (or both disagree with the author), discuss it against the policy text. Then write one line per resolved item in `resolutions.jsonl`:

- the policy settles it: `{"hand_id": "...", "label": "<settled label>", "note": "<why, citing the rule>"}`;
- it was ambiguous and you rewrote it: add `"state": {...}` with the rewritten records and request (and label the rewritten version blind again if the change is more than a word);
- it cannot be settled: `{"hand_id": "...", "drop": true, "note": "..."}`, and write a replacement item for the same quota cell.

**No item stays in the set while its label is disputed.** If the policy text itself caused the dispute, propose a change to `decisions_policy_v1.md`; a change to the policy means rebuilding and re-checking the whole set.

### Step 5: merge and rebuild

```bash
python data/decisions_annotation.py merge-hand drafts.jsonl hand_A.jsonl hand_B.jsonl \
    --resolutions resolutions.jsonl --out data/decisions_hand_v1.jsonl
python data/build_decisions.py
python -m pytest tests/test_decisions.py tests/test_data.py
```

`decisions_hand_v1.jsonl` keeps both annotators' labels (`label_a`, `label_b`), the final `label` and the `resolution` note, uncompressed so its diffs are readable. The builder puts the hand items in the reserved slots and fills only the slots that are still empty; nothing else in the set changes. Then copy the new `sha256` and `bytes` from `decisions_v1_stats.json` into `_variables.yml` (`datasets.decisions`) and `data/README.md`, and the agreement numbers into the README's "Labels and agreement" table. Labs 11, 12 and 14 load the file by hash, so they pick up the new version once it is on `main`.

## Job 2: the 60-item template audit

The sample is fixed and already written: [`decisions_audit_sheet_v1.jsonl`](decisions_audit_sheet_v1.jsonl) (regenerate with `python data/decisions_annotation.py audit-sheet` only after a rebuild). It holds 5 template items per family and difficulty tag, 2 from `train` and 3 from `dev` or `test`, drawn with a fixed seed from the items that do not change when hand items are added, in a shuffled order, blind (no label, rule, rationale or difficulty).

1. Each annotator copies the sheet (`audit_A.jsonl`, `audit_B.jsonl`) and labels every line, independently.
2. Run `python data/decisions_annotation.py agreement audit_A.jsonl audit_B.jsonl --reference engine`.
3. Trace every disagreement with the rule engine. If both annotators disagree with the engine, or one does and is right on reading the policy, it is a **template or rule bug**: fix `build_decisions.py` (or the policy), rebuild, and re-run the tests. If the engine is right, it is a **human slip**: log it.
4. Report in `data/README.md`: agreement and kappa (A vs B, each vs the engine) before any fix, the bugs found and fixed, and the slips. Target: at least 95% agreement with the engine before fixes.
