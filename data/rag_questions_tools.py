"""Tools for the people who write and check Workshop RAG Questions v1.

The question file is data/rag_questions_v1.jsonl. Specification: briefs/13-rag.md, decision (b);
instructions: data/rag_questions_TEMPLATE.md. Standard library only, no network, no model. Nothing
here writes, proposes or labels a question.

    python data/rag_questions_tools.py validate data/rag_questions_v1.jsonl  # all checks
    python data/rag_questions_tools.py find "exact text from a page"         # where it occurs
    python data/rag_questions_tools.py checker-sheet IN.jsonl OUT.jsonl      # questions only
    python data/rag_questions_tools.py agreement AUTHOR.jsonl CHECKER.jsonl  # before resolution

tests/test_rag_questions.py runs `validate` on the real file once it exists, and on the
agent-written fixture (tests/fixtures/rag_questions_fixture.json) until then.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from collections import Counter
from pathlib import Path

DATA = Path(__file__).resolve().parent
CORPUS = DATA / "workshop_lectures_v1.jsonl.gz"
SPLITS = {"dev": 30, "test": 50}
KINDS = {"lookup": 0.40, "specific": 0.30, "multi": 0.15, "unanswerable": 0.15}
FIELDS = {
    "id",
    "split",
    "kind",
    "question",
    "evidence",
    "answer",
    "key_facts",
    "author",
    "checker",
    "notes",
}
OPTIONAL = {"would_be_in"}  # unanswerable items: the page(s) where the fact would have been
MAX_QUOTE_WORDS = 60
# A question sharing this many consecutive non-stopword tokens with one of its quotes is flagged.
OVERLAP_RUN = 4
# A fixed English stopword list for the overlap flag (function words only), so the flag does not
# depend on any library's list.
STOPWORDS = frozenset(
    (
        "a an and are as at be been being but by can could did do does doing for from had has "
        "have having he her here hers him his how i if in into is it its itself just me more "
        "most my no nor not of off on once only or other our out over own same she should so "
        "some such than that the their them then there these they this those through to too "
        "under until up very was we were what when where which while who whom why will with "
        "would you your"
    ).split()
)


def load_corpus(path: Path = CORPUS) -> dict[str, str]:
    """{slug: text} of Workshop Lectures v1."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return {r["slug"]: r["text"] for r in map(json.loads, f)}


def load_items(path: Path) -> list[dict]:
    """Items of a .jsonl question file, or of the fixture's {"notice", "items"} JSON."""
    text = Path(path).read_text(encoding="utf-8")
    if Path(path).suffix == ".json":
        return json.loads(text)["items"]
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def content_tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


def longest_shared_run(question: str, quote: str) -> int:
    """Length of the longest run of consecutive non-stopword tokens that the two texts share."""
    a, b = content_tokens(question), content_tokens(quote)
    best = 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, start=1):
            if x == y:
                cur[j] = prev[j - 1] + 1
                best = max(best, cur[j])
        prev = cur
    return best


def schema_errors(items: list[dict]) -> list[str]:
    errors = []
    ids = Counter(it.get("id") for it in items)
    errors += [f"duplicate id {i}" for i, n in ids.items() if n > 1]
    for it in items:
        i = it.get("id", "?")
        missing, extra = FIELDS - set(it), set(it) - FIELDS - OPTIONAL
        if missing or extra:
            errors.append(f"{i}: missing {sorted(missing)}, unexpected {sorted(extra)}")
            continue
        if it["split"] not in SPLITS:
            errors.append(f"{i}: split {it['split']!r}")
        if it["kind"] not in KINDS:
            errors.append(f"{i}: kind {it['kind']!r}")
        if not isinstance(it["question"], str) or not it["question"].strip():
            errors.append(f"{i}: empty question")
        ev = it["evidence"]
        if it["kind"] == "unanswerable":
            if ev != []:
                errors.append(f"{i}: an unanswerable item has evidence []")
        elif not (isinstance(ev, list) and ev and all(isinstance(g, list) and g for g in ev)):
            errors.append(f"{i}: evidence must be a non-empty list of non-empty groups")
        else:
            if it["kind"] == "multi" and len(ev) < 2:
                errors.append(f"{i}: a multi item needs at least two groups")
            for g in ev:
                for span in g:
                    if set(span) != {"slug", "quote"} or not span["quote"].strip():
                        errors.append(f"{i}: a span is {{'slug', 'quote'}} with a non-empty quote")
                    elif len(span["quote"].split()) > MAX_QUOTE_WORDS:
                        errors.append(f"{i}: quote longer than {MAX_QUOTE_WORDS} words")
        if not (
            isinstance(it["key_facts"], list)
            and all(isinstance(f, str) and f for f in it["key_facts"])
        ):
            errors.append(f"{i}: key_facts must be a list of non-empty strings")
        for who in ("author", "checker"):
            if not (isinstance(it[who], str) and re.fullmatch(r"[A-Z][A-Z0-9]{1,3}", it[who])):
                errors.append(f"{i}: {who} must be initials")
        if it["author"] == it["checker"]:
            errors.append(f"{i}: author and checker must be different people")
    return errors


def quote_errors(items: list[dict], corpus: dict[str, str]) -> list[str]:
    """Every quote must occur exactly once in its page of the snapshot. This is also what catches a
    corpus rebuild that changed or duplicated the text a question was written against."""
    errors = []
    for it in items:
        for g in it.get("evidence") or []:
            for span in g:
                text = corpus.get(span.get("slug"))
                if text is None:
                    errors.append(f"{it['id']}: no page {span.get('slug')!r} in the snapshot")
                    continue
                n = text.count(span["quote"])
                if n != 1:
                    errors.append(
                        f"{it['id']}: quote occurs {n} times in {span['slug']}: "
                        f"{span['quote'][:60]!r}"
                    )
        for slug in it.get("would_be_in", []):
            if slug not in corpus:
                errors.append(f"{it['id']}: would_be_in names no page {slug!r}")
    return errors


def spec_errors(items: list[dict], corpus: dict[str, str]) -> list[str]:
    """Split sizes, kind shares (within one item), key_facts share, lecture coverage."""
    errors = []
    for split, n in SPLITS.items():
        its = [it for it in items if it["split"] == split]
        if len(its) != n:
            errors.append(f"{split}: {len(its)} items, expected {n}")
        kinds = Counter(it["kind"] for it in its)
        for kind, share in KINDS.items():
            if abs(kinds[kind] - share * n) > 1:
                errors.append(f"{split}: {kinds[kind]} {kind} items, expected {share * n:g} ± 1")
    answerable = [it for it in items if it["kind"] != "unanswerable"]
    with_facts = sum(bool(it["key_facts"]) for it in answerable)
    if answerable and with_facts < 0.7 * len(answerable):
        errors.append(
            f"only {with_facts} of {len(answerable)} answerable items have key_facts (need 70%)"
        )
    cover = Counter()
    for it in items:
        slugs = {s["slug"] for g in it["evidence"] for s in g} | set(it.get("would_be_in", []))
        cover.update(slugs)
    for slug in sorted(s for s in corpus if s[:2].isdigit()):
        if cover[slug] < 4:
            errors.append(f"{slug}: covered by {cover[slug]} questions, need at least 4")
    return errors


def overlap_flags(items: list[dict], run: int = OVERLAP_RUN) -> list[tuple[str, int]]:
    """(id, run length) of every question that copies `run` or more consecutive non-stopword tokens
    from one of its own quotes. The authors rewrite those questions."""
    flags = []
    for it in items:
        quotes = [s["quote"] for g in it["evidence"] for s in g]
        longest = max((longest_shared_run(it["question"], q) for q in quotes), default=0)
        if longest >= run:
            flags.append((it["id"], longest))
    return flags


def validate(path: Path, full_spec: bool = True) -> list[str]:
    items, corpus = load_items(path), load_corpus()
    errors = schema_errors(items)
    if not errors:
        errors += quote_errors(items, corpus)
        errors += [
            f"{i}: question shares {n} consecutive words with its quote"
            for i, n in overlap_flags(items)
        ]
        if full_spec:
            errors += spec_errors(items, corpus)
    return errors


def agreement(author: list[dict], checker: list[dict]) -> tuple[int, int]:
    """(agreeing, answerable): the checker's evidence agrees with an item when one of the checker's
    quotes overlaps one of the author's spans on the same page (step 3 of the protocol)."""
    corpus = load_corpus()
    found = {c["id"]: [s for g in c.get("evidence") or [] for s in g] for c in checker}

    def span(s):
        start = corpus[s["slug"]].find(s["quote"])
        return s["slug"], start, start + len(s["quote"])

    agree = n = 0
    for it in author:
        if it["kind"] == "unanswerable":
            continue
        n += 1
        mine = [span(s) for g in it["evidence"] for s in g]
        theirs = [
            span(s) for s in found.get(it["id"], []) if s["quote"] in corpus.get(s["slug"], "")
        ]
        agree += any(
            a[0] == b[0] and min(a[2], b[2]) > max(a[1], b[1]) for a in mine for b in theirs
        )
    return agree, n


NOT_FOUND = "not in the corpus"


def unanswerable_agreement(author: list[dict], checker: list[dict]) -> tuple[int, int]:
    """(agreeing, unanswerable): the checker agrees with an unanswerable item when they also
    found nothing: no evidence, and the answer "not in the corpus" (step 3 of the protocol)."""
    sheet = {c["id"]: c for c in checker}
    agree = n = 0
    for it in author:
        if it["kind"] != "unanswerable":
            continue
        n += 1
        c = sheet.get(it["id"], {})
        answer = (c.get("answer") or "").strip().rstrip(".").lower()
        agree += not c.get("evidence") and answer == NOT_FOUND
    return agree, n


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("validate").add_argument("path", type=Path)
    sub.add_parser("find").add_argument("text")
    p = sub.add_parser("checker-sheet")
    p.add_argument("src", type=Path)
    p.add_argument("dst", type=Path)
    p = sub.add_parser("agreement")
    p.add_argument("author", type=Path)
    p.add_argument("checker", type=Path)
    args = parser.parse_args()
    if args.cmd == "validate":
        errors = validate(args.path)
        print("\n".join(errors) or "OK: every check passed")
        return 1 if errors else 0
    if args.cmd == "find":
        for slug, text in load_corpus().items():
            starts = [m.start() for m in re.finditer(re.escape(args.text), text)]
            if starts:
                print(f"{slug}: {len(starts)} occurrence(s) at {starts[:10]}")
        return 0
    if args.cmd == "checker-sheet":
        sheet = [
            {"id": it["id"], "question": it["question"], "evidence": [], "answer": ""}
            for it in load_items(args.src)
        ]
        args.dst.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in sheet), encoding="utf-8"
        )
        print(f"wrote {len(sheet)} questions to {args.dst} (no evidence, answers or kinds)")
        return 0
    author, checker = load_items(args.author), load_items(args.checker)
    agree, n = agreement(author, checker)
    print(
        f"evidence agreement before resolution: {agree} of {n} answerable items "
        f"({agree / max(n, 1):.1%})"
    )
    agree, n = unanswerable_agreement(author, checker)
    print(
        f"the checker also found nothing: {agree} of {n} unanswerable items "
        f"({agree / max(n, 1):.1%})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
