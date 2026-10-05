"""Validator for Workshop Capstone Questions v1 (the 45 new questions of Lab 15) and its manifest.

Spec: briefs/15-capstone.md, "Decision: the fixed evaluation set" (item schema, labeling protocol,
validator, manifest). Until people write data/capstone_questions_v1.jsonl, the item checks run on
tests/fixtures/capstone_questions_fixture.json: an agent-written fixture for exercising code, not an
evaluation set; no number from it is quoted anywhere. The full-specification checks (split sizes,
kind counts, memory bait, ID collisions with Lab 13's real file, the manifest) apply only to the
real files. Standard library and PyYAML only; nothing here touches the network.
"""

import importlib.util
import json
import re
import unittest
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
V = yaml.safe_load((ROOT / "_variables.yml").read_text(encoding="utf-8"))
LECTURES = V["datasets"]["lectures"]
REAL = DATA / "capstone_questions_v1.jsonl"
FIXTURE = ROOT / "tests" / "fixtures" / "capstone_questions_fixture.json"
LAB13_REAL = DATA / "rag_questions_v1.jsonl"
LAB13_FIXTURE = ROOT / "tests" / "fixtures" / "rag_questions_fixture.json"
MANIFEST = DATA / "capstone_eval_v1.json"

# The specification (brief 15): dev 15 and test 30 new items; per split, unanswerable 9 / 18 and
# reading 6 / 12; at least 13 of the 27 unanswerable items are memory bait.
SPLITS = {"dev": 15, "test": 30}
KIND_COUNTS = {
    "dev": {"unanswerable": 9, "reading": 6},
    "test": {"unanswerable": 18, "reading": 12},
}
MIN_MEMORY_BAIT = 13
FIELDS = {"id", "split", "kind", "question", "evidence", "answer", "key_facts", "author", "checker"}
OPTIONAL = {"notes", "would_be", "absent_terms", "outside_source", "memory_bait"}


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tools = _module("rag_questions_tools", DATA / "rag_questions_tools.py")
builder = _module("build_capstone_eval", DATA / "build_capstone_eval.py")
CORPUS = tools.load_corpus()
SNAPSHOT_TEXT = "\n".join(CORPUS.values()).casefold()


def schema_errors(items: list[dict]) -> list[str]:
    """The capstone item schema: Lab 13's, plus would_be, absent_terms, outside_source and
    memory_bait."""
    errors = [
        f"duplicate id {i}" for i, n in Counter(it.get("id") for it in items).items() if n > 1
    ]
    for it in items:
        i = it.get("id", "?")
        missing, extra = FIELDS - set(it), set(it) - FIELDS - OPTIONAL
        if missing or extra:
            errors.append(f"{i}: missing {sorted(missing)}, unexpected {sorted(extra)}")
            continue
        if not re.fullmatch(r"c\d{3}", str(i)):
            errors.append(f"{i}: IDs are c001, c002, ... (Lab 13's are q001 ...)")
        if it["split"] not in SPLITS:
            errors.append(f"{i}: split {it['split']!r}")
        if not isinstance(it["question"], str) or not it["question"].strip():
            errors.append(f"{i}: empty question")
        facts = it["key_facts"]
        if not (isinstance(facts, list) and all(isinstance(f, str) and f for f in facts)):
            errors.append(f"{i}: key_facts must be a list of non-empty strings")
        if it["kind"] == "unanswerable":
            if it["evidence"] != [] or facts != []:
                errors.append(f"{i}: an unanswerable item has evidence [] and key_facts []")
            wb = it.get("would_be")
            if not (isinstance(wb, dict) and set(wb) == {"slug", "where"} and wb["slug"] in CORPUS):
                errors.append(f"{i}: would_be must be {{slug (a page of the snapshot), where}}")
            terms = it.get("absent_terms")
            if not (
                isinstance(terms, list) and terms and all(isinstance(t, str) and t for t in terms)
            ):
                errors.append(f"{i}: absent_terms must be a non-empty list of strings")
            if not isinstance(it.get("memory_bait"), bool):
                errors.append(f"{i}: memory_bait must be true or false on an unanswerable item")
            if it.get("memory_bait") and not str(it.get("outside_source", "")).strip():
                errors.append(f"{i}: a memory-bait item names its outside_source")
        elif it["kind"] == "reading":
            ev = it["evidence"]
            if not (isinstance(ev, list) and ev and all(isinstance(g, list) and g for g in ev)):
                errors.append(f"{i}: evidence must be a non-empty list of non-empty groups")
            else:
                for g in ev:
                    for span in g:
                        if set(span) != {"slug", "quote"} or not str(span["quote"]).strip():
                            errors.append(f"{i}: a span is {{'slug', 'quote'}} with a quote")
                        elif len(span["quote"].split()) > tools.MAX_QUOTE_WORDS:
                            errors.append(f"{i}: quote longer than {tools.MAX_QUOTE_WORDS} words")
            if not facts:
                errors.append(f"{i}: every new answerable item needs key_facts (brief 15)")
            if it.get("memory_bait"):
                errors.append(f"{i}: memory_bait is false or absent on an answerable item")
        else:
            errors.append(f"{i}: kind {it['kind']!r} (new items are 'unanswerable' or 'reading')")
        for who in ("author", "checker"):
            if not (isinstance(it[who], str) and re.fullmatch(r"[A-Z][A-Z0-9]{1,3}", it[who])):
                errors.append(f"{i}: {who} must be initials")
        if it["author"] == it["checker"]:
            errors.append(f"{i}: author and checker must be different people")
    return errors


def absence_errors(items: list[dict]) -> list[str]:
    """A mechanical check of absence: no absent_term occurs in the snapshot (case-insensitive)."""
    return [
        f"{it['id']}: absent term {t!r} occurs in the snapshot"
        for it in items
        if it["kind"] == "unanswerable"
        for t in it.get("absent_terms") or []
        if t.casefold() in SNAPSHOT_TEXT
    ]


def spec_errors(items: list[dict]) -> list[str]:
    errors = []
    for split, n in SPLITS.items():
        its = [it for it in items if it["split"] == split]
        if len(its) != n:
            errors.append(f"{split}: {len(its)} items, expected {n}")
        kinds = Counter(it["kind"] for it in its)
        for kind, count in KIND_COUNTS[split].items():
            if kinds[kind] != count:
                errors.append(f"{split}: {kinds[kind]} {kind} items, expected {count}")
    bait = sum(bool(it.get("memory_bait")) for it in items if it["kind"] == "unanswerable")
    if bait < MIN_MEMORY_BAIT:
        errors.append(f"{bait} memory-bait items, need at least {MIN_MEMORY_BAIT}")
    return errors


def lab13_ids() -> set[str]:
    if LAB13_REAL.exists():
        return {it["id"] for it in tools.load_items(LAB13_REAL)}
    return {it["id"] for it in tools.load_items(LAB13_FIXTURE)}


class Questions(unittest.TestCase):
    """The real question file when it exists; the agent-written fixture until then."""

    path = REAL if REAL.exists() else FIXTURE

    def items(self):
        return tools.load_items(self.path)

    def test_fixture_is_labelled_and_small(self):
        fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
        self.assertIn("AGENT-WRITTEN FIXTURE", fixture["notice"])
        self.assertIn("NOT AN EVALUATION SET", fixture["notice"])
        self.assertIn("NO NUMBER FROM IT IS QUOTED ANYWHERE", fixture["notice"])
        self.assertLessEqual(len(fixture["items"]), 10)

    def test_fixture_was_written_against_this_snapshot(self):
        fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
        self.assertEqual(fixture["corpus_sha256"], LECTURES["sha256"])

    def test_schema(self):
        self.assertEqual(schema_errors(self.items()), [])

    def test_every_quote_occurs_exactly_once(self):
        self.assertEqual(tools.quote_errors(self.items(), CORPUS), [])

    def test_absent_terms_are_absent_from_the_snapshot(self):
        self.assertEqual(absence_errors(self.items()), [])

    def test_no_question_copies_its_quote(self):
        flags = tools.overlap_flags(self.items())
        self.assertEqual(flags, [], f"rewrite these questions in other words: {flags}")

    def test_ids_do_not_collide_with_lab13(self):
        self.assertEqual({it["id"] for it in self.items()} & lab13_ids(), set())

    def test_the_validator_catches_bad_items(self):
        bad = [
            {
                "id": "c900",
                "split": "dev",
                "kind": "unanswerable",
                "question": "q",
                "evidence": [],
                "answer": "",
                "key_facts": [],
                "would_be": {"slug": "nope", "where": "x"},
                "absent_terms": ["Workshop"],
                "memory_bait": True,
                "outside_source": "",
                "author": "AB",
                "checker": "AB",
            },
            {
                "id": "c901",
                "split": "test",
                "kind": "reading",
                "question": "q",
                "evidence": [[{"slug": "references", "quote": "x"}]],
                "answer": "a",
                "key_facts": [],
                "memory_bait": True,
                "author": "AB",
                "checker": "CD",
            },
        ]
        errors = " | ".join(schema_errors(bad))
        for expected in (
            "would_be",
            "outside_source",
            "different people",
            "needs key_facts",
            "memory_bait is false",
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, errors)
        self.assertEqual(
            absence_errors(bad[:1]), ["c900: absent term 'Workshop' occurs in the snapshot"]
        )

    def test_the_briefs_example_absent_term_is_in_the_snapshot(self):
        """Brief 15's schema example lists "4.5 million" as an absent term; the v1 snapshot
        contains it,
        so an item using it would be rejected (reported to the Academic Director)."""
        self.assertIn("4.5 million", SNAPSHOT_TEXT)

    @unittest.skipUnless(REAL.exists(), "data/capstone_questions_v1.jsonl not written yet (people)")
    def test_full_specification(self):
        self.assertEqual(spec_errors(self.items()), [])

    @unittest.skipUnless(REAL.exists(), "data/capstone_questions_v1.jsonl not written yet (people)")
    def test_questions_are_written_against_a_final_snapshot(self):
        self.assertNotEqual(
            LECTURES.get("status"),
            "provisional",
            "freeze and rebuild Workshop Lectures v1 first; then Lab 13's questions; then these",
        )


@unittest.skipUnless(
    MANIFEST.exists(), "data/capstone_eval_v1.json not built yet (needs both files)"
)
class Manifest(unittest.TestCase):
    def test_rebuilds_byte_for_byte(self):
        self.assertEqual(builder.build(), MANIFEST.read_bytes())

    def test_hashes_and_splits(self):
        m = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(m["lab13_sha256"], builder.sha256(LAB13_REAL))
        self.assertEqual(m["capstone_sha256"], builder.sha256(REAL))
        self.assertEqual(m["snapshot_sha256"], LECTURES["sha256"])
        self.assertEqual({s: len(ids) for s, ids in m["splits"].items()}, builder.SPLITS)
        self.assertEqual(m["costs"], {"wrong": 5, "abstain": 1})


if __name__ == "__main__":
    unittest.main()
