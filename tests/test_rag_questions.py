"""Validator for Workshop RAG Questions v1 and checks on the Workshop Lectures v1 snapshot.

Spec: briefs/13-rag.md, decisions (a) and (b). Until people write data/rag_questions_v1.jsonl, the
question checks run on tests/fixtures/rag_questions_fixture.json: an agent-written fixture for
exercising code, not an evaluation set (the full-spec checks, split sizes, kind shares and coverage,
apply only to the real file). Nothing here touches the network.
"""

import gzip
import importlib.util
import json
import os
import re
import subprocess
import unittest
from pathlib import Path
from unittest import mock

import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
V = yaml.safe_load((ROOT / "_variables.yml").read_text(encoding="utf-8"))
LECTURES = V["datasets"]["lectures"]
REAL = DATA / "rag_questions_v1.jsonl"
FIXTURE = ROOT / "tests" / "fixtures" / "rag_questions_fixture.json"


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tools = _module("rag_questions_tools", DATA / "rag_questions_tools.py")
builder = _module("build_lectures_corpus", DATA / "build_lectures_corpus.py")
CORPUS = tools.load_corpus()


def _commit_available(commit: str) -> bool:
    result = subprocess.run(
        ["git", "-C", str(ROOT), "cat-file", "-e", f"{commit}^{{commit}}"], capture_output=True
    )
    return result.returncode == 0


class Snapshot(unittest.TestCase):
    def test_pages(self):
        with gzip.open(DATA / LECTURES["file"], "rt", encoding="utf-8") as f:
            records = [json.loads(line) for line in f]
        self.assertEqual(len(records), LECTURES["documents"])
        self.assertEqual([r["module"] for r in records], list(range(1, 13)) + [None])
        self.assertEqual(sum(len(r["text"]) for r in records), LECTURES["characters"])
        for r in records:
            self.assertEqual(r["source_commit"], LECTURES["source_commit"])
            self.assertTrue(r["text"].startswith(f"# {r['title']}\n"), r["slug"])
            for leftover in ("{{<", "<!--", "\n:::"):
                self.assertNotIn(leftover, r["text"], r["slug"])
        self.assertNotIn("13-rag", [r["slug"] for r in records])

    def test_builder_pins_the_recorded_commit(self):
        self.assertEqual(builder.SOURCE_COMMIT, LECTURES["source_commit"])

    @unittest.skipUnless(
        _commit_available(LECTURES["source_commit"]), "source commit not in this clone's history"
    )
    def test_snapshot_rebuilds_byte_for_byte(self):
        # Content, not compressed bytes: gzip output depends on the zlib build, which differs
        # between machines. tests/test_data.py checks the committed file's hash.
        blob, _ = builder.build(LECTURES["source_commit"])
        committed = (DATA / LECTURES["file"]).read_bytes()
        self.assertEqual(gzip.decompress(blob), gzip.decompress(committed))

    def test_notebook_quotes_the_snapshot_hash(self):
        nb = json.loads((ROOT / "notebooks" / "13-rag.ipynb").read_text(encoding="utf-8"))
        source = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
        self.assertEqual(source.count(f'"{LECTURES["sha256"]}"'), 2)  # loader and interface cells
        self.assertNotIn(
            "rag_questions_fixture", source, "the notebook must never load the fixture"
        )

    def test_readme_loading_snippet(self):
        readme = (DATA / "README.md").read_text(encoding="utf-8")
        blocks = re.findall(r"```python\n(.*?)```", readme, flags=re.DOTALL)
        # The loading contract (first block, defines fetch), then the lectures cell that uses it.
        code = blocks[0] + "\n" + next(b for b in blocks if "def load_lectures" in b)
        namespace: dict = {}

        def no_network(*args, **kwargs):
            raise OSError("network disabled in tests")

        with (
            mock.patch.dict(os.environ, {"NLP_LLMS_DATA": str(DATA)}),
            mock.patch("urllib.request.urlopen", no_network),
        ):
            exec(compile(code, "data/README.md", "exec"), namespace)  # noqa: S102
            docs = namespace["load_lectures"]()
        self.assertEqual(len(docs), LECTURES["documents"])
        self.assertEqual(set(docs[0]), {"doc_id", "title", "text"})


class OverlapFlag(unittest.TestCase):
    def test_runs(self):
        quote = "The dot products grow with the dimension of the keys."
        self.assertEqual(tools.longest_shared_run("Why is attention scaled?", quote), 0)
        # stopwords are skipped: "dot products grow ... dimension" is a run of four content tokens
        self.assertEqual(tools.longest_shared_run("Do dot products grow with dimension?", quote), 4)
        self.assertEqual(tools.longest_shared_run("dot products shrink with dimension", quote), 2)

    def test_flagged_item(self):
        item = {
            "id": "x",
            "question": "Why do the dot products grow with the dimension of the keys?",
            "evidence": [[{"slug": "s", "quote": "The dot products grow with the dimension."}]],
        }
        self.assertEqual(tools.overlap_flags([item]), [("x", 4)])


class Agreement(unittest.TestCase):
    def test_unanswerable_items_agree_only_when_the_checker_found_nothing(self):
        author = [
            {"id": "q1", "kind": "unanswerable"},
            {"id": "q2", "kind": "unanswerable"},
            {"id": "q3", "kind": "unanswerable"},
            {"id": "q4", "kind": "lookup"},
        ]
        checker = [
            {"id": "q1", "evidence": [], "answer": "Not in the corpus"},
            {"id": "q2", "evidence": [], "answer": "about 4.5 million pairs"},
            {"id": "q3", "evidence": [[{"slug": "x", "quote": "y"}]], "answer": "not in the corpus"},
        ]
        self.assertEqual(tools.unanswerable_agreement(author, checker), (1, 3))


class Questions(unittest.TestCase):
    """The real question file when it exists; the agent-written fixture until then."""

    path = REAL if REAL.exists() else FIXTURE

    def test_fixture_is_labelled_and_small(self):
        fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
        self.assertIn("AGENT-WRITTEN FIXTURE", fixture["notice"])
        self.assertIn("NOT AN EVALUATION SET", fixture["notice"])
        self.assertLessEqual(len(fixture["items"]), 10)

    def test_schema(self):
        self.assertEqual(tools.schema_errors(tools.load_items(self.path)), [])

    def test_every_quote_occurs_exactly_once(self):
        self.assertEqual(tools.quote_errors(tools.load_items(self.path), CORPUS), [])

    def test_no_question_copies_its_quote(self):
        flags = tools.overlap_flags(tools.load_items(self.path))
        self.assertEqual(flags, [], f"rewrite these questions in other words: {flags}")

    def test_fixture_was_written_against_this_snapshot(self):
        fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
        self.assertEqual(fixture["corpus_sha256"], LECTURES["sha256"])

    @unittest.skipUnless(
        REAL.exists(), "data/rag_questions_v1.jsonl not written yet (needs two people)"
    )
    def test_full_specification(self):
        self.assertEqual(tools.spec_errors(tools.load_items(REAL), CORPUS), [])

    @unittest.skipUnless(
        REAL.exists(), "data/rag_questions_v1.jsonl not written yet (needs two people)"
    )
    def test_questions_are_written_against_a_final_snapshot(self):
        self.assertNotEqual(
            LECTURES.get("status"),
            "provisional",
            "rebuild Workshop Lectures v1 from the finished references.qmd first",
        )
        entry = V["datasets"].get("rag_questions", {})
        self.assertEqual(entry.get("corpus_sha256"), LECTURES["sha256"])


if __name__ == "__main__":
    unittest.main()
