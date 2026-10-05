"""Checks on Lab 15's scoring (scripts/capstone_score.py), its collector and its manifest builder.

- the notebook's cell "Scoring: do not edit" restates the module's block verbatim, and the hash the
  notebook expects is the module's;
- the scoring reproduces hand-computed numbers (lecture 15's equations) and the sign-test values;
- the collector's four checks (scripts/collect_capstone.py);
- the manifest builder (data/build_capstone_eval.py) is deterministic, stratified and refuses to
  build while its inputs are missing or the snapshot is provisional.

Standard library and PyYAML only, as the CI test job has. Nothing here touches the network.
"""

import contextlib
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
V = yaml.safe_load((ROOT / "_variables.yml").read_text(encoding="utf-8"))


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S = _module("capstone_score", ROOT / "scripts" / "capstone_score.py")
C = _module("collect_capstone", ROOT / "scripts" / "collect_capstone.py")
B = _module("build_capstone_eval", ROOT / "data" / "build_capstone_eval.py")
NB = json.loads((ROOT / "notebooks" / "15-capstone.ipynb").read_text(encoding="utf-8"))
CODE = ["".join(c["source"]) for c in NB["cells"] if c["cell_type"] == "code"]

EV = [[{"slug": "01-text-as-data", "quote": "x", "start": 0, "end": 1}]]


def item(i, kind="A"):
    return {
        "id": i,
        "evidence": [] if kind == "U" else EV,
        "key_facts": ["alpha"] if kind == "A" else [],
        "source": "s",
        "kind": kind,
    }


def rec(i, output, final, usd=None):
    return {"id": i, "output": output, "final": final, "sources": ["src"], "usd": usd,
            "latency_s": 1.0, "n_llm": 1, "n_dec": 2}  # fmt: skip


def judge(claim, sources):
    return 0.0 if ("Beta" in claim or "Gamma" in claim) else 1.0


class NotebookCell(unittest.TestCase):
    def test_scoring_cell_is_the_module_block(self):
        cell = next(c for c in CODE if "# ==== Scoring: do not edit ====" in c)
        self.assertIn(S.block_source(), cell)

    def test_expected_hash_is_the_modules(self):
        selftest = next(c for c in CODE if "Self-test:" in c and "SCORING_SHA256 =" in c)
        expected = re.search(r'SCORING_SHA256 = "([0-9a-f]{64})"', selftest).group(1)
        self.assertEqual(expected, S.scoring_hash())

    def test_block_is_self_contained_and_never_asks_the_decider(self):
        block = S.block_source()
        self.assertNotIn("decide(", block)
        self.assertNotIn("\nimport ", block)
        self.assertNotIn("REFERENCE_JUDGE =", block)
        for name in ("ABSTAIN", "LOSS"):  # defined outside the block, identically in the notebook
            self.assertIn(f"\n{name} = ", "\n" + "\n".join(CODE).replace(S.block_source(), ""))

    def test_notebook_constants_match_the_module(self):
        code = "\n".join(CODE)
        self.assertIn(f'ABSTAIN = "{S.ABSTAIN}"', code)
        self.assertIn("LOSS = dict(wrong=5, abstain=1)", code)


class Scoring(unittest.TestCase):
    def test_hand_computed_numbers(self):
        """The ten hand-made records of the notebook's self-test, check (xi)."""
        items = [item(f"a{i}") for i in range(1, 7)] + [item(f"u{i}", "U") for i in range(1, 4)]
        items.append(item("r1", "R"))
        records = [
            rec("a1", "answer", "Alpha is the scale [1]."),
            rec("a2", "answer", "Beta is the scale [1]."),
            rec("a3", "abstain", S.ABSTAIN),
            rec("a4", "error", None),
            rec("a5", "answer", "ALPHA [2]."),
            rec("a6", "answer", "I cannot answer from the provided sources"),
            rec("u1", "abstain", S.ABSTAIN),
            rec("u2", "answer", "Gamma [1]."),
            rec("u3", "abstain", S.ABSTAIN),
            rec("r1", "answer", "Delta [1]."),
        ]
        p = S.score(records, items, judge=judge)["pooled"]
        self.assertTrue(math.isclose(p["acc"]["value"], 2 / 6))
        self.assertTrue(math.isclose(p["acc"]["se"], math.sqrt((1 / 3) * (2 / 3) / 6)))
        self.assertTrue(math.isclose(p["acc_answered"]["value"], 2 / 3))
        self.assertTrue(math.isclose(p["abs_A"]["value"], 2 / 6))
        self.assertTrue(math.isclose(p["abs_U"]["value"], 2 / 3))
        self.assertEqual((p["uns"]["n"], p["uns"]["value"]), (5, 2 / 5))
        self.assertTrue(math.isclose(p["cost_bar"]["value"], 17 / 9))
        self.assertEqual((p["n_scored"], p["n_reported_only"], p["n_errors"]), (9, 1, 1))
        self.assertIsNone(p["usd_per_q"])
        self.assertEqual(p["calls_per_q"], 3)
        self.assertEqual((p["latency_median"], p["latency_p95"]), (1.0, 1.0))

    def test_cost_rule(self):
        self.assertEqual(S.question_cost(rec("x", "abstain", S.ABSTAIN), item("x", "U")), 0)
        self.assertEqual(S.question_cost(rec("x", "abstain", S.ABSTAIN), item("x")), 1)
        self.assertEqual(S.question_cost(rec("x", "answer", "anything"), item("x", "U")), 5)
        self.assertEqual(S.question_cost(rec("x", "error", None), item("x", "U")), 5)
        self.assertEqual(S.question_cost(rec("x", "answer", "alpha"), item("x")), 0)
        # A paraphrased refusal is an answer, and here a wrong one.
        self.assertEqual(
            S.question_cost(rec("x", "answer", "The sources do not say."), item("x")), 5
        )

    def test_sign_test_values_of_the_lecture(self):
        self.assertEqual(round(S.sign_test(8, 1), 4), 0.0391)
        self.assertEqual(round(S.sign_test(7, 2), 4), 0.1797)
        self.assertEqual(S.sign_test(4, 3), 1.0)
        self.assertEqual(S.sign_test(0, 0), 1.0)

    def test_compare_and_flips(self):
        items = [item(f"c{i}") for i in range(11)]
        base = [rec(f"c{i}", "answer", "wrong" if i < 8 else "alpha") for i in range(11)]
        new = [rec(f"c{i}", "answer", "alpha" if i < 8 or i > 8 else "wrong") for i in range(11)]
        c = S.compare(base, new, items, judge=judge)
        self.assertEqual((c["gained"], c["lost"], c["n_paired"]), (8, 1, 11))
        self.assertEqual(round(c["p"], 4), 0.0391)
        self.assertTrue(math.isclose(c["deltas"]["acc"], 10 / 11 - 3 / 11))
        self.assertEqual(S.flips(base, new, items), 9)
        self.assertEqual(S.flips(base, base, items), 0)

    def test_usd_is_none_unless_every_record_has_one(self):
        items = [item("a"), item("b")]
        self.assertIsNone(S.score([rec("a", "answer", "x", 0.1), rec("b", "answer", "x")], items,
                                  judge=judge)["pooled"]["usd_per_q"])  # fmt: skip
        both = S.score(
            [rec("a", "answer", "x", 0.1), rec("b", "answer", "x", 0.3)], items, judge=judge
        )
        self.assertTrue(math.isclose(both["pooled"]["usd_per_q"], 0.2))


def submission(
    tmp, *, path_class="open-cpu", test_runs=2, manifest="m" * 64, time_ok=True, pair="1"
):
    items = [item("a"), item("u", "U")]
    run = [rec("a", "answer", "alpha"), rec("u", "abstain", S.ABSTAIN)]
    context = {
        "path_class": path_class,
        "backends": {"generator": "g", "decider": "d", "retriever": "r", "judge": "j"},
        "eval_set": "test fixture",
        "manifest_sha256": manifest,
        "config_diff": {},
        "edited_cells": [],
        "notebook": "test",
        "test_runs": test_runs,
        "test_seconds": {"baseline": 1.0, "final": 1.0 if time_ok else 700.0},
        "budgets": {"k_max_llm": 4, "k_max_dec": 4, "test_seconds_max": 600},
    }
    runs = {"baseline": {"dev": run, "test": run}, "final": {"dev": run, "test": run}}
    share = {"changed": "c", "predicted": "p", "happened": "h"}
    hypothesis = {"component": "V1", "change": "x", "metric": "cost_bar", "direction": "down",
                  "size": "1"}  # fmt: skip
    cwd = os.getcwd()
    try:
        os.chdir(tmp)
        with contextlib.redirect_stdout(io.StringIO()):
            sub = S.submission(
                pair, hypothesis, runs, items=items, context=context, share_text=share, judge=judge
            )
            S.share_card(sub)
    finally:
        os.chdir(cwd)
    return sub


class Collector(unittest.TestCase):
    def test_statuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.json"
            manifest.write_bytes(b"{}\n")
            good = hashlib.sha256(b"{}\n").hexdigest()
            cases = {
                "ranked": submission(tmp, manifest=good),
                "code check only": submission(tmp, manifest=good, path_class="stub", pair="2"),
                "test used for development": submission(tmp, manifest=good, test_runs=3, pair="3"),
                "over budget": submission(tmp, manifest=good, time_ok=False, pair="4"),
                "not comparable": submission(tmp, pair="5"),  # wrong manifest hash
            }
            for status, sub in cases.items():
                with self.subTest(status=status):
                    self.assertEqual(C.check(sub, good)[0], status)
            tampered = dict(cases["ranked"], scoring_hash="0" * 64)
            self.assertEqual(C.check(tampered, good)[0], "not comparable")
            self.assertEqual(C.check(cases["ranked"], None)[0], "not comparable")
            files = sorted(Path(tmp).glob("capstone_*.json"))
            self.assertEqual(len(files), 5)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(C.main([tmp, "--manifest", str(manifest)]), 0)
            text = out.getvalue()
            self.assertIn("== open-cpu (4 submissions) ==", text)
            self.assertIn("== stub (1 submissions) ==", text)
            self.assertIn("With m = 1 ranked pairs", text)


class ManifestBuilder(unittest.TestCase):
    def items(self, source, n_auto, n_reported, n_unans, split):
        p = f"{source}-{split}"
        return (
            [{"id": f"{p}-a{i}", "split": split, "kind": "lookup" if i % 2 else "multi",
              "evidence": EV, "key_facts": ["k"]} for i in range(n_auto)]
            + [{"id": f"{p}-r{i}", "split": split, "kind": "lookup", "evidence": EV,
                "key_facts": []} for i in range(n_reported)]
            + [{"id": f"{p}-u{i}", "split": split, "kind": "unanswerable", "evidence": [],
                "key_facts": []} for i in range(n_unans)]
        )  # fmt: skip

    def build(self, **kw):
        lab13 = self.items("lab13", 20, 3, 5, "dev") + self.items("lab13", 34, 4, 8, "test")
        new = self.items("new", 6, 0, 9, "dev") + self.items("new", 12, 0, 18, "test")
        return B.build_manifest(
            lab13, new, lab13_sha256="a", new_sha256="b", snapshot_sha256="c", **kw
        )

    def test_deterministic_and_complete(self):
        m = self.build()
        self.assertEqual(B.manifest_bytes(m), B.manifest_bytes(self.build()))
        self.assertEqual({s: len(ids) for s, ids in m["splits"].items()}, {"dev": 43, "test": 76})
        self.assertEqual(len(m["reported_only"]), 7)
        self.assertEqual(len(m["scored"]) + len(m["reported_only"]), 119)
        self.assertEqual(m["costs"], {"wrong": 5, "abstain": 1})
        self.assertNotEqual(B.manifest_bytes(m), B.manifest_bytes(self.build(seed=1)))

    def test_cpu_subsets_are_stratified(self):
        m = self.build()
        for split, (n_auto, n_unans) in {"dev": (8, 4), "test": (16, 8)}.items():
            subset = m["cpu_subset"][split]
            with self.subTest(split=split):
                self.assertEqual(sum("-a" in i for i in subset), n_auto)
                self.assertEqual(sum("-u" in i for i in subset), n_unans)
                self.assertFalse(
                    any("-r" in i for i in subset), "reported-only items are not scored"
                )
                self.assertEqual({i.split("-")[0] for i in subset}, {"lab13", "new"})
                self.assertTrue(set(subset) <= set(m["splits"][split]))

    def test_id_collision_is_refused(self):
        lab13 = self.items("x", 2, 0, 1, "dev")
        with self.assertRaises(ValueError):
            B.build_manifest(lab13, lab13, lab13_sha256="a", new_sha256="b", snapshot_sha256="c")

    def test_allocation(self):
        alloc = B._manifest_allocate({("lab13", "a"): 10, ("lab13", "b"): 8, ("new", "c"): 6}, 8)
        self.assertEqual(alloc, {("lab13", "a"): 3, ("lab13", "b"): 3, ("new", "c"): 2})
        alloc = B._manifest_allocate({("lab13", "u"): 30, ("new", "u"): 1}, 2)
        self.assertEqual(alloc, {("lab13", "u"): 1, ("new", "u"): 1}, "both sources represented")
        self.assertEqual(sum(B._manifest_allocate({("a", "x"): 2}, 5).values()), 2)

    def test_reads_the_snapshot_status_and_refuses(self):
        self.assertEqual(B.snapshot_status(), V["datasets"]["lectures"]["status"])
        if not (B.LAB13.exists() and B.CAPSTONE.exists()) or B.snapshot_status() == "provisional":
            with self.assertRaises(SystemExit):
                B.build()


if __name__ == "__main__":
    unittest.main()
