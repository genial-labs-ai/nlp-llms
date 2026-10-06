"""Readiness metadata (_variables.yml) and run records (runs/*.json) are well formed and
agree with each other, and the generated readiness pages are current."""

import sys
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import gen_tables as g  # noqa: E402
import readiness  # noqa: E402
import run_records  # noqa: E402

V = yaml.safe_load((ROOT / "_variables.yml").read_text(encoding="utf-8"))
R = V["readiness"]
ENVS = set(R["envs"])
NOTEBOOKS = {p.stem for p in (ROOT / "notebooks").glob("*.ipynb")}
LECTURE_STATES = {"drafted", "reviewed", "piloted", "final"}
LAB_STATES = {"none", "written", "reviewed", "piloted", "final"}
MODULE_FIELDS = {"lecture", "lab", "runtime", "estimate_minutes", "accounts", "cost", "fallback"}
OPTIONAL_FIELDS = {"optional_accounts", "gaps"}


class Records(unittest.TestCase):
    def test_every_file_is_valid(self):
        files = sorted((ROOT / "runs").glob("*.json"))
        self.assertTrue(files)
        for path in files:
            self.assertEqual(run_records.validate_file(path, ENVS, NOTEBOOKS), [], path.name)

    def test_backfilled_records_never_count_as_teaching_evidence(self):
        report = readiness.build(V, run_records.load_all())
        for slug, e in report["evidence"].items():
            if e["teaching"] is not None:
                self.assertNotEqual(e["teaching"]["source"], "backfill", slug)

    def test_validator_rejects_bad_records(self):
        bad = ROOT / "tests" / "fixtures" / "runs-bad.json"
        errors = run_records.validate_file(bad, ENVS, NOTEBOOKS)
        joined = "\n".join(errors)
        for expected in (
            "credential",
            "env",
            "no notebook",
            "date",
            "scope_note",
            "content_sha",
            "status",
        ):
            self.assertIn(expected, joined)


class Metadata(unittest.TestCase):
    def test_top_level(self):
        self.assertIn(R["teaching_env"], ENVS)
        self.assertTrue(R["envs"][R["teaching_env"]].get("teaching"))
        self.assertIn(R["release_path"], R["paths"])
        self.assertEqual(set(R["paths"]), set(run_records.PATHS))
        self.assertGreater(R["max_run_age_days"], 0)

    def test_every_module_has_readiness(self):
        for key, m in V["modules"].items():
            r = m.get("readiness")
            self.assertIsNotNone(r, key)
            self.assertEqual(MODULE_FIELDS - set(r), set(), key)
            self.assertEqual(set(r) - MODULE_FIELDS - OPTIONAL_FIELDS, set(), key)
            self.assertIn(r["lecture"], LECTURE_STATES, key)
            self.assertIn(r["lab"], LAB_STATES, key)
            self.assertEqual(r["lab"] == "none", not m.get("notebook", True), key)
            self.assertIn(r["runtime"], ENVS, key)
            self.assertTrue(R["envs"][r["runtime"]].get("teaching"), key)
            self.assertIn(r["fallback"]["kind"], R["fallback_kinds"], key)
            self.assertTrue(r["fallback"]["note"], key)
            self.assertGreater(r["estimate_minutes"], 0, key)

    def test_items_evaluate_and_name_real_modules(self):
        numbers = {m["n"] for m in V["modules"].values()}
        for item in readiness.items(V):
            self.assertTrue(set(item["modules"]) <= numbers, item["id"])
            self.assertIsInstance(item["closed"], bool, item["id"])

    def test_items_check_existing_inputs(self):
        for key, item in R["items"].items():
            check = item["check"]
            for field in ("json", "absent"):
                if field in check:
                    self.assertTrue((ROOT / check[field]).exists(), key)


class Generated(unittest.TestCase):
    def test_readiness_includes_are_current(self):
        report = readiness.build(V, run_records.load_valid(ENVS))
        for name, body in {
            "readiness.md": g.readiness_table(V, report),
            "readiness-summary.md": g.readiness_summary(report),
        }.items():
            committed = (ROOT / "_includes" / name).read_text(encoding="utf-8")
            self.assertEqual(committed, f"{g.NOTICE}\n\n{body.rstrip()}\n", name)

    def test_summary_never_claims_readiness_without_colab_runs(self):
        report = readiness.build(V, run_records.load_all())
        s = report["summary"]
        text = g.readiness_status(report)
        if s["teaching"] < s["labs"] or s["items_open"]:
            self.assertIn("not yet ready to teach", text)

    def test_output_does_not_depend_on_the_clock(self):
        text = g.readiness_table(V, readiness.build(V, run_records.load_all()))
        self.assertNotIn("today", text.lower())


def record(**fields) -> dict:
    """A synthetic, valid record for the evidence rules."""
    base = {
        "notebook": "05-transformer-from-scratch",
        "date": "2026-10-06",
        "source": "colab",
        "env": "colab-t4",
        "path": "open",
        "mode": "worked",
        "scope": "notebook",
        "status": "pass",
        "seconds": 300,
        "evidence": "test",
        "file": "test.json",
        "content_sha": run_records.content_sha("05-transformer-from-scratch"),
    }
    return {**base, **fields}


class EvidenceRules(unittest.TestCase):
    """The rules the readiness page and the release check rely on."""

    SLUG, RUNTIME = "05-transformer-from-scratch", "colab-t4"

    def ev(self, *records):
        return readiness.evidence(V, self.SLUG, self.RUNTIME, list(records))

    def test_a_newer_failure_replaces_an_older_pass(self):
        e = self.ev(record(date="2026-10-06"), record(date="2026-10-10", status="fail"))
        self.assertEqual(e["teaching"]["status"], "fail")
        self.assertFalse(readiness.passed(e["teaching"]))

    def test_only_the_modules_own_runtime_counts_as_teaching(self):
        for env in ("colab-cpu", "own-laptop", "mac-m1pro"):
            e = self.ev(record(env=env))
            self.assertIsNone(e["teaching"], env)
            self.assertTrue(readiness.passed(e["real"]), env)

    def test_a_run_against_older_code_is_stale(self):
        e = self.ev(record(content_sha="0" * 16))
        self.assertTrue(e["teaching"]["stale"])
        self.assertFalse(readiness.passed(e["teaching"]))

    def test_backfill_and_offline_never_count_as_teaching(self):
        self.assertIsNone(self.ev(record(source="backfill", content_sha=None))["teaching"])
        self.assertIsNone(self.ev(record(path="offline"))["teaching"])

    def test_ci_sentence_uses_only_the_newest_batch(self):
        old = [
            record(notebook=s, env="gha-ubuntu", path="offline", date="2026-10-06")
            for s in ("01-text-as-data", "02-word-vectors", "03-sequence-models")
        ]
        new = [
            record(notebook="01-text-as-data", env="gha-ubuntu", path="offline", date="2026-10-20"),
            record(
                notebook="02-word-vectors",
                env="gha-ubuntu",
                path="offline",
                date="2026-10-20",
                status="fail",
            ),
        ]
        s = readiness.build(V, old + new)["summary"]
        self.assertEqual((s["ci_date"], s["ci_passed"], s["ci_failed"]), ("2026-10-20", 1, 1))

    def test_ready_needs_every_lab_on_its_runtime_and_no_open_work(self):
        labs = [m for m in V["modules"].values() if m.get("notebook", True)]
        records = [
            record(
                notebook=m["slug"],
                env=m["readiness"]["runtime"],
                content_sha=run_records.content_sha(m["slug"]),
            )
            for m in labs
        ]
        s = readiness.build(V, records)["summary"]
        self.assertEqual(s["teaching"], len(labs))
        self.assertEqual(s["ready"], s["items_open"] == 0)


if __name__ == "__main__":
    unittest.main()
