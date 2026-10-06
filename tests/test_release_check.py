"""scripts/release_check.py: a lab blocks the release unless its newest teaching run,
of the current code, passed within the age limit; open items always block."""

import datetime as dt
import sys
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import readiness  # noqa: E402
import release_check  # noqa: E402
import run_records  # noqa: E402

V = yaml.safe_load((ROOT / "_variables.yml").read_text(encoding="utf-8"))
M = V["modules"]["m01"]
SLUG = M["slug"]
AS_OF = dt.date(2026, 10, 6)


def teaching_run(**overrides) -> dict:
    """A record that counts as teaching evidence for Lab 1, unless overridden."""
    record = {
        "source": "test_notebooks",
        "env": M["readiness"]["runtime"],
        "path": V["readiness"]["release_path"],
        "mode": "worked",
        "notebook": SLUG,
        "date": "2026-10-05",
        "scope": "notebook",
        "status": "pass",
        "seconds": 300.0,
        "settings": {},
        "content_sha": run_records.content_sha(SLUG),
        "file": "test.json",
        "index": 0,
    }
    return {**record, **overrides}


def lab1(records: list[dict]) -> str | None:
    evidence = readiness.build(V, records)["evidence"][SLUG]
    return release_check.lab_blocker(V, M, evidence, AS_OF)


class LabBlockers(unittest.TestCase):
    def test_a_fresh_passing_teaching_run_clears_the_lab(self):
        self.assertIsNone(lab1([teaching_run()]))

    def test_no_run_blocks(self):
        self.assertIn("no teaching run", lab1([]))

    def test_a_run_of_other_code_blocks(self):
        self.assertIn("predates", lab1([teaching_run(content_sha="0" * 16)]))

    def test_a_failed_run_blocks(self):
        self.assertIn("failed", lab1([teaching_run(status="fail")]))

    def test_an_old_run_blocks(self):
        limit = V["readiness"]["max_run_age_days"]
        old = (AS_OF - dt.timedelta(days=limit + 1)).isoformat()
        self.assertIn("days old", lab1([teaching_run(date=old)]))
        edge = (AS_OF - dt.timedelta(days=limit)).isoformat()
        self.assertIsNone(lab1([teaching_run(date=edge)]))

    def test_runs_that_are_not_teaching_evidence_do_not_clear_the_lab(self):
        for name, value in [
            ("env", "mac-m1pro"),
            ("path", "offline"),
            ("mode", "learner"),
            ("scope", "partial"),
            ("source", "backfill"),
            ("settings", {"NLP_LLMS_QUICK": "1"}),
        ]:
            with self.subTest(name):
                self.assertIn("no teaching run", lab1([teaching_run(**{name: value})]))


class Release(unittest.TestCase):
    def test_open_items_block(self):
        found = release_check.blockers(V, [], AS_OF)
        open_items = [i for i in readiness.items(V) if not i["closed"]]
        for item in open_items:
            self.assertTrue(any(b.startswith(f"Item {item['id']} ") for b in found), item["id"])

    def test_every_lab_is_checked(self):
        found = release_check.blockers(V, [], AS_OF)
        labs = [m for m in V["modules"].values() if m.get("notebook", True)]
        self.assertEqual(sum(b.startswith("Module ") for b in found), len(labs))


if __name__ == "__main__":
    unittest.main()
