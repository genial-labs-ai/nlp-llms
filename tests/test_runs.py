"""Readiness metadata (_variables.yml) and run records (runs/*.json) are well formed and
agree with each other, and the generated readiness pages are current."""

import json
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
        "index": 0,
    }
    base["content_sha"] = run_records.content_sha(fields.get("notebook", base["notebook"]))
    return {**base, **fields}


class Validation(unittest.TestCase):
    def check(self, text: str) -> list[str]:
        path = ROOT / "tests" / "fixtures" / "_tmp_run.json"
        path.write_text(text, encoding="utf-8")
        try:
            return run_records.validate_file(path, ENVS, NOTEBOOKS)
        finally:
            path.unlink()

    def test_wrong_shapes_are_reported_not_raised(self):
        self.assertIn("list of runs", "\n".join(self.check("[1, 2]")))
        self.assertIn("must be an object", "\n".join(self.check('{"schema": 1, "runs": ["x"]}')))

    def test_a_tool_record_needs_a_content_sha(self):
        batch = (
            '{"schema": 1, "date": "2026-10-06", "source": "colab", "env": "colab-t4",'
            ' "path": "open", "mode": "worked", "evidence": "x", "runs": [{"notebook":'
            ' "01-text-as-data", "scope": "notebook", "status": "pass", "seconds": 1}]}'
        )
        self.assertIn("content_sha", "\n".join(self.check(batch)))

    def test_impossible_future_and_unhashable_values_are_reported(self):
        def batch(**over):
            entry = {
                "notebook": "01-text-as-data",
                "scope": "notebook",
                "status": "pass",
                "seconds": 1,
                **over,
            }
            return json.dumps(
                {
                    "schema": 1,
                    "date": "2026-10-06",
                    "source": "backfill",
                    "env": "colab-t4",
                    "path": "open",
                    "mode": "worked",
                    "evidence": "x",
                    "runs": [entry],
                }
            )

        self.assertIn("not a real date", "\n".join(self.check(batch(date="2026-13-40"))))
        self.assertIn("future", "\n".join(self.check(batch(date="2999-01-01"))))
        self.assertIn("env", "\n".join(self.check(batch(env=["colab-t4"]))))
        self.assertIn("notebook", "\n".join(self.check(batch(notebook={"slug": "x"}))))

    def test_item_checks_never_crash(self):
        for check in (
            {"absent": "no/such/file.qmd", "pattern": "x"},
            {"json": "data/decisions_v1_stats.json", "key": "split", "at_least": 1},
            {"var": "repo.owner"},
        ):
            closed, why = readiness.item_closed(V, {"check": check})
            self.assertFalse(closed, check)

    def test_malformed_json_is_reported_not_raised(self):
        self.assertIn("not valid JSON", "\n".join(self.check('{"schema": 1,}')))

    def test_settings_values_must_be_strings(self):
        batch = (
            '{"schema": 1, "date": "2026-10-06", "source": "backfill", "env": "colab-t4",'
            ' "path": "open", "mode": "worked", "evidence": "x", "settings": {"NLP_LLMS_QUICK": 1},'
            ' "runs": [{"notebook": "01-text-as-data", "scope": "notebook", "status": "pass",'
            ' "seconds": 1}]}'
        )
        self.assertIn("settings", "\n".join(self.check(batch)))

    def test_seconds_must_not_be_a_boolean(self):
        batch = (
            '{"schema": 1, "date": "2026-10-06", "source": "colab", "env": "colab-t4",'
            ' "path": "open", "mode": "worked", "evidence": "x", "runs": [{"notebook":'
            ' "01-text-as-data", "scope": "notebook", "status": "pass", "seconds": true}]}'
        )
        self.assertIn("seconds", "\n".join(self.check(batch)))

    def test_ordinary_words_are_not_credentials(self):
        self.assertIsNone(run_records.SECRET.search("mask-and-attend-to-previous-tokens"))
        self.assertIsNotNone(run_records.SECRET.search('"sk-abcdefghijklmnopqrstuvwx"'))


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
            self.assertTrue(readiness.passed(e["other"]), env)

    def test_a_run_against_older_code_is_stale(self):
        e = self.ev(record(content_sha="0" * 16))
        self.assertTrue(e["teaching"]["stale"])
        self.assertFalse(readiness.passed(e["teaching"]))

    def test_backfill_and_offline_never_count_as_teaching(self):
        self.assertIsNone(self.ev(record(source="backfill", content_sha=None))["teaching"])
        self.assertIsNone(self.ev(record(path="offline"))["teaching"])

    def test_keyed_quick_and_learner_runs_are_not_teaching_evidence(self):
        for fields in (
            {"path": "keyed"},
            {"settings": {"NLP_LLMS_QUICK": "1"}},
            {"mode": "learner"},
            {"scope": "partial", "scope_note": "Part A"},
        ):
            e = self.ev(record(**fields))
            self.assertIsNone(e["teaching"], fields)
            self.assertIsNotNone(e["other"], fields)

    def test_a_backfill_on_the_runtime_is_shown_not_dropped(self):
        e = self.ev(record(source="backfill", content_sha=None))
        self.assertIsNotNone(e["other"])

    def test_a_same_day_failure_beats_a_pass(self):
        e = self.ev(record(status="fail", index=0), record(status="pass", index=1))
        self.assertEqual(e["teaching"]["status"], "fail")

    def test_staleness_is_checked_on_every_row(self):
        e = self.ev(record(env="mac-m1pro", content_sha="0" * 16))
        self.assertTrue(e["other"]["stale"])
        self.assertFalse(readiness.passed(e["other"]))

    def test_a_tool_record_without_a_hash_is_stale_as_teaching_evidence(self):
        e = self.ev(record(content_sha=None))
        self.assertTrue(e["teaching"]["stale"])

    def test_a_newer_partial_failure_replaces_an_older_full_pass(self):
        e = self.ev(
            record(env="mac-m1pro", date="2026-10-05"),
            record(
                env="mac-m1pro", date="2026-10-20", scope="partial", scope_note="x", status="fail"
            ),
        )
        self.assertEqual(e["other"]["status"], "fail")

    def test_full_settings_beat_quick_on_the_same_day(self):
        e = self.ev(
            record(env="mac-m1pro", index=0),
            record(env="mac-m1pro", index=1, settings={"NLP_LLMS_QUICK": "1"}),
        )
        self.assertNotIn("settings", e["other"])

    def test_a_stale_same_day_failure_does_not_hide_a_current_pass(self):
        e = self.ev(record(status="fail", content_sha="0" * 16, index=0), record(index=1))
        self.assertTrue(readiness.passed(e["teaching"]))

    def test_an_offline_ci_run_does_not_hide_a_real_path_ci_pass(self):
        e = self.ev(
            record(env="gha-ubuntu", date="2026-10-05", content_sha=None, source="backfill"),
            record(env="gha-ubuntu", date="2026-10-06", path="offline"),
        )
        self.assertEqual(e["ci"]["path"], "offline")
        self.assertTrue(readiness.passed(e["ci_real"]))

    def test_no_records_reads_sensibly(self):
        text = g.readiness_status(readiness.build(V, []))
        self.assertIn("No runs are recorded yet", text)
        self.assertNotIn("None", text)

    def test_a_missing_check_input_is_open_work_not_a_crash(self):
        closed, why = readiness.item_closed(
            V, {"check": {"json": "nope.json", "key": "k", "at_least": 1}}
        )
        self.assertFalse(closed)
        closed, why = readiness.item_closed(V, {"check": {"var": "no.such.key", "equals": 1}})
        self.assertFalse(closed)

    def test_a_newer_stale_run_never_hides_an_older_current_pass(self):
        e = self.ev(
            record(date="2026-10-06"),
            record(date="2026-10-09", status="fail", content_sha="0" * 16),
        )
        self.assertTrue(readiness.passed(e["teaching"]))

    def test_a_newer_partial_pass_keeps_an_older_full_pass_end_to_end(self):
        records = [
            record(env="mac-m1pro", date="2026-10-04"),
            record(env="mac-m1pro", date="2026-10-05", scope="partial", scope_note="x"),
        ]
        s = readiness.build(V, records)["summary"]
        self.assertEqual((s["real"], s["real_partial_only"]), (1, 0))

    def test_ci_runs_are_their_own_row(self):
        e = self.ev(record(env="gha-ubuntu", path="offline"))
        self.assertIsNone(e["other"])
        self.assertEqual(e["ci"]["env"], "gha-ubuntu")

    def test_a_lab_passing_on_colab_is_not_also_counted_in_part(self):
        labs = [m for m in V["modules"].values() if m.get("notebook", True)]
        records = [record(), record(env="mac-m1pro", scope="partial", scope_note="x")]
        s = readiness.build(V, records)["summary"]
        self.assertEqual(s["teaching"], 1)
        self.assertEqual(s["real_partial_only"], 0)
        self.assertEqual(len(labs), s["labs"])

    def test_ci_counts_the_newest_run_of_each_notebook(self):
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
        self.assertEqual(
            (s["ci_from"], s["ci_to"], s["ci_passed"], s["ci_failed"]),
            ("2026-10-06", "2026-10-20", 2, 1),
        )

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
