"""Consistency checks on _variables.yml and the files derived from it."""

import re
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
V = yaml.safe_load((ROOT / "_variables.yml").read_text(encoding="utf-8"))
MODULE_FIELDS = {"n", "slug", "day", "minutes", "title", "summary", "objectives", "stack"}


class Modules(unittest.TestCase):
    def test_every_module_has_all_fields(self):
        for key, m in V["modules"].items():
            self.assertEqual(MODULE_FIELDS - set(m), set(), key)
            self.assertTrue(m["objectives"], key)

    def test_keys_numbers_and_slugs_agree(self):
        for key, m in V["modules"].items():
            self.assertEqual(key, f"m{m['n']:02d}")
            self.assertRegex(m["slug"], rf"^{m['n']:02d}-[a-z0-9]+(-[a-z0-9]+)*$")

    def test_numbers_are_consecutive(self):
        numbers = sorted(m["n"] for m in V["modules"].values())
        self.assertEqual(numbers, list(range(1, len(numbers) + 1)))

    def test_every_module_has_a_lecture_page(self):
        for m in V["modules"].values():
            self.assertTrue((ROOT / "lectures" / f"{m['slug']}.qmd").exists(), m["slug"])


class Days(unittest.TestCase):
    def test_each_day_fills_the_module_slots(self):
        slots = sum(1 for s in V["schedule"]["slots"] if s["kind"] == "module")
        for key, d in V["days"].items():
            self.assertEqual(len(d["slots"]), slots, key)

    def test_slots_reference_modules_of_that_day(self):
        for d in V["days"].values():
            for slot in d["slots"]:
                key = slot["module"] if isinstance(slot, dict) else slot
                self.assertEqual(V["modules"][key]["day"], d["n"], key)

    def test_every_module_is_scheduled(self):
        scheduled = set()
        for d in V["days"].values():
            for slot in d["slots"]:
                scheduled.add(slot["module"] if isinstance(slot, dict) else slot)
        self.assertEqual(scheduled, set(V["modules"]))

    def test_day_count_and_pages(self):
        self.assertEqual(len(V["days"]), V["workshop"]["days"])
        for d in V["days"].values():
            self.assertTrue((ROOT / f"day-{d['n']}.qmd").exists())


class Notebooks(unittest.TestCase):
    def test_every_notebook_matches_a_slug(self):
        slugs = {m["slug"] for m in V["modules"].values()} | {V["setup"]["slug"]}
        for path in (ROOT / "notebooks").glob("*.ipynb"):
            self.assertIn(path.stem, slugs)

    def test_no_api_key_is_committed(self):
        pattern = re.compile(r"sk-[A-Za-z0-9_-]{20,}")
        for path in (ROOT / "notebooks").glob("*.ipynb"):
            self.assertIsNone(pattern.search(path.read_text(encoding="utf-8")), path.name)


if __name__ == "__main__":
    unittest.main()
