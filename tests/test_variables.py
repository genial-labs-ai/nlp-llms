"""Consistency checks on _variables.yml and the files derived from it."""

import re
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
V = yaml.safe_load((ROOT / "_variables.yml").read_text(encoding="utf-8"))
MODULE_FIELDS = {"n", "slug", "day", "minutes", "title", "summary", "objectives", "stack"}


def slot_key(slot) -> str:
    return slot["module"] if isinstance(slot, dict) else slot


def day_slots(d: dict) -> list:
    """A day's optional self-serve slot, then its module slots."""
    return ([d["self_serve"]] if d.get("self_serve") else []) + d["slots"]


def has_notebook(m: dict) -> bool:
    return m.get("notebook", True)


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
        # Module 0 (coding agents in the terminal) opens Day 1; the labs are 1 onward.
        numbers = sorted(m["n"] for m in V["modules"].values())
        self.assertEqual(numbers, list(range(0, len(numbers))))

    def test_every_module_has_a_lecture_page(self):
        for m in V["modules"].values():
            self.assertTrue((ROOT / "lectures" / f"{m['slug']}.qmd").exists(), m["slug"])

    def test_notebook_flag(self):
        for key, m in V["modules"].items():
            self.assertIn(m.get("notebook", True), (True, False), key)
            if not has_notebook(m):
                self.assertFalse((ROOT / "notebooks" / f"{m['slug']}.ipynb").exists(), key)

    def test_module_zero_has_no_notebook(self):
        self.assertFalse(has_notebook(V["modules"]["m00"]))


class Days(unittest.TestCase):
    def test_each_day_fills_the_module_slots(self):
        slots = sum(1 for s in V["schedule"]["slots"] if s["kind"] == "module")
        for key, d in V["days"].items():
            self.assertEqual(len(d["slots"]), slots, key)

    def test_slots_reference_modules_of_that_day(self):
        for d in V["days"].values():
            for slot in day_slots(d):
                key = slot_key(slot)
                self.assertEqual(V["modules"][key]["day"], d["n"], key)

    def test_every_module_is_scheduled(self):
        scheduled = set()
        for d in V["days"].values():
            for slot in day_slots(d):
                scheduled.add(slot_key(slot))
        self.assertEqual(scheduled, set(V["modules"]))

    def test_self_serve_slot(self):
        kinds = [s["kind"] for s in V["schedule"]["slots"]]
        self.assertLessEqual(set(kinds), {"self_serve", "opening", "module", "break"})
        users = [d for d in V["days"].values() if d.get("self_serve")]
        self.assertEqual(kinds.count("self_serve"), 1 if users else 0)
        if users:
            slot = next(s for s in V["schedule"]["slots"] if s["kind"] == "self_serve")
            start, end = (int(t[:2]) * 60 + int(t[3:]) for t in (slot["start"], slot["end"]))
            for d in users:
                m = V["modules"][slot_key(d["self_serve"])]
                self.assertEqual(m["minutes"], end - start, d["n"])

    def test_slots_are_back_to_back(self):
        slots = V["schedule"]["slots"]
        for a, b in zip(slots, slots[1:], strict=False):
            self.assertEqual(a["end"], b["start"])

    def test_day_count_and_pages(self):
        self.assertEqual(len(V["days"]), V["workshop"]["days"])
        for d in V["days"].values():
            self.assertTrue((ROOT / f"day-{d['n']}.qmd").exists())


class Notebooks(unittest.TestCase):
    def test_every_notebook_matches_a_slug(self):
        slugs = {m["slug"] for m in V["modules"].values() if has_notebook(m)}
        slugs |= {V["setup"]["slug"]}
        for path in (ROOT / "notebooks").glob("*.ipynb"):
            self.assertIn(path.stem, slugs)

    def test_no_api_key_is_committed(self):
        pattern = re.compile(r"sk-[A-Za-z0-9_-]{20,}")
        for path in (ROOT / "notebooks").glob("*.ipynb"):
            self.assertIsNone(pattern.search(path.read_text(encoding="utf-8")), path.name)


class AgentsIntro(unittest.TestCase):
    """Module 0's npm package names and versions (registry.npmjs.org, dated)."""

    def test_keys_and_shapes(self):
        a = V["agents_intro"]
        self.assertRegex(a["checked"], r"^\d{4}-\d{2}-\d{2}$")
        self.assertRegex(a["threejs"], r"^0\.\d+\.\d+$")
        self.assertEqual(
            a["clis"],
            {
                "claude_code": "@anthropic-ai/claude-code",
                "codex": "@openai/codex",
                "gemini": "@google/gemini-cli",
            },
        )
        self.assertEqual(set(a["node"]), set(a["clis"]))
        for key in ("claude_code", "codex", "gemini_cli"):  # versions live in `assistants`
            self.assertRegex(V["assistants"][key], r"^\d+\.\d+\.\d+$")


if __name__ == "__main__":
    unittest.main()
