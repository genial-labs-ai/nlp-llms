"""Consistency checks on _variables.yml and the files derived from it."""

import re
import sys
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import gen_tables as g  # noqa: E402  (units, placements, minutes_of)

V = yaml.safe_load((ROOT / "_variables.yml").read_text(encoding="utf-8"))
MODULE_FIELDS = {"n", "slug", "day", "minutes", "title", "summary", "objectives", "stack"}


def slot_key(slot) -> str:
    return slot["module"] if isinstance(slot, dict) else slot


def to_minutes(hhmm: str) -> int:
    return int(hhmm[:2]) * 60 + int(hhmm[3:])


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
        # Module 0 (coding agents in the terminal) is pre-work; the labs are 1 onward.
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


CLOCKS = V["schedule"]["clocks"]
SLOT_KINDS = {"clinic", "opening", "module", "break", "closing"}


class Clocks(unittest.TestCase):
    def test_slot_kinds_and_labels(self):
        for name, clk in CLOCKS.items():
            for s in clk["slots"]:
                self.assertIn(s["kind"], SLOT_KINDS, (name, s))
                self.assertEqual("label" in s, s["kind"] == "break", (name, s))
                self.assertIn(s.get("part"), (None, "lecture", "lab"), (name, s))
                if "part" in s:
                    self.assertEqual(s["kind"], "module", (name, s))

    def test_slots_are_back_to_back(self):
        for name, clk in CLOCKS.items():
            slots = clk["slots"]
            for a, b in zip(slots, slots[1:], strict=False):
                self.assertEqual(a["end"], b["start"], name)
                self.assertLess(to_minutes(a["start"]), to_minutes(a["end"]), name)

    def test_shape(self):
        for name, clk in CLOCKS.items():
            self.assertEqual(list(clk["shape"])[:2], ["lecture", "lab"], name)
            self.assertLessEqual(set(clk["shape"]), {"lecture", "lab", "debrief"}, name)

    def test_unit_minutes_equal_the_shape(self):
        # Slot arithmetic per clock, without the generator: a whole module slot is the
        # shape's total; a lecture part is the lecture; the lab part after it is the rest.
        for name, clk in CLOCKS.items():
            total = sum(clk["shape"].values())
            for s in clk["slots"]:
                if s["kind"] != "module":
                    continue
                length = to_minutes(s["end"]) - to_minutes(s["start"])
                expected = {
                    None: total,
                    "lecture": clk["shape"]["lecture"],
                    "lab": total - clk["shape"]["lecture"],
                }[s.get("part")]
                self.assertEqual(length, expected, (name, s))

    def test_days_use_known_clocks(self):
        for key, d in V["days"].items():
            self.assertIn(d["clock"], CLOCKS, key)

    def test_opening_and_closing_labels(self):
        for key, d in V["days"].items():
            kinds = {s["kind"] for s in CLOCKS[d["clock"]]["slots"]}
            for kind in ("opening", "closing"):
                self.assertEqual(kind in d, kind in kinds, (key, kind))


class Days(unittest.TestCase):
    def test_each_day_fills_its_clock_units(self):
        for key, d in V["days"].items():
            self.assertEqual(len(d["slots"]), len(g.units(CLOCKS[d["clock"]])), key)

    def test_slots_reference_modules_of_that_day(self):
        for d in V["days"].values():
            for slot in d["slots"]:
                key = slot_key(slot)
                self.assertEqual(V["modules"][key]["day"], d["n"], key)

    def test_every_module_is_scheduled_once_or_is_prework(self):
        placed = {}
        for d in g.days_in_order(V):
            for p in g.placements(V, d):
                placed.setdefault(p["key"], set()).add(d["n"])
        for key, m in V["modules"].items():
            if m["day"] == 0:
                self.assertNotIn(key, placed, key)
            else:
                self.assertEqual(placed.get(key), {m["day"]}, key)

    def test_clinic(self):
        clinics = [d for d in V["days"].values() if d.get("clinic")]
        for d in clinics:
            kinds = [s["kind"] for s in CLOCKS[d["clock"]]["slots"]]
            self.assertEqual(kinds.count("clinic"), 1, d["n"])
            # The clinic is drop-in help with pre-work.
            self.assertEqual(V["modules"][slot_key(d["clinic"])]["day"], 0, d["n"])
        prework = {key for key, m in V["modules"].items() if m["day"] == 0}
        self.assertEqual({slot_key(d["clinic"]) for d in clinics}, prework)

    def test_minutes_equal_what_the_clock_gives(self):
        # Pages read modules.mNN.minutes; it must agree with the clocks (pre-work excepted:
        # Module 0's minutes are a planning estimate).
        for key, m in V["modules"].items():
            if m["day"] == 0:
                continue
            self.assertEqual(m["minutes"], g.minutes_of(V, key), key)
        self.assertEqual(V["modules"]["m01"]["minutes"], 95)
        self.assertEqual(V["modules"]["m08"]["minutes"], 120)
        self.assertEqual(V["modules"]["m15"]["minutes"], 240)

    def test_day_count_and_pages(self):
        self.assertEqual(len(V["days"]), V["workshop"]["days"])
        self.assertEqual(sorted(d["n"] for d in V["days"].values()), list(range(1, 6)))
        for d in V["days"].values():
            self.assertTrue((ROOT / f"day-{d['n']}.qmd").exists())
        self.assertTrue((ROOT / "day-5.qmd").exists())


class Site(unittest.TestCase):
    """_quarto.yml lists every day page and its Days menu follows _variables.yml."""

    @classmethod
    def setUpClass(cls):
        cls.quarto = yaml.safe_load((ROOT / "_quarto.yml").read_text(encoding="utf-8"))

    def test_render_list_has_every_day_page(self):
        render = self.quarto["project"]["render"]
        for d in V["days"].values():
            self.assertIn(f"day-{d['n']}.qmd", render)

    def test_days_menu(self):
        left = self.quarto["website"]["navbar"]["left"]
        menu = next(item["menu"] for item in left if item.get("text") == "Days")
        hrefs = [item["href"] for item in menu]
        self.assertEqual(hrefs[0], "lectures/00-coding-agents.qmd")  # pre-work first
        days = g.days_in_order(V)
        self.assertEqual(hrefs[1:], [f"day-{d['n']}.qmd" for d in days])
        texts = [item["text"] for item in menu[1:]]
        self.assertEqual(texts, [f"Day {d['n']} · {d['short']}" for d in days])


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
