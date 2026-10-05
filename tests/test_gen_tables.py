"""scripts/gen_tables.py: the generated includes agree with _variables.yml and are current."""

import re
import sys
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import gen_tables as g  # noqa: E402

V = g.load_variables()
INCLUDES = ROOT / "_includes"


def committed(name: str) -> str:
    return (INCLUDES / name).read_text(encoding="utf-8")


def generated(body: str) -> str:
    return f"{g.NOTICE}\n\n{body.rstrip()}\n"


class Current(unittest.TestCase):
    """What the generator would write is what is committed (the CI drift gate, as a test)."""

    def test_includes_are_current(self):
        expected = {
            "facts.md": g.facts_strip(V),
            "days.md": g.days_cards(V),
            "path.md": g.path_steps(V),
            "schedule.md": g.schedule_table(V),
            "notebooks.md": g.notebooks_index(V),
        }
        for d in g.days_in_order(V):
            expected[f"day-{d['n']}.md"] = g.day_cards(V, d)
        for key, m in g.modules_in_order(V):
            expected[f"module-{m['n']:02d}.md"] = g.module_block(V, key, m)
        for name, body in expected.items():
            self.assertEqual(committed(name), generated(body), name)

    def test_sidebar_is_current(self):
        self.assertEqual(committed("sidebar.yml"), f"{g.YAML_NOTICE}\n{g.sidebar_yaml(V)}")


class Sidebar(unittest.TestCase):
    def test_lists_every_lecture_once_in_module_order(self):
        sidebar = yaml.safe_load(g.sidebar_yaml(V))["website"]["sidebar"][0]
        self.assertEqual(len(sidebar["contents"]), V["workshop"]["days"])
        hrefs = [item["href"] for section in sidebar["contents"] for item in section["contents"]]
        self.assertEqual(hrefs, [f"lectures/{m['slug']}.qmd" for _, m in g.modules_in_order(V)])
        for href in hrefs:
            self.assertTrue((ROOT / href).exists(), href)


class Clock(unittest.TestCase):
    def test_module_clock(self):
        self.assertEqual(g.module_clock(V, "m00"), g.self_serve_time(V, "m00"))
        slots = [s for s in V["schedule"]["slots"] if s["kind"] == "module"]
        self.assertEqual(g.module_clock(V, "m01"), f"{slots[0]['start']}–{slots[0]['end']}")
        # The capstone fills the last two slots of Day 4: one card, one span.
        self.assertEqual(g.module_clock(V, "m15"), f"{slots[2]['start']}–{slots[3]['end']}")

    def test_lecture_and_lab_fill_a_module_slot(self):
        w = V["workshop"]
        for s in V["schedule"]["slots"]:
            if s["kind"] == "module":
                length = g.to_minutes(s["end"]) - g.to_minutes(s["start"])
                self.assertEqual(length, w["lecture_minutes"] + w["lab_minutes"], s)

    def test_schedule_lab_starts_after_the_lecture(self):
        w = V["workshop"]
        pairs = re.findall(
            r"Lecture (\d\d:\d\d)\]\{\.slot-lecture\}\[Lab (\d\d:\d\d)\]", g.schedule_table(V)
        )
        self.assertTrue(pairs)
        for lecture, lab in pairs:
            self.assertEqual(g.to_minutes(lab) - g.to_minutes(lecture), w["lecture_minutes"])


class Content(unittest.TestCase):
    def test_path_has_one_step_per_module(self):
        path = g.path_steps(V)
        self.assertEqual(path.count("{.path-num}"), len(V["modules"]))
        for m in V["modules"].values():
            self.assertIn(f"](lectures/{m['slug']}.qmd){{.path-title}}", path)

    def test_module_header_links_resolve_from_lectures(self):
        # The header is included from lectures/, so its internal links are project-absolute.
        for key, m in g.modules_in_order(V):
            block = g.module_block(V, key, m)
            self.assertIn(f"](/day-{m['day']}.qmd)", block)
            for objective in m["objectives"]:
                self.assertIn(f"- {objective}", block)

    def test_only_modules_with_a_notebook_get_a_colab_link(self):
        index = g.notebooks_index(V)
        for m in V["modules"].values():
            link = f"{V['repo']['colab_base']}/{m['slug']}.ipynb"
            self.assertEqual(link in index, g.has_notebook(m), m["slug"])

    def test_fenced_divs_are_balanced(self):
        blocks = [g.facts_strip(V), g.days_cards(V), g.path_steps(V), g.notebooks_index(V)]
        blocks += [g.day_cards(V, d) for d in g.days_in_order(V)]
        blocks += [g.module_block(V, key, m) for key, m in g.modules_in_order(V)]
        for block in blocks:
            lines = block.splitlines()
            opens = sum(1 for line in lines if line.startswith("::: {"))
            closes = sum(1 for line in lines if line == ":::")
            self.assertEqual(opens, closes, block[:60])


if __name__ == "__main__":
    unittest.main()
