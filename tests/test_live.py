"""Every lecture's live plan (front matter `live`, scripts/live_plan.py) covers its
sections, names real activity blocks, holds enough activity time, and fills the module's
lecture minutes exactly; the generated tables are current."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import gen_tables as g  # noqa: E402
import live_plan  # noqa: E402

V = g.load_variables()
# Modules 1 to 14 have a lecture. Module 0 is pre-work and Module 15 is the capstone.
LECTURED = [(key, m) for key, m in g.modules_in_order(V) if 1 <= m["n"] <= 14]
MIN_ACTIVITY_MINUTES = 8


def lecture_minutes(key: str) -> int:
    return g.shape(V, key)["lecture"]


class Plans(unittest.TestCase):
    def test_every_plan_is_complete_and_fills_the_lecture(self):
        for key, m in LECTURED:
            with self.subTest(m["slug"]):
                self.assertEqual(live_plan.problems(m["slug"], lecture_minutes(key)), [])

    def test_activities_are_inside_the_budget(self):
        for _key, m in LECTURED:
            with self.subTest(m["slug"]):
                _, activities = live_plan.totals(live_plan.rows(m["slug"]))
                self.assertGreaterEqual(activities, MIN_ACTIVITY_MINUTES)

    def test_each_lecture_uses_its_demo_in_the_room(self):
        for _key, m in LECTURED:
            lecture = live_plan.read(m["slug"])
            demos = [b for b, kind in lecture["blocks"].items() if kind == "demo"]
            planned = {a["block"] for r in live_plan.rows(m["slug"]) for a in r["activities"]}
            with self.subTest(m["slug"]):
                self.assertTrue(set(demos) & planned or not demos, "the demo is not planned")

    def test_the_lecture_includes_its_generated_table(self):
        for _key, m in LECTURED:
            text = (ROOT / "lectures" / f"{m['slug']}.qmd").read_text(encoding="utf-8")
            with self.subTest(m["slug"]):
                self.assertIn(f"{{{{< include /_includes/live-{m['n']:02d}.md >}}}}", text)
                self.assertNotIn("\n## Timing\n", text)


class Parsing(unittest.TestCase):
    PAGE = """---
live:
  - {section: 1, minutes: 3, activities: [{block: chk-a, minutes: 1}, {block: demo-b, minutes: 1}]}
---

## 1. Only section

::: {#chk-a .self-check}
:::

::: {.demo #demo-b}
:::

::: {.callout-tip}
:::
"""

    def page(self, text: str):
        """Run the test body against one temporary lecture, x.qmd."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        (Path(tmp.name) / "x.qmd").write_text(text, encoding="utf-8")
        patch = mock.patch.object(live_plan, "LECTURES", Path(tmp.name))
        patch.start()
        self.addCleanup(patch.stop)

    def test_ids_are_found_in_either_attribute_order(self):
        self.page(self.PAGE)
        self.assertEqual(live_plan.read("x")["blocks"], {"chk-a": "self-check", "demo-b": "demo"})
        self.assertEqual(live_plan.problems("x", 5), [])

    def test_a_malformed_page_gives_readable_problems_and_errors(self):
        self.page(self.PAGE.replace("{section: 1, minutes: 3,", "{sectoin: 1, minutes: 3,"))
        found = live_plan.problems("x", 5)
        self.assertEqual(len(found), 1)
        self.assertIn("entry 1", found[0])
        for build in (live_plan.table, live_plan.pace_rows):
            with self.assertRaisesRegex(ValueError, "entry 1"):
                build("x")

    def test_a_block_planned_under_another_section_is_reported(self):
        moved = self.PAGE.replace("::: {.demo #demo-b}", "## Summary\n\n::: {.demo #demo-b}")
        self.page(moved)
        self.assertEqual(
            live_plan.problems("x", 5),
            ["x: activity block #demo-b is planned under section 1 but sits in section None"],
        )

    def test_duplicate_ids_are_reported(self):
        self.page(self.PAGE + "\n::: {#chk-a .self-check}\n:::\n")
        self.assertIn(
            "x: more than one activity block has the id #chk-a", live_plan.problems("x", 5)
        )

    def test_callout_titles_and_fenced_examples_are_not_sections_or_blocks(self):
        extra = (
            '\n::: {.callout-note collapse="true"}\n## 2. A numbered callout title\n:::\n'
            "\n```markdown\n## 3. Inside a fence\n::: {#chk-z .self-check}\n:::\n```\n"
        )
        self.page(self.PAGE + extra)
        lecture = live_plan.read("x")
        self.assertEqual(list(lecture["sections"]), [1])
        self.assertNotIn("chk-z", lecture["blocks"])

    def test_a_malformed_entry_is_reported_not_raised(self):
        bad = [
            {"sectoin": 3, "minutes": 5},
            {"section": 4, "minutes": 5, "activities": [{"blok": "x"}]},
            # YAML's true is not a section number, and minutes are never negative.
            {"section": True, "minutes": -5},
            {"section": 5, "minutes": 1, "activities": [{"block": "b", "minutes": 0}]},
        ]
        found = live_plan.shape_problems("x", bad)
        self.assertEqual(
            [f.split(" (")[0] for f in found], [f"x: `live` entry {i}" for i in range(1, 5)]
        )
        self.assertEqual(live_plan.shape_problems("x", [{"section": 1, "minutes": 0}]), [])


class Generated(unittest.TestCase):
    def test_tables_are_current(self):
        for _key, m in LECTURED:
            for name, body in (
                (f"live-{m['n']:02d}.md", live_plan.table(m["slug"])),
                (f"pace-{m['n']:02d}.md", live_plan.pace_rows(m["slug"])),
            ):
                path = ROOT / "_includes" / name
                with self.subTest(name):
                    self.assertTrue(path.exists())
                    self.assertEqual(
                        path.read_text(encoding="utf-8"),
                        f"{g.LIVE_NOTICE.format(slug=m['slug'])}\n\n{body.rstrip()}\n",
                    )


if __name__ == "__main__":
    unittest.main()
