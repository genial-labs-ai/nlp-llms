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

    def test_ids_are_found_in_either_attribute_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "x.qmd").write_text(self.PAGE, encoding="utf-8")
            with mock.patch.object(live_plan, "LECTURES", Path(tmp)):
                self.assertEqual(
                    live_plan.read("x")["blocks"], {"chk-a": "self-check", "demo-b": "demo"}
                )
                self.assertEqual(live_plan.problems("x", 5), [])

    def test_a_malformed_entry_is_reported_not_raised(self):
        bad = [
            {"sectoin": 3, "minutes": 5},
            {"section": 4, "minutes": 5, "activities": [{"blok": "x"}]},
        ]
        found = live_plan.shape_problems("x", bad)
        self.assertEqual(len(found), 2)
        self.assertIn("entry 1", found[0])
        self.assertIn("entry 2", found[1])


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
