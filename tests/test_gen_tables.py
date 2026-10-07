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
            "schedule.md": g.schedule(V),
            "module-shape.md": g.module_shape(V),
            "welcome-days.md": g.welcome_days(V),
            "welcome-clocks.md": g.welcome_clocks(V),
            "welcome-data.md": g.welcome_data(V),
            "notebooks.md": g.notebooks_index(V),
        }
        for d in g.days_in_order(V):
            expected[f"day-{d['n']}.md"] = g.day_cards(V, d)
            expected[f"run-{d['n']}.md"] = g.run_sheet(V, d)
        for key, m in g.modules_in_order(V):
            expected[f"module-{m['n']:02d}.md"] = g.module_block(V, key, m)
        for name, body in expected.items():
            self.assertEqual(committed(name), generated(body), name)

    def test_run_sheets(self):
        # Every fallback kind has a short form, and each day page shows its run sheet.
        self.assertEqual(set(g.FALLBACK_SHORT), set(V["readiness"]["fallback_kinds"]))
        for d in g.days_in_order(V):
            page = (ROOT / f"day-{d['n']}.qmd").read_text(encoding="utf-8")
            sheet = g.run_sheet(V, d)
            with self.subTest(day=d["n"]):
                self.assertIn(f"{{{{< include /_includes/run-{d['n']}.md >}}}}", page)
                for _key, m in g.day_modules(V, d):
                    self.assertIn(f"lectures/{m['slug']}.qmd", sheet)

    def test_sidebar_is_current(self):
        self.assertEqual(committed("sidebar.yml"), f"{g.YAML_NOTICE}\n{g.sidebar_yaml(V)}")


class Sidebar(unittest.TestCase):
    def test_lists_every_lecture_once_in_module_order(self):
        sidebar = yaml.safe_load(g.sidebar_yaml(V))["website"]["sidebar"][0]
        sections = [s["section"] for s in sidebar["contents"]]
        # Pre-work (Module 0) first, then one section per day.
        self.assertEqual(sections[0], g.PREWORK)
        self.assertEqual(sections[1:], [f"Day {d['n']} · {d['short']}" for d in g.days_in_order(V)])
        self.assertEqual(len(sections) - 1, V["workshop"]["days"])
        hrefs = [item["href"] for section in sidebar["contents"] for item in section["contents"]]
        self.assertEqual(hrefs, [f"lectures/{m['slug']}.qmd" for _, m in g.modules_in_order(V)])
        for href in hrefs:
            self.assertTrue((ROOT / href).exists(), href)


def clock_named(name: str) -> dict:
    return V["schedule"]["clocks"][name]


class Clock(unittest.TestCase):
    def test_units(self):
        # Day 1 (standard): four one-slot units. Days 2-5 (long): the middle unit is
        # a lecture slot before lunch and a lab slot after it.
        self.assertEqual([len(u) for u in g.units(clock_named("standard"))], [1, 1, 1, 1])
        long_units = g.units(clock_named("long"))
        self.assertEqual([len(u) for u in long_units], [1, 2, 1])
        self.assertEqual([s.get("part") for s in long_units[1]], ["lecture", "lab"])

    def test_units_reject_an_unclosed_or_orphan_part(self):
        lecture = {"start": "09:00", "end": "10:00", "kind": "module", "part": "lecture"}
        lab = {"start": "10:00", "end": "11:00", "kind": "module", "part": "lab"}
        whole = {"start": "11:00", "end": "12:00", "kind": "module"}
        for slots in ([lecture], [lab], [lecture, whole, lab]):
            with self.assertRaises(ValueError):
                g.units({"slots": slots})

    def test_every_unit_fits_its_clock_shape(self):
        for name, clk in V["schedule"]["clocks"].items():
            total = sum(clk["shape"].values())
            for unit in g.units(clk):
                self.assertEqual(sum(g.span(s) for s in unit), total, (name, unit))
                for s in unit:
                    # The parts that fall in a slot fill it exactly.
                    self.assertEqual(
                        sum(m for _, _, m in g.segments(clk["shape"], s)), g.span(s), (name, s)
                    )
                if len(unit) == 2:
                    self.assertEqual(g.span(unit[0]), clk["shape"]["lecture"], name)

    def test_placements_match_units_one_to_one(self):
        for d in g.days_in_order(V):
            self.assertEqual(len(d["slots"]), len(g.units(g.clock_of(V, d))), d["n"])
            self.assertEqual(len(g.placements(V, d)), len(d["slots"]), d["n"])
        broken = dict(V["days"]["d2"], slots=V["days"]["d2"]["slots"][:-1])
        with self.assertRaises(ValueError):
            g.placements(V, broken)

    def test_module_clock(self):
        self.assertEqual(g.module_clock(V, "m00"), "08:00–09:00")  # the Day 1 clinic
        self.assertEqual(g.module_clock(V, "m01"), "09:10–10:45")
        self.assertEqual(g.module_clock(V, "m05"), "09:15–11:15")
        # Module B of a long day spans lunch.
        self.assertEqual(g.module_clock(V, "m06"), "11:30–12:25 · 13:25–14:30")
        self.assertEqual(g.module_clock(V, "m07"), "14:45–16:45")
        # The capstone fills the last two units of Day 5.
        self.assertEqual(g.module_clock(V, "m15"), "11:30–12:25 · 13:25–14:30 · 14:45–16:45")

    def test_minutes_and_timing(self):
        self.assertEqual(g.minutes_of(V, "m00"), 0)  # pre-work has no module slot
        self.assertEqual(g.minutes_of(V, "m01"), 95)
        self.assertEqual(g.minutes_of(V, "m06"), 120)
        self.assertEqual(g.minutes_of(V, "m15"), 240)
        self.assertEqual(g.timing(V, "m04", V["modules"]["m04"]), "95 minutes (45 lecture, 50 lab)")
        self.assertEqual(
            g.timing(V, "m08", V["modules"]["m08"]), "120 minutes (55 lecture, 55 lab, 10 debrief)"
        )
        self.assertEqual(g.timing(V, "m15", V["modules"]["m15"]), "240 minutes")
        self.assertIsNone(g.shape(V, "m15"))
        self.assertIsNone(g.shape(V, "m00"))
        self.assertIn("pre-work", g.timing(V, "m00", V["modules"]["m00"]))

    def test_schedule_bars_fill_their_row(self):
        # In every timetable cell with a lecture/lab/debrief bar, the first part starts at the
        # row's start, each part starts where the last ended, and the last ends at the row's end.
        hidden = r"(?:\{\.visually-hidden\}\])?"  # the debrief's label is visually hidden
        part = re.compile(
            r"(Lecture|Lab|Debrief) (\d\d:\d\d)\]" + hidden + r"\{[^}]*flex-grow: (\d+)"
        )
        table = g.schedule(V)
        checked = 0
        for row in re.findall(r"^\| (\d\d:\d\d)–(\d\d:\d\d) \|(.*)\|$", table, re.MULTILINE):
            start, end, cells = row
            for cell in cells.split(" | "):
                parts = part.findall(cell)
                if not parts:
                    continue
                t = g.to_minutes(start)
                for _, at, minutes in parts:
                    self.assertEqual(g.to_minutes(at), t, cell)
                    t += int(minutes)
                self.assertEqual(t, g.to_minutes(end), cell)
                checked += 1
        # One bar per clock slot of every module with a lecture/lab shape (not the capstone).
        expected = sum(
            len(p["slots"])
            for d in g.days_in_order(V)
            for p in g.placements(V, d)
            if g.shape(V, p["key"])
        )
        self.assertEqual(checked, expected)
        # Day 1: 4. Days 2 to 4: 4 each (module B has two rows). Day 5: 1 (Module 14).
        self.assertEqual(expected, 17)
        self.assertEqual(table.count("{.slot-closing}"), 4)
        self.assertEqual(table.count("{.slot-clinic}"), 1)
        self.assertEqual(table.count("::: {.table-wide .timetable"), len(g.clock_groups(V)))

    def test_one_grid_per_run_of_days_on_a_clock(self):
        groups = [(name, [d["n"] for d in days]) for name, days in g.clock_groups(V)]
        self.assertEqual(groups, [("standard", [1]), ("long", [2, 3, 4, 5])])
        self.assertIn("## Day 1 · 95-minute modules", g.schedule(V))
        self.assertIn("## Days 2–5 · 120-minute modules", g.schedule(V))

    def test_facts_take_their_ranges_from_the_clocks(self):
        facts = g.facts_strip(V)
        self.assertIn(f"**{V['workshop']['days']}** days", facts)
        self.assertIn("**45–55** min lectures, **50–55** min labs", facts)


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
            target = "/setup.qmd#module-0" if g.is_prework(m) else f"/day-{m['day']}.qmd"
            self.assertIn(f"]({target})", block)
            for objective in m["objectives"]:
                self.assertIn(f"- {objective}", block)

    def test_only_modules_with_a_notebook_get_a_colab_link(self):
        index = g.notebooks_index(V)
        for m in V["modules"].values():
            link = f"{V['repo']['colab_base']}/{m['slug']}.ipynb"
            self.assertEqual(link in index, g.has_notebook(m), m["slug"])

    def test_fenced_divs_are_balanced(self):
        blocks = [g.facts_strip(V), g.days_cards(V), g.path_steps(V), g.notebooks_index(V)]
        blocks += [g.schedule(V)]
        blocks += [g.day_cards(V, d) for d in g.days_in_order(V)]
        blocks += [g.module_block(V, key, m) for key, m in g.modules_in_order(V)]
        for block in blocks:
            lines = block.splitlines()
            opens = sum(1 for line in lines if line.startswith("::: {"))
            closes = sum(1 for line in lines if line == ":::")
            self.assertEqual(opens, closes, block[:60])


if __name__ == "__main__":
    unittest.main()
