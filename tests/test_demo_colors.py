"""The lecture demos draw with the theme's colors, so they follow the light/dark toggle.

An Observable JS cell names a color as a CSS custom property (`"var(--demo-blue)"`),
never as a hex value, and every property it names is defined on :root in custom.scss,
where custom-dark.scss gives the dark values.
"""

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LECTURES = sorted((ROOT / "lectures").glob("*.qmd"))
OJS = re.compile(r"^```\{ojs\}.*?^```", re.M | re.S)
HEX = re.compile(r"""["'`]#[0-9a-fA-F]{3,8}["'`]""")
USED = re.compile(r"var\((--demo-[\w-]+)\)")
DEFINED = re.compile(r"^\s*(--demo-[\w-]+):", re.M)


def ojs_cells(path: Path) -> list[str]:
    return OJS.findall(path.read_text(encoding="utf-8"))


class DemoColors(unittest.TestCase):
    def test_lectures_have_demos(self):
        self.assertTrue(any(ojs_cells(p) for p in LECTURES))

    def test_no_hex_colors_in_demo_code(self):
        for path in LECTURES:
            for cell in ojs_cells(path):
                with self.subTest(path.name):
                    self.assertEqual(HEX.findall(cell), [], "use a --demo-* custom property")

    def test_every_property_used_is_defined(self):
        defined = set(DEFINED.findall((ROOT / "custom.scss").read_text(encoding="utf-8")))
        for path in LECTURES:
            used = {name for cell in ojs_cells(path) for name in USED.findall(cell)}
            with self.subTest(path.name):
                self.assertEqual(used - defined, set())


if __name__ == "__main__":
    unittest.main()
