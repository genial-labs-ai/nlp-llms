"""The intro slides (welcome.qmd): rendered, linked, and themed with the site's colors."""

import re
import sys
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import gen_tables as g  # noqa: E402

V = yaml.safe_load((ROOT / "_variables.yml").read_text(encoding="utf-8"))
DECK = (ROOT / "welcome.qmd").read_text(encoding="utf-8")
FRONT = yaml.safe_load(DECK.split("---")[1])


def tokens(path: Path) -> dict[str, str]:
    """$nl-name: #hex pairs in an SCSS file's defaults layer."""
    text = path.read_text(encoding="utf-8").split("/*-- scss:rules --*/")[0]
    return dict(re.findall(r"^\$(nl-[a-z-]+):\s*(#[0-9a-f]{6})", text, flags=re.M))


class Deck(unittest.TestCase):
    def test_is_a_revealjs_deck_in_the_render_list(self):
        self.assertIn("revealjs", FRONT["format"])
        quarto = yaml.safe_load((ROOT / "_quarto.yml").read_text(encoding="utf-8"))
        self.assertIn("welcome.qmd", quarto["project"]["render"])

    def test_uses_the_generated_includes(self):
        for name in ("welcome-days.md", "welcome-clocks.md", "welcome-data.md", "module-shape.md"):
            self.assertIn(f"{{{{< include /_includes/{name} >}}}}", DECK, name)
            self.assertTrue((ROOT / "_includes" / name).exists(), name)

    def test_theme_colors_follow_custom_scss(self):
        site = tokens(ROOT / "custom.scss")
        deck = tokens(ROOT / "slides.scss")
        self.assertTrue(deck)
        for name, value in deck.items():
            self.assertEqual(value, site.get(name), name)

    def test_title_slide_band_is_the_navy_token(self):
        navy = tokens(ROOT / "custom.scss")["nl-navy"]
        self.assertEqual(FRONT["title-slide-attributes"]["data-background-color"], navy)

    def test_schedule_links_the_opening_slot_to_the_deck(self):
        self.assertEqual(V["days"]["d1"]["opening_page"], "welcome.qmd")
        self.assertIn("[Welcome, setup check](welcome.qmd)", g.schedule(V))

    def test_pages_link_to_the_deck(self):
        for page in ("index.qmd", "prepare.qmd", "schedule.qmd", "day-1.qmd", "teach.qmd"):
            self.assertIn("(welcome.qmd)", (ROOT / page).read_text(encoding="utf-8"), page)


if __name__ == "__main__":
    unittest.main()
