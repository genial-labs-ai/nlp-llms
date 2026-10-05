"""No page, notebook or script may point participants at a lookalike TypeSafe package.

The official names are in _variables.yml `packages`; see briefs/jev-verification.md.
"""

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Unaffiliated or unregistered names that a participant could install by mistake.
LOOKALIKE = re.compile(
    r"pip install[^\n]*\b(typesafe-ai|typesafe-sdk-python|llama-index-jev|"
    r"llama-index-postprocessor-jev|typesafe-client|jev)\b"
)
SOURCES = ["*.qmd", "lectures/*.qmd", "notebooks/*.ipynb", "scripts/*.py", "README.md"]


class PackageNames(unittest.TestCase):
    def test_no_lookalike_installs(self):
        for pattern in SOURCES:
            for path in sorted(ROOT.glob(pattern)):
                text = path.read_text(encoding="utf-8")
                with self.subTest(path=str(path.relative_to(ROOT))):
                    self.assertIsNone(LOOKALIKE.search(text))

    def test_pattern_catches_the_lookalikes(self):
        self.assertIsNotNone(LOOKALIKE.search("%pip install -q typesafe-ai==0.1.0"))
        self.assertIsNotNone(LOOKALIKE.search("pip install jev"))
        self.assertIsNone(LOOKALIKE.search("%pip install -q typesafe-sdk==0.7.2"))


if __name__ == "__main__":
    unittest.main()
