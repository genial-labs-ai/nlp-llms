"""Lab 9: the Lab 9 -> Lab 10 interface is restated verbatim, and the data builder works.

The interface section of briefs/09-preference-learning.md is the contract between Lab 9,
Lab 10 and data/build_lab09_preferences.py. These checks fail when the notebook or the
build script drifts from it. The builder test runs only where torch is installed (the
`test` dependency group has no torch) and uses the offline stand-in, never the Hub.
"""

import gzip
import importlib.util
import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
BRIEF = (ROOT / "briefs" / "09-preference-learning.md").read_text(encoding="utf-8")
INTERFACE = BRIEF.split("## Lab 9 → Lab 10 interface", 1)[1].split("\n## ", 1)[0]
BLOCKS = re.findall(r"```python\n(.*?)```", INTERFACE, flags=re.DOTALL)
BUILD = ROOT / "data" / "build_lab09_preferences.py"
MODELS = yaml.safe_load((ROOT / "_variables.yml").read_text(encoding="utf-8"))["models"]


def notebook_code() -> str:
    path = ROOT / "notebooks" / "09-preference-learning.ipynb"
    nb = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")


class Interface(unittest.TestCase):
    def test_brief_has_the_expected_blocks(self):
        # constants, function signatures, gold rule, reward-model class, checkpoint layout
        self.assertEqual(len(BLOCKS), 5)

    def test_constants_and_gold_rule_are_verbatim(self):
        code, build = notebook_code(), BUILD.read_text(encoding="utf-8")
        for block in (BLOCKS[0], BLOCKS[2]):
            self.assertIn(block.strip(), code)
            self.assertIn(block.strip(), build)

    def test_function_and_class_signatures(self):
        code = notebook_code()
        signatures = re.findall(r"^(def \w+\(.*?\):)", BLOCKS[1], flags=re.MULTILINE)
        self.assertEqual(len(signatures), 6)
        signatures += re.findall(r"^\s*(def \w+\(.*?\):)", BLOCKS[3], flags=re.MULTILINE)
        signatures.append("class RewardModel(nn.Module):")
        for signature in signatures:
            with self.subTest(signature=signature):
                self.assertIn(signature, code)

    def test_checkpoint_format(self):
        code = notebook_code()
        self.assertIn('"format": "nlp-llms/lab09-reward-model", "version": 2', code)
        for key in ("heldout_acc", "oracle_acc", "acc_gold_order_untied", "spearman_gold"):
            self.assertIn(f'"{key}"', code)

    def test_build_script_uses_the_pinned_model(self):
        match = re.search(r'^MODEL = "([^"]+)"', BUILD.read_text(encoding="utf-8"), re.MULTILINE)
        self.assertEqual(match.group(1), MODELS["causal_lm"])


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch is not installed")
class OfflineBuild(unittest.TestCase):
    def run_build(self, *args):
        return subprocess.run(
            [sys.executable, str(BUILD), "--offline-tiny", *args],
            capture_output=True,
            text=True,
            timeout=300,
        )

    def test_stand_in_files_follow_the_interface(self):
        with tempfile.TemporaryDirectory() as out:
            result = self.run_build("--out", out, "--train-pairs", "800", "--heldout-pairs", "100")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            doc = json.loads((Path(out) / "lab09_prompts.json").read_text(encoding="utf-8"))
            self.assertTrue(doc["build"]["stand_in"])
            self.assertEqual(
                [len(doc["prompts"][s]) for s in ("train", "heldout", "eval")], [160, 32, 64]
            )
            blob = (Path(out) / "lab09_preferences.jsonl.gz").read_bytes()
            pairs = [json.loads(line) for line in gzip.decompress(blob).decode().splitlines()]
            self.assertEqual(len(pairs), 900)
            self.assertTrue(
                all(len(p["chosen_ids"]) == 24 and len(p["prompt_ids"]) == 8 for p in pairs)
            )

    def test_stand_in_is_refused_inside_data(self):
        result = self.run_build()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("outside data/", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
