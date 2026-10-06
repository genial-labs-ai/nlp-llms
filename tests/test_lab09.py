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


DATASETS = yaml.safe_load((ROOT / "_variables.yml").read_text(encoding="utf-8"))["datasets"]
LAB09_KEYS = {
    "prompts": "lab09_prompts",
    "preferences": "lab09_preferences",
    "reward_model": "lab09_reward_model",
}  # notebook key -> _variables.yml datasets key


def notebook_code(slug: str = "09-preference-learning") -> str:
    path = ROOT / "notebooks" / f"{slug}.ipynb"
    nb = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")


def restated_gold():
    """GOLD and gold_reward exactly as the brief's interface states them."""
    namespace = {"re": re}
    exec(BLOCKS[2], namespace)  # noqa: S102
    return namespace["GOLD"], namespace["gold_reward"]


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


class CommittedFiles(unittest.TestCase):
    """The committed Lab 9 files, the hashes both notebooks pin and _variables.yml agree."""

    def test_both_notebooks_pin_the_recorded_hashes(self):
        lab09, lab10 = notebook_code(), notebook_code("10-rlhf")
        for key, dataset in LAB09_KEYS.items():
            sha = DATASETS[dataset]["sha256"]
            with self.subTest(file=dataset):
                self.assertIn(f'"{sha}"', lab10)
                if key != "reward_model":  # Lab 9 writes the reward model; it loads only the data
                    self.assertIn(f'"{sha}"', lab09)

    def test_files_follow_the_restated_gold_rule(self):
        gold, gold_reward = restated_gold()
        doc = json.loads((ROOT / "data" / "lab09_prompts.json").read_text(encoding="utf-8"))
        self.assertEqual(doc["gold"], gold)
        self.assertFalse(doc["build"]["stand_in"])
        self.assertEqual(doc["model"], MODELS["causal_lm"])
        self.assertEqual(doc["revision"], MODELS["causal_lm_revision"])
        blob = (ROOT / "data" / "lab09_preferences.jsonl.gz").read_bytes()
        pairs = [json.loads(line) for line in gzip.decompress(blob).decode("utf-8").splitlines()]
        self.assertEqual(len(pairs), sum(doc["build"]["pairs"].values()))
        for p in pairs:
            self.assertEqual(gold_reward(p["prompt"], p["chosen"]), p["gold_chosen"])
            self.assertEqual(gold_reward(p["prompt"], p["rejected"]), p["gold_rejected"])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is not installed")
    def test_reward_model_matches_the_data(self):
        import hashlib

        import torch

        ckpt = torch.load(ROOT / "data" / "lab09_reward_model.pt", weights_only=True)
        self.assertEqual(ckpt["gold"], restated_gold()[0])
        for key in ("prompts", "preferences"):
            self.assertEqual(ckpt["data_sha256"][key], DATASETS[LAB09_KEYS[key]]["sha256"], key)
        self.assertEqual(ckpt["policy"]["revision"], MODELS["causal_lm_revision"])
        path = ROOT / "data" / "lab09_reward_model.pt"
        self.assertEqual(
            hashlib.sha256(path.read_bytes()).hexdigest(), DATASETS["lab09_reward_model"]["sha256"]
        )


def literal_assignments(source: str, names: set[str]) -> dict:
    """{name: value} for every `name = <literal>` in source, at any depth."""
    import ast

    out = {}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in names:
                try:
                    out[target.id] = ast.literal_eval(node.value)
                except ValueError:
                    pass
    return out


def notebook_cells(slug: str) -> list[str]:
    nb = json.loads((ROOT / "notebooks" / f"{slug}.ipynb").read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


class Frames(unittest.TestCase):
    """The 16 subjects and 16 frames are written out in the builder and again in the offline
    stand-ins of Labs 9 and 10; all three must equal what the committed data were built from."""

    def test_stand_ins_use_the_committed_frames(self):
        prompts = json.loads((ROOT / "data" / "lab09_prompts.json").read_text(encoding="utf-8"))
        built = literal_assignments(BUILD.read_text(encoding="utf-8"), {"SUBJECTS", "FRAMES"})
        self.assertEqual(built["SUBJECTS"], prompts["subjects"])
        self.assertEqual(built["FRAMES"], prompts["frames"])
        for slug in ("09-preference-learning", "10-rlhf"):
            stand_ins = []  # the cells that define both, which is the stand-in builder
            for cell in notebook_cells(slug):
                try:
                    found = literal_assignments(cell, {"subjects", "frames"})
                except SyntaxError:  # a cell with a Colab shell command
                    continue
                if {"subjects", "frames"} <= set(found):
                    stand_ins.append(found)
            with self.subTest(slug):
                self.assertEqual(len(stand_ins), 1)
                self.assertEqual(stand_ins[0]["subjects"], prompts["subjects"])
                self.assertEqual(stand_ins[0]["frames"], prompts["frames"])


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
