"""Checks on data/baselines.json, the measured baselines that later labs cite."""

import json
import re
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DOC = json.loads((ROOT / "data" / "baselines.json").read_text(encoding="utf-8"))
V = yaml.safe_load((ROOT / "_variables.yml").read_text(encoding="utf-8"))
FIELDS = {"id", "notebook", "dataset", "split", "metrics", "settings", "date"}


class Baselines(unittest.TestCase):
    def test_entries_are_complete(self):
        ids = [b["id"] for b in DOC["baselines"]]
        self.assertEqual(len(ids), len(set(ids)), "duplicate id")
        for b in DOC["baselines"]:
            self.assertEqual(FIELDS - set(b), set(), b["id"])
            self.assertTrue((ROOT / b["notebook"]).exists(), b["notebook"])
            self.assertIn(b["dataset"], V["datasets"], b["id"])
            self.assertIn(b["split"], {"train", "val", "test"}, b["id"])
            self.assertRegex(b["date"], r"^\d{4}-\d{2}-\d{2}$", b["id"])
            self.assertTrue(b["metrics"], b["id"])

    def test_lab_1_checkpoint_matches_the_file(self):
        """The Lab 1 notebook asserts the recorded character n-gram values."""
        source = (ROOT / "notebooks" / "01-text-as-data.ipynb").read_text(encoding="utf-8")
        for b in DOC["baselines"]:
            if b["id"].startswith("lab01.char_ngram."):
                expected = f"{b['settings']['n']}: {b['metrics']['nats_per_char']}"
                self.assertIn(expected, source, b["id"])

    def test_lab_2_checkpoints_match_the_file(self):
        """Lab 2 asserts Lab 1's recorded TF-IDF row, and its recorded SGNS row clears its floor."""
        source = (ROOT / "notebooks" / "02-word-vectors.ipynb").read_text(encoding="utf-8")
        by_id = {b["id"]: b for b in DOC["baselines"]}
        lab1 = by_id["lab01.tfidf_logreg"]["metrics"]
        for metric in ("accuracy", "macro_f1"):
            self.assertIn(f'tfidf_test[\\"{metric}\\"] - {lab1[metric]}', source, metric)
        floor = float(re.search(r"ACCURACY_FLOOR = ([0-9.]+)", source).group(1))
        self.assertGreaterEqual(by_id["lab02.sgns_avg_ffn"]["metrics"]["accuracy"], floor)


if __name__ == "__main__":
    unittest.main()
