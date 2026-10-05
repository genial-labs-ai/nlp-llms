"""Checks on scripts/lab10_seed_protocol.py, Lab 10's threshold rule, on made-up runs.

Nothing here runs the notebook: the numbers below are invented to test the arithmetic of the rule.
"""

import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import lab10_seed_protocol as protocol  # noqa: E402


def fake_run(seed, gold_gap=0.5, hack_drift=40.0, jitter=0.0):
    """One made-up run in the format the notebook writes."""
    j = jitter * (seed + 1)
    return {
        "seed": seed,
        "hp": {"BETA": 0.1, "LR": 3e-05, "DPO_LR": 5e-06, "CLIP": 1.0},
        "fast": False,
        "offline_test_stand_in": None,
        "device": "cuda",
        "gpu": "Tesla T4",
        "gpu_peak_gb": 3.0,
        "versions": {"torch": "x", "transformers": "y", "cuda": "z"},
        "date": "2026-10-05",
        "check_failures": [],
        "seconds": {"total so far": 400.0},
        "samples": {},
        "policies": {
            "reference": {"score": 0.0, "gold": 0.2, "drift": 0.0, "distinct2": 0.9},
            "rlhf_beta": {"score": 1.0 + j, "gold": 0.8, "drift": 5.0 + j, "distinct2": 0.85},
            "rlhf_beta0": {
                "score": 1.5,
                "gold": 0.8 - gold_gap - j,
                "drift": hack_drift,
                "distinct2": 0.6,
            },
            "dpo": {"score": 0.4, "gold": 0.5, "drift": 3.0, "distinct2": 0.88},
        },
        "heldout_accuracy": {"reference": 0.66, "dpo": 0.62 + j, "gold rule": 0.7},
        "stretch": [
            {"beta": b, "steps": 100, "score": 1.0, "gold": 0.5, "drift": d, "distinct2": 0.8}
            for b, d in ((0.0, 40.0), (0.025, 20.0), (0.1, 5.0), (0.4, 2.0), (1.6, 0.5))
        ]
        + [{"beta": 0.1, "steps": None, "method": "DPO", "score": 0.4, "gold": 0.5, "drift": 3.0}],
    }


class ThresholdRule(unittest.TestCase):
    def test_clear_signature_gives_half_the_smallest_gap(self):
        runs = {f"seed{s}": fake_run(s, jitter=0.01) for s in protocol.SEEDS}
        repeat = fake_run(0, jitter=0.011)
        result = protocol.derive_thresholds(runs, repeat)
        t = result["thresholds"]
        gains = [r["policies"]["rlhf_beta"]["score"] for r in runs.values()]
        self.assertAlmostEqual(t["reward_gain_min"], min(gains) / 2)
        self.assertAlmostEqual(
            t["hack_gold_margin"], 0.5 / 2 + 0.01 / 2
        )  # seed 0 has the smallest gap
        self.assertAlmostEqual(t["dpo_acc_min"], 0.5 + (0.62 + 0.01 - 0.5) / 2)
        self.assertTrue(all(rule["assertable"] for rule in result["rules"].values()))
        self.assertIs(t["sweep_monotone"], True)
        drift_beta_max = max(r["policies"]["rlhf_beta"]["drift"] for r in runs.values())
        self.assertGreater(t["drift_max"], drift_beta_max)
        self.assertLess(t["drift_max"], 40.0)

    def test_one_wrong_sign_blocks_the_assertion(self):
        runs = {f"seed{s}": fake_run(s) for s in protocol.SEEDS}
        runs["seed3"] = fake_run(3, gold_gap=-0.1)  # gold rose without the penalty on one seed
        result = protocol.derive_thresholds(runs, fake_run(0))
        self.assertFalse(result["rules"]["hack_gold_margin"]["assertable"])
        self.assertIsNone(result["thresholds"]["hack_gold_margin"])

    def test_gap_below_twice_the_noise_floor_blocks_the_assertion(self):
        runs = {f"seed{s}": fake_run(s, gold_gap=0.05) for s in protocol.SEEDS}
        repeat = fake_run(0, gold_gap=0.0)  # the repeat of seed 0 differs by 0.05 > half the gap
        result = protocol.derive_thresholds(runs, repeat)
        self.assertFalse(result["rules"]["hack_gold_margin"]["assertable"])

    def test_notebook_has_the_same_threshold_keys(self):
        nb = json.loads((ROOT / "notebooks" / "10-rlhf.ipynb").read_text(encoding="utf-8"))
        source = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
        block = re.search(r"^THRESHOLDS = dict\((.*?)^\)", source, flags=re.S | re.M).group(1)
        keys = set(re.findall(r"^\s+(\w+)=", block, flags=re.M))
        runs = {f"seed{s}": fake_run(s) for s in protocol.SEEDS}
        derived = set(protocol.derive_thresholds(runs, fake_run(0))["thresholds"])
        self.assertEqual(keys, derived)

    def test_summarize_writes_the_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            for s in protocol.SEEDS:
                (out / f"seed{s}.json").write_text(json.dumps(fake_run(s)))
            (out / "seed0-repeat.json").write_text(json.dumps(fake_run(0)))
            self.assertEqual(protocol.main(["summarize", "--out", tmp]), 0)
            summary = json.loads((out / "summary.json").read_text())
            self.assertIn("thresholds", summary)
            self.assertTrue((out / "reward_drift_points.json").exists())


if __name__ == "__main__":
    unittest.main()
