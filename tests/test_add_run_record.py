"""scripts/add_run_record.py: machine and path classes from a pasted run record."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import add_run_record as arr  # noqa: E402
import run_records  # noqa: E402


class Classes(unittest.TestCase):
    def test_colab_machines(self):
        base = {"colab_release": "release-colab_20261001"}
        self.assertEqual(arr.env_of({**base, "gpu": "Tesla T4"}, None), "colab-t4")
        self.assertEqual(arr.env_of({**base, "gpu": None}, None), "colab-cpu")
        with self.assertRaises(SystemExit):
            arr.env_of({**base, "gpu": "NVIDIA L4"}, None)
        with self.assertRaises(SystemExit):
            arr.env_of({"gpu": "Tesla T4"}, None)

    def test_every_offline_flag_makes_an_offline_run(self):
        for flag in run_records.OFFLINE_FLAGS:
            self.assertEqual(arr.path_of({"settings": {flag: "1"}}, None), "offline", flag)
        self.assertEqual(arr.path_of({"settings": {}, "provider": "open"}, None), "open")
        self.assertEqual(arr.path_of({"settings": {}, "provider": "anthropic"}, None), "keyed")
        fell_back = {"settings": {}, "provider": "open", "fallbacks": ["USING StubProvider"]}
        self.assertEqual(arr.path_of(fell_back, None), "offline")


if __name__ == "__main__":
    unittest.main()
