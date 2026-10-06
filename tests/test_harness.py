"""The generated exercise harness (scripts/harness.py), run in a real IPython shell."""

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import harness  # noqa: E402

try:
    from IPython.core.interactiveshell import InteractiveShell
except ImportError:  # the render job's test environment has no IPython
    InteractiveShell = None

SOURCE = harness.harness_source("toy", "0" * 16, {1: ("double",)})


@unittest.skipIf(InteractiveShell is None, "IPython is not installed")
class Harness(unittest.TestCase):
    def setUp(self):
        os.environ.pop("NLP_LLMS_WORKED", None)
        InteractiveShell.clear_instance()
        self.ip = InteractiveShell.instance(history_length=0)
        self.calls = []
        self.other = lambda *args: self.calls.append(args)
        self.ip.events.register("post_run_cell", self.other)

    def tearDown(self):
        InteractiveShell.clear_instance()

    def cell(self, code):
        return self.ip.run_cell(code, store_history=False)

    def test_other_hooks_survive_the_harness_and_its_reruns(self):
        self.cell(SOURCE)
        self.cell(SOURCE)
        callbacks = self.ip.events.callbacks["post_run_cell"]
        self.assertIn(self.other, callbacks)
        ours = [c for c in callbacks if type(getattr(c, "__self__", None)).__name__ == "_Workshop"]
        self.assertEqual(len(ours), 1)

    def test_a_solution_does_not_replace_your_code(self):
        self.cell(SOURCE)
        self.cell("def double(x):\n    return x + x + 1")
        self.cell("@workshop.solution(1)\ndef double(x):\n    return 2 * x")
        self.assertEqual(self.ip.user_ns["double"](3), 7)
        self.cell("workshop.use_reference(1)")
        self.assertEqual(self.ip.user_ns["double"](3), 6)

    def test_worked_mode_binds_the_reference(self):
        os.environ["NLP_LLMS_WORKED"] = "1"
        try:
            self.cell(SOURCE)
            self.cell("def double(x):\n    raise NotImplementedError('TODO 1')")
            self.cell("@workshop.solution(1)\ndef double(x):\n    return 2 * x")
            self.assertEqual(self.ip.user_ns["double"](3), 6)
        finally:
            os.environ.pop("NLP_LLMS_WORKED", None)

    def test_checkpoints_say_whose_code_they_checked(self):
        self.cell(SOURCE)
        self.cell("def double(x):\n    return 2 * x")
        self.cell("@workshop.solution(1)\ndef double(x):\n    return 2 * x")
        self.cell("workshop.checkpoint(1)\nassert double(2) == 4")
        self.assertEqual(self.ip.user_ns["workshop"].results["1"], (True, "your code"))
        self.cell("workshop.use_reference(1)")
        self.cell("workshop.checkpoint(1, label='1b')\nassert double(2) == 4")
        self.assertEqual(
            self.ip.user_ns["workshop"].results["1b"], (True, "the REFERENCE solution")
        )

    def test_rerunning_the_harness_starts_a_new_run(self):
        self.cell(SOURCE)
        self.cell("workshop.checkpoint(label='x')\nassert False")
        self.cell(SOURCE)
        self.assertEqual(self.ip.user_ns["workshop"].results, {})


if __name__ == "__main__":
    unittest.main()
