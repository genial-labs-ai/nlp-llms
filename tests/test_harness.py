"""The generated exercise harness (scripts/harness.py), run in a real IPython shell."""

import json
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
        import builtins

        builtins.print = self._print

    _print = print

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

    def test_a_check_on_provided_code_names_no_owner_and_no_reference(self):
        self.cell(SOURCE)
        self.cell("def double(x):\n    raise NotImplementedError('TODO 1')")
        result = self.cell("workshop.checkpoint(label='uses 1')\ndouble(2)")
        self.assertFalse(result.success)
        self.assertEqual(self.ip.user_ns["workshop"].results["uses 1"], (False, None))

    def test_a_skipped_todo_is_not_replaced_by_the_reference(self):
        self.cell(SOURCE)
        self.cell("@workshop.solution(1)\ndef double(x):\n    return 2 * x")
        result = self.cell("double(2)")
        self.assertIsInstance(result.error_in_exec, NotImplementedError)
        self.assertIn("TODO 1", str(result.error_in_exec))

    def test_unticking_worked_example_restores_your_code(self):
        self.cell(SOURCE)
        self.cell("def double(x):\n    return x + x + 1")
        self.cell("WORKED_EXAMPLE = True")
        self.cell("@workshop.solution(1)\ndef double(x):\n    return 2 * x")
        self.assertEqual(self.ip.user_ns["double"](3), 6)
        self.cell("WORKED_EXAMPLE = False")
        self.cell("@workshop.solution(1)\ndef double(x):\n    return 2 * x")
        self.assertEqual(self.ip.user_ns["double"](3), 7)

    def test_a_reference_bound_long_ago_is_still_recognized(self):
        self.cell(SOURCE)
        self.cell("def double(x):\n    return x + x + 1")
        self.cell("@workshop.solution(1)\ndef double(x):\n    return 2 * x")
        self.cell("workshop.use_reference(1)")
        for _ in range(5):
            self.cell("@workshop.solution(1)\ndef double(x):\n    return 2 * x")
        self.cell("workshop.checkpoint(1)\nassert double(2) == 4")
        self.assertEqual(self.ip.user_ns["workshop"].results["1"], (True, "the REFERENCE solution"))

    def test_the_run_record_has_a_date_and_passes_on_checkpoints(self):
        self.cell(SOURCE)
        self.cell("workshop.checkpoint(label='x')\nassert False")
        self.cell(SOURCE)  # Run all reruns the harness first, so the run starts clean
        self.cell("workshop.checkpoint(label='x')\nassert True")
        self.cell(
            "import io, contextlib, json\n_buf = io.StringIO()\n"
            "with contextlib.redirect_stdout(_buf):\n    workshop.run_record()"
        )
        text = self.ip.user_ns["_buf"].getvalue()
        record = json.loads(text[text.index("{") : text.rindex("}") + 1])
        self.assertEqual(record["status"], "pass")
        self.assertRegex(record["date"], r"^\d{4}-\d{2}-\d{2}$")
        self.assertNotIn("NLP_LLMS_DATA", record["settings"])

    def record(self):
        self.cell(
            "import io, contextlib\n_buf = io.StringIO()\n"
            "with contextlib.redirect_stdout(_buf):\n    workshop.run_record()"
        )
        text = self.ip.user_ns["_buf"].getvalue()
        return json.loads(text[text.index("{") : text.rindex("}") + 1])

    def test_a_fallback_printed_by_a_lab_is_in_the_record(self):
        self.cell(SOURCE)
        self.cell("print('Could not load Qwen; USING StubProvider')")
        self.cell("workshop.checkpoint(label='x')\nassert True")
        record = self.record()
        self.assertIn("USING StubProvider", record["fallbacks"])

    def test_a_run_without_checkpoints_or_with_errors_does_not_pass(self):
        self.cell(SOURCE)
        self.assertEqual(self.record()["status"], "fail")
        self.cell("workshop.checkpoint(label='x')\nassert True")
        self.cell("raise ValueError('a provided cell failed')")
        self.assertEqual(self.record()["status"], "fail")

    def test_a_plain_scalar_cannot_be_a_solution_value(self):
        self.cell(SOURCE)
        result = self.cell("K = workshop.solution_value(1, 'K', 5)")
        self.assertIsInstance(result.error_in_exec, TypeError)

    def test_rerunning_the_harness_starts_a_new_run(self):
        self.cell(SOURCE)
        self.cell("workshop.checkpoint(label='x')\nassert False")
        self.cell(SOURCE)
        self.assertEqual(self.ip.user_ns["workshop"].results, {})


if __name__ == "__main__":
    unittest.main()
