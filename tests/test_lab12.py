"""Checks on notebooks/12-rlcd-jev.ipynb.

Static checks (standard library and PyYAML): the policy text and route descriptions are the
decision set's; the data-loading cells are data/README.md's, unchanged; the calibration helpers
restated from Lab 11 are word for word Lab 11's (one documented deviation); the recorded toy-model
values agree with data/baselines.json; the honesty labels and banners are present; no TypeSafe
client is built without a key.

Behavior of the local decider (needs torch, typesafe-sdk 0.7.2, scipy and matplotlib; skipped
otherwise): the notebook's own cells are executed and LocalDecider is held to the behavior table of
briefs/12-rlcd-jev.md, including UnsupportedQuestion for questions the toy model was not trained on.
"""

import ast
import asyncio
import importlib.util
import json
import os
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
NOTEBOOK = ROOT / "notebooks" / "12-rlcd-jev.ipynb"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


bd = _load("build_decisions", DATA / "build_decisions.py")
NB = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
CODE_CELLS = ["".join(c["source"]) for c in NB["cells"] if c["cell_type"] == "code"]
CODE = "\n".join(CODE_CELLS)
TEXT = "\n".join("".join(c["source"]) for c in NB["cells"])
BASELINES = {
    b["id"]: b
    for b in json.loads((DATA / "baselines.json").read_text(encoding="utf-8"))["baselines"]
}
STATS = json.loads(bd.STATS.read_text(encoding="utf-8"))


LAB11_NOTEBOOK = ROOT / "notebooks" / "11-calibration.ipynb"
LAB11_RESTATED = (
    "log_softmax",
    "reliability_bins",
    "ece",
    "brier",
    "brier_binary",
    "noise_floor",
    "risk_coverage",
    "plot_reliability",
)


def _functions(notebook: dict) -> dict:
    """Name -> source of the last top-level definition in a notebook's code cells (a solution
    cell follows its stub, so the solution wins). IPython magics are skipped."""
    out = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        lines = "".join(cell["source"]).splitlines()
        src = "\n".join(line for line in lines if not line.lstrip().startswith(("%", "!")))
        try:
            flags = ast.PyCF_ONLY_AST | ast.PyCF_ALLOW_TOP_LEVEL_AWAIT
            tree = compile(src, "cell", "exec", flags=flags)
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                out[node.name] = ast.get_source_segment(src, node)
    return out


def cell_with(marker: str) -> str:
    matches = [c for c in CODE_CELLS if marker in c]
    assert len(matches) == 1, f"expected one code cell containing {marker!r}, found {len(matches)}"
    return matches[0]


class Restatements(unittest.TestCase):
    def test_policy_verbatim(self):
        match = re.search(r'^POLICY_TEXT = """\\\n(.*?)"""', CODE, flags=re.DOTALL | re.MULTILINE)
        self.assertIsNotNone(match, "Lab 12 defines POLICY_TEXT as a triple-quoted string")
        self.assertEqual(match.group(1).strip(), bd.policy_text())

    def test_route_descriptions_equal_the_builders(self):
        match = re.search(r"^ROUTE_DESCRIPTIONS = (\{.*?^\})", CODE, flags=re.DOTALL | re.MULTILINE)
        self.assertIsNotNone(match, "Lab 12 defines ROUTE_DESCRIPTIONS once, as a dict literal")
        self.assertEqual(eval(match.group(1)), bd.route_descriptions())  # noqa: S307
        options = re.search(r"^ROUTE_OPTIONS = (\[.*?\])", CODE, flags=re.MULTILINE)
        self.assertEqual(eval(options.group(1)), bd.ROUTE_OPTIONS)  # noqa: S307

    def test_loader_cells_are_the_readmes(self):
        readme = (DATA / "README.md").read_text(encoding="utf-8")
        blocks = re.findall(r"```python\n(.*?)```", readme, flags=re.DOTALL)
        loader = blocks[0]
        decisions = next(b for b in blocks if "def load_decisions" in b)
        self.assertIn(loader.strip(), [c.strip() for c in CODE_CELLS])
        self.assertIn(decisions.strip(), [c.strip() for c in CODE_CELLS])

    def test_decision_set_status(self):
        self.assertIn(f'DECISIONS_STATUS = "{STATS["status"]}"', CODE)
        self.assertIn(f'DECISIONS_NAME = "{STATS["name"]}"', CODE)

    def test_lab11_helpers_word_for_word(self):
        """The calibration helpers are Lab 11's, word for word; fit_temperature differs only by
        the documented log_bounds argument."""
        lab11 = _functions(json.loads(LAB11_NOTEBOOK.read_text(encoding="utf-8")))
        lab12 = _functions(NB)
        for name in LAB11_RESTATED:
            with self.subTest(function=name):
                self.assertEqual(lab12[name], lab11[name])
        widened = (
            lab11["fit_temperature"]
            .replace(
                "def fit_temperature(val_logits, val_labels):",
                "def fit_temperature(val_logits, val_labels, log_bounds=(-3, 3)):",
            )
            .replace("bounds=(-3, 3), method", "bounds=log_bounds, method")
        )
        self.assertNotEqual(widened, lab11["fit_temperature"])
        self.assertEqual(lab12["fit_temperature"], widened)


class RecordedValues(unittest.TestCase):
    def test_toy_values_match_baselines(self):
        """Checkpoint 1 asserts the values recorded under lab12.toy.*."""
        block = re.search(r"RECORDED_TOY = \{(.*?)^\}", CODE, flags=re.DOTALL | re.MULTILINE)
        recorded = eval("{" + block.group(1) + "}")  # noqa: S307
        ids = {"accuracy": "lab12.toy.accuracy_reward", "Brier": "lab12.toy.brier_reward"}
        for (reward, split, group), values in recorded.items():
            entry = BASELINES[ids[reward]]
            metrics = {"test": entry["metrics"], "train": entry["train_metrics"]}[split]
            suffix = "" if group == "pooled" else f"_{group}"
            expected = tuple(
                metrics[f"{name}{suffix}"] for name in ("accuracy", "mean_p_hat", "ece", "brier")
            )
            with self.subTest(reward=reward, split=split, group=group):
                self.assertEqual(values, expected)

    def test_threshold_values_match_baselines(self):
        entry = BASELINES["lab12.local_thresholds"]
        tau_esc, tau_act = entry["settings"]["chosen_on_dev"]
        self.assertIn(f"np.allclose(_chosen, ({tau_esc}, {tau_act})", CODE)
        self.assertIn(f"- {entry['metrics']['cost_per_case_chosen_on_dev']:.4f})", CODE)

    def test_entries_name_the_decision_set(self):
        ids = ("lab12.toy.accuracy_reward", "lab12.toy.brier_reward", "lab12.local_thresholds")
        for bid in ids:
            self.assertEqual(BASELINES[bid]["dataset"], "decisions")
            self.assertIn("not Jev", json.dumps(BASELINES[bid]["settings"]))


class Honesty(unittest.TestCase):
    def test_labels_and_banners(self):
        self.assertIn('TOY_LABEL = "toy model (our illustration)"', CODE)
        self.assertIn('"local toy decider (not Jev)"', CODE)
        self.assertIn("Our illustration, not TypeSafe's method.", TEXT)
        self.assertIn("TypeSafe has not published how RLCD works.", TEXT)
        self.assertIn("System One models are trained for calibrated decisions", TEXT)
        self.assertIn("They measure our toy model, not Jev. Do not quote them as Jev's.", CODE)
        self.assertIn("What this lab showed and what it did not", TEXT)

    def test_jev_label_only_on_the_keyed_path(self):
        self.assertIn(
            'DECIDER_LABEL = "Jev" if JEV_PATH == "keyed" else "local toy decider (not Jev)"', CODE
        )

    def test_no_client_without_a_key(self):
        for call in re.findall(r"(?:Async)?TypeSafeClient\([^)]*\)", CODE):
            with self.subTest(call=call):
                self.assertIn("api_key=", call)
        construct = cell_with('JEV = AsyncTypeSafeClient(api_key=KEYS["typesafe"]')
        self.assertIn('if JEV_PATH == "keyed":', construct)

    def test_never_debug_logging(self):
        self.assertNotIn("logging.DEBUG", CODE)
        self.assertIn('logging.getLogger("typesafe_sdk").setLevel(logging.WARNING)', CODE)


def _available(*modules):
    return all(importlib.util.find_spec(m) is not None for m in modules)


@unittest.skipUnless(
    _available("torch", "typesafe_sdk", "scipy", "matplotlib", "httpx2"),
    "needs torch, typesafe-sdk, scipy and matplotlib",
)
class LocalDeciderBehavior(unittest.TestCase):
    """The behavior table of briefs/12-rlcd-jev.md, on the notebook's own cells."""

    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("NLP_LLMS_DATA", str(DATA))
        setup = cell_with("%pip install -q typesafe-sdk")
        setup = "\n".join(line for line in setup.splitlines() if not line.lstrip().startswith("%"))
        cells = [
            setup,
            cell_with('JEV_MODEL = "jev-latest"'),
            cell_with("def fetch("),
            cell_with("def load_decisions"),
            cell_with("DECISIONS = load_decisions()"),
            cell_with('POLICY_TEXT = """'),
            cell_with("def featurize("),
            cell_with("class LocalDecider"),
        ]
        ns: dict = {}
        import matplotlib

        matplotlib.use("Agg")
        for source in cells:
            exec(compile(source, str(NOTEBOOK), "exec"), ns)  # noqa: S102
        cls.ns = ns
        cls.policy_item = next(it for it in ns["DEV"] if it["family"] == "policy")
        cls.route_item = next(it for it in ns["DEV"] if it["family"] == "route")

    def decider(self, **kwargs):
        return self.ns["LocalDecider"](self.ns["TOY_REF"], self.ns["featurize"], **kwargs)

    def test_every_policy_question_is_supported_and_route_is_not_a_noul(self):
        kind = self.ns["policy_question_type"]
        items = self.ns["TRAIN"] + self.ns["DEV"] + self.ns["TEST"]
        for question in {it["question"] for it in items if it["family"] == "policy"}:
            self.assertIsNotNone(kind(question), question)
        self.assertIsNone(kind(self.route_item["question"]))

    def test_noul(self):
        Noul = self.ns["Noul"]
        resp = self.decider().system_one(
            {"policy": "ignored", **self.policy_item["state"]},
            {"decision": Noul(instructions=self.policy_item["question"])},
            model="some-other-model",
        )
        answer = resp.answers["decision"]
        self.assertIsInstance(answer, self.ns["NoulAnswer"])
        self.assertFalse(hasattr(answer, "confidence"))
        self.assertTrue(0 <= answer.noul <= 1)
        self.assertEqual(resp.model, "local-toy-decider-v1")
        self.assertEqual(resp.request_id, "local-000001")
        self.assertIsNone(resp.usage.input_tokens)
        self.assertIsNone(resp.usage.output_tokens)

    def test_unsupported_nouls(self):
        Noul, Unsupported = self.ns["Noul"], self.ns["UnsupportedQuestion"]
        for question in (
            Noul(instructions="Does the policy allow sending this email to this address?"),
            Noul(instructions="Is the answer supported by the sources?"),
            Noul(),
            {"type": "noul", "instructions": "Is this request risky?"},
        ):
            with self.subTest(question=question), self.assertRaises(Unsupported):
                self.decider().system_one(self.policy_item["state"], {"guard": question})

    def test_choice(self):
        Choice = self.ns["Choice"]
        criteria = {o: "description" for o in reversed(self.ns["ROUTE_OPTIONS"])}
        resp = self.decider().system_one(
            self.route_item["state"],
            {"route": Choice(instructions=self.route_item["question"], criteria=criteria)},
        )
        answer = resp.answers["route"]
        self.assertEqual(list(answer.probabilities), list(criteria))
        self.assertAlmostEqual(sum(answer.probabilities.values()), 1, delta=1e-6)
        self.assertEqual(answer.choice, max(answer.probabilities, key=answer.probabilities.get))
        p = answer.probabilities[answer.choice]
        self.assertAlmostEqual(answer.confidence, (p - 0.2) / 0.8, places=12)

    def test_unsupported_choice_and_score(self):
        Choice, Score = self.ns["Choice"], self.ns["Score"]
        Unsupported = self.ns["UnsupportedQuestion"]
        for question in (
            Choice(instructions="Which?", criteria={"a": None, "b": None}),
            Choice(criteria={o: None for o in self.ns["ROUTE_OPTIONS"][:4]}),
            Score(criteria=["low", "high"]),
            {"type": "score", "criteria": ["low", "high"]},
        ):
            with self.subTest(question=question), self.assertRaises(Unsupported):
                self.decider().system_one(self.route_item["state"], {"q": question})

    def test_state_and_questions_are_checked(self):
        Noul, TypeSafeError = self.ns["Noul"], self.ns["TypeSafeError"]
        question = {"decision": Noul(instructions=self.policy_item["question"])}
        for state in (
            {"today": "2027-01-01"},
            {**self.policy_item["state"], "label": "yes"},
            "a string",
        ):
            with self.subTest(state=state), self.assertRaises(ValueError):
                self.decider().system_one(state, question)
        with self.assertRaises(TypeSafeError):
            self.decider().system_one(self.policy_item["state"], {})

    def test_raw_dicts_rounding_determinism_and_context_manager(self):
        Noul = self.ns["Noul"]
        q = self.policy_item["question"]
        state = self.policy_item["state"]
        a = self.decider().system_one(state, {"d": Noul(instructions=q)}).answers["d"].noul
        b = self.decider().system_one(state, {"d": {"type": "noul", "instructions": q}})
        self.assertEqual(a, b.answers["d"].noul)
        rounded = self.decider(round_to=2).system_one(state, {"d": Noul(instructions=q)})
        self.assertEqual(rounded.answers["d"].noul, round(a, 2))
        with self.decider() as d:
            resp = d.system_one(state, {"d": Noul(instructions=q)})
            self.assertTrue(resp.request_id.startswith("local-"))

    def test_response_model_and_async_wrapper(self):
        Noul, SystemOneResponse = self.ns["Noul"], self.ns["SystemOneResponse"]

        class Typed(SystemOneResponse):
            pass

        state, q = self.policy_item["state"], self.policy_item["question"]
        resp = self.decider().system_one(state, {"d": Noul(instructions=q)}, response_model=Typed)
        self.assertIsInstance(resp, Typed)

        async def run():
            async with self.ns["AsyncLocalDecider"](self.decider()) as d:
                out = await d.system_one(state, {"d": Noul(instructions=q)})
                await d.aclose()
                return out

        self.assertIsInstance(asyncio.run(run()), SystemOneResponse)


if __name__ == "__main__":
    unittest.main()
