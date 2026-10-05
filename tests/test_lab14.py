"""Checks on notebooks/14-agents.ipynb.

Static checks (standard library and PyYAML only, as the CI test job has):

- the cells Lab 14 restates from Lab 12 and Lab 8 are identical to theirs, definition by
  definition, and the data-loading cells are data/README.md's;
- package pins and the decision set's status repeat _variables.yml and the statistics file;
- the toy router's recorded values agree with data/baselines.json (lab14.local_router);
- the honesty labels, banners and safety settings are present (no TypeSafe classifier without a
  key, no DEBUG logging, recursion_limit on every run, no prebuilt agent or experimental
  middleware).
"""

import ast
import json
import re
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = ROOT / "notebooks"
V = yaml.safe_load((ROOT / "_variables.yml").read_text(encoding="utf-8"))
BASELINES = {
    b["id"]: b
    for b in json.loads((ROOT / "data" / "baselines.json").read_text(encoding="utf-8"))["baselines"]
}
STATS = json.loads((ROOT / "data" / "decisions_v1_stats.json").read_text(encoding="utf-8"))

# Names Lab 14 restates word for word.
FROM_LAB12 = [
    "POLICY_TEXT",
    "ROUTE_OPTIONS",
    "ROUTE_DESCRIPTIONS",
    "featurize",
    "ToyDecider",
    "train_toy",
    "reward_brier_ref",
    "TOY_REF",
    "UnsupportedQuestion",
    "LocalDecider",
    "AsyncLocalDecider",
    "policy_question_type",
    "jev_state",
    "chosen_answer",
    "expected_costs",
    "action_thresholds",
]
FROM_LAB08 = [
    "ToolCall",
    "Reply",
    "tool_result",
    "Tool",
    "strict_schema",
    "json_span",
    "first_json_object",
    "with_backoff",
    "Provider",
    "FakeProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "TOOL_PROTOCOL",
    "LocalProvider",
    "parse_openai",
    "ANTHROPIC_STOPS",
    "parse_anthropic",
    "validation_message",
    "execute",
    "StubProvider",
    "STUB_BANNER",
    "make_provider",
    "label",
]


def cells(slug: str) -> list[dict]:
    return json.loads((NOTEBOOKS / f"{slug}.ipynb").read_text(encoding="utf-8"))["cells"]


def code_cells(slug: str) -> list[str]:
    return ["".join(c["source"]) for c in cells(slug) if c["cell_type"] == "code"]


def definitions(slug: str) -> dict[str, str]:
    """Top-level definitions by name, source text including decorators; the last one wins."""
    found = {}
    for source in code_cells(slug):
        clean = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("%"))
        try:
            tree = ast.parse(clean)
        except SyntaxError:
            continue
        lines = clean.splitlines()
        for node in tree.body:
            names = []
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                names = [node.name]
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    elts = target.elts if isinstance(target, ast.Tuple) else [target]
                    names += [e.id for e in elts if isinstance(e, ast.Name)]
            start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
            text = "\n".join(lines[start - 1 : node.end_lineno])
            for name in names:
                found[name] = text
    return found


LAB14 = code_cells("14-agents")
CODE = "\n".join(LAB14)
TEXT = "\n".join("".join(c["source"]) for c in cells("14-agents"))
DEFS14 = definitions("14-agents")


class Restatements(unittest.TestCase):
    def test_lab12_definitions_are_verbatim(self):
        lab12 = definitions("12-rlcd-jev")
        for name in FROM_LAB12:
            with self.subTest(name=name):
                self.assertIn(name, DEFS14)
                self.assertEqual(DEFS14[name], lab12[name])

    def test_lab12_provided_cells_are_whole(self):
        """Lab 12's policy, toy-model and local-decider cells appear unchanged, as whole cells."""
        lab12 = code_cells("12-rlcd-jev")
        for marker in ('POLICY_TEXT = """', "def featurize(", "class LocalDecider"):
            source = next(c for c in lab12 if marker in c)
            with self.subTest(marker=marker):
                self.assertIn(source, LAB14)

    def test_lab08_definitions_are_verbatim(self):
        lab08 = definitions("08-llm-apis")
        for name in FROM_LAB08:
            with self.subTest(name=name):
                self.assertIn(name, DEFS14)
                self.assertEqual(DEFS14[name], lab08[name])

    def test_lab08_provider_cell_is_whole(self):
        source = next(c for c in code_cells("08-llm-apis") if "The provider wrapper (reused" in c)
        self.assertIn(source, LAB14)

    def test_loader_cells_are_the_readmes(self):
        readme = (ROOT / "data" / "README.md").read_text(encoding="utf-8")
        blocks = re.findall(r"```python\n(.*?)```", readme, flags=re.DOTALL)
        decisions = next(b for b in blocks if "def load_decisions" in b)
        stripped = [c.strip() for c in LAB14]
        self.assertIn(blocks[0].strip(), stripped)
        self.assertIn(decisions.strip(), stripped)

    def test_lab13_retriever_cell_is_whole(self):
        """Lab 13's 'reused by Labs 14 and 15' cell, unchanged; Lab 14 calls its BM25 path only."""
        source = next(c for c in code_cells("13-rag") if "REUSED BY LABS 14 AND 15" in c)
        self.assertIn(source, LAB14)
        builds = re.findall(r"build_retriever\(([^)]*)\)", CODE.replace(source, ""))
        self.assertTrue(builds)
        for args in builds:
            with self.subTest(args=args):
                self.assertIn('method="bm25"', args)


class Settings(unittest.TestCase):
    def test_pins_repeat_variables(self):
        p = V["packages"]
        line = next(c for c in LAB14 if "%pip install" in c)
        for package, key in (
            ("langgraph", "langgraph"),
            ("langchain-core", "langchain_core"),
            ("langchain-typesafe", "langchain_typesafe"),
            ("typesafe-sdk", "typesafe_sdk"),
            ("llama-index-core", "llama_index_core"),
            ("llama-index-retrievers-bm25", "llama_index_retrievers_bm25"),
        ):
            with self.subTest(package=package):
                self.assertIn(f"{package}=={p[key]}", line)

    def test_decision_set_status(self):
        self.assertIn(f'DECISIONS_STATUS = "{STATS["status"]}"', CODE)
        self.assertIn(f'DECISIONS_NAME = "{STATS["name"]}"', CODE)

    def test_thresholds_and_limits(self):
        self.assertIn("ROUTER_COSTS = dict(wrong=4, ask=2.5, miss=2, esc=1)", CODE)
        self.assertIn("GUARD_COSTS = dict(wrong=20, ask=0.5, miss=4, esc=3)", CODE)
        self.assertIn('RUN_CFG = {"recursion_limit": 40}', CODE)
        self.assertIn("K_MAX = 4", CODE)
        # Every graph invoke carries the run configuration: thread() adds RUN_CFG, and every `cfg`
        # and `_cfg` in the notebook comes from thread(). Runnable invokes take {"state": ...}.
        self.assertIn('return {"configurable": {"thread_id": name}, **RUN_CFG}', CODE)
        for call in re.findall(r"\.invoke\(([^\n]*)", CODE):
            if call.startswith('{"state"'):
                continue
            with self.subTest(call=call):
                self.assertRegex(call, r"RUN_CFG|thread\(|\bcfg\)|_cfg\)")
        for assignment in re.findall(r"^\s*_?cfg = (.*)$", CODE, flags=re.MULTILINE):
            with self.subTest(assignment=assignment):
                self.assertTrue(assignment.startswith("thread("), assignment)

    def test_router_values_match_baselines(self):
        entry = BASELINES["lab14.local_router"]
        block = re.search(r"RECORDED_ROUTER = \{(.*?)^\}", CODE, flags=re.DOTALL | re.MULTILINE)
        recorded = eval("{" + block.group(1) + "}")  # noqa: S307
        for split, metrics in (("dev", entry["val_metrics"]), ("test", entry["metrics"])):
            with self.subTest(split=split):
                self.assertEqual(
                    recorded[split],
                    (metrics["accuracy"], metrics["followed"], metrics["accuracy_when_followed"]),
                )
        self.assertEqual(entry["dataset"], "decisions")
        self.assertIn("not Jev", json.dumps(entry["settings"]))

    def test_router_test_accuracy_is_lab12s(self):
        """The router is Lab 12's toy model on the same route items and question."""
        lab12 = BASELINES["lab12.toy.brier_reward"]["metrics"]["accuracy_route"]
        self.assertEqual(BASELINES["lab14.local_router"]["metrics"]["accuracy"], lab12)


class Honesty(unittest.TestCase):
    def test_labels_and_banners(self):
        for text in (
            '"local toy decider (not Jev)"',
            '"Qwen log-prob decider (not Jev)"',
            '"stub (test double)"',
            "those models, not Jev. Do not quote them as Jev's.",
            "These numbers measure the notebook's code, not any model. Do not quote them.",
            "component estimate",
            "transport test",
        ):
            with self.subTest(text=text):
                self.assertIn(text, CODE)
        self.assertIn("What this lab showed and what it did not", TEXT)

    def test_no_classifier_without_a_key(self):
        constructions = re.findall(r"TypeSafeClassifier\(", CODE)
        self.assertEqual(len(constructions), 1, "one construction, inside make_keyed_classifier")
        cell = next(c for c in LAB14 if 'make_keyed_classifier(KEYS["typesafe"])' in c)
        self.assertIn('if JEV_PATH == "keyed":', cell)
        self.assertIn('JEV_PATH = "keyed" if KEYS["typesafe"] else "local"', CODE)

    def test_retry_only_transient_errors(self):
        self.assertIn(
            "TRANSIENT = (TypeSafeRateLimitError, TypeSafeAPIConnectionError, "
            "TypeSafeInternalServerError)",
            CODE,
        )
        self.assertIn("retry_if_exception_type=TRANSIENT", CODE)

    def test_beta_warning_silenced_locally(self):
        self.assertIn('warnings.simplefilter("ignore", LangChainBetaWarning)', CODE)
        self.assertIn("with warnings.catch_warnings():", CODE)
        self.assertNotIn("filterwarnings", CODE)

    def test_never_debug_logging(self):
        self.assertNotIn("logging.DEBUG", CODE)
        self.assertIn('logging.getLogger("typesafe_sdk").setLevel(logging.WARNING)', CODE)

    def test_no_prebuilt_agent_or_middleware(self):
        for name in ("create_react_agent", "create_agent(", "AutoModeMiddleware(", "ToolNode("):
            with self.subTest(name=name):
                self.assertNotIn(name, CODE)
        self.assertNotIn("draw_mermaid_png", CODE)

    def test_guard_question_avoids_the_toy_keywords(self):
        """The guard question must not be answered by Lab 12's P5 head."""
        question = re.search(r"GUARD_QUESTION = \((.*?)\)\n", CODE, flags=re.DOTALL).group(1)
        text = "".join(re.findall(r'"(.*?)"', question)).lower()
        for keyword in ("accepted", "full refund", "transfer", "catering", "approval", "do next"):
            with self.subTest(keyword=keyword):
                self.assertNotIn(keyword, text)

    def test_hard_rule_assertion_present(self):
        self.assertIn('assert mail["to"] == mail["record"]', CODE)
        self.assertIn("HARD_RULES = True", CODE)

    def test_decider_not_called_in_human_review(self):
        body = re.search(
            r"    def human_review\(state\):\n(.*?)\n\n    def ", CODE, flags=re.DOTALL
        ).group(1)
        self.assertIn("interrupt(", body)
        self.assertNotIn("decide(", body)
        self.assertNotIn("guard_check(", body)


if __name__ == "__main__":
    unittest.main()
