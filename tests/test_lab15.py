"""Checks on notebooks/15-capstone.ipynb.

Static checks (standard library and PyYAML only, as the CI test job has):

- the cells Lab 15 restates from Labs 8, 11, 12, 13 and 14 are identical to theirs, definition by
  definition, and Lab 13's "reused by Labs 14 and 15" cell is restated whole; the manifest functions
  repeat data/build_capstone_eval.py;
- Lab 14's decide() comes without its LocalDecider branch (brief 15, (b)): no decision set, no toy;
- package pins repeat _variables.yml; thresholds, budgets and recursion_limit are the lecture's;
- the honesty labels and banners are present, no TypeSafe client is built without a key, the
  typesafe_sdk logger is never set to DEBUG, and the notebook never loads an agent-written fixture.
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

FROM_LAB08 = [
    "STOPS", "ToolCall", "Reply", "tool_result", "Tool", "strict_schema", "json_span",
    "first_json_object", "with_backoff", "Provider", "FakeProvider", "OpenAIProvider",
    "AnthropicProvider", "TOOL_PROTOCOL", "LocalProvider", "parse_openai", "ANTHROPIC_STOPS",
    "parse_anthropic", "validation_message", "execute",
]  # fmt: skip
FROM_LAB11 = ["reliability_bins", "ece", "plot_reliability", "choose_threshold"]
FROM_LAB12 = ["UnsupportedQuestion", "expected_costs", "action_thresholds"]
FROM_LAB13 = [
    "RETRIEVER_MODELS", "LECTURES", "ABSTAIN", "Passage", "load_corpus", "format_sources",
    "Retriever", "build_retriever", "fetch", "locate_spans", "SENTENCE_END", "make_probes",
    "extract", "STOPWORDS", "content_words", "first_sentence", "stub_rag_answer", "SOURCE_BLOCK",
    "StubProvider", "stub_judge", "STUB_BANNER", "make_provider", "label", "RAG_SYSTEM",
    "generate", "CITATION", "is_abstention", "split_claims", "normalize", "covers", "recall_at_k",
    "ST_VERSION", "load_sentence_transformers", "_lsa", "embed_texts", "LIEmbedding",
]  # fmt: skip
FROM_LAB14 = [
    "get_secret", "SECRETS", "RUN_CFG", "QWEN_LABEL", "STUB_LABEL", "TRANSIENT", "DECIDER_ERRORS",
    "classifier_response", "to_classifier_response", "with_transient_retry",
    "make_keyed_classifier", "make_decide", "StubDecider", "QwenDecider", "as_lab8_tool",
    "reply_entry", "to_lab8", "snapshot_before", "thread",
]  # fmt: skip
FROM_BUILDER = ["_manifest_allocate", "build_manifest", "manifest_bytes"]
NOT_REUSED = [
    "LocalDecider", "AsyncLocalDecider", "featurize", "ToyDecider", "train_toy", "TOY_REF",
    "load_decisions", "POLICY_TEXT", "SimulatedHuman", "run_with_human", "agent_metrics",
]  # fmt: skip


def cells(slug: str) -> list[dict]:
    return json.loads((NOTEBOOKS / f"{slug}.ipynb").read_text(encoding="utf-8"))["cells"]


def code_cells(slug: str) -> list[str]:
    return ["".join(c["source"]) for c in cells(slug) if c["cell_type"] == "code"]


def definitions_of(sources: list[str]) -> dict[str, str]:
    """Top-level definitions by name, source text including decorators (except the
    harness marker); the last one wins."""
    found = {}
    for source in sources:
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
            # The harness marker on a solution (@workshop.solution(N)) is not part of the
            # restated code: a later lab restates the definition as provided code.
            decorators = [
                d
                for d in getattr(node, "decorator_list", [])
                if not ast.unparse(d).startswith("workshop.solution")
            ]
            start = min([node.lineno] + [d.lineno for d in decorators])
            text = "\n".join(lines[start - 1 : node.end_lineno])
            for name in names:
                found[name] = text
    return found


def without_docstring(source: str) -> str:
    """ast.dump of a function with its docstring removed."""
    node = ast.parse(source).body[0]
    if ast.get_docstring(node) is not None:
        node.body = node.body[1:]
    return ast.dump(node)


LAB15 = code_cells("15-capstone")
CODE = "\n".join(LAB15)
TEXT = "\n".join("".join(c["source"]) for c in cells("15-capstone"))
DEFS15 = definitions_of(LAB15)


def value(name: str):
    """The literal value of a top-level assignment in the notebook."""
    return ast.literal_eval(ast.parse(DEFS15[name]).body[0].value)


class Restatements(unittest.TestCase):
    def check(self, slug_or_defs, names):
        theirs = (
            definitions_of(code_cells(slug_or_defs))
            if isinstance(slug_or_defs, str)
            else slug_or_defs
        )
        for name in names:
            with self.subTest(name=name):
                self.assertIn(name, DEFS15)
                self.assertEqual(DEFS15[name], theirs[name])

    def test_lab08_definitions_are_verbatim(self):
        self.check("08-llm-apis", FROM_LAB08)

    def test_lab11_definitions_are_verbatim(self):
        self.check("11-calibration", FROM_LAB11)

    def test_lab12_definitions_are_verbatim(self):
        self.check("12-rlcd-jev", FROM_LAB12)

    def test_lab13_definitions_are_verbatim(self):
        self.check("13-rag", FROM_LAB13)

    def test_lab13_is_correct_differs_only_in_its_docstring(self):
        """The scoring cell restates is_correct with its docstring shortened to fit the line length
        of scripts/ (ruff); the code is Lab 13's."""
        lab13 = definitions_of(code_cells("13-rag"))["is_correct"]
        self.assertEqual(without_docstring(DEFS15["is_correct"]), without_docstring(lab13))

    def test_lab14_definitions_are_verbatim(self):
        self.check("14-agents", FROM_LAB14)

    def test_builder_definitions_are_verbatim(self):
        builder = (ROOT / "data" / "build_capstone_eval.py").read_text(encoding="utf-8")
        self.check(definitions_of([builder]), FROM_BUILDER)

    def test_whole_cells(self):
        """Lab 8's provider cell and Lab 13's interface, stub, extract and make_provider cells
        appear
        unchanged, as whole cells."""
        for slug, marker in (
            ("08-llm-apis", "The provider wrapper (reused"),
            ("13-rag", "REUSED BY LABS 14 AND 15"),
            ("13-rag", "StubProvider: a deterministic TEST DOUBLE for offline and CI runs"),
            ("13-rag", "Lab 8's validate-and-retry loop (Solution 2)"),
            ("13-rag", "Lab 8's make_provider and label, restated"),
            ("14-agents", "Restated from Lab 8, word for word: Solution 1"),
        ):
            source = next(c for c in code_cells(slug) if marker in c)
            with self.subTest(slug=slug, marker=marker):
                self.assertIn(source, LAB15)

    def test_decide_without_the_local_decider(self):
        for name in NOT_REUSED:
            with self.subTest(name=name):
                self.assertNotIn(name, DEFS15)
        self.assertNotIn("decisions_v1", CODE)
        self.assertIn(
            "Lab 15 asks no decision-set question; the toy model would raise UnsupportedQuestion "
            "for "
            "both questions.",
            CODE,
        )
        self.assertIn("decide = make_decide(RunnableLambda(_decider), DECIDER_LABEL)", CODE)
        self.assertIn('decide = make_decide(JEV_RUNNABLE, "Jev ({model})")', CODE)

    def test_stub_decider_rules_read_only_the_state(self):
        body = DEFS15["stub_noul"]
        for forbidden in ("label", "gold", "key_facts", "evidence", '"kind"', "answerable"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, body.split('"""')[2])


class Settings(unittest.TestCase):
    def test_pins_repeat_variables(self):
        p = V["packages"]
        line = next(c for c in LAB15 if "%pip install" in c)
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

    def test_costs_thresholds_and_budgets(self):
        for line in (
            "LOSS = dict(wrong=5, abstain=1)",
            "TAU_PLAN = 0.5",
            'TAU_VERIFY = 1 - LOSS["abstain"] / LOSS["wrong"]',
            "R_MAX, K_MAX_LLM, K_MAX_DEC = 1, 4, 4",
            'RUN_CFG = {"recursion_limit": 40}',
            "assert action_thresholds(dict(wrong=LOSS['wrong'], ask=1, miss=1, "
            "esc=LOSS['abstain']))".replace("'", '"'),
        ):
            with self.subTest(line=line):
                self.assertIn(line, CODE)

    def test_the_two_questions(self):
        """The instructions of brief 15, (c), complete: question names are not sent to the model."""
        self.assertEqual(
            value("NEEDS_TWO"),
            "Does answering this question need evidence from two different modules or documents of "
            "the workshop's lecture pages, rather than one passage?",
        )
        self.assertEqual(
            value("SUPPORTED"),
            "Is every factual claim in the answer supported by the passages below? Background "
            "knowledge does not count as support.",
        )

    def test_every_invoke_carries_the_run_configuration(self):
        self.assertIn('return {"configurable": {"thread_id": name}, **RUN_CFG}', CODE)
        for call in re.findall(r"\.invoke\(([^\n]*)", CODE):
            if call.startswith('{"state"'):  # a decider runnable, not the graph
                continue
            with self.subTest(call=call):
                self.assertRegex(call, r"RUN_CFG|thread\(")

    def test_offline_flag(self):
        self.assertIn('os.environ.get("NLP_LLMS_LAB15_OFFLINE") == "1"', CODE)

    def test_open_model_decodes_greedily(self):
        self.assertIn("do_sample=False", CODE)
        self.assertNotIn("do_sample=True", CODE)


class Honesty(unittest.TestCase):
    def test_labels_and_banners(self):
        for text in (
            'QWEN_LABEL = "Qwen log-prob decider (not Jev)"',
            'STUB_LABEL = "stub (test double)"',
            "stand-in (not a neural model)",
            "These numbers measure the notebook's code, not any model. Do not quote them.",
            "Reference judge not validated against human labels",
            "PLUMBING PROBES, not questions",
            "transport test",
        ):
            with self.subTest(text=text[:60]):
                self.assertIn(text, CODE)
        self.assertEqual(
            value("NO_KEY_BANNER"),
            "No TypeSafe key: the planner's and verifier's probabilities come from a small open "
            "language model scored by the probability of ' yes'. They measure that model, not Jev. "
            "Do not quote them as Jev's.",
        )
        self.assertIn("What this capstone showed and what it did not", TEXT)
        self.assertIn("our guess before any run", TEXT)

    def test_no_rlcd(self):
        """Jev is described only through its interface; RLCD is not named (brief 15)."""
        self.assertNotIn("RLCD", TEXT)

    def test_no_classifier_or_client_without_a_key(self):
        self.assertEqual(len(re.findall(r"TypeSafeClassifier\(", CODE)), 1)
        cell = next(c for c in LAB15 if 'make_keyed_classifier(KEYS["typesafe"])' in c)
        self.assertIn('if JEV_PATH == "keyed":', cell)
        self.assertIn(
            'JEV_PATH = "keyed" if KEYS["typesafe"] and PROVIDER != "open" else "local"', CODE
        )
        interface = next(c for c in LAB15 if "REUSED BY LABS 14 AND 15" in c)
        for call in re.findall(r"AsyncTypeSafeClient\(([^\n]*)", CODE.replace(interface, "")):
            with self.subTest(call=call):
                self.assertIn("api_key=", call)

    def test_retry_only_transient_errors(self):
        self.assertIn("retry_if_exception_type=TRANSIENT", CODE)

    def test_never_debug_logging(self):
        self.assertNotIn("logging.DEBUG", CODE)
        self.assertIn('logging.getLogger("typesafe_sdk").setLevel(logging.WARNING)', CODE)

    def test_no_prebuilt_agent_or_middleware(self):
        for name in ("create_react_agent", "create_agent(", "AutoModeMiddleware(", "ToolNode("):
            with self.subTest(name=name):
                self.assertNotIn(name, CODE)
        self.assertNotIn("draw_mermaid_png()", CODE)

    def test_never_loads_a_fixture(self):
        self.assertNotIn("rag_questions_fixture", CODE)
        self.assertNotIn("capstone_questions_fixture", CODE)
        self.assertNotIn("tests/fixtures", CODE)

    def test_snapshot_hash_is_quoted_twice(self):
        sha = V["datasets"]["lectures"]["sha256"]
        self.assertEqual(CODE.count(f'"{sha}"'), 2)  # Lab 13's interface cell and DATASETS

    def test_results_are_printed_under_banners(self):
        cell = next(c for c in LAB15 if "def print_scores(" in c)
        self.assertIn("banners()", cell)
        self.assertGreaterEqual(len(re.findall(r"^\s*banners\(\)", CODE, flags=re.M)), 3)

    def test_one_exercise_with_a_folded_solution(self):
        todo = next(i for i, c in enumerate(LAB15) if c.startswith("# TODO 1"))
        self.assertTrue(LAB15[todo + 1].startswith("#@title Solution 1"))
        self.assertEqual(sum(c.startswith("# TODO") for c in LAB15), 1)

    def test_self_test_has_fourteen_checks(self):
        cell = next(c for c in LAB15 if "Self-test:" in c)
        numerals = "i ii iii iv v vi vii viii ix x xi xii xiii xiv".split()
        self.assertEqual(re.findall(r'_ok\.append\("(\w+)"\)', cell), numerals)


if __name__ == "__main__":
    unittest.main()
