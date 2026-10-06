"""Every lab's exercises use the harness: solutions never replace participants' code, stubs
raise when called, and every checkpoint says whose code it checked (CONTRIBUTING.md)."""

import ast
import json
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import harness  # noqa: E402
import run_records  # noqa: E402

NOTEBOOKS = sorted((ROOT / "notebooks").glob("*.ipynb"))
TODO = re.compile(r"^#\s*TODO\s*(\d+|\(stretch\))", re.I | re.M)
TITLE = re.compile(r"^#@title\s+Solution\s*(\d+|\(stretch\))", re.I)


def load(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["cells"]


def text(cell: dict) -> str:
    return "".join(cell["source"])


def tags(cell: dict) -> list[str]:
    return cell.get("metadata", {}).get("tags", [])


def number(token: str):
    return "stretch" if token.lower() == "(stretch)" else int(token)


def is_marker(dec: ast.expr, n=None) -> bool:
    return (
        isinstance(dec, ast.Call)
        and isinstance(dec.func, ast.Attribute)
        and dec.func.attr == "solution"
        and isinstance(dec.func.value, ast.Name)
        and dec.func.value.id == "workshop"
        and (
            n is None
            or (dec.args and isinstance(dec.args[0], ast.Constant) and dec.args[0].value == n)
        )
    )


def definitions(tree: ast.Module) -> dict[str, ast.AST]:
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
    }


def checkpoint_call(cell: dict) -> ast.Call | None:
    tree = ast.parse(text(cell))
    first = tree.body[0] if tree.body else None
    if (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Call)
        and isinstance(first.value.func, ast.Attribute)
        and first.value.func.attr == "checkpoint"
        and isinstance(first.value.func.value, ast.Name)
        and first.value.func.value.id == "workshop"
    ):
        return first.value
    return None


class Generated(unittest.TestCase):
    def test_every_notebook_has_the_harness_and_summary_cells(self):
        for path in NOTEBOOKS:
            cells = load(path)
            with self.subTest(path.name):
                self.assertEqual(cells[1].get("id"), harness.HARNESS_ID)
                self.assertEqual(cells[-2].get("id"), harness.SUMMARY_ID)
                self.assertEqual(text(cells[-2]), harness.SUMMARY)

    def test_the_harness_names_the_exercises_and_the_code_hash(self):
        for path in NOTEBOOKS:
            cells = load(path)
            expected = harness.harness_source(
                path.stem, run_records.sha_of_cells(cells), harness.exercises_of(cells)
            )
            with self.subTest(path.name):
                self.assertEqual(text(cells[1]), expected)


class Exercises(unittest.TestCase):
    def test_each_stub_is_followed_by_its_marked_folded_solution(self):
        for path in NOTEBOOKS:
            cells = load(path)
            for i, cell in enumerate(cells):
                if "exercise" not in tags(cell):
                    continue
                stub = text(cell)
                n = number(TODO.search(stub).group(1))
                solution = cells[i + 1]
                with self.subTest(f"{path.stem} TODO {n}"):
                    self.assertIn("solution", tags(solution))
                    self.assertEqual(number(TITLE.match(text(solution)).group(1)), n)
                    self.assertEqual(solution["metadata"].get("cellView"), "form")
                    self.assertTrue(solution["metadata"].get("jupyter", {}).get("source_hidden"))
                    stub_defs = definitions(ast.parse(stub))
                    sol_tree = ast.parse(text(solution))
                    redefined = set(stub_defs) & set(definitions(sol_tree))
                    for name, node in definitions(sol_tree).items():
                        markers = [d for d in node.decorator_list if is_marker(d)]
                        if name in redefined:
                            self.assertTrue(markers, f"{name} replaces the stub: mark it")
                            self.assertTrue(is_marker(node.decorator_list[0], n), name)
                    given = {
                        node.targets[0].id: ast.get_source_segment(stub, node.value)
                        for node in ast.parse(stub).body
                        if isinstance(node, ast.Assign)
                        and len(node.targets) == 1
                        and isinstance(node.targets[0], ast.Name)
                    }
                    for node in sol_tree.body:
                        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
                            continue
                        target = node.targets[0]
                        if not isinstance(target, ast.Name) or target.id not in given:
                            continue
                        value = ast.get_source_segment(text(solution), node.value)
                        marked = "workshop.solution_value(" in value
                        # A value the stub gives word for word is not part of the exercise.
                        self.assertTrue(
                            marked or value == given[target.id],
                            f"a plain assignment replaces {target.id}: mark it",
                        )

    def test_pure_stubs_raise_not_implemented(self):
        for path in NOTEBOOKS:
            for cell in load(path):
                if "exercise" not in tags(cell):
                    continue
                n = number(TODO.search(text(cell)).group(1))
                for node in ast.walk(ast.parse(text(cell))):
                    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                        body = node.body
                        if (
                            body
                            and isinstance(body[0], ast.Expr)
                            and isinstance(getattr(body[0], "value", None), ast.Constant)
                            and isinstance(body[0].value.value, str)
                        ):
                            body = body[1:]
                        with self.subTest(f"{path.stem} {node.name}"):
                            self.assertFalse(
                                len(body) == 1
                                and isinstance(body[0], ast.Expr)
                                and isinstance(body[0].value, ast.Constant)
                                and body[0].value.value is Ellipsis,
                                f"TODO {n}: a pure stub must raise NotImplementedError",
                            )

    def test_every_checkpoint_cell_starts_by_naming_what_it_checks(self):
        for path in NOTEBOOKS:
            for cell in load(path):
                if "checkpoint" in tags(cell):
                    with self.subTest(f"{path.stem} {cell.get('id')}"):
                        self.assertIsNotNone(checkpoint_call(cell))

    def test_every_exercise_has_a_checkpoint(self):
        for path in NOTEBOOKS:
            cells = load(path)
            exercises = {
                number(TODO.search(text(c)).group(1)) for c in cells if "exercise" in tags(c)
            }
            checked = set()
            for cell in cells:
                call = checkpoint_call(cell) if "checkpoint" in tags(cell) else None
                if call is not None and call.args and isinstance(call.args[0], ast.Constant):
                    checked.add(call.args[0].value)
            with self.subTest(path.stem):
                self.assertEqual(exercises - checked, set())

    def test_exercise_names_are_unique_within_a_lab(self):
        # The harness tells your code from a reference by exercise; a name shared by two
        # exercises would blur that.
        for path in NOTEBOOKS:
            seen = {}
            for n, names in harness.exercises_of(load(path)).items():
                for name in names:
                    with self.subTest(f"{path.stem} {name}"):
                        self.assertNotIn(name, seen, f"in exercises {seen.get(name)} and {n}")
                    seen[name] = n

    def test_no_notebook_says_solutions_replace_your_code(self):
        for path in NOTEBOOKS:
            with self.subTest(path.stem):
                self.assertNotIn("replace your functions", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
