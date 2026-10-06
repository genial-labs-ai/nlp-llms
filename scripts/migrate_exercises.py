"""Move every lab to the exercise harness (five-day revision, Phase 2). Idempotent.

For each notebook it:
  - marks each solution's definitions with @workshop.solution(N), so the folded solution
    stores the reference instead of replacing the participant's code;
  - turns each pure stub body (a lone `...`) into `raise NotImplementedError("TODO N")`;
  - starts each checkpoint cell with workshop.checkpoint(...), so the cell reports whose
    code it checked;
  - rewrites the "Run all replaces your functions" sentence in the introduction.

The notebook's generated cells (header, harness, summary, footer) are left to
scripts/gen_notebooks.py. Run with --dry-run to print the checkpoint map for review.

Run:  uv run --group site python scripts/migrate_exercises.py [--dry-run] [SLUG ...]
then: uv run --group site python scripts/gen_notebooks.py
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

TODO = re.compile(r"^#\s*TODO\s*(\d+|\(stretch\))", re.I | re.M)
SOLUTION = re.compile(r"#@title\s+Solution\s*(\d+|\(stretch\))", re.I)
CHECKPOINT = re.compile(r"^#\s*Checkpoint\s+([0-9]+[A-Za-z]?)\b", re.I)

# Checkpoint cells that are not "Checkpoint N" for the exercise just above them, by
# cell id: (exercise number or None for code the lab provides, label). Reviewed by hand.
OVERRIDES: dict[str, dict[str, tuple[object, str]]] = {
    "00-setup": {"5aac06c8": (None, "setup")},
    "01-text-as-data": {"card-check": (None, "baselines")},
    "02-word-vectors": {"cell-048": (None, "baselines")},
    "03-sequence-models": {"lab03-027": (None, "3A"), "lab03-049": (None, "LSTM against trigram")},
    "04-seq2seq-attention": {"compare-check": (None, "bottleneck")},
    "05-transformer-from-scratch": {"lab05-032": (None, "leak test")},
    "07-finetuning-lora": {"2233d662": (5, "after training")},
    "08-llm-apis": {"stretch-check": (None, "stretch")},
    "09-preference-learning": {"lab09-021": (2, "2a and 2b")},
    "11-calibration": {"stretch-check": ("stretch", "stretch")},
    "12-rlcd-jev": {"ab7123f0": (None, "local decider"), "9e88449c": (None, "stretch")},
    "15-capstone": {"15dac415": (1, "self-test")},
    "14-agents": {
        "5af815c7": (None, "code-level assertions"),
        "6bf193be": (None, "router"),
        "8cc62d9c": (None, "transport test"),
        "1ea915b0": (6, "stretch A"),
        "7bdc5162": (7, "stretch B"),
        "7428ab50": (8, "stretch C"),
        "59e194d3": (9, "stretch D"),
    },
}
# Checkpoints that a stub cannot be checked against: they run after training on the
# participant's code, or change state the rest of the lab relies on.
NO_VERIFY = {("07-finetuning-lora", "2233d662")}

INTRO_OLD = re.compile(
    r"(If you use Run all|With Run all), the solution(?: cell)?s run after your cells and replace"
    r" your (?:functions|code), so the notebook always completes[.;] (?:T|t)o test your own code,"
    r" run (?:the|your) `# TODO` cell and then the checkpoint, skipping the solution\."
)
INTRO_NEW = (
    "The folded solution cells never replace your code: each checkpoint checks your own"
    " functions and says so. If you are stuck, `workshop.use_reference(N)` lets you go on"
    " with exercise N's reference solution, and the checkpoint then says it checked the"
    " reference. To run the whole lab as a worked example, tick `WORKED_EXAMPLE` in the"
    " harness cell at the top; a worked run shows how the lab goes, not that you did it."
)


def first_line(text: str) -> str:
    return text.splitlines()[0] if text else ""


def source(cell: dict) -> str:
    return "".join(cell["source"])


def set_source(cell: dict, text: str) -> None:
    lines = text.split("\n")
    cell["source"] = [line + "\n" for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])


def top_level_names(tree: ast.Module) -> dict[str, ast.AST]:
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def assigned_names(tree: ast.Module) -> set[str]:
    return {
        node.targets[0].id
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    }


def pure_stub_bodies(tree: ast.Module) -> list[ast.Expr]:
    """The `...` statements that are the whole body (after a docstring) of any function in
    a stub cell: top level, a method, or a function nested in a provided factory."""
    out = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = fn.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            body = body[1:]
        if (
            len(body) == 1
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and body[0].value.value is Ellipsis
        ):
            out.append(body[0])
    return out


def number(token: str) -> object:
    return "stretch" if token.lower() == "(stretch)" else int(token)


def assigned_sources(text: str) -> dict[str, str]:
    """{name: source of the assigned value} for the top-level single-name assignments."""
    tree = ast.parse(text)
    return {
        node.targets[0].id: ast.get_source_segment(text, node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    }


def migrate_stub(text: str, n: object) -> tuple[str, set[str], set[str]]:
    tree = ast.parse(text)
    lines = text.split("\n")
    for expr in pure_stub_bodies(tree):
        line = lines[expr.lineno - 1]
        indent = line[: len(line) - len(line.lstrip())]
        if line.strip() == "...":
            lines[expr.lineno - 1] = f'{indent}raise NotImplementedError("TODO {n}")'
    return "\n".join(lines), set(top_level_names(tree)), assigned_sources(text)


def migrate_solution(text: str, n: object, defs: set[str], values: dict[str, str]) -> str:
    if "workshop.solution" in text:
        return text
    tree = ast.parse(text)
    lines = text.split("\n")
    arg = repr(n)
    edits = []  # (line index, kind, payload)
    for name, node in top_level_names(tree).items():
        if name in defs:
            first = min([d.lineno for d in node.decorator_list] + [node.lineno])
            edits.append((first - 1, "insert", f"@workshop.solution({arg})"))
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in values
            and node.targets[0].id not in defs
            # A value the stub already gives, word for word, is not part of the exercise.
            and ast.get_source_segment(text, node.value) != values[node.targets[0].id]
        ):
            edits.append((node, "wrap", node.targets[0].id))
    for index, kind, payload in sorted(
        edits, key=lambda e: e[0] if isinstance(e[0], int) else e[0].lineno, reverse=True
    ):
        if kind == "insert":
            lines.insert(index, payload)
        else:
            v = index.value
            start, end = v.lineno - 1, v.end_lineno - 1
            lines[end] = lines[end][: v.end_col_offset] + ")" + lines[end][v.end_col_offset :]
            lines[start] = (
                lines[start][: v.col_offset]
                + f'workshop.solution_value({arg}, "{payload}", '
                + lines[start][v.col_offset :]
            )
    return "\n".join(lines)


def checkpoint_call(n: object, label: str | None) -> str:
    if n is None:
        return f'workshop.checkpoint(label="{label}")'
    if label is None or label == str(n):
        return f"workshop.checkpoint({n!r})"
    return f'workshop.checkpoint({n!r}, label="{label}")'


def migrate_checkpoint(text: str, call: str) -> str:
    if "workshop.checkpoint(" in text:
        return text
    lines = text.split("\n")
    i = 0
    while i < len(lines) and (lines[i].startswith("#") or not lines[i].strip()):
        i += 1
    lines.insert(i, call)
    return "\n".join(lines)


def plan_checkpoint(slug: str, cell: dict, current: object) -> tuple[object, str | None]:
    if cell.get("id") in OVERRIDES.get(slug, {}):
        return OVERRIDES[slug][cell["id"]]
    first = source(cell).lstrip()
    m = CHECKPOINT.match(first)
    if m and current is not None and m.group(1).rstrip("abcdefABCDEF") == str(current):
        return current, m.group(1)
    return None, (m.group(1) if m else None)


ATTACH = "LoRALinear.merged_weight = merged_weight  # attach it as a method"
ATTACH_LATE = (
    "LoRALinear.merged_weight = lambda layer: merged_weight(layer)  # calls whichever"
    " merged_weight is bound: yours, or the reference after use_reference(2)"
)


def special_cases(slug: str, cells: list) -> None:
    """Hand-written edits the general rules cannot make."""
    if slug == "07-finetuning-lora":
        # The stub and the solution both attached merged_weight to the class, so the
        # solution's attachment would bypass the harness. Attach once, late, by name.
        for cell in cells:
            text = source(cell)
            if ATTACH in text:
                set_source(
                    cell, text.replace("\n\n\n" + ATTACH, "").replace(ATTACH, "").rstrip("\n")
                )
        for cell in cells:
            text = source(cell)
            if (
                "checkpoint" in cell.get("metadata", {}).get("tags", [])
                and text.startswith("# Checkpoint 2\n")
                and ATTACH_LATE not in text
            ):
                set_source(
                    cell,
                    text.replace("# Checkpoint 2\n", "# Checkpoint 2\n" + ATTACH_LATE + "\n", 1),
                )
    if slug == "15-capstone":
        for cell in cells:
            text = source(cell)
            tags = cell.setdefault("metadata", {}).setdefault("tags", [])
            if text.startswith("# TODO 1: complete this function.") and "exercise" not in tags:
                tags.append("exercise")
            if text.startswith("#@title Solution 1") and "solution" not in tags:
                tags.append("solution")
                cell["metadata"]["cellView"] = "form"
                cell["metadata"].setdefault("jupyter", {})["source_hidden"] = True
            if text.startswith("# The system self-test:") and "checkpoint" not in tags:
                tags.append("checkpoint")


def migrate(path: Path, dry_run: bool) -> list[str]:
    nb = json.loads(path.read_text(encoding="utf-8"))
    slug = path.stem
    special_cases(slug, nb["cells"])
    report, current, stubs, seen = [], None, {}, set()
    cells = nb["cells"]
    for i, cell in enumerate(cells):
        tags = cell.get("metadata", {}).get("tags", [])
        text = source(cell)
        if cell["cell_type"] == "markdown" and INTRO_OLD.search(text):
            set_source(cell, INTRO_OLD.sub(INTRO_NEW, text))
        if cell["cell_type"] != "code":
            continue
        if "exercise" in tags:
            m = TODO.search(text)
            if m:
                current = number(m.group(1))
                new, defs, values = migrate_stub(text, current)
                stubs[current] = (defs, values)
                set_source(cell, new)
        elif "solution" in tags:
            m = SOLUTION.search(text)
            n = number(m.group(1)) if m else current
            defs, values = stubs.get(n, (set(), {}))
            set_source(cell, migrate_solution(text, n, defs, values))
        elif "checkpoint" in tags:
            n, label = plan_checkpoint(slug, cell, current)
            if label is None:
                label = str(n) if n is not None else f"cell {cell.get('id', i)}"
            base, k = label, 2
            while label in seen:
                first = source(cell).splitlines()[0] if source(cell) else ""
                paren = re.search(r"\(([^)]+)\)|(follow-up)", first)
                hint = (paren.group(1) or paren.group(2)) if paren else None
                label = f"{base} ({hint})" if hint and k == 2 else f"{base}, part {k}"
                k += 1
            seen.add(label)
            if label == str(n):
                label = None
            call = checkpoint_call(n, label)
            report.append(
                f"{slug:30} {cell.get('id', i)!s:12} -> {call:55} | {first_line(text)[:60]}"
            )
            set_source(cell, migrate_checkpoint(text, call))
            if (slug, cell.get("id")) in NO_VERIFY and "no-verify" not in tags:
                tags.append("no-verify")
    if not dry_run:
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--notebooks", type=Path, default=ROOT / "notebooks")
    parser.add_argument("slugs", nargs="*")
    args = parser.parse_args()
    paths = sorted(args.notebooks.glob("[01][0-9]-*.ipynb"))
    if args.slugs:
        paths = [p for p in paths if p.stem in args.slugs]
    for path in paths:
        for line in migrate(path, args.dry_run):
            print(line)


if __name__ == "__main__":
    main()
