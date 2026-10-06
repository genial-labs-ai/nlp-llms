"""Execute the notebooks top to bottom in a real kernel.

Each notebook is run from a temporary directory, so nothing is written back to
notebooks/. By default the run is a worked example (NLP_LLMS_WORKED=1): every
@workshop.solution(N) binds the reference solution, so a full run exercises the
solutions and every checkpoint. No API keys are needed: labs fall back to open
models when none are set.

Run:  uv run --group execute python scripts/test_notebooks.py [slug ...]

Options (used by .github/workflows/publish.yml and health.yml):
  --save DIR             write each executed notebook, with its outputs, to DIR
  --expect-hub           fail a notebook that ran but fell back from its open-model path
                         (see FALLBACK_MARKERS), so a green run means the Hub path ran
  --learner              run as a participant who has written nothing: the run must stop
                         at a checkpoint with the harness's recovery message ("not written
                         yet" or "failed on your code"), not in a provided cell
  --verify-checkpoints   before each exercise's first checkpoint, run a copy of it on the
                         unfinished stub; it must fail, or the checkpoint cannot tell a
                         finished exercise from an unfinished one
  --record DIR           write a run record batch (runs/README.md) to DIR; give the
                         machine with --env (a key of readiness.envs in _variables.yml)
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import functools
import json
import os
import platform
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

import nbformat
import yaml
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError, CellTimeoutError, DeadKernelError

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = ROOT / "notebooks"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness  # noqa: E402
import run_records  # noqa: E402

# Per-cell limit. CI raises it: CPU training cells in Labs 2 and 7 run long.
TIMEOUT_SECONDS = int(os.environ.get("NLP_LLMS_CELL_TIMEOUT", "900"))

# Text a lab prints when it could not load an open model and carried on with a
# stand-in. Labs 8, 11, 13 and 14 fall back silently on purpose (a participant
# without Hub access still finishes), so with --expect-hub these count as failures.
FALLBACK_MARKERS = harness.FALLBACK_MARKERS
# Environment variables that put a lab on its offline path (publish.yml's notebooks job).
OFFLINE_FLAGS = run_records.OFFLINE_FLAGS
# What the harness prints when a checkpoint stops a participant who has not finished an
# exercise: a pure stub raised NotImplementedError, or a partial stub gave a wrong answer.
LEARNER_MESSAGES = ("is not written yet", "you have not written yet", "failed on your code")


def printed(nb: nbformat.NotebookNode) -> str:
    """Everything the notebook printed."""
    return "".join(
        "".join(out.get("text", ""))
        for cell in nb.cells
        if cell.cell_type == "code"
        for out in cell.get("outputs", [])
        if out.get("output_type") == "stream"
    )


def fallbacks(nb: nbformat.NotebookNode) -> list[str]:
    """The fallback markers found in the notebook's printed output."""
    text = printed(nb)
    return [marker.strip() for marker in FALLBACK_MARKERS if marker in text]


def checkpoint_exercise(cell) -> object | None:
    """N from a checkpoint cell's `workshop.checkpoint(N, ...)`, or None for a checkpoint
    of provided code (`workshop.checkpoint(label=...)`)."""
    code = "\n".join(
        line for line in cell.source.splitlines() if not line.lstrip().startswith(("%", "!"))
    )
    try:
        tree = ast.parse(code)
    except SyntaxError:  # other IPython syntax: read the first statement by pattern
        m = re.search(r"workshop\.checkpoint\((\d+|'[^']*'|\"[^\"]*\")?", cell.source)
        return ast.literal_eval(m.group(1)) if m and m.group(1) else None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "checkpoint"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "workshop"
        ):
            return (
                node.args[0].value if node.args and isinstance(node.args[0], ast.Constant) else None
            )
    return None


def with_verification(nb: nbformat.NotebookNode) -> int:
    """Insert, before each exercise's first checkpoint, a copy of it run on the stub.
    Returns how many checkpoints will be verified."""
    cells, seen, count = [], set(), 0
    for cell in nb.cells:
        tags = cell.get("metadata", {}).get("tags", [])
        n = checkpoint_exercise(cell) if cell.cell_type == "code" and "checkpoint" in tags else None
        if n is not None and n not in seen and "no-verify" not in tags:
            seen.add(n)
            count += 1
            copy = nbformat.v4.new_code_cell(cell.source)
            copy.metadata["tags"] = ["raises-exception"]
            cells += [
                nbformat.v4.new_code_cell(f"workshop._verify_begin({n!r})"),
                copy,
                nbformat.v4.new_code_cell(f"workshop._verify_end({n!r})"),
            ]
        cells.append(cell)
    nb.cells = cells
    return count


def cell_seconds(cell) -> float:
    ex = cell.get("metadata", {}).get("execution", {})
    start, end = ex.get("iopub.status.busy"), ex.get("shell.execute_reply")
    if not start or not end:
        return 0.0
    parse = dt.datetime.fromisoformat
    return max(
        0.0,
        (parse(end.replace("Z", "+00:00")) - parse(start.replace("Z", "+00:00"))).total_seconds(),
    )


def phases(nb: nbformat.NotebookNode) -> dict[str, float]:
    """Seconds by the first matching cell tag: setup, exercise, solution, checkpoint,
    generated; anything untagged (provided runs, training) is `other`."""
    order = ("setup", "exercise", "solution", "checkpoint", "generated")
    out: dict[str, float] = {}
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        tags = cell.get("metadata", {}).get("tags", [])
        key = next((t for t in order if t in tags), "other")
        out[key] = round(out.get(key, 0.0) + cell_seconds(cell), 1)
    return out


def run(path: Path, save: Path | None, verify: bool) -> dict:
    nb = nbformat.read(path, as_version=4)
    verified = with_verification(nb) if verify else 0
    started = time.monotonic()
    ok, error = True, ""
    with tempfile.TemporaryDirectory() as workdir:
        client = NotebookClient(
            nb,
            timeout=TIMEOUT_SECONDS,
            kernel_name="python3",
            resources={"metadata": {"path": workdir}},
            record_timing=True,
        )
        try:
            client.execute()
        except CellExecutionError as exc:
            ok, error = False, str(exc)
        except (CellTimeoutError, DeadKernelError) as exc:  # not CellExecutionError subclasses
            ok, error = False, f"{type(exc).__name__}: {exc}"
    if save is not None:
        save.mkdir(parents=True, exist_ok=True)
        nbformat.write(nb, save / path.name)
    return {
        "ok": ok,
        "seconds": time.monotonic() - started,
        "error": error,
        "fallbacks": fallbacks(nb),
        "learner_message": any(
            line.startswith("[workshop] Checkpoint") and any(m in line for m in LEARNER_MESSAGES)
            for line in printed(nb).splitlines()
        ),
        "phases": phases(nb),
        "verified": verified,
    }


@functools.cache
def has_exercises(path: Path) -> bool:
    nb = nbformat.read(path, as_version=4)
    return any("exercise" in c.get("metadata", {}).get("tags", []) for c in nb.cells)


def reads_offline_flags(path: Path) -> bool:
    """Whether a notebook has an offline path at all: one that reads none of the flags in
    its own cells runs its real path even in an offline batch. The generated harness cell
    names every flag, so it is not counted."""
    nb = nbformat.read(path, as_version=4)
    text = "".join(c.source for c in nb.cells if c.get("id") not in harness.GENERATED_CODE_IDS)
    return any(flag in text for flag in OFFLINE_FLAGS)


def status_of(path: Path, r: dict, learner: bool) -> str:
    """The same verdict the console prints: a learner run passes when it stops as expected."""
    if learner and has_exercises(path):
        return "pass" if (not r["ok"] and r["learner_message"]) else "fail"
    return "pass" if r["ok"] else "fail"


def entry_path(path: Path, r: dict, offline: bool) -> dict:
    """Per-notebook corrections to the batch's path. A run that fell back to stand-ins ran
    test doubles, whatever the flags; a notebook with no offline path ran its only path."""
    if r["fallbacks"]:
        return {"path": "offline", "note": "fell back: " + "; ".join(r["fallbacks"])}
    if offline and not reads_offline_flags(path):
        quick = {k: v for k, v in [("NLP_LLMS_QUICK", os.environ.get("NLP_LLMS_QUICK"))] if v}
        return {
            "path": "open",
            "settings": quick,
            "note": "reads none of the offline flags: its only path",
        }
    return {}


def write_record(
    directory: Path, env: str, results: list[tuple[Path, dict]], learner: bool
) -> Path:
    """One run record batch for this invocation (runs/README.md)."""
    offline = any(os.environ.get(flag) for flag in OFFLINE_FLAGS)
    keyed = any(
        os.environ.get(k) for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "TYPESAFE_API_KEY")
    )
    today = dt.date.today().isoformat()
    batch = {
        "schema": run_records.SCHEMA,
        "date": today,
        "source": "test_notebooks",
        "env": env,
        "env_detail": f"{platform.platform()}, Python {platform.python_version()}",
        "path": "offline" if offline else "keyed" if keyed else "open",
        "mode": "learner" if learner else "worked",
        "settings": {k: os.environ[k] for k in harness.RECORDED_SETTINGS if os.environ.get(k)},
        "evidence": "scripts/test_notebooks.py --record",
        "runs": [
            {
                "notebook": path.stem,
                "scope": "notebook",
                "status": status_of(path, r, learner),
                "seconds": round(r["seconds"], 1),
                "content_sha": run_records.content_sha(path.stem),
                "phases": r["phases"],
                **entry_path(path, r, offline),
            }
            for path, r in results
        ],
    }
    name = f"{today}-{env}-{batch['path']}-{int(time.time())}.json"
    body = json.dumps(batch, indent=2) + "\n"
    envs = set(
        yaml.safe_load((ROOT / "_variables.yml").read_text(encoding="utf-8"))["readiness"]["envs"]
    )
    _, problems = run_records.check_batch(
        name, body, envs, {p.stem for p in NOTEBOOKS.glob("*.ipynb")}
    )
    if problems:
        raise SystemExit("Run record not written:\n  " + "\n  ".join(problems))
    directory.mkdir(parents=True, exist_ok=True)
    out = directory / name
    out.write_text(body, encoding="utf-8")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("slugs", nargs="*", help="notebook stems; default: all")
    parser.add_argument("--save", type=Path, help="write executed notebooks here")
    parser.add_argument("--expect-hub", action="store_true", help="fail on a fallback")
    parser.add_argument("--learner", action="store_true", help="run with nothing written")
    parser.add_argument("--verify-checkpoints", action="store_true")
    parser.add_argument("--record", type=Path, help="write a run record batch here")
    parser.add_argument("--env", help="the machine, a key of readiness.envs (with --record)")
    args = parser.parse_args()
    if args.record and not args.env:
        parser.error("--record needs --env")

    # Notebooks run in a temporary directory, so point the data loader
    # (data/README.md) at a copy of the repository's data instead of the network. A copy,
    # because the loader caches what it downloads (Lab 7's Dolly file, on its real path)
    # into that folder, and a run must never add files to data/.
    scratch = tempfile.TemporaryDirectory()
    if "NLP_LLMS_DATA" not in os.environ:
        data = Path(scratch.name) / "data"
        shutil.copytree(ROOT / "data", data)
        os.environ["NLP_LLMS_DATA"] = str(data)
    os.environ["NLP_LLMS_WORKED"] = "0" if args.learner else "1"
    wanted = set(args.slugs)
    paths = [p for p in sorted(NOTEBOOKS.glob("*.ipynb")) if not wanted or p.stem in wanted]
    missing = wanted - {p.stem for p in paths}
    if missing:
        print(f"No such notebook: {', '.join(sorted(missing))}")
        return 2
    failures, results = 0, []
    if args.learner:
        # Only labs with exercises have something for a learner run to stop at.
        paths = [p for p in paths if has_exercises(p)]
    for path in paths:
        r = run(path, args.save, args.verify_checkpoints)
        results.append((path, r))
        if args.learner and has_exercises(path):
            # A participant who has written nothing must be stopped at a checkpoint, with the
            # harness's recovery message, not by a provided cell that calls the stub first.
            stopped = not r["ok"] and r["learner_message"]
            status = "PASS" if stopped else "FAIL"
            print(
                f"{status}  {path.name}  learner run "
                + ("stopped as expected" if stopped else "did not stop at an unwritten TODO"),
                flush=True,
            )
            failures += not stopped
            continue
        fell_back = r["ok"] and args.expect_hub and bool(r["fallbacks"])
        status = "FAIL" if not r["ok"] else "FALLBACK" if fell_back else "PASS"
        note = (
            f"  fell back: {'; '.join(r['fallbacks'])}"
            if r["fallbacks"] and (fell_back or not r["ok"])
            else ""
        )
        extra = (
            f"  ({r['verified']} checkpoints verified on stubs)" if args.verify_checkpoints else ""
        )
        print(f"{status}  {path.name}  {r['seconds']:.1f}s{extra}{note}", flush=True)
        if not r["ok"] or fell_back:
            failures += 1
        if not r["ok"]:
            print(r["error"])
    if args.record:
        print(f"Run record: {write_record(args.record, args.env, results, args.learner)}")
    print(f"{len(paths) - failures} of {len(paths)} notebooks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
