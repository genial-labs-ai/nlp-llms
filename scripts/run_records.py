"""Load and validate run records: the evidence that a notebook ran (runs/README.md).

A file holds one batch: top-level fields apply to every entry of `runs`, and an entry
may override them. `load_all()` returns one flat dict per entry, with `file` set to the
batch's file name.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
NOTEBOOKS = ROOT / "notebooks"

SCHEMA = 1
SOURCES = ("test_notebooks", "colab", "backfill")
PATHS = ("offline", "open", "keyed")
MODES = ("worked", "learner", "unknown")
SCOPES = ("notebook", "partial")
STATUSES = ("pass", "fail")
# Fields an entry inherits from its batch.
INHERITED = (
    "date",
    "source",
    "env",
    "env_detail",
    "path",
    "mode",
    "settings",
    "commit",
    "evidence",
    "content_sha",
    "packages",
)
REQUIRED = (
    "date",
    "source",
    "env",
    "path",
    "mode",
    "notebook",
    "scope",
    "status",
    "seconds",
    "evidence",
)
ALLOWED = set(INHERITED) | {
    "notebook",
    "scope",
    "scope_note",
    "status",
    "seconds",
    "phases",
    "note",
    "file",
    "index",
}
BATCH_FIELDS = set(INHERITED) | {"schema", "runs"}
DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
# Anything that looks like a credential must never be committed in a record.
SECRET = re.compile(
    r"(?<![A-Za-z0-9])(sk-[A-Za-z0-9_-]{20,}|sk-ant-[A-Za-z0-9_-]+|hf_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16})"
)


def content_sha(slug: str) -> str:
    """A short hash of a notebook's code cells: a run record made against other code than
    the committed notebook's is stale. (The generated header and footer are markdown.)"""
    nb = json.loads((NOTEBOOKS / f"{slug}.ipynb").read_text(encoding="utf-8"))
    code = ["".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "code"]
    return hashlib.sha256("\n\x1e\n".join(code).encode("utf-8")).hexdigest()[:16]


def records_of(batch: dict, name: str) -> list[dict]:
    """The flat records of one parsed batch: shared fields, then each entry's own."""
    shared = {k: batch[k] for k in INHERITED if k in batch}
    return [
        {**shared, **entry, "file": name, "index": index}
        for index, entry in enumerate(batch.get("runs", []))
    ]


def load_file(path: Path) -> list[dict]:
    return records_of(json.loads(path.read_text(encoding="utf-8")), path.name)


def load_all(runs_dir: Path = RUNS) -> list[dict]:
    records = []
    for path in sorted(runs_dir.glob("*.json")):
        records.extend(load_file(path))
    return records


def check_batch(name: str, text: str, envs: set[str], notebooks: set[str]):
    """(records, problems) for one batch file's text; records is None when unreadable."""
    errors = []
    if SECRET.search(text):
        errors.append(f"{name}: contains something that looks like a credential")
    try:
        batch = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, errors + [f"{name}: not valid JSON ({exc})"]
    if not isinstance(batch, dict) or not isinstance(batch.get("runs"), list):
        return None, errors + [f"{name}: must be an object with a list of runs"]
    if not all(isinstance(entry, dict) for entry in batch["runs"]):
        return None, errors + [f"{name}: every entry of runs must be an object"]
    if batch.get("schema") != SCHEMA:
        errors.append(f"{name}: schema must be {SCHEMA}")
    if not batch["runs"]:
        errors.append(f"{name}: no runs")
    for field in sorted(set(batch) - BATCH_FIELDS):
        errors.append(f"{name}: unknown top-level field {field} (only shared fields go here)")
    records = records_of(batch, name)
    for i, r in enumerate(records):
        where = f"{name} runs[{i}] ({r.get('notebook')})"
        for field in REQUIRED:
            if field not in r:
                errors.append(f"{where}: missing {field}")
        for field in r:
            if field not in ALLOWED:
                errors.append(f"{where}: unknown field {field}")
        checks = [
            ("source", SOURCES),
            ("path", PATHS),
            ("mode", MODES),
            ("scope", SCOPES),
            ("status", STATUSES),
        ]
        for field, allowed in checks:
            if field in r and r[field] not in allowed:
                errors.append(f"{where}: {field} {r[field]!r} not in {allowed}")
        if r.get("env") not in envs:
            errors.append(f"{where}: env {r.get('env')!r} is not a key of readiness.envs")
        if r.get("notebook") not in notebooks:
            errors.append(f"{where}: no notebook named {r.get('notebook')!r}")
        if not DATE.match(str(r.get("date", ""))):
            errors.append(f"{where}: date must be YYYY-MM-DD")
        seconds = r.get("seconds")
        is_number = isinstance(seconds, (int, float)) and not isinstance(seconds, bool)
        if seconds is not None and not (is_number and seconds >= 0):
            errors.append(f"{where}: seconds must be a non-negative number or null")
        if r.get("scope") == "partial" and not r.get("scope_note"):
            errors.append(f"{where}: a partial run needs a scope_note")
        settings = r.get("settings")
        if settings is not None and not (
            isinstance(settings, dict) and all(isinstance(x, str) for x in settings.values())
        ):
            errors.append(f"{where}: settings must be an object of strings, as in the environment")
        if r.get("source") != "backfill" and not r.get("content_sha"):
            errors.append(f"{where}: a record made by a tool needs the notebook's content_sha")
        if r.get("source") == "backfill" and r.get("content_sha"):
            errors.append(f"{where}: a backfilled record cannot carry a content_sha")
    return records, errors


def validate_file(path: Path, envs: set[str], notebooks: set[str]) -> list[str]:
    """Problems with one batch file, as readable strings (empty when it is valid)."""
    return check_batch(path.name, path.read_text(encoding="utf-8"), envs, notebooks)[1]


def load_valid(envs: set[str], runs_dir: Path = RUNS) -> list[dict]:
    """All records, each file read and parsed once; stops with every problem listed."""
    notebooks = {p.stem for p in NOTEBOOKS.glob("*.ipynb")}
    records, errors = [], []
    for path in sorted(runs_dir.glob("*.json")):
        batch_records, batch_errors = check_batch(
            path.name, path.read_text(encoding="utf-8"), envs, notebooks
        )
        errors += batch_errors
        records += batch_records or []
    if errors:
        raise SystemExit("Invalid run records:\n  " + "\n  ".join(errors))
    return records
