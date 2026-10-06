"""Load and validate run records: the evidence that a notebook ran (runs/README.md).

A file holds one batch: top-level fields apply to every entry of `runs`, and an entry
may override them. `load_all()` returns one flat dict per entry, with `file` set to the
batch's file name.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"

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
}
DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
# Anything that looks like a credential must never be committed in a record.
SECRET = re.compile(r"(sk-[A-Za-z0-9_-]{16,}|sk-ant-|hf_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16})")


def load_file(path: Path) -> list[dict]:
    batch = json.loads(path.read_text(encoding="utf-8"))
    shared = {k: batch[k] for k in INHERITED if k in batch}
    out = []
    for entry in batch.get("runs", []):
        record = {**shared, **entry, "file": path.name}
        out.append(record)
    return out


def load_all(runs_dir: Path = RUNS) -> list[dict]:
    records = []
    for path in sorted(runs_dir.glob("*.json")):
        records.extend(load_file(path))
    return records


def validate_file(path: Path, envs: set[str], notebooks: set[str]) -> list[str]:
    """Problems with one batch file, as readable strings (empty when it is valid)."""
    errors = []
    text = path.read_text(encoding="utf-8")
    if SECRET.search(text):
        errors.append(f"{path.name}: contains something that looks like a credential")
    batch = json.loads(text)
    if batch.get("schema") != SCHEMA:
        errors.append(f"{path.name}: schema must be {SCHEMA}")
    if not batch.get("runs"):
        errors.append(f"{path.name}: no runs")
    for i, r in enumerate(load_file(path)):
        where = f"{path.name} runs[{i}] ({r.get('notebook')})"
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
        if seconds is not None and not (isinstance(seconds, (int, float)) and seconds >= 0):
            errors.append(f"{where}: seconds must be a non-negative number or null")
        if r.get("scope") == "partial" and not r.get("scope_note"):
            errors.append(f"{where}: a partial run needs a scope_note")
        if r.get("source") == "backfill" and r.get("content_sha"):
            errors.append(f"{where}: a backfilled record cannot carry a content_sha")
    return errors
