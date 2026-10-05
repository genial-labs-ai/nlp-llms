"""Merge the pairs' capstone submissions and check each one (the instructor's collector).

Usage:  python scripts/collect_capstone.py SUBMISSIONS_DIR_OR_FILES... [--manifest PATH]

Each pair hands in capstone_<pair>_<path class>.json, written by the last cell of
notebooks/15-capstone.ipynb. This script checks every file (briefs/15-capstone.md, "What a pair
submits"):

1. the scoring hash equals the hash of scripts/capstone_score.py, the costs are the fixed ones and
   the manifest hash equals data/capstone_eval_v1.json's; otherwise "not comparable";
2. `test` was run exactly twice; otherwise "test used for development";
3. the budgets were respected; otherwise "over budget" (shown, not ranked);
4. `stub` submissions are "code check only" and are never ranked.

It prints one table per path class, grouped by menu code, then the multiple-comparisons note.
Standard library only; nothing is sent anywhere.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "capstone_eval_v1.json"
PATH_CLASSES = ["keyed-jev", "keyed-llm", "open-t4", "open-cpu", "stub"]
MENU = ["R1", "R2", "R3", "P1", "P2", "V1", "V2", "V3", "G1", "T1", "M1", "other"]


def _scoring_module():
    spec = importlib.util.spec_from_file_location(
        "capstone_score", ROOT / "scripts" / "capstone_score.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SCORING = _scoring_module()


def expected_manifest_sha256(path: Path = MANIFEST) -> str | None:
    """SHA-256 of the registered manifest's bytes, or None while it does not exist."""
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def check(sub: dict, manifest_sha256: str | None) -> tuple[str, list[str]]:
    """(status, reasons). Status is one of: ranked, over budget, test used for development,
    not comparable, code check only."""
    reasons = []
    if sub.get("scoring_hash") != SCORING.scoring_hash():
        reasons.append("scoring hash differs from scripts/capstone_score.py")
    if sub.get("loss") != SCORING.LOSS or sub.get("abstain_sentence") != SCORING.ABSTAIN:
        reasons.append("costs or abstention sentence differ from the fixed ones")
    if manifest_sha256 is None:
        reasons.append("no registered manifest (data/capstone_eval_v1.json does not exist yet)")
    elif sub.get("manifest_sha256") != manifest_sha256:
        reasons.append("manifest hash differs from data/capstone_eval_v1.json")
    if sub.get("path_class") not in PATH_CLASSES:
        reasons.append(f"unknown path class {sub.get('path_class')!r}")
    if reasons:
        return "not comparable", reasons
    if sub["path_class"] == "stub":
        return "code check only", ["stub path: measures the notebook's code, not a model"]
    if sub.get("test_runs") != 2:
        return "test used for development", [f"test was run {sub.get('test_runs')} times, not 2"]
    budget = sub.get("budget", {})
    over = [k for k in ("calls_ok", "time_ok", "usd_ok") if budget.get(k) is False]
    if over:
        return "over budget", [f"budget check failed: {', '.join(over)}"]
    return "ranked", []


def load(paths: list[Path]) -> list[dict]:
    files = []
    for p in paths:
        files += sorted(p.glob("capstone_*.json")) if p.is_dir() else [p]
    subs = []
    for f in files:
        sub = json.loads(f.read_text(encoding="utf-8"))
        sub["_file"] = f.name
        subs.append(sub)
    return subs


def _d(x, fmt="+.3f"):
    return "n/a" if x is None else format(x, fmt)


def row(sub: dict, status: str) -> dict:
    c = sub["compare"]["test"]
    d = c["deltas"]
    return {
        "pair": sub["pair"],
        "code": sub["hypothesis"].get("component", "?"),
        "status": status,
        "d_cost": _d(d["cost_bar"]),
        "d_acc": _d(d["acc"]),
        "d_absU": _d(d["abs_U"]),
        "d_absA": _d(d["abs_A"]),
        "d_uns": _d(d["uns"]),
        "g/l/p": f"{c['gained']}/{c['lost']}/{c['p']:.3f}",
        "flips": "n/a" if sub.get("flips") is None else str(sub["flips"]),
        "d_usd": _d(d["usd_per_q"], "+.4f"),
        "d_lat_s": _d(d["latency_median"], "+.2f"),
    }


def table(rows: list[dict]) -> str:
    cols = list(rows[0])
    width = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    lines = ["  ".join(c.ljust(width[c]) for c in cols)]
    lines += ["  ".join(str(r[c]).ljust(width[c]) for c in cols) for r in rows]
    return "\n".join(lines)


def report(subs: list[dict], manifest_sha256: str | None) -> str:
    out = []
    checked = [(sub, *check(sub, manifest_sha256)) for sub in subs]
    for sub, status, reasons in checked:
        if reasons:
            out.append(f"{sub['_file']}: {status}: {'; '.join(reasons)}")
    order = {code: i for i, code in enumerate(MENU)}
    for path_class in PATH_CLASSES + ["(other)"]:
        group = [
            (s, st)
            for s, st, _ in checked
            if (s.get("path_class") if s.get("path_class") in PATH_CLASSES else "(other)")
            == path_class
        ]
        if not group:
            continue
        group.sort(key=lambda x: (order.get(x[0]["hypothesis"].get("component"), 99), x[0]["pair"]))
        out.append(f"\n== {path_class} ({len(group)} submissions) ==")
        out.append(table([row(s, st) for s, st in group]))
        m = sum(st == "ranked" for _, st in group)
        out.append(
            f"With m = {m} ranked pairs each tested at 0.05, about {0.05 * m:.1f} significant "
            "results are expected by chance."
        )
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args(argv)
    subs = load(args.paths)
    if not subs:
        print("No capstone_*.json files found.")
        return 1
    print(report(subs, expected_manifest_sha256(args.manifest)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
