"""Can this commit be released as ready to teach? Lists every blocker, exits 1 if any.

A release needs, as of the given date:

- for every lab, a passing teaching-eligible run of its current code on the module's
  own runtime (scripts/readiness.py: worked, whole notebook, release path, no QUICK,
  not a backfill), no older than readiness.max_run_age_days;
- every readiness item closed (readiness.items in _variables.yml).

It reads only the repository and the date, so the same commit and date always give
the same answer. The site deploys from main regardless: this gates the GitHub
Release (.github/workflows/release.yml), not the deploy.

    uv run --group site python scripts/release_check.py [--as-of 2026-10-06]
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import readiness  # noqa: E402
import run_records  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def lab_blocker(v: dict, m: dict, e: dict, as_of: dt.date) -> str | None:
    """Why module m's lab is not ready to teach, or None."""
    r = m["readiness"]
    env = v["readiness"]["envs"][r["runtime"]]["name"]
    run = e["teaching"]
    if run is None:
        path = v["readiness"]["release_path"]
        return f"no teaching run on {env} ({path} path, worked, whole notebook)"
    if run.get("stale"):
        return f"the newest teaching run ({run['date']}) predates the notebook's current code"
    if run["status"] != "pass":
        return f"the newest teaching run failed ({run['date']}, {env})"
    age = (as_of - dt.date.fromisoformat(run["date"])).days
    limit = v["readiness"]["max_run_age_days"]
    if age > limit:
        return f"the newest teaching run is {age} days old ({run['date']}); the limit is {limit}"
    return None


def blockers(v: dict, records: list[dict], as_of: dt.date) -> list[str]:
    report = readiness.build(v, records)
    out = []
    for m in sorted(v["modules"].values(), key=lambda m: m["n"]):
        if not m.get("notebook", True):
            continue
        why = lab_blocker(v, m, report["evidence"][m["slug"]], as_of)
        if why:
            out.append(f"Module {m['n']} ({m['title']}): {why}")
    for item in report["items"]:
        if not item["closed"]:
            out.append(f"Item {item['id']} ({item['title']}): {item['why']}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--as-of",
        type=dt.date.fromisoformat,
        default=dt.date.today(),
        help="the release date (default: today)",
    )
    args = parser.parse_args()
    v = yaml.safe_load((ROOT / "_variables.yml").read_text(encoding="utf-8"))
    records = run_records.load_valid(set(v["readiness"]["envs"]))
    found = blockers(v, records, args.as_of)
    if found:
        lines = [f"Not ready to release as of {args.as_of}: {len(found)} blocker(s).", ""]
        lines += [f"- {b}" for b in found]
    else:
        lines = [f"Ready to release as of {args.as_of}: no blockers."]
    text = "\n".join(lines)
    print(text)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write(f"## Release check\n\n{text}\n")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
