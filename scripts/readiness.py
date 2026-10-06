"""What is known about whether each module can be taught, and on what evidence.

Reads the readiness fields in _variables.yml (readiness: and modules.mNN.readiness) and
the run records in runs/. Used by scripts/gen_tables.py to write the readiness tables
(and, once it exists, by the release check, which also applies readiness.max_run_age_days).
Nothing here reads the clock: the output depends only on the
repository, so the generated pages do not drift from one day to the next.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import run_records

ROOT = Path(__file__).resolve().parent.parent


def lookup(v: dict, dotted: str):
    node = v
    for part in dotted.split("."):
        node = node[part]
    return node


def item_closed(v: dict, item: dict) -> tuple[bool, str]:
    """(closed, why) for one blocking item, by its mechanical check."""
    check = item["check"]
    if "exists" in check:
        paths = check["exists"] if isinstance(check["exists"], list) else [check["exists"]]
        missing = [p for p in paths if not (ROOT / p).exists()]
        if missing:
            return False, ", ".join(f"`{p}`" for p in missing) + " does not exist"
        return True, ", ".join(f"`{p}`" for p in paths) + " exists"
    if "var" in check:
        try:
            value = lookup(v, check["var"])
        except (KeyError, TypeError):
            return False, f"`{check['var']}` is missing from `_variables.yml`"
        return value == check["equals"], f"`{check['var']}` is `{value}`"
    if "json" in check:
        try:
            data = json.loads((ROOT / check["json"]).read_text(encoding="utf-8"))
            value = data[check["key"]]
        except (OSError, ValueError, KeyError, TypeError):
            return False, f"`{check['json']}` or its `{check['key']}` is missing"
        return value >= check["at_least"], f"`{check['key']}` is {value} of {check['at_least']}"
    if "absent" in check:
        text = (ROOT / check["absent"]).read_text(encoding="utf-8")
        present = re.search(check["pattern"], text) is not None
        return (
            not present,
            f"`{check['absent']}` {'still has' if present else 'no longer has'} its marker",
        )
    if check.get("manual"):
        return bool(check.get("done")), "done" if check.get("done") else "not done"
    raise ValueError(f"unknown check: {check}")


def items(v: dict) -> list[dict]:
    out = []
    for key, item in v["readiness"]["items"].items():
        closed, why = item_closed(v, item)
        out.append(
            {
                "id": key,
                "title": item["title"],
                "modules": item["modules"],
                "closed": closed,
                "why": why,
            }
        )
    return out


FALLBACK_LABELS = {"none": "One path", "open-model": "Open model", "toy": "Toy model"}


def _quick(r: dict) -> bool:
    return (r.get("settings") or {}).get("NLP_LLMS_QUICK") == "1"


def _newest(records: list[dict]) -> dict | None:
    """The most recent record. On the same date a record of the current code wins over a
    stale one, then a failure over a pass, then the whole notebook over a part, then full
    settings over QUICK, then the later entry of a batch: a same-day pass never hides a
    same-day failure of the same code."""
    if not records:
        return None
    return max(
        records,
        key=lambda r: (
            r["date"],
            not r.get("stale"),
            r["status"] == "fail",
            r["scope"] == "notebook",
            not _quick(r),
            r["file"],
            r["index"],
        ),
    )


def _with_staleness(r: dict, current_sha: str, missing_is_stale: bool) -> dict:
    """The record, flagged stale when it was made against other code than the notebook's."""
    sha = r.get("content_sha")
    if (sha and sha != current_sha) or (not sha and missing_is_stale):
        return {**r, "stale": True}
    return r


def teaching_eligible(v: dict, r: dict, runtime: str) -> bool:
    """Whether a record could show that a lab runs as taught: the whole notebook, worked
    (solutions bound), on the release path with no QUICK shortcuts, on the module's own
    runtime, recorded by a tool rather than copied from a report."""
    return (
        r["env"] == runtime
        and r["path"] == v["readiness"]["release_path"]
        and r["scope"] == "notebook"
        and r["mode"] == "worked"
        and r["source"] != "backfill"
        and not _quick(r)
    )


def evidence(v: dict, slug: str, runtime: str, records: list[dict]) -> dict:
    """The newest record of each kind for one notebook, passing or failing.

    teaching: the newest teaching-eligible record (see teaching_eligible). A missing
        content_sha counts as stale.
    other: the newest other run of the real path (open or keyed), whole or part, off CI.
    ci: the newest run on a CI runner, on any path; ci_real: the newest that used no
        test doubles.
    A newer failure replaces an older pass. Records without a content_sha (backfills)
    cannot be checked for staleness outside the teaching row.
    """
    current = run_records.content_sha(slug)
    ci_envs = {k for k, e in v["readiness"]["envs"].items() if e.get("ci")}
    mine = [r for r in records if r["notebook"] == slug]
    eligible = [
        _with_staleness(r, current, missing_is_stale=True)
        for r in mine
        if teaching_eligible(v, r, runtime)
    ]
    rest = [
        _with_staleness(r, current, False) for r in mine if not teaching_eligible(v, r, runtime)
    ]
    real = [r for r in rest if r["path"] in ("open", "keyed")]
    ci = [r for r in rest if r["env"] in ci_envs]
    return {
        "teaching": _newest(eligible),
        "other": _newest([r for r in real if r["env"] not in ci_envs]),
        "ci": _newest(ci),
        "ci_real": _newest([r for r in ci if r["path"] != "offline"]),
    }


def passed(r: dict | None) -> bool:
    return r is not None and r["status"] == "pass" and not r.get("stale")


def duration(seconds: float | None) -> str:
    if seconds is None:
        return "time not recorded"
    if seconds < 100:
        return f"{seconds:.0f} s"
    minutes = seconds / 60
    if minutes < 10:
        return f"{minutes:.1f} min"
    if minutes < 120:
        return f"{minutes:.0f} min"
    return f"{minutes / 60:.1f} h"


def describe(v: dict, r: dict) -> str:
    """'2026-10-06, Apple M1 Pro laptop, 4.4 min' for one record."""
    env = v["readiness"]["envs"][r["env"]]["name"]
    text = f"{r['date']}, {env}, {duration(r['seconds'])}"
    if r["status"] == "fail":
        text = f"**failed** {text}"
    if r.get("stale"):
        text += ", before the notebook last changed"
    if r["scope"] == "partial":
        text += f" ({r['scope_note']})"
    if _quick(r) and r["path"] != "offline":
        text += ", QUICK settings"
    if r["path"] == "keyed":
        text += ", with API keys"
    return text


def build(v: dict, records: list[dict]) -> dict:
    """Everything the readiness pages need, computed once."""
    labs = [m for m in v["modules"].values() if m.get("notebook", True)]
    ev = {m["slug"]: evidence(v, m["slug"], m["readiness"]["runtime"], records) for m in labs}
    notebooks = [m["slug"] for m in labs] + [v["setup"]["slug"]]
    all_items = items(v)
    ci_envs = {k for k, e in v["readiness"]["envs"].items() if e.get("ci")}
    ci_latest = [
        _newest([r for r in records if r["notebook"] == slug and r["env"] in ci_envs])
        for slug in notebooks
    ]
    ci_latest = [r for r in ci_latest if r is not None]
    not_taught = [e for e in ev.values() if not passed(e["teaching"])]

    def real_run(e: dict, scope: str) -> bool:
        """Whether the newest real-path run off the teaching runtime passed at this scope,
        on another machine or on the CI runner."""
        return any(passed(r) and r["scope"] == scope for r in (e["other"], e["ci_real"]) if r)

    summary = {
        "labs": len(labs),
        "teaching": len(labs) - len(not_taught),
        "real": sum(1 for e in not_taught if real_run(e, "notebook")),
        "real_partial_only": sum(
            1 for e in not_taught if not real_run(e, "notebook") and real_run(e, "partial")
        ),
        "notebooks": len(notebooks),
        "items_open": sum(1 for i in all_items if not i["closed"]),
        "items": len(all_items),
        "as_of": max((r["date"] for r in records), default=None),
        "ci_from": min((r["date"] for r in ci_latest), default=None),
        "ci_to": max((r["date"] for r in ci_latest), default=None),
        "ci_passed": sum(1 for r in ci_latest if r["status"] == "pass"),
        "ci_failed": sum(1 for r in ci_latest if r["status"] == "fail"),
        "ci_doubles": sum(1 for r in ci_latest if r["status"] == "pass" and r["path"] == "offline"),
    }
    summary["ready"] = summary["teaching"] == summary["labs"] and summary["items_open"] == 0
    return {"evidence": ev, "items": all_items, "summary": summary}
