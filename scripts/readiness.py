"""What is known about whether each module can be taught, and on what evidence.

Reads the readiness fields in _variables.yml (readiness: and modules.mNN.readiness) and
the run records in runs/. Used by scripts/gen_tables.py to write the readiness tables
and by scripts/release_check.py, which also applies readiness.max_run_age_days.
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
            verb = " does not exist" if len(missing) == 1 else " do not exist"
            return False, ", ".join(f"`{p}`" for p in missing) + verb
        return True, ", ".join(f"`{p}`" for p in paths) + " exists"
    if "var" in check:
        try:
            value = lookup(v, check["var"])
        except (KeyError, TypeError):
            return False, f"`{check['var']}` is missing from `_variables.yml`"
        if "equals" not in check:
            return False, f"the check on `{check['var']}` names no value to equal"
        return value == check["equals"], f"`{check['var']}` is `{value}`"
    if "json" in check:
        try:
            data = json.loads((ROOT / check["json"]).read_text(encoding="utf-8"))
            value = data[check["key"]]
        except (OSError, ValueError, KeyError, TypeError):
            return False, f"`{check['json']}` or its `{check['key']}` is missing"
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return False, f"`{check['key']}` in `{check['json']}` is not a number"
        return value >= check["at_least"], f"`{check['key']}` is {value} of {check['at_least']}"
    if "absent" in check:
        try:
            text = (ROOT / check["absent"]).read_text(encoding="utf-8")
        except OSError:
            return False, f"`{check['absent']}` does not exist"
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
    """The most telling record. A record of the current code always wins over a stale one,
    which says nothing about the current code; then the most recent; on the same date a
    failure over a pass, then the whole notebook over a part, then full settings over
    QUICK, then the later entry of a batch: a same-day pass never hides a same-day failure."""
    if not records:
        return None
    return max(
        records,
        key=lambda r: (
            not r.get("stale"),
            r["date"],
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
    # A learner-mode run stops at a checkpoint on purpose: it is evidence of the harness,
    # not of the lab running end to end, so only worked runs count below.
    rest = [
        _with_staleness(r, current, False)
        for r in mine
        if not teaching_eligible(v, r, runtime) and r["mode"] == "worked"
    ]
    real = [r for r in rest if r["path"] in ("open", "keyed")]
    ci = [r for r in rest if r["env"] in ci_envs]
    off_ci = [r for r in real if r["env"] not in ci_envs]
    ci_real = [r for r in ci if r["path"] != "offline"]
    return {
        "teaching": _newest(eligible),
        "other": _newest(off_ci),
        "ci": _newest(ci),
        "ci_real": _newest(ci_real),
        # For the summary: the newest whole-notebook real run, and the newest real run of
        # any scope, so that a newer partial pass does not hide an older full one.
        "real_full": _newest([r for r in off_ci + ci_real if r["scope"] == "notebook"]),
        "real_any": _newest(off_ci + ci_real),
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
    setup = {"slug": v["setup"]["slug"], "readiness": {"runtime": v["setup"].get("runtime")}}
    ev = {
        m["slug"]: evidence(v, m["slug"], m["readiness"]["runtime"], records)
        for m in [*labs, setup]
    }
    lab_ev = [ev[m["slug"]] for m in labs]
    all_items = items(v)
    ci_latest = [e["ci"] for e in ev.values() if e["ci"] is not None]
    not_taught = [e for e in lab_ev if not passed(e["teaching"])]

    def end_to_end(e: dict) -> bool:
        """The newest whole-notebook real run passed, and nothing failed after it."""
        full, newest = e["real_full"], e["real_any"]
        later_failure = (
            full is not None
            and newest is not None
            and newest is not full
            and newest["status"] == "fail"
            and newest["date"] >= full["date"]
        )
        return passed(full) and not later_failure

    summary = {
        "labs": len(labs),
        "teaching": len(labs) - len(not_taught),
        "real": sum(1 for e in not_taught if end_to_end(e)),
        "real_quick": sum(1 for e in not_taught if end_to_end(e) and _quick(e["real_full"])),
        "real_partial_only": sum(
            1 for e in not_taught if not end_to_end(e) and passed(e["real_any"])
        ),
        "notebooks": len(ev),
        "items_open": sum(1 for i in all_items if not i["closed"]),
        "items": len(all_items),
        "as_of": max((r["date"] for r in records), default=None),
        "ci_from": min((r["date"] for r in ci_latest), default=None),
        "ci_to": max((r["date"] for r in ci_latest), default=None),
        "ci_passed": sum(1 for r in ci_latest if passed(r)),
        "ci_failed": sum(1 for r in ci_latest if r["status"] == "fail" and not r.get("stale")),
        "ci_stale": sum(1 for r in ci_latest if r.get("stale")),
        "ci_doubles": sum(1 for r in ci_latest if passed(r) and r["path"] == "offline"),
    }
    summary["setup_ready"] = passed(ev[setup["slug"]]["teaching"])
    summary["ready"] = (
        summary["teaching"] == summary["labs"]
        and summary["setup_ready"]
        and summary["items_open"] == 0
    )
    return {
        "evidence": {k: e for k, e in ev.items() if k != setup["slug"]},
        "setup": ev[setup["slug"]],
        "items": all_items,
        "summary": summary,
    }
