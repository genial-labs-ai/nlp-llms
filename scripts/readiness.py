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
        path = ROOT / check["exists"]
        return (
            path.exists(),
            f"`{check['exists']}` {'exists' if path.exists() else 'does not exist'}",
        )
    if "var" in check:
        value = lookup(v, check["var"])
        return value == check["equals"], f"`{check['var']}` is `{value}`"
    if "json" in check:
        value = json.loads((ROOT / check["json"]).read_text(encoding="utf-8"))[check["key"]]
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


def _newest(records: list[dict]) -> dict | None:
    """The most recent record. Ties on the same date go to a failure, then to the whole
    notebook over a part, then to the later entry of a batch: a same-day pass never hides
    a same-day failure."""
    if not records:
        return None
    return max(
        records,
        key=lambda r: (
            r["date"],
            r["status"] == "fail",
            r["scope"] == "notebook",
            r["file"],
            r["index"],
        ),
    )


def _mark_stale(r: dict | None, current_sha: str) -> dict | None:
    """Flag a record made against other code than the committed notebook's."""
    if r is not None and r.get("content_sha") and r["content_sha"] != current_sha:
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
        and (r.get("settings") or {}).get("NLP_LLMS_QUICK") != "1"
    )


def evidence(v: dict, slug: str, runtime: str, records: list[dict]) -> dict:
    """The newest record of each kind for one notebook, passing or failing.

    teaching: a teaching-eligible record (see teaching_eligible); stale when its
        content_sha no longer matches the notebook.
    real, real_partial: every other run of the real path (open or keyed), whole or part.
    code: the offline path (test doubles).
    A newer failure replaces an older pass. Records without a content_sha (backfills)
    cannot be checked for staleness.
    """
    current = run_records.content_sha(slug)
    mine = [r for r in records if r["notebook"] == slug]
    real = [r for r in mine if r["path"] in ("open", "keyed")]
    eligible = [r for r in real if teaching_eligible(v, r, runtime)]
    others = [r for r in real if not teaching_eligible(v, r, runtime)]
    return {
        "teaching": _mark_stale(_newest(eligible), current) if eligible else None,
        "real": _mark_stale(_newest([r for r in others if r["scope"] == "notebook"]), current),
        "real_partial": _mark_stale(
            _newest([r for r in others if r["scope"] == "partial"]), current
        ),
        "code": _mark_stale(_newest([r for r in mine if r["path"] == "offline"]), current),
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
    if (r.get("settings") or {}).get("NLP_LLMS_QUICK") == "1" and r["path"] != "offline":
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
    ci = [r for r in records if r["env"] in ci_envs]
    ci_date = max((r["date"] for r in ci), default=None)
    ci_latest = {}
    for slug in {r["notebook"] for r in ci if r["date"] == ci_date}:
        ci_latest[slug] = _newest([r for r in ci if r["notebook"] == slug and r["date"] == ci_date])
    summary = {
        "labs": len(labs),
        "teaching": sum(1 for e in ev.values() if passed(e["teaching"])),
        "real": sum(1 for e in ev.values() if passed(e["real"])),
        "real_partial_only": sum(
            1 for e in ev.values() if e["real"] is None and passed(e["real_partial"])
        ),
        "notebooks": len(notebooks),
        "items_open": sum(1 for i in all_items if not i["closed"]),
        "items": len(all_items),
        "as_of": max((r["date"] for r in records), default=None),
        "ci_date": ci_date,
        "ci_passed": sum(1 for r in ci_latest.values() if r["status"] == "pass"),
        "ci_failed": sum(1 for r in ci_latest.values() if r["status"] == "fail"),
        "ci_doubles": sum(
            1 for r in ci_latest.values() if r["status"] == "pass" and r["path"] == "offline"
        ),
    }
    summary["ready"] = summary["teaching"] == summary["labs"] and summary["items_open"] == 0
    return {"evidence": ev, "items": all_items, "summary": summary}
