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
    """The most recent record; among records of one day, the most complete."""
    if not records:
        return None
    return max(records, key=lambda r: (r["date"], r["scope"] == "notebook", r["file"]))


def evidence(v: dict, slug: str, runtime: str, records: list[dict]) -> dict:
    """The newest record of each kind for one notebook, passing or failing.

    teaching: the whole notebook, on its real path, on the runtime the module is designed
        for, recorded by a tool (never a backfill). It is marked stale when the notebook's
        code has changed since (its content_sha no longer matches).
    real, real_partial: the real path on any other machine.
    code: the offline path (test doubles).
    A newer failing record replaces an older pass, so a broken lab is never shown as fine.
    """
    mine = [r for r in records if r["notebook"] == slug]
    real = [r for r in mine if r["path"] in ("open", "keyed")]
    teaching = _newest(
        [
            r
            for r in real
            if r["env"] == runtime and r["scope"] == "notebook" and r["source"] != "backfill"
        ]
    )
    if teaching is not None and teaching.get("content_sha") != run_records.content_sha(slug):
        teaching = {**teaching, "stale": True}
    elsewhere = [r for r in real if r["env"] != runtime]
    return {
        "teaching": teaching,
        "real": _newest([r for r in elsewhere if r["scope"] == "notebook"]),
        "real_partial": _newest([r for r in elsewhere if r["scope"] == "partial"]),
        "code": _newest([r for r in mine if r["path"] == "offline"]),
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
    ci = [r for r in records if r["env"] == "gha-ubuntu"]
    ci_date = max((r["date"] for r in ci), default=None)
    ci_latest = [r for r in ci if r["date"] == ci_date]
    ci_passed = {r["notebook"] for r in ci_latest if r["status"] == "pass"}
    summary = {
        "labs": len(labs),
        "teaching": sum(1 for e in ev.values() if passed(e["teaching"])),
        "real": sum(1 for e in ev.values() if passed(e["real"])),
        "real_partial_only": sum(
            1 for e in ev.values() if not passed(e["real"]) and passed(e["real_partial"])
        ),
        "notebooks": len(notebooks),
        "items_open": sum(1 for i in all_items if not i["closed"]),
        "items": len(all_items),
        "as_of": max((r["date"] for r in records), default=None),
        "ci_date": ci_date,
        "ci_passed": len(ci_passed),
        "ci_failed": len({r["notebook"] for r in ci_latest if r["status"] == "fail"} - ci_passed),
        "ci_doubles": len(
            {r["notebook"] for r in ci_latest if r["status"] == "pass" and r["path"] == "offline"}
        ),
    }
    summary["ready"] = summary["teaching"] == summary["labs"] and summary["items_open"] == 0
    return {"evidence": ev, "items": all_items, "summary": summary}
