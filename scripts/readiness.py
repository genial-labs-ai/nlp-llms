"""What is known about whether each module can be taught, and on what evidence.

Reads the readiness fields in _variables.yml (readiness: and modules.mNN.readiness) and
the run records in runs/. Used by scripts/gen_tables.py to write the readiness tables,
and by the release check. Nothing here reads the clock: the output depends only on the
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


def open_items(v: dict, n: int) -> list[dict]:
    return [i for i in items(v) if not i["closed"] and n in i["modules"]]


def _newest(records: list[dict]) -> dict | None:
    """The most recent record; among records of one day, the most complete."""
    if not records:
        return None
    return max(records, key=lambda r: (r["date"], r["scope"] == "notebook", r["file"]))


def evidence(v: dict, slug: str, records: list[dict]) -> dict:
    """The newest passing record of each kind for one notebook."""
    runtime_envs = {k for k, e in v["readiness"]["envs"].items() if e.get("teaching")}
    mine = [r for r in records if r["notebook"] == slug and r["status"] == "pass"]
    real = [r for r in mine if r["path"] in ("open", "keyed")]
    return {
        # On a runtime a delivery uses, the whole notebook, on its real path, recorded by a tool.
        "teaching": _newest(
            [
                r
                for r in real
                if r["env"] in runtime_envs
                and r["scope"] == "notebook"
                and r["source"] != "backfill"
            ]
        ),
        "real": _newest([r for r in real if r["scope"] == "notebook"]),
        "real_partial": _newest([r for r in real if r["scope"] == "partial"]),
        "code": _newest([r for r in mine if r["path"] == "offline"]),
    }


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
    if r["scope"] == "partial":
        text += f" ({r['scope_note']})"
    if r.get("settings", {}).get("NLP_LLMS_QUICK") == "1" and r["path"] != "offline":
        text += ", QUICK settings"
    if r["path"] == "keyed":
        text += ", with API keys"
    return text


def summary(v: dict, records: list[dict]) -> dict:
    """Counts behind the one-paragraph status on the landing, FAQ and teach pages."""
    labs = [(key, m) for key, m in v["modules"].items() if m.get("notebook", True)]
    ev = {m["slug"]: evidence(v, m["slug"], records) for _, m in labs}
    notebooks = [m["slug"] for _, m in labs] + [v["setup"]["slug"]]
    code_or_real = [
        s for s in notebooks if any(r["notebook"] == s and r["status"] == "pass" for r in records)
    ]
    all_items = items(v)
    return {
        "labs": len(labs),
        "teaching": sum(1 for e in ev.values() if e["teaching"]),
        "real": sum(1 for e in ev.values() if e["real"]),
        "real_partial_only": sum(1 for e in ev.values() if not e["real"] and e["real_partial"]),
        "notebooks": len(notebooks),
        "ran_somewhere": len(code_or_real),
        "items_open": sum(1 for i in all_items if not i["closed"]),
        "items": len(all_items),
        "as_of": max((r["date"] for r in records), default=None),
    }


def load(v: dict) -> list[dict]:
    return run_records.load_all()
