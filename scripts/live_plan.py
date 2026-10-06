"""Each lecture's live teaching sequence: what is taught in the room, in what order, and
for how long, including the activities (predictions, checks, the demo) that the minutes
must hold.

The plan lives in the lecture's front matter, next to the text it times:

    live:
      - {section: 1, minutes: 5}
      - {section: 2, minutes: 7, activities: [{block: chk-sdpa, minutes: 2}]}
      - {section: 6, minutes: 4, activities: [{block: demo-scaling, minutes: 4}]}

`minutes` is the exposition time of the numbered section (`## N. Title`); each activity
names a block on the page by its id (`::: {#chk-sdpa .self-check}`, `.demo`, `.predict`
or `.discuss`) and the minutes it takes in the room. Exposition and activities together
fill the module's lecture minutes exactly (45 on Day 1, 55 on Days 2 to 5).

A numbered section left out of the plan is reference material: its heading carries
`{.reference}`, and it stays on the page, styled, for reading after class.

scripts/gen_tables.py writes `_includes/live-NN.md` (the lecture's timing table) and
`_includes/pace-NN.md` (the pace sheet's lecture rows) from this; tests/test_live.py
checks it. filters/pedagogy.lua marks the planned blocks so the page shows which ones
are used in the room.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
LECTURES = ROOT / "lectures"

KINDS = {"self-check": "Check yourself", "demo": "Demo", "predict": "Predict", "discuss": "Discuss"}
SECTION = re.compile(r"^## (\d+)\. (.+?)\s*(\{[^}]*\})?\s*$", re.M)
BLOCK = re.compile(r"^:::+\s*\{#([\w-]+)([^}]*)\}", re.M)


def read(slug: str) -> dict:
    """Front matter, numbered sections and identified activity blocks of a lecture."""
    text = (LECTURES / f"{slug}.qmd").read_text(encoding="utf-8")
    front = {}
    if text.startswith("---\n"):
        end = text.index("\n---", 4)
        front = yaml.safe_load(text[4:end]) or {}
    sections = {
        int(m.group(1)): {"title": m.group(2), "reference": ".reference" in (m.group(3) or "")}
        for m in SECTION.finditer(text)
    }
    blocks = {}
    for m in BLOCK.finditer(text):
        kinds = [k for k in KINDS if f".{k}" in m.group(2).split()]
        if kinds:
            blocks[m.group(1)] = kinds[0]
    return {"front": front, "sections": sections, "blocks": blocks}


def rows(slug: str) -> list[dict]:
    """The plan in order, with section titles and activity kinds filled in."""
    lecture = read(slug)
    out = []
    for entry in lecture["front"].get("live") or []:
        n = entry["section"]
        out.append(
            {
                "n": n,
                "title": lecture["sections"].get(n, {}).get("title", "?"),
                "minutes": entry["minutes"],
                "activities": [
                    {
                        "block": a["block"],
                        "kind": lecture["blocks"].get(a["block"], "?"),
                        "minutes": a["minutes"],
                    }
                    for a in entry.get("activities", [])
                ],
            }
        )
    return out


def totals(plan: list[dict]) -> tuple[int, int]:
    """(exposition minutes, activity minutes)."""
    exposition = sum(r["minutes"] for r in plan)
    activities = sum(a["minutes"] for r in plan for a in r["activities"])
    return exposition, activities


def problems(slug: str, lecture_minutes: int) -> list[str]:
    """Everything wrong with a lecture's plan, as readable strings."""
    lecture = read(slug)
    plan = lecture["front"].get("live")
    if not plan:
        return [f"{slug}: no `live` plan in the front matter"]
    out = []
    planned = [e["section"] for e in plan]
    if len(planned) != len(set(planned)):
        out.append(f"{slug}: a section is planned twice")
    for n, section in lecture["sections"].items():
        if section["reference"] and n in planned:
            out.append(f"{slug}: section {n} is marked .reference but is in the plan")
        if not section["reference"] and n not in planned:
            out.append(f"{slug}: section {n} is neither planned nor marked .reference")
    for n in planned:
        if n not in lecture["sections"]:
            out.append(f"{slug}: the plan names section {n}, which does not exist")
    if planned != sorted(planned):
        out.append(f"{slug}: the plan is not in section order")
    used = [a["block"] for e in plan for a in e.get("activities", [])]
    for block in used:
        if block not in lecture["blocks"]:
            out.append(f"{slug}: activity block #{block} is not an identified activity block")
    if len(used) != len(set(used)):
        out.append(f"{slug}: an activity block is planned twice")
    exposition, activities = totals(rows(slug))
    if exposition + activities != lecture_minutes:
        out.append(
            f"{slug}: the plan fills {exposition + activities} minutes"
            f" ({exposition} exposition + {activities} activities), not {lecture_minutes}"
        )
    return out


def _activity(a: dict) -> str:
    return f"[{KINDS.get(a['kind'], a['kind'])}](#{a['block']}) ({a['minutes']} min)"


def table(slug: str) -> str:
    """The lecture's timing table: minute ranges, sections and activities in the room."""
    plan = rows(slug)
    lecture = read(slug)
    lines = [
        "| Minutes | Section | In the room |",
        "|---|---|---|",
    ]
    t = 0
    for r in plan:
        length = r["minutes"] + sum(a["minutes"] for a in r["activities"])
        activity = "; ".join(_activity(a) for a in r["activities"]) or "—"
        lines.append(f"| {t}–{t + length} | {r['n']}. {r['title']} | {activity} |")
        t += length
    exposition, activities = totals(plan)
    lines.append(
        f"| **{t}** | **Total** | **{exposition} minutes of exposition,"
        f" {activities} of activities** |"
    )
    reference = [
        f"{n}. {s['title']}" for n, s in sorted(lecture["sections"].items()) if s["reference"]
    ]
    out = ["::: {.live-plan}", "\n".join(lines), ":::"]
    if reference:
        out += [
            "",
            "Not taught in the room: "
            + "; ".join(f"*{title}*" for title in reference)
            + ". These sections are marked **Reference**; read them after the session.",
        ]
    out += [
        "",
        "Checks, demos and predictions listed here are part of the live session and its"
        " minutes. The others on the page, and collapsed callouts marked **Optional**, are"
        " for reading afterwards.",
    ]
    return "\n".join(out)


def pace_rows(slug: str) -> str:
    """The pace sheet's lecture rows: minute ranges from the start of the module's slot."""
    lines = ["| Minutes | Segment | In the room |", "|---|---|---|"]
    t = 0
    for r in rows(slug):
        length = r["minutes"] + sum(a["minutes"] for a in r["activities"])
        activity = "; ".join(
            f"{KINDS.get(a['kind'], a['kind'])} ({a['minutes']} min)" for a in r["activities"]
        )
        lines.append(f"| {t}–{t + length} | {r['n']}. {r['title']} | {activity or '—'} |")
        t += length
    return "\n".join(lines)
