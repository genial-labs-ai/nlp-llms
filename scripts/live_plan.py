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
checks it. filters/live.lua marks the planned blocks so the page shows which ones are
used in the room.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
LECTURES = ROOT / "lectures"

KINDS = {"self-check": "Check yourself", "demo": "Demo", "predict": "Predict", "discuss": "Discuss"}
SECTION = re.compile(r"^## (\d+)\. (.+?)\s*(\{[^}]*\})?\s*$")
OPEN_DIV = re.compile(r"^:::+\s*\S")
CLOSE_DIV = re.compile(r"^:::+\s*$")
FENCE = re.compile(r"^(`{3,}|~{3,})")


def read(slug: str) -> dict:
    """Front matter, numbered sections and identified activity blocks of a lecture.

    `blocks` maps each activity block's id to its kind, and `block_sections` to the
    numbered section it sits in (None outside one: before the first, or under an
    unnumbered heading such as Summary); `duplicates` lists ids used by more than one
    activity block. Only top-level headings are sections: a `##` title inside a
    callout is not, and neither is anything inside fenced code."""
    text = (LECTURES / f"{slug}.qmd").read_text(encoding="utf-8")
    front = {}
    body = text
    if text.startswith("---\n"):
        end = text.index("\n---", 4)
        front = yaml.safe_load(text[4:end]) or {}
        body = text[end + 4 :]
    sections, blocks, block_sections, duplicates = {}, {}, {}, []
    current, depth, fence = None, 0, None
    for line in body.splitlines():
        if fence:
            if line.startswith(fence) and not line[len(fence) :].strip():
                fence = None
            continue
        opened = FENCE.match(line)
        if opened:
            fence = opened.group(1)
            continue
        if depth == 0 and line.startswith("## "):
            m = SECTION.match(line)
            current = int(m.group(1)) if m else None
            if m:
                sections[current] = {
                    "title": m.group(2),
                    "reference": ".reference" in (m.group(3) or ""),
                }
        if CLOSE_DIV.match(line):
            depth = max(depth - 1, 0)
        elif OPEN_DIV.match(line):
            depth += 1
            attrs = re.match(r"^:::+\s*\{([^}]*)\}", line)
            # Pandoc attributes in any order: {#chk-x .self-check} or {.self-check #chk-x}.
            words = attrs.group(1).split() if attrs else []
            ids = [w[1:] for w in words if w.startswith("#")]
            kinds = [k for k in KINDS if f".{k}" in words]
            if ids and kinds:
                if ids[0] in blocks:
                    duplicates.append(ids[0])
                blocks[ids[0]] = kinds[0]
                block_sections[ids[0]] = current
    return {
        "front": front,
        "sections": sections,
        "blocks": blocks,
        "block_sections": block_sections,
        "duplicates": duplicates,
    }


def rows(slug: str, lecture: dict | None = None) -> list[dict]:
    """The plan in order, with section titles and activity kinds filled in. A plan
    whose entries are malformed raises ValueError with the readable problems."""
    lecture = lecture or read(slug)
    found = shape_problems(slug, lecture["front"].get("live") or [])
    if found:
        raise ValueError("\n".join(found))
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


def shape_problems(slug: str, plan) -> list[str]:
    """Entries that are not {section: int, minutes: int, activities: [{block, minutes}]}."""
    if not isinstance(plan, list):
        return [f"{slug}: `live` must be a list of sections"]
    out = []
    for i, e in enumerate(plan, 1):
        where = f"{slug}: `live` entry {i} ({e!r})"
        if not isinstance(e, dict) or set(e) - {"section", "minutes", "activities"}:
            out.append(f"{where} must have only the keys section, minutes and activities")
            continue
        if not (is_count(e.get("section")) and e["section"] > 0 and is_count(e.get("minutes"))):
            out.append(f"{where} needs a positive integer section and whole minutes (0 or more)")
        activities = e.get("activities", [])
        if not isinstance(activities, list) or not all(
            isinstance(a, dict)
            and set(a) == {"block", "minutes"}
            and isinstance(a["block"], str)
            and is_count(a["minutes"])
            and a["minutes"] > 0
            for a in activities
        ):
            out.append(f"{where}: each activity is {{block: <id>, minutes: <positive integer>}}")
    return out


def is_count(x) -> bool:
    """A whole number of 0 or more; YAML's true and false are not numbers here."""
    return type(x) is int and x >= 0


def problems(slug: str, lecture_minutes: int) -> list[str]:
    """Everything wrong with a lecture's plan, as readable strings."""
    lecture = read(slug)
    plan = lecture["front"].get("live")
    if not plan:
        return [f"{slug}: no `live` plan in the front matter"]
    out = shape_problems(slug, plan)
    if out:
        return out
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
    for e in plan:
        for a in e.get("activities", []):
            where = lecture["block_sections"].get(a["block"], e["section"])
            if where != e["section"]:
                out.append(
                    f"{slug}: activity block #{a['block']} is planned under section"
                    f" {e['section']} but sits in section {where}"
                )
    for block in sorted(set(lecture["duplicates"])):
        out.append(f"{slug}: more than one activity block has the id #{block}")
    if len(used) != len(set(used)):
        out.append(f"{slug}: an activity block is planned twice")
    exposition, activities = totals(rows(slug, lecture))
    if exposition + activities != lecture_minutes:
        out.append(
            f"{slug}: the plan fills {exposition + activities} minutes"
            f" ({exposition} exposition + {activities} activities), not {lecture_minutes}"
        )
    return out


def _activity(a: dict, link: bool) -> str:
    kind = KINDS.get(a["kind"], a["kind"])
    return (
        f"[{kind}](#{a['block']}) ({a['minutes']} min)" if link else f"{kind} ({a['minutes']} min)"
    )


def _timed_rows(plan: list[dict], link: bool) -> tuple[list[str], int]:
    """Table rows with minute ranges from the start of the lecture, and the end minute.
    Shared by the lecture's table (activities linked) and the pace sheet (plain)."""
    lines = []
    t = 0
    for r in plan:
        length = r["minutes"] + sum(a["minutes"] for a in r["activities"])
        activity = "; ".join(_activity(a, link) for a in r["activities"]) or "—"
        lines.append(f"| {t}–{t + length} | {r['n']}. {r['title']} | {activity} |")
        t += length
    return lines, t


def table(slug: str, lecture: dict | None = None) -> str:
    """The lecture's timing table: minute ranges, sections and activities in the room."""
    lecture = lecture or read(slug)
    plan = rows(slug, lecture)
    body, t = _timed_rows(plan, link=True)
    lines = ["| Minutes | Section | In the room |", "|---|---|---|", *body]
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


def pace_rows(slug: str, lecture: dict | None = None) -> str:
    """The pace sheet's lecture rows: minute ranges from the start of the module's slot."""
    body, _ = _timed_rows(rows(slug, lecture), link=False)
    return "\n".join(["| Minutes | Segment | In the room |", "|---|---|---|", *body])
