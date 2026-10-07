"""Build Workshop Lectures v1: a frozen plain-text snapshot of lecture pages 01-12
and references.qmd, the document set of Labs 13, 14 and 15.

Specification: briefs/13-rag.md, decision (a). Standard library plus PyYAML; no
network, no model. The pages are read from one git commit (SOURCE_COMMIT), not
from the working tree, so the snapshot can be rebuilt byte for byte at any later
commit while the lectures keep changing.

Rules for `text` (documented in data/README.md):

- the YAML front matter is removed and the page title (and subtitle, if any) is
  written as a first-level heading;
- `{{< include /_includes/module-NN.md >}}` becomes the module's summary and
  objectives from _variables.yml, as plain text;
- `{{< var a.b.c >}}` is resolved from _variables.yml at the same commit;
- the "## Live plan" section (the lecture's in-room timetable, generated from its
  front matter) is deleted up to the next second-level heading: it is logistics, not
  content;
- heading attributes such as `{.reference}` are removed from heading lines (outside
  fenced code, where a `#` line is a comment);
- Observable JS cells (```` ```{ojs} ```` blocks, with or without options; the
  interactive demos' code) are deleted; the demos' prose stays;
- HTML comments (the figure specs) are deleted; figure captions and alt text stay;
- callout fence lines (`::: {.callout-...}` and `:::`) are deleted; a callout's
  title, if it has one, is kept as a line of its own; the content stays;
- headings, tables, code and LaTeX are kept as written; trailing spaces are
  removed and runs of blank lines are collapsed to one.

One JSON object per page, keys sorted: slug, module (1-12, or null for the
reading list), title, text, source_commit, source_sha256 (of the .qmd bytes).
The gzip header has mtime 0 and no file name.

Run:   python data/build_lectures_corpus.py            # writes the snapshot
       python data/build_lectures_corpus.py --check    # rebuilds in memory and compares
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "workshop_lectures_v1.jsonl.gz"

# The commit the v1 pages are read from: the merge of the five-day revision's last
# lecture edits (PR #16, 2026-10-06). Lecture 13 itself is not in the corpus
# (briefs/13-rag.md). Earlier builds read ec97bea (before references.qmd was finished)
# and 3ba37bc (before the five-day revision).
# STILL PROVISIONAL: lecture 12's quotations of TypeSafe's documentation await sign-off,
# and lectures may still change after their Colab T4 runs and spoken dry runs. If a page
# changes, rebuild from the new commit and update _variables.yml (sha256, bytes,
# source_commit, characters, status) before anyone writes a question against the snapshot
# (data/README.md, "Status: provisional").
SOURCE_COMMIT = "31d5d92cd1d5ac7c12b05f547caa6d56ca55765d"
LECTURE = re.compile(r"^lectures/(0[1-9]|1[0-2])-[a-z0-9-]+\.qmd$")
REFERENCES = "references.qmd"

FRONT_MATTER = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
INCLUDE = re.compile(r"\{\{<\s*include\s+/_includes/module-(\d\d)\.md\s*>\}\}")
VAR = re.compile(r"\{\{<\s*var\s+([A-Za-z0-9_.]+)\s*>\}\}")
LIVE_PLAN = re.compile(r"^## Live plan\n.*?(?=^## |\Z)", re.MULTILINE | re.DOTALL)
OJS_CELL = re.compile(r"^```\{ojs[^}\n]*\}.*?^```[ \t]*\n?", re.MULTILINE | re.DOTALL)
# Spaces and tabs only around the attributes: `\s` would also eat the line break and
# the blank line after the heading.
HEADING_ATTRS = re.compile(r"^(#{1,6} .*?)[ \t]*\{[^}\n]*\}[ \t]*$")
CODE_FENCE = re.compile(r"^(`{3,}|~{3,})")

FENCE_OPEN = re.compile(r"^\s*:::+\s*\{[^}]*\}\s*$")
FENCE_CLOSE = re.compile(r"^\s*:::+\s*$")
TITLE_ATTR = re.compile(r'title="([^"]*)"')


def strip_heading_attrs(body: str) -> str:
    """Remove `{...}` attributes from heading lines outside fenced code, where a `#` line
    is a comment and its braces are code."""
    out, fence = [], None
    for line in body.split("\n"):
        opened = CODE_FENCE.match(line)
        if fence:
            if opened and opened.group(1)[0] == fence[0] and len(opened.group(1)) >= len(fence):
                fence = None
        elif opened:
            fence = opened.group(1)
        else:
            line = HEADING_ATTRS.sub(r"\1", line)
        out.append(line)
    return "\n".join(out)


def git_show(commit: str, path: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{commit}:{path}"],
        check=True,
        capture_output=True,
    ).stdout


def git_ls(commit: str) -> list[str]:
    out = subprocess.run(
        ["git", "-C", str(ROOT), "ls-tree", "-r", "--name-only", commit],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return out.splitlines()


def lookup(variables: dict, dotted: str):
    value = variables
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(f"_variables.yml has no {dotted!r}")
        value = value[part]
    return value


def resolve_vars(text: str, variables: dict) -> str:
    return VAR.sub(lambda m: str(lookup(variables, m.group(1))), text)


def module_block(variables: dict, nn: str) -> str:
    """Plain-text stand-in for _includes/module-NN.md: summary and objectives."""
    m = variables["modules"][f"m{nn}"]
    lines = [
        f"Module {m['n']} · Day {m['day']} · {m['minutes']} minutes · {', '.join(m['stack'])}",
        "",
        m["summary"],
        "",
        "## Learning objectives",
        "",
        "By the end of this module you can:",
        "",
    ]
    lines += [f"- {o}" for o in m["objectives"]]
    return "\n".join(lines)


def clean(source: str, variables: dict) -> tuple[str, str]:
    """Return (title, text) for one .qmd page."""
    match = FRONT_MATTER.match(source)
    if not match:
        raise ValueError("page has no YAML front matter")
    meta = yaml.safe_load(resolve_vars(match.group(1), variables))
    body = source[match.end() :]
    body = COMMENT.sub("", body)
    body = strip_heading_attrs(body)  # first, so "## Live plan {.x}" is found below
    body = LIVE_PLAN.sub("", body)
    body = OJS_CELL.sub("", body)
    body = INCLUDE.sub(lambda m: module_block(variables, m.group(1)), body)
    body = resolve_vars(body, variables)
    if "{{<" in body:
        raise ValueError(f"unresolved shortcode: {body[body.index('{{<') :][:60]!r}")
    lines = []
    for line in body.splitlines():
        if FENCE_OPEN.match(line):
            title = TITLE_ATTR.search(line)
            if title:
                lines.append(title.group(1))
            continue
        if FENCE_CLOSE.match(line):
            continue
        lines.append(line.rstrip())
    title = str(meta["title"])
    head = [f"# {title}"]
    if meta.get("subtitle"):
        head += ["", str(meta["subtitle"])]
    text = "\n".join(head + [""] + lines)
    text = re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"
    return title, text


def build(commit: str = SOURCE_COMMIT) -> tuple[bytes, list[dict]]:
    """Return (gzip bytes, records) for the snapshot read from `commit`."""
    variables = yaml.safe_load(git_show(commit, "_variables.yml").decode("utf-8"))
    paths = sorted(p for p in git_ls(commit) if LECTURE.match(p)) + [REFERENCES]
    records = []
    for path in paths:
        raw = git_show(commit, path)
        title, text = clean(raw.decode("utf-8"), variables)
        slug = Path(path).stem
        module = int(slug[:2]) if path != REFERENCES else None
        records.append(
            {
                "slug": slug,
                "module": module,
                "title": title,
                "text": text,
                "source_commit": commit,
                "source_sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    if [r["module"] for r in records] != list(range(1, 13)) + [None]:
        raise ValueError("expected lectures 01-12 and the reading list")
    jsonl = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records)
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, mtime=0, compresslevel=9) as gz:
        gz.write(jsonl.encode("utf-8"))
    return buffer.getvalue(), records


def summary(blob: bytes, records: list[dict]) -> str:
    chars = sum(len(r["text"]) for r in records)
    words = sum(len(r["text"].split()) for r in records)
    rows = [f"{r['slug']:<32} {len(r['text']):>7} chars" for r in records]
    return "\n".join(
        rows
        + [
            f"documents   {len(records)}",
            f"characters  {chars}",
            f"words       {words}",
            f"bytes       {len(blob)} (gzip); {len(gzip.decompress(blob))} uncompressed",
            f"sha256      {hashlib.sha256(blob).hexdigest()}",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--commit", default=SOURCE_COMMIT, help="git commit to read the pages from")
    parser.add_argument("--check", action="store_true", help="rebuild in memory and compare")
    args = parser.parse_args()
    blob, records = build(args.commit)
    print(summary(blob, records))
    if args.check:
        if not OUT.exists():
            print(f"{OUT.name} does not exist")
            return 1
        committed = OUT.read_bytes()
        if committed == blob:
            print("OK: the committed snapshot is identical to a fresh build")
            return 0
        same_content = gzip.decompress(committed) == gzip.decompress(blob)
        print(
            "DIFFERENT: "
            + ("same content, different compression (zlib build)" if same_content else "content")
        )
        return 1
    OUT.write_bytes(blob)
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
