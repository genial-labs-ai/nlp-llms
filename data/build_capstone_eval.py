"""Build data/capstone_eval_v1.json, the manifest of Workshop Capstone Questions v1 (Lab 15).

Specification: briefs/15-capstone.md, "The manifest". Standard library only, seed 0, deterministic:
the same three input files always give the same bytes.

    python data/build_capstone_eval.py            # writes data/capstone_eval_v1.json
    python data/build_capstone_eval.py --check    # rebuilds in memory, compares with the file

The manifest records the hashes of the two question files and of the corpus snapshot, the ordered
IDs of `dev` (45) and `test` (80), the scored IDs (answerable with key facts, or unanswerable), the
stratified CPU subsets (`dev` 8 + 4, `test` 16 + 8), the costs and the scoring version.

It refuses to build while an input is missing, while the corpus snapshot is "provisional" in
_variables.yml, or while the split sizes differ from the specification. Nothing here writes,
proposes or labels a question: both question files are written and checked by people.

notebooks/15-capstone.ipynb restates `_manifest_allocate`, `build_manifest` and `manifest_bytes`
word for word (tests/test_lab15.py checks it): it builds a probe manifest with them while the real
question files do not exist, and checks the real manifest's hashes once they do.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parent
ROOT = DATA.parent
LAB13 = DATA / "rag_questions_v1.jsonl"
CAPSTONE = DATA / "capstone_questions_v1.jsonl"
SNAPSHOT = DATA / "workshop_lectures_v1.jsonl.gz"
OUT = DATA / "capstone_eval_v1.json"
SPLITS = {"dev": 45, "test": 80}


def _manifest_allocate(strata, n):
    """Split n slots over strata {(source, kind): size} in proportion to size, by largest
    remainder (ties by key), never more than a stratum holds; then, if a source got no slot
    but has items, move one slot to it from the largest allocation of another source."""
    total = sum(strata.values())
    n = min(n, total)
    alloc = {key: 0 for key in strata}
    if n == 0:
        return alloc
    quota = {key: n * size / total for key, size in strata.items()}
    for key in strata:
        alloc[key] = min(int(quota[key]), strata[key])
    order = sorted(strata, key=lambda key: (-(quota[key] - int(quota[key])), key))
    i = 0
    while sum(alloc.values()) < n:
        key = order[i % len(order)]
        if alloc[key] < strata[key]:
            alloc[key] += 1
        i += 1
    sources = sorted({key[0] for key in strata})
    if n >= len(sources):
        for source in sources:
            if sum(v for key, v in alloc.items() if key[0] == source) == 0:
                donors = [key for key in alloc if key[0] != source and alloc[key] > 0]
                donor = max(donors, key=lambda key: (alloc[key], key))
                takers = [key for key in strata if key[0] == source]
                taker = max(takers, key=lambda key: (strata[key], key))
                alloc[donor] -= 1
                alloc[taker] += 1
    return alloc


def build_manifest(
    lab13_items, new_items, *, lab13_sha256, new_sha256, snapshot_sha256, seed=0, cpu_sizes=None
):
    """The manifest as a dict. Items are dicts with id, split, kind, evidence and key_facts;
    `lab13_items` are Lab 13's questions (source "lab13"), `new_items` the capstone's ("new").
    cpu_sizes: {split: (auto-checkable answerable, unanswerable)}."""
    cpu_sizes = cpu_sizes or {"dev": (8, 4), "test": (16, 8)}
    sources = {it["id"]: "lab13" for it in lab13_items} | {it["id"]: "new" for it in new_items}
    if len(sources) != len(lab13_items) + len(new_items):
        raise ValueError("question IDs collide between or within the two files")
    items = list(lab13_items) + list(new_items)
    auto = [it["id"] for it in items if it["evidence"] and it.get("key_facts")]
    unanswerable = [it["id"] for it in items if not it["evidence"]]
    rng = random.Random(seed)
    splits, cpu_subset = {}, {}
    for split in ("dev", "test"):
        in_split = [it for it in items if it["split"] == split]
        splits[split] = [it["id"] for it in in_split]
        chosen = set()
        for pool, n in ((set(auto), cpu_sizes[split][0]), (set(unanswerable), cpu_sizes[split][1])):
            strata = {}
            for it in in_split:
                if it["id"] in pool:
                    strata.setdefault((sources[it["id"]], it["kind"]), []).append(it["id"])
            alloc = _manifest_allocate({key: len(ids) for key, ids in strata.items()}, n)
            for key in sorted(strata):
                chosen |= set(rng.sample(sorted(strata[key]), alloc[key]))
        cpu_subset[split] = [i for i in splits[split] if i in chosen]
    scored = set(auto) | set(unanswerable)
    return {
        "name": "Workshop Capstone Questions v1: manifest",
        "scoring_version": "v1",
        "seed": seed,
        "lab13_sha256": lab13_sha256,
        "capstone_sha256": new_sha256,
        "snapshot_sha256": snapshot_sha256,
        "splits": splits,
        "sources": sources,
        "scored": [it["id"] for it in items if it["id"] in scored],
        "reported_only": [it["id"] for it in items if it["id"] not in scored],
        "cpu_subset": cpu_subset,
        "costs": {"wrong": 5, "abstain": 1},
    }


def manifest_bytes(manifest):
    """The manifest file's exact bytes: sorted keys, one-space indent, UTF-8, final newline."""
    return (json.dumps(manifest, indent=1, sort_keys=True, ensure_ascii=False) + "\n").encode()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_items(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def snapshot_status() -> str | None:
    """`status` of datasets.lectures in _variables.yml, read without PyYAML (standard library)."""
    text = (ROOT / "_variables.yml").read_text(encoding="utf-8")
    block = re.search(r"^  lectures:\n((?:    .*\n|\s*#.*\n)+)", text, flags=re.M)
    found = re.search(r'^    status: "?([\w-]+)"?', block.group(1), flags=re.M) if block else None
    return found.group(1) if found else None


def build() -> bytes:
    missing = [p.name for p in (LAB13, CAPSTONE, SNAPSHOT) if not p.exists()]
    if missing:
        raise SystemExit(
            f"Missing input(s): {', '.join(missing)}. Both question files are "
            "written and checked by people (briefs/13-rag.md, briefs/15-capstone.md)."
        )
    if snapshot_status() == "provisional":
        raise SystemExit(
            "The corpus snapshot is still provisional in _variables.yml: freeze it, "
            "then write the questions, then build the manifest."
        )
    lab13, new = load_items(LAB13), load_items(CAPSTONE)
    manifest = build_manifest(
        lab13,
        new,
        lab13_sha256=sha256(LAB13),
        new_sha256=sha256(CAPSTONE),
        snapshot_sha256=sha256(SNAPSHOT),
    )
    sizes = {split: len(ids) for split, ids in manifest["splits"].items()}
    if sizes != SPLITS:
        raise SystemExit(f"Split sizes {sizes}, expected {SPLITS}")
    return manifest_bytes(manifest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="compare with the committed file")
    args = parser.parse_args()
    blob = build()
    if args.check:
        same = OUT.exists() and OUT.read_bytes() == blob
        print("manifest up to date" if same else "manifest differs from a rebuild")
        return 0 if same else 1
    OUT.write_bytes(blob)
    print(f"wrote {OUT.relative_to(ROOT)}: sha256 {hashlib.sha256(blob).hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
