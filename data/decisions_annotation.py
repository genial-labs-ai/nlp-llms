"""Human annotation of Workshop Desk Decisions v1: blind sheets, agreement and Cohen's kappa.

The decision set's labels come from a rule engine (template items) or from two people
(hand-written items). This script supports the two human steps of the specification
(briefs/11-calibration.md, "How the labels are made"); the instructions for the people
doing them are in data/decisions_hand_TEMPLATE.md. Standard library only.

    # Step 3, the template audit: 60 template items, stratified, labelled blind by two people.
    python data/decisions_annotation.py audit-sheet  # writes data/decisions_audit_sheet_v1.jsonl
    #   each annotator copies the sheet, fills "label" on every line, and returns it
    python data/decisions_annotation.py agreement A.jsonl B.jsonl --reference engine

    # Step 2, the hand-written items: blind sheet from the authors' drafts, then agreement.
    python data/decisions_annotation.py hand-sheet drafts.jsonl --out hand_sheet.jsonl
    python data/decisions_annotation.py agreement A.jsonl B.jsonl --reference drafts.jsonl
    python data/decisions_annotation.py merge-hand drafts.jsonl A.jsonl B.jsonl \\
        --resolutions resolutions.jsonl --out data/decisions_hand_v1.jsonl

`agreement` prints the raw agreement and Cohen's kappa between the two annotators, and of
each annotator against the reference (the rule engine's label for the audit, the author's
intended label for hand items), overall and per question family, and lists every
disagreement. Record the numbers in data/README.md ("Labels and agreement").
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_decisions as bd  # noqa: E402

AUDIT_SHEET = HERE / "decisions_audit_sheet_v1.jsonl"
AUDIT_PER_STRATUM = {"train": 2, "eval": 3}  # 12 strata (family x difficulty) x 5 = 60 items
BLIND_FIELDS = ("id", "records", "request", "question", "options")


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    text = "".join(json.dumps(r, sort_keys=False, ensure_ascii=True) + "\n" for r in rows)
    Path(path).write_text(text, encoding="utf-8")


def committed_items() -> list[dict]:
    return [
        json.loads(line)
        for line in gzip.decompress(bd.OUT.read_bytes()).decode("utf-8").splitlines()
    ]


def blind(item: dict, item_id: str | None = None) -> dict:
    """What an annotator sees: records, request, question and options. Nothing else."""
    state = item["state"]
    records = {k: state[k] for k in ("today", "event", "registration")}
    return {
        "id": item_id or item["id"],
        "records": records,
        "request": state["request"],
        "question": item["question"],
        "options": item["options"],
        "label": None,
        "notes": "",
    }


def main_template_ids(items: list[dict]) -> set[str]:
    """Template items that stay fixed when hand items are added (not slot fill-ins)."""
    stats = json.loads(bd.STATS.read_text(encoding="utf-8"))
    fills = set(stats["fill_in_ids"])
    return {it["id"] for it in items if it["source"] == "template" and it["id"] not in fills}


def audit_sample(items: list[dict]) -> list[dict]:
    """60 template items: per family and difficulty tag, 2 from train and 3 from dev or test.

    Seeded (random.Random(f"{SEED}:audit")) and drawn only from the main template items,
    so the sample does not change when hand items fill the reserved slots.
    """
    rng = random.Random(f"{bd.SEED}:audit")
    keep = main_template_ids(items)
    chosen = []
    for family in bd.FAMILIES:
        for diff in bd.DIFFICULTIES:
            stratum = [
                it
                for it in items
                if it["id"] in keep and it["family"] == family and it["difficulty"] == diff
            ]
            train = [it for it in stratum if it["split"] == "train"]
            evals = [it for it in stratum if it["split"] != "train"]
            chosen += rng.sample(train, AUDIT_PER_STRATUM["train"])
            chosen += rng.sample(evals, AUDIT_PER_STRATUM["eval"])
    rng.shuffle(chosen)  # the sheet does not reveal the strata
    return chosen


def cohen_kappa(a: list[str], b: list[str]) -> float:
    """Cohen's kappa of two label lists: (p_o - p_e) / (1 - p_e). NaN if p_e is 1."""
    assert len(a) == len(b) and a
    n = len(a)
    p_o = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    p_e = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return float("nan") if p_e == 1 else (p_o - p_e) / (1 - p_e)


def compare(name: str, a: dict, b: dict, families: dict) -> dict:
    """Agreement and kappa of two {id: label} maps over their common ids."""
    ids = sorted(set(a) & set(b))
    out = {"pair": name, "n": len(ids)}
    for fam in [None, *bd.FAMILIES]:
        sub = [i for i in ids if fam is None or families.get(i) == fam]
        if not sub:
            continue
        la, lb = [a[i] for i in sub], [b[i] for i in sub]
        key = fam or "all"
        out[key] = {
            "n": len(sub),
            "agreement": sum(x == y for x, y in zip(la, lb, strict=True)) / len(sub),
            "kappa": cohen_kappa(la, lb),
        }
    out["disagreements"] = [(i, a[i], b[i]) for i in ids if a[i] != b[i]]
    return out


def read_labels(path: Path) -> tuple[dict, dict]:
    labels, families = {}, {}
    for row in load_jsonl(path):
        if row.get("label") not in row["options"]:
            raise SystemExit(f"{path}: {row['id']} has no valid label ({row.get('label')!r})")
        labels[row["id"]] = row["label"]
        families[row["id"]] = "policy" if row["options"] == bd.POLICY_OPTIONS else "route"
    return labels, families


def cmd_audit_sheet(args) -> int:
    sample = audit_sample(committed_items())
    write_jsonl(args.out, [blind(it) for it in sample])
    print(f"wrote {args.out}: {len(sample)} items, blind (no label, rule, rationale or difficulty)")
    return 0


def cmd_hand_sheet(args) -> int:
    rows = []
    for d in load_jsonl(args.drafts):
        rows.append(
            blind(
                {"state": d["state"], "question": d["question"], "options": d["options"]},
                d["hand_id"],
            )
        )
    random.Random(f"{bd.SEED}:hand-sheet").shuffle(rows)
    write_jsonl(args.out, rows)
    print(f"wrote {args.out}: {len(rows)} items, blind (no intended label, rule or rationale)")
    return 0


def cmd_agreement(args) -> int:
    a, fam_a = read_labels(args.a)
    b, fam_b = read_labels(args.b)
    if set(a) != set(b):
        raise SystemExit("the two sheets do not cover the same items")
    families = fam_a | fam_b
    reports = [compare("A vs B", a, b, families)]
    if args.reference == "engine":
        ref = {it["id"]: it["label"] for it in committed_items() if it["id"] in a}
        name = "rule engine"
    elif args.reference:
        ref = {d["hand_id"]: d["intended_label"] for d in load_jsonl(args.reference)}
        name = "author"
    else:
        ref = None
    if ref is not None:
        reports += [
            compare(f"A vs {name}", a, ref, families),
            compare(f"B vs {name}", b, ref, families),
        ]
    for r in reports:
        print(f"\n{r['pair']}  (n = {r['n']})")
        for key in ("all", *bd.FAMILIES):
            if key in r:
                s = r[key]
                print(f"  {key:<7} n={s['n']:<4} agreement={s['agreement']:.3f}", end="")
                print(f"  kappa={s['kappa']:.3f}")
        for i, x, y in r["disagreements"]:
            print(f"  disagree {i}: {x} / {y}")
    return 0


def cmd_merge_hand(args) -> int:
    drafts = {d["hand_id"]: d for d in load_jsonl(args.drafts)}
    a, _ = read_labels(args.a)
    b, _ = read_labels(args.b)
    resolutions = (
        {r["hand_id"]: r for r in load_jsonl(args.resolutions)} if args.resolutions else {}
    )
    out, unresolved = [], []
    for hid in sorted(drafts):
        d, la, lb = drafts[hid], a.get(hid), b.get(hid)
        if la is None or lb is None:
            raise SystemExit(f"{hid}: labelled by only one annotator")
        res = resolutions.get(hid)
        if res and res.get("drop"):
            continue
        if la == lb and not res:
            label, note = la, ""
        elif res and res.get("label"):
            label, note = res["label"], res.get("note", "")
            if res.get("state"):  # the item was rewritten to settle the dispute
                d = d | {"state": res["state"]}
        else:
            unresolved.append(hid)
            continue
        out.append(
            {
                "hand_id": hid,
                "split": d["split"],
                "family": d["family"],
                "difficulty": d["difficulty"],
                "state": d["state"],
                "question": d["question"],
                "options": d["options"],
                "label_a": la,
                "label_b": lb,
                "label": label,
                "resolution": note,
                "rule": d["rule"],
                "rationale": d["rationale"],
                "author": d["author"],
            }
        )
    if unresolved:
        raise SystemExit(
            "disputed items need a resolution (label, rewrite or drop): " + ", ".join(unresolved)
        )
    write_jsonl(args.out, out)
    bd.load_hand(Path(args.out))  # validates every field the builder needs
    print(f"wrote {args.out}: {len(out)} hand items; now run python data/build_decisions.py")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("audit-sheet", help="write the blind sheet of the 60-item template audit")
    p.add_argument("--out", type=Path, default=AUDIT_SHEET)
    p.set_defaults(fn=cmd_audit_sheet)
    p = sub.add_parser("hand-sheet", help="write a blind sheet from the authors' hand-item drafts")
    p.add_argument("drafts", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(fn=cmd_hand_sheet)
    p = sub.add_parser("agreement", help="agreement and Cohen's kappa of two filled sheets")
    p.add_argument("a", type=Path)
    p.add_argument("b", type=Path)
    p.add_argument("--reference", help="'engine' (audit) or the drafts file (hand items)")
    p.set_defaults(fn=cmd_agreement)
    p = sub.add_parser(
        "merge-hand", help="combine drafts and both sheets into decisions_hand_v1.jsonl"
    )
    p.add_argument("drafts", type=Path)
    p.add_argument("a", type=Path)
    p.add_argument("b", type=Path)
    p.add_argument("--resolutions", type=Path)
    p.add_argument("--out", type=Path, default=bd.HAND)
    p.set_defaults(fn=cmd_merge_hand)
    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
