"""Checks on Workshop Desk Decisions v1 (data/decisions_v1.jsonl.gz), shared by Labs 11, 12 and 14.

Rebuilds the set in memory and compares bytes; checks the schema, split sizes, label
balance and difficulty mix of the specification (briefs/11-calibration.md, "Shared
decision set"); re-derives the policy labels with an implementation written here from the
policy text, independent of the builder's rule engine; checks that no text has an email
address outside example.org; and checks the annotation tooling. Standard library and
PyYAML only, no network.
"""

import gzip
import hashlib
import importlib.util
import json
import os
import re
import string
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from datetime import date
from pathlib import Path
from unittest import mock

import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


bd = _load("build_decisions", DATA / "build_decisions.py")
ann = _load("decisions_annotation", DATA / "decisions_annotation.py")
V = yaml.safe_load((ROOT / "_variables.yml").read_text(encoding="utf-8"))
BLOB = bd.OUT.read_bytes()
ITEMS = [json.loads(line) for line in gzip.decompress(BLOB).decode("utf-8").splitlines()]
STATS = json.loads(bd.STATS.read_text(encoding="utf-8"))
README = (DATA / "README.md").read_text(encoding="utf-8")
ITEM_FIELDS = {
    "id",
    "split",
    "family",
    "source",
    "difficulty",
    "state",
    "question",
    "options",
    "label",
    "rule",
    "rationale",
}
EVENT_FIELDS = {"id", "topic", "city", "start_date", "seats", "registered", "remote", "fee_eur"}
REG_FIELDS = {"name", "email", "status", "paid_eur"}


def by_split(split):
    return [it for it in ITEMS if it["split"] == split]


class Rebuild(unittest.TestCase):
    def test_rebuild_is_byte_identical(self):
        items, blob, stats = bd.build()
        # Content, not compressed bytes: gzip output depends on the zlib build, which differs
        # between machines. tests/test_data.py checks the committed file's hash.
        self.assertEqual(
            gzip.decompress(blob), gzip.decompress(BLOB), "content differs: rebuild the set"
        )
        self.assertEqual(bd.stats_bytes(stats), bd.STATS.read_bytes())

    def test_rebuild_ignores_the_hash_seed(self):
        """A fresh interpreter with another PYTHONHASHSEED finds the committed files up to date."""
        env = dict(os.environ, PYTHONHASHSEED="12345")
        result = subprocess.run(
            [sys.executable, str(DATA / "build_decisions.py"), "--check"],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_variables_entry(self):
        d = V["datasets"]["decisions"]
        self.assertEqual(d["name"], bd.NAME)
        self.assertEqual(d["modules"], [11, 12, 14])
        self.assertEqual(d["file"], bd.OUT.name)
        self.assertEqual(d["sha256"], STATS["sha256"])
        self.assertEqual(d["bytes"], STATS["bytes"])
        self.assertEqual(d["splits"], bd.SPLITS)
        self.assertEqual(d["route_options"], bd.ROUTE_OPTIONS)
        self.assertEqual(d["families"], list(bd.FAMILIES))
        self.assertEqual(d["seed"], bd.SEED)


class Schema(unittest.TestCase):
    def test_fields(self):
        for it in ITEMS:
            self.assertEqual(set(it), ITEM_FIELDS, it["id"])
            self.assertEqual(
                set(it["state"]), {"today", "event", "registration", "request"}, it["id"]
            )
            self.assertEqual(set(it["state"]["event"]), EVENT_FIELDS, it["id"])
            reg = it["state"]["registration"]
            self.assertTrue(reg is None or set(reg) == REG_FIELDS, it["id"])
            self.assertIn(it["family"], bd.FAMILIES)
            self.assertIn(it["source"], ("template", "hand"))
            self.assertIn(it["difficulty"], bd.DIFFICULTIES)
            self.assertIn(it["label"], it["options"])

    def test_ids_are_unique_and_ordered(self):
        for split in bd.SPLITS:
            ids = [it["id"] for it in by_split(split)]
            self.assertEqual(ids, [f"dec-{split}-{i:04d}" for i in range(1, len(ids) + 1)])

    def test_questions_and_options(self):
        policy_qs = {bd.QUESTIONS[r]: r for r in bd.RULES}
        for it in ITEMS:
            if it["family"] == "policy":
                self.assertEqual(it["options"], ["yes", "no"])
                self.assertIn(it["question"], policy_qs)
                if it["source"] == "template":
                    self.assertEqual(policy_qs[it["question"]], it["rule"])
            else:
                self.assertEqual(
                    it["options"], ["retrieve", "calculate", "send_email", "ask_user", "escalate"]
                )
                self.assertEqual(it["question"], "What should the assistant do next?")


class SplitsAndBalance(unittest.TestCase):
    def test_sizes(self):
        self.assertEqual(len(ITEMS), 2400)
        for split, n in bd.SPLITS.items():
            items = by_split(split)
            self.assertEqual(len(items), n, split)
            fam = Counter(it["family"] for it in items)
            self.assertEqual(fam, {"policy": n // 2, "route": n // 2}, split)
        self.assertTrue(all(it["source"] == "template" for it in by_split("train")))

    def test_hand_slots(self):
        n_hand = sum(it["source"] == "hand" for it in ITEMS)
        self.assertEqual(STATS["hand_items"], n_hand)
        self.assertEqual(len(STATS["fill_in_ids"]), 80 - n_hand)
        if n_hand < 80:
            self.assertEqual(STATS["status"], "v1-template-only")
        for split, slots in (("dev", 20), ("test", 60)):
            tail = {it["id"] for it in by_split(split)[-slots:]}
            fills = {i for i in STATS["fill_in_ids"] if i.startswith(f"dec-{split}-")}
            hand = {it["id"] for it in by_split(split) if it["source"] == "hand"}
            self.assertEqual(fills | hand, tail, "the reserved slots are the last ids of the split")

    def test_policy_labels_are_balanced(self):
        for split in bd.SPLITS:
            labels = Counter(it["label"] for it in by_split(split) if it["family"] == "policy")
            n = sum(labels.values())
            if STATS["hand_items"] == 0:
                self.assertEqual(labels["yes"], labels["no"], split)
            else:
                self.assertLessEqual(abs(labels["yes"] - n / 2), 0.04 * n, split)

    def test_route_labels_near_20_percent(self):
        for split in bd.SPLITS:
            labels = Counter(it["label"] for it in by_split(split) if it["family"] == "route")
            n = sum(labels.values())
            for option in bd.ROUTE_OPTIONS:
                self.assertGreaterEqual(labels[option] / n, 0.16, (split, option))
                self.assertLessEqual(labels[option] / n, 0.24, (split, option))

    def test_difficulty_mix_in_dev_and_test(self):
        for split in ("dev", "test"):
            tags = {}
            for family in bd.FAMILIES:
                tags[family] = Counter(
                    it["difficulty"] for it in by_split(split) if it["family"] == family
                )
                n = sum(tags[family].values())
                self.assertAlmostEqual(
                    tags[family]["plain"] / n, 0.40, delta=0.03, msg=(split, family)
                )
                for tag in bd.DIFFICULTIES[1:]:
                    self.assertGreaterEqual(tags[family][tag] / n, 0.08, (split, family, tag))
            self.assertLess(
                tags["policy"]["missing"], tags["route"]["missing"], "missing mainly in route"
            )

    def test_policy_rules_are_spread(self):
        for split in bd.SPLITS:
            rules = Counter(
                it["rule"]
                for it in by_split(split)
                if it["family"] == "policy" and it["source"] == "template"
            )
            self.assertEqual(set(rules), set(bd.RULES))
            self.assertLessEqual(max(rules.values()) - min(rules.values()), 2, split)


# An implementation of the policy questions written from the policy text, independent of the
# builder's engine. It reads only the records and the request.
NAME_RE = re.compile("|".join(re.escape(n) for n in bd.NAMES))


def days_before(state):
    return (
        date.fromisoformat(state["event"]["start_date"]) - date.fromisoformat(state["today"])
    ).days


def expected_policy_label(rule, state):
    ev, reg, d = state["event"], state["registration"], days_before(state)
    if rule == "P1":
        return "yes" if d >= 7 and ev["registered"] < ev["seats"] else "no"
    if rule == "P4":
        return "no" if ev["remote"] else "yes"
    if reg is None or reg["status"] != "confirmed":
        return "no"
    if rule == "P2":
        return "yes" if d >= 14 else "no"
    if rule == "P3":
        named = [n for n in NAME_RE.findall(state["request"]) if n != reg["name"]]
        return "yes" if named and d >= 2 else "no"
    if rule == "P5":
        refund = reg["paid_eur"] if d >= 14 else reg["paid_eur"] / 2 if d >= 7 else 0
        return "yes" if refund <= 500 else "no"
    raise ValueError(rule)


class Labels(unittest.TestCase):
    def test_policy_labels_follow_from_the_records(self):
        for it in ITEMS:
            if it["family"] == "policy" and it["source"] == "template":
                self.assertEqual(
                    it["label"], expected_policy_label(it["rule"], it["state"]), it["id"]
                )

    def test_every_template_label_is_the_rule_engines(self):
        items, _ = bd.build_items(bd.load_hand())
        committed = {it["id"]: it for it in ITEMS}
        for it in items:
            if it["source"] != "template":
                continue
            self.assertEqual(committed[it["id"]]["label"], it["label"])
            if it["family"] == "policy":
                label, _ = bd.policy_label(it["rule"], it["state"], it["_intent"])
            else:
                label, _, _ = bd.route_label(it["state"], it["_intent"])
            self.assertEqual(label, it["label"], it["id"])

    def test_route_spot_checks(self):
        """Properties of the routing order that can be read off the records."""
        for it in ITEMS:
            if it["family"] != "route" or it["source"] != "template":
                continue
            if it["difficulty"] == "injection":
                self.assertEqual(it["label"], "escalate", it["id"])
            if it["label"] == "ask_user":
                self.assertEqual(it["difficulty"], "missing", it["id"])
            if it["state"]["registration"] is None and it["label"] in ("send_email",):
                self.assertIn(
                    "@example.org", it["state"]["request"], "a new registration gives an address"
                )

    def test_conflict_items_state_something_the_records_contradict(self):
        for it in ITEMS:
            if it["difficulty"] == "conflict" and it["source"] == "template":
                self.assertIn("(P8)", it["rationale"], it["id"])


class Content(unittest.TestCase):
    def test_no_email_outside_example_org(self):
        for line in gzip.decompress(BLOB).decode("utf-8").splitlines():
            for address in re.findall(r"[\w.+-]*@[\w.-]*", line):
                self.assertTrue(address.rstrip(".").endswith("@example.org"), address)

    def test_names_come_from_the_invented_list(self):
        for it in ITEMS:
            reg = it["state"]["registration"]
            if reg is not None and it["source"] == "template":
                self.assertIn(reg["name"], bd.NAMES)
                self.assertEqual(reg["email"], bd.email_of(reg["name"]))

    def test_amounts_written_as_eur(self):
        for it in ITEMS:
            self.assertNotIn("$", it["state"]["request"])
            self.assertNotIn("€", it["state"]["request"])

    def test_dev_and_test_wordings_are_held_out_from_train(self):
        def segments(templates):
            out = set()
            for t in templates:
                for literal, _, _, _ in string.Formatter().parse(t.replace(" || ", " ")):
                    out |= {
                        s.strip() for s in re.split(r"[.?!,;:]", literal) if len(s.strip()) >= 18
                    }
            return out

        def literal(t):
            return "\x00".join(
                lit for lit, _, _, _ in string.Formatter().parse(t.replace(" || ", " "))
            )

        train, evals, train_text, eval_text = set(), set(), [], []
        for kind in bd.W.values():
            train |= segments(kind["train"])
            evals |= segments(kind["eval"])
            train_text += [literal(t) for t in kind["train"]]
            eval_text += [literal(t) for t in kind["eval"]]
        # A segment is forbidden in a split if no wording of that split contains it.
        only_train = {s for s in train if not any(s in t for t in eval_text)}
        only_eval = {s for s in evals if not any(s in t for t in train_text)}
        self.assertGreater(len(only_train), 30)
        self.assertGreater(len(only_eval), 20)
        for it in ITEMS:
            text = it["state"]["request"]
            if it["source"] != "template":
                continue
            forbidden = only_train if it["split"] != "train" else only_eval
            for seg in forbidden:
                self.assertNotIn(seg, text, (it["id"], seg))

    def test_rule_and_rationale_never_in_the_request(self):
        for it in ITEMS:
            self.assertNotIn("(P", it["state"]["request"])


class Policy(unittest.TestCase):
    def test_policy_is_short_and_versioned(self):
        text = bd.policy_text()
        self.assertLessEqual(len(text.split()), 500)
        self.assertEqual(STATS["policy_sha256"], hashlib.sha256(text.encode("utf-8")).hexdigest())
        for rule in ("P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"):
            self.assertIn(f"\n{rule} ", "\n" + text)
        for option in bd.ROUTE_OPTIONS:
            self.assertIn(f"{option}:", text)

    def test_route_descriptions_come_from_the_policy(self):
        d = bd.route_descriptions()
        self.assertEqual(list(d), bd.ROUTE_OPTIONS)
        for option, text in d.items():
            self.assertIn(f"{option}: {text}", bd.policy_text())


LAB11 = ROOT / "notebooks" / "11-calibration.ipynb"


@unittest.skipUnless(LAB11.exists(), "Lab 11 notebook not written yet")
class Lab11Restatements(unittest.TestCase):
    """Lab 11 restates the policy and the route descriptions; Lab 12 copies them from Lab 11."""

    @classmethod
    def setUpClass(cls):
        nb = json.loads(LAB11.read_text(encoding="utf-8"))
        cls.code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")

    def test_policy_verbatim(self):
        match = re.search(r'DECISION_POLICY = """\\\n(.*?)"""', self.code, flags=re.DOTALL)
        self.assertIsNotNone(match, "Lab 11 defines DECISION_POLICY as a triple-quoted string")
        self.assertEqual(match.group(1).strip(), bd.policy_text())

    def test_route_descriptions_verbatim(self):
        match = re.search(
            r"^ROUTE_DESCRIPTIONS = (\{.*?^\})", self.code, flags=re.DOTALL | re.MULTILINE
        )
        self.assertIsNotNone(match, "Lab 11 defines ROUTE_DESCRIPTIONS once, as a dict literal")
        self.assertEqual(eval(match.group(1)), bd.route_descriptions())  # noqa: S307


class ReadmeLoader(unittest.TestCase):
    def test_load_decisions_snippet(self):
        """The decision-set block of data/README.md loads the committed copy with no network."""
        blocks = re.findall(r"```python\n(.*?)```", README, flags=re.DOTALL)
        code = blocks[0] + "\n" + next(b for b in blocks if "def load_decisions" in b)
        ns: dict = {}

        def no_network(*args, **kwargs):
            raise OSError("network disabled in tests")

        with (
            mock.patch.dict(os.environ, {"NLP_LLMS_DATA": str(DATA)}),
            mock.patch("urllib.request.urlopen", no_network),
        ):
            exec(compile(code, "data/README.md", "exec"), ns)  # noqa: S102
            decisions = ns["load_decisions"]()
        self.assertEqual({k: len(v) for k, v in decisions.items()}, bd.SPLITS)
        self.assertIn(STATS["sha256"], README)


class Annotation(unittest.TestCase):
    def test_kappa(self):
        self.assertEqual(
            ann.cohen_kappa(["yes", "no", "yes", "no"], ["yes", "no", "yes", "no"]), 1.0
        )
        # p_o = 0.5, p_e = 0.5 -> kappa 0
        self.assertAlmostEqual(
            ann.cohen_kappa(["yes", "yes", "no", "no"], ["yes", "no", "yes", "no"]), 0.0
        )
        # A worked example: p_o = 0.7, p_e = 0.5*0.6 + 0.5*0.4 = 0.5 -> kappa 0.4
        a = ["yes"] * 5 + ["no"] * 5
        b = ["yes", "yes", "yes", "yes", "no", "yes", "yes", "no", "no", "no"]
        self.assertAlmostEqual(ann.cohen_kappa(a, b), 0.4)

    def test_audit_sample(self):
        sample = ann.audit_sample(ITEMS)
        self.assertEqual(len(sample), 60)
        self.assertEqual(len({it["id"] for it in sample}), 60)
        self.assertEqual(sample, ann.audit_sample(ITEMS), "seeded")
        keep = ann.main_template_ids(ITEMS)
        self.assertTrue(all(it["id"] in keep for it in sample), "main template items only")
        strata = Counter((it["family"], it["difficulty"]) for it in sample)
        self.assertEqual(set(strata.values()), {5})
        self.assertEqual(sum(it["split"] == "train" for it in sample), 24)

    def test_committed_audit_sheet_is_blind_and_current(self):
        rows = ann.load_jsonl(ann.AUDIT_SHEET)
        self.assertEqual([r["id"] for r in rows], [it["id"] for it in ann.audit_sample(ITEMS)])
        for r in rows:
            self.assertEqual(
                set(r), {"id", "records", "request", "question", "options", "label", "notes"}
            )
            self.assertIsNone(r["label"])

    def test_hand_items_fill_the_slots_and_leave_the_rest_alone(self):
        """Hand items (fixtures copied from template items, not data) replace only slot items."""
        base, _ = bd.build_items([])
        donors = [it for it in base if it["split"] == "dev" and it["source"] == "template"][:6]
        hand = []
        for i, it in enumerate(donors):
            hand.append(
                {
                    "hand_id": f"h{i:03d}",
                    "split": "dev",
                    "family": it["family"],
                    "difficulty": it["difficulty"],
                    "state": it["state"],
                    "question": it["question"],
                    "options": it["options"],
                    "label_a": it["label"],
                    "label_b": it["label"],
                    "label": it["label"],
                    "resolution": "",
                    "rule": it["rule"],
                    "rationale": it["rationale"],
                    "author": "fixture",
                }
            )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hand.jsonl"
            path.write_text("".join(json.dumps(h) + "\n" for h in hand), encoding="utf-8")
            loaded = bd.load_hand(path)
        items, blob, stats = bd.build(hand=loaded)
        self.assertEqual(stats["hand_items"], 6)
        self.assertEqual(len(stats["fill_in_ids"]), 74)
        old = {it["id"]: it for it in ITEMS}
        fills = set(STATS["fill_in_ids"])
        for it in items:
            if it["id"] not in fills:
                self.assertEqual(it, old[it["id"]], "main template items must not change")
        self.assertEqual(sum(it["source"] == "hand" for it in items), 6)
        self.assertEqual(Counter(it["split"] for it in items), Counter(bd.SPLITS))

    def test_hand_file_rules(self):
        bad = {
            "hand_id": "h1",
            "split": "dev",
            "family": "policy",
            "difficulty": "plain",
            "state": {"request": "write to someone@gmail.com"},
            "question": bd.QUESTIONS["P2"],
            "options": ["yes", "no"],
            "label_a": "yes",
            "label_b": "no",
            "label": "yes",
            "resolution": "",
            "rule": "P2",
            "rationale": "",
            "author": "x",
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hand.jsonl"
            path.write_text(json.dumps(bad) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "resolution"):
                bd.load_hand(path)
            path.write_text(
                json.dumps(bad | {"resolution": "settled by P2"}) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "example.org"):
                bd.load_hand(path)


if __name__ == "__main__":
    unittest.main()
