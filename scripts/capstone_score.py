"""Lab 15's scoring as a module: the fixed cell "Scoring: do not edit" of 15-capstone.ipynb.

The block between the BEGIN and END markers below is restated verbatim in that notebook cell, and
tests/test_capstone_score.py fails if the two differ. The notebook writes `scoring_hash()` into
every submission; scripts/collect_capstone.py recomputes it from this file and refuses to compare
a submission whose hash differs. Specification: briefs/15-capstone.md, section (e); equations from
lecture 15 (eq-cap-acc, eq-cap-abstain, eq-cap-unsupported, eq-cap-cost, eq-sign-test).

Standard library only, so the CI test job (PyYAML and nbformat only) can import it.

The block reads three names defined outside it, here and in the notebook: ABSTAIN (Lab 13's
abstention sentence), LOSS (lecture eq-cap-cost) and REFERENCE_JUDGE (the notebook's setup sets
it to the NLI judge, or to the stub judge offline); and the standard-library modules below.
"""

import hashlib
import inspect
import json
import math
import re
import statistics
import unicodedata

ABSTAIN = "I cannot answer from the provided sources."  # Lab 13's abstention sentence
LOSS = dict(wrong=5, abstain=1)  # lecture eq-cap-cost; fixed for every pair
REFERENCE_JUDGE = None  # the notebook sets it; here, pass judge= explicitly

# ---- BEGIN SCORING (restated verbatim in notebooks/15-capstone.ipynb) ----
# Lab 13's answer checks, restated word for word, so that this cell's hash covers every rule used.
CITATION = re.compile(r"\s*\[\d+(?:\s*,\s*\d+)*\]")


def is_abstention(answer):
    return " ".join(answer.split()).strip().rstrip(".") == ABSTAIN.rstrip(".")


def split_claims(answer):
    """The answer's sentences, citations removed (lecture section 9: one sentence is one claim)."""
    if is_abstention(answer):
        return []
    text = " ".join(CITATION.sub("", answer).split())
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def normalize(text):
    return unicodedata.normalize("NFKC", text).casefold()


def is_correct(answer, item):
    """True/False if the item has key_facts (any one must appear); None if not auto-checkable."""
    if not item.get("key_facts"):
        return None
    return any(normalize(f) in normalize(answer) for f in item["key_facts"])


def answer_class(record):
    """z_i: "error" if the run failed; "abstain" if the delivered text is the abstention sentence
    (exact match after whitespace normalization: a paraphrased refusal is an answer); "answer"."""
    if record.get("output") == "error":
        return "error"
    return "abstain" if is_abstention(record.get("final") or "") else "answer"


def item_class(item):
    """A: answerable, with key facts (scored); U: unanswerable, no evidence (scored);
    R: answerable without key facts (reported, not scored)."""
    if not item["evidence"]:
        return "U"
    return "A" if item.get("key_facts") else "R"


def correctness(record, item):
    """o_i: 1 if the output answers an A question and contains one of its key facts, else 0."""
    ok = item_class(item) == "A" and answer_class(record) == "answer"
    return int(ok and bool(is_correct(record["final"], item)))


def question_cost(record, item, loss=LOSS):
    """One scored question's term of eq-cap-cost. An error counts as a wrong answer (Module 8)."""
    if answer_class(record) == "abstain":
        return loss["abstain"] if item_class(item) == "A" else 0
    return loss["wrong"] * (1 - correctness(record, item))


def unsupported(record, judge):
    """1 if a delivered answer has a sentence that the reference judge scores below 1/2 against
    the passages it was written from (eq-cap-unsupported), else 0."""
    claims = split_claims(record["final"])
    return int(bool(claims) and min(judge(c, record["sources"]) for c in claims) < 0.5)


def _rate(hits, n):
    """A rate with its denominator and Module 8's standard error sqrt(r (1 - r) / N)."""
    if n == 0:
        return {"value": None, "se": None, "n": 0}
    r = hits / n
    return {"value": r, "se": math.sqrt(r * (1 - r) / n), "n": n}


def _mean(values):
    """A mean with its denominator and standard error (population SD / sqrt(N))."""
    if not values:
        return {"value": None, "se": None, "n": 0}
    n = len(values)
    return {"value": sum(values) / n, "se": statistics.pstdev(values) / math.sqrt(n), "n": n}


def _quantile(values, q):
    """The q-quantile, linear interpolation between order statistics (NumPy's default)."""
    if not values:
        return None
    xs = sorted(values)
    h = (len(xs) - 1) * q
    lo = math.floor(h)
    return xs[lo] + (h - lo) * (xs[min(lo + 1, len(xs) - 1)] - xs[lo])


def _block(pairs, loss, judge):
    """The five numbers and the capstone cost on a list of (record, item) pairs."""
    A = [(r, it) for r, it in pairs if item_class(it) == "A"]
    U = [(r, it) for r, it in pairs if item_class(it) == "U"]
    answered_A = [(r, it) for r, it in A if answer_class(r) == "answer"]
    delivered = [r for r, _ in pairs if answer_class(r) == "answer"]
    return {
        "acc": _rate(sum(correctness(r, it) for r, it in A), len(A)),
        "acc_answered": _rate(sum(correctness(r, it) for r, it in answered_A), len(answered_A)),
        "abs_U": _rate(sum(answer_class(r) == "abstain" for r, _ in U), len(U)),
        "abs_A": _rate(sum(answer_class(r) == "abstain" for r, _ in A), len(A)),
        "uns": _rate(sum(unsupported(r, judge) for r in delivered), len(delivered)),
        "cost_bar": _mean([question_cost(r, it, loss) for r, it in A + U]),
        "n_scored": len(A) + len(U),
        "n_reported_only": sum(item_class(it) == "R" for _, it in pairs),
        "n_errors": sum(answer_class(r) == "error" for r, _ in pairs),
    }


def score(records, items, *, loss=LOSS, judge=None):
    """Pooled, by source and by kind, each number with its N and standard error; plus dollars
    and calls per question and the latency median and 95th percentile (pooled). Correctness
    and support are computed here, from the records, never by the system."""
    judge = judge or REFERENCE_JUDGE
    by_id = {it["id"]: it for it in items}
    pairs = [(r, by_id[r["id"]]) for r in records]
    out = {"pooled": _block(pairs, loss, judge), "by_source": {}, "by_kind": {}}
    for key, field in (("by_source", "source"), ("by_kind", "kind")):
        for value in sorted({it[field] for _, it in pairs}):
            subset = [(r, it) for r, it in pairs if it[field] == value]
            out[key][value] = _block(subset, loss, judge)
    usd = [r["usd"] for r in records]
    latency = [r["latency_s"] for r in records]
    calls = [r["n_llm"] + r["n_dec"] for r in records]
    out["pooled"]["usd_per_q"] = None if None in usd or not usd else sum(usd) / len(usd)
    out["pooled"]["calls_per_q"] = sum(calls) / len(calls) if calls else None
    out["pooled"]["latency_median"] = _quantile(latency, 0.5)
    out["pooled"]["latency_p95"] = _quantile(latency, 0.95)
    return out


def sign_test(gained, lost):
    """Two-sided exact sign test on the questions that changed (eq-sign-test), with math.comb."""
    n = gained + lost
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, j) for j in range(max(gained, lost), n + 1)) / 2**n)


def _paired_costs(run_a, run_b, items, loss):
    by_id = {it["id"]: it for it in items}
    a, b = {r["id"]: r for r in run_a}, {r["id"]: r for r in run_b}
    ids = [i for i in a if i in b and item_class(by_id[i]) != "R"]
    return [(question_cost(a[i], by_id[i], loss), question_cost(b[i], by_id[i], loss)) for i in ids]


def compare(base, new, items, *, loss=LOSS, judge=None):
    """Per-question capstone cost, baseline against new, on the scored questions of both runs:
    gained (cost fell), lost (cost rose), the sign test, and the change in every pooled number."""
    costs = _paired_costs(base, new, items, loss)
    gained = sum(y < x for x, y in costs)
    lost = sum(y > x for x, y in costs)
    sb = score(base, items, loss=loss, judge=judge)["pooled"]
    sn = score(new, items, loss=loss, judge=judge)["pooled"]
    deltas = {}
    for m in ("acc", "acc_answered", "abs_U", "abs_A", "uns", "cost_bar"):
        x, y = sb[m]["value"], sn[m]["value"]
        deltas[m] = None if x is None or y is None else y - x
    for m in ("usd_per_q", "calls_per_q", "latency_median", "latency_p95"):
        deltas[m] = None if sb[m] is None or sn[m] is None else sn[m] - sb[m]
    p = sign_test(gained, lost)
    return {"n_paired": len(costs), "gained": gained, "lost": lost, "p": p, "deltas": deltas}


def flips(run_a, run_b, items, *, loss=LOSS):
    """Scored questions whose per-question cost differs between two runs (the run-to-run floor)."""
    return sum(x != y for x, y in _paired_costs(run_a, run_b, items, loss))


def scoring_hash():
    """SHA-256 of the source of every function in this cell (inspect.getsource), the citation
    pattern, the abstention sentence and the costs. Written into every submission."""
    text = "\n".join(inspect.getsource(f) for f in SCORING_FUNCTIONS)
    text += CITATION.pattern + ABSTAIN + json.dumps(LOSS, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def submission(
    pair, hypothesis, runs, *, items, context, share_text, recorded_test=None, judge=None, loss=LOSS
):
    """Writes capstone_<pair>_<path class>.json and returns it. `runs` holds the record lists
    {"baseline": {"dev", "test"}, "final": {"dev", "test"}}; `context` what the notebook knows:
    path class, backend labels, configuration diff, manifest hash, test runs, budgets, seconds."""
    budgets = context["budgets"]
    scores = {
        tag: {
            split: score(runs[tag][split], items, loss=loss, judge=judge)
            for split in ("dev", "test")
        }
        for tag in ("baseline", "final")
    }
    base_usd = scores["baseline"]["test"]["pooled"]["usd_per_q"]
    final_usd = scores["final"]["test"]["pooled"]["usd_per_q"]
    tested = runs["baseline"]["test"] + runs["final"]["test"]
    sub = {
        "pair": str(pair),
        "path_class": context["path_class"],
        "backends": dict(context["backends"]),
        "eval_set": context["eval_set"],
        "hypothesis": dict(hypothesis),
        "config_diff": context["config_diff"],
        "edited_cells": list(context["edited_cells"]),
        "scoring_hash": scoring_hash(),
        "manifest_sha256": context["manifest_sha256"],
        "notebook": context["notebook"],
        "loss": dict(loss),
        "abstain_sentence": ABSTAIN,
        "score": scores,
        "compare": {
            split: compare(
                runs["baseline"][split], runs["final"][split], items, loss=loss, judge=judge
            )
            for split in ("test", "dev")
        },
        "flips": None
        if recorded_test is None
        else flips(runs["baseline"]["test"], recorded_test, items, loss=loss),
        "test_runs": context["test_runs"],
        "budget": {
            "calls_ok": all(
                r["n_llm"] <= budgets["k_max_llm"] and r["n_dec"] <= budgets["k_max_dec"]
                for r in tested
            ),
            "test_seconds": dict(context["test_seconds"]),
            "time_ok": all(
                s <= budgets["test_seconds_max"] for s in context["test_seconds"].values()
            ),
            "usd_ok": None if base_usd is None or final_usd is None else final_usd <= 2 * base_usd,
        },
        "share_text": dict(share_text),
    }
    sub["file"] = f"capstone_{sub['pair']}_{sub['path_class']}.json"
    with open(sub["file"], "w", encoding="utf-8") as f:
        json.dump(sub, f, indent=1, sort_keys=True)
    return sub


def _fmt(metric):
    if metric["value"] is None:
        return "n/a"
    return f"{metric['value']:.3f} ± {metric['se']:.3f} (N = {metric['n']})"


def share_card(sub):
    """Prints the share card: the four backend labels above the numbers, test before and after."""
    print("=" * 96)
    print(f"Capstone share card · pair {sub['pair']} · path class {sub['path_class']}")
    for role, name in sub["backends"].items():
        print(f"  {role:<10} {name}")
    print(f"  evaluation set: {sub['eval_set']}")
    print("-" * 96)
    before = sub["score"]["baseline"]["test"]["pooled"]
    after = sub["score"]["final"]["test"]["pooled"]
    rows = (
        ("cost_bar", "capstone cost (eq-cap-cost)"),
        ("acc", "accuracy on A"),
        ("abs_U", "abstention on U"),
        ("abs_A", "abstention on A"),
        ("uns", "unsupported answers"),
    )
    for m, name in rows:
        print(f"  {name:<30} {_fmt(before[m]):<28} -> {_fmt(after[m])}")
    c = sub["compare"]["test"]
    print(
        f"  gained {c['gained']}, lost {c['lost']}, sign-test p = {c['p']:.4f} "
        f"(N paired = {c['n_paired']})"
    )
    print(f"  run-to-run flips against the recorded baseline: {sub['flips']}")
    if before["usd_per_q"] is None or after["usd_per_q"] is None:
        usd = "n/a (local)"
    else:
        usd = f"${before['usd_per_q']:.4f} -> ${after['usd_per_q']:.4f}"
    print(
        f"  cost per question {usd}; calls per question "
        f"{before['calls_per_q']:.2f} -> {after['calls_per_q']:.2f}"
    )
    print(
        f"  latency median {before['latency_median']:.2f} s -> "
        f"{after['latency_median']:.2f} s; test runs {sub['test_runs']}"
    )
    print("-" * 96)
    print(f"  changed [{sub['hypothesis']['component']}]: {sub['share_text']['changed']}")
    print(f"  predicted: {sub['share_text']['predicted']}")
    print(f"  happened: {sub['share_text']['happened']}")
    manifest = str(sub["manifest_sha256"])[:16]
    print(f"  scoring hash {sub['scoring_hash'][:16]}... manifest {manifest}...")
    print("=" * 96)


SCORING_FUNCTIONS = (
    is_abstention,
    split_claims,
    normalize,
    is_correct,
    answer_class,
    item_class,
    correctness,
    question_cost,
    unsupported,
    _rate,
    _mean,
    _quantile,
    _block,
    score,
    sign_test,
    _paired_costs,
    compare,
    flips,
    scoring_hash,
    submission,
    _fmt,
    share_card,
)
# ---- END SCORING ----

BEGIN, END = "# ---- BEGIN SCORING", "# ---- END SCORING ----"


def block_source(path=__file__):
    """The text from the BEGIN marker to the END marker, as the notebook restates it."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    return text[text.index(BEGIN) : text.index(END) + len(END)]
