"""Lab 10's seed protocol: run the notebook over seeds, then derive its thresholds.

This is the protocol of briefs/10-rlhf.md ("Reliability"). It runs notebooks/10-rlhf.ipynb top to
bottom, solutions included, once per seed, each in a fresh kernel (a fresh CUDA context), and
collects the numbers each run writes. It then applies the brief's threshold rule:

  - a comparison may be asserted only if it has the right sign on all five seeds (0-4) and its
    smallest gap is at least twice the noise floor (the difference between two runs of seed 0);
  - its threshold is half the smallest gap. Thresholds are never set from seed 0 alone, and never
    loosened to make a run pass: if a comparison fails the rule, change the design and rerun.

Where to run it: a Colab T4 (or another CUDA machine) that can reach huggingface.co, with Lab 9's
three files in data/ (lab09_prompts.json, lab09_preferences.jsonl.gz, lab09_reward_model.pt).

Usage, from the repository root:

  python scripts/lab10_seed_protocol.py run --out lab10_protocol
      Seeds 0-4, then seed 0 again (about 10 minutes each on a T4, estimated). Finished runs are
      skipped, so the command can be repeated after a disconnect. --seeds 0 1 runs a subset;
      --no-repeat skips the repeat of seed 0; --skip-stretch drops the beta sweep.
  python scripts/lab10_seed_protocol.py run --out lab10_protocol --robustness rm1.pt rm2.pt
      Step 4 of the protocol: seed 0 with the reward models of Lab 9's seeds 1 and 2.
  python scripts/lab10_seed_protocol.py summarize --out lab10_protocol
      Prints the per-seed table, the noise floor and the THRESHOLDS to paste into the notebook's
      setup cell; writes summary.json and reward_drift_points.json (seed 0's stretch, for
      images/10-reward-drift.svg) into the output directory.

Design changes go in as --overrides '{"LR": 2e-5, "BETA": 0.05}' (keys of HP in the notebook).
Every run of one protocol must use the same overrides; summarize refuses to mix them.

On a CPU machine (step 6 of the protocol: the FAST path, seeds 0-2) add --allow-cpu; such runs
check only that the notebook completes, and summarize does not derive thresholds from them.

On Colab (Runtime -> Change runtime type -> T4 GPU), in a notebook cell; writing to Drive keeps
finished runs across a disconnect:

  from google.colab import drive; drive.mount("/content/drive")
  !git clone https://github.com/project-delphi/nlp-llms /content/nlp-llms
  %cd /content/nlp-llms
  !pip -q install nbclient nbformat transformers==5.18.0
  !python scripts/lab10_seed_protocol.py run --out /content/drive/MyDrive/lab10_protocol
  !python scripts/lab10_seed_protocol.py summarize --out /content/drive/MyDrive/lab10_protocol

The six runs take about an hour (estimated). Each run also saves its executed notebook, with the
side-by-side samples and plots, next to its results file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "notebooks" / "10-rlhf.ipynb"
SEEDS = [0, 1, 2, 3, 4]
METRICS = ("score", "gold", "drift", "distinct2")


# ---------------------------------------------------------------------------------------------
# Running the notebook


def run_notebook(label: str, out: Path, env: dict[str, str], timeout: int) -> bool:
    """Execute the notebook once with `env` added to the environment. Returns True on success."""
    import nbformat
    from nbclient import NotebookClient

    results = out / f"{label}.json"
    nb = nbformat.read(NOTEBOOK, as_version=4)
    saved = dict(os.environ)
    os.environ.update(env)
    os.environ["NLP_LLMS_LAB10_RESULTS"] = str(results)
    os.environ.setdefault("NLP_LLMS_DATA", str(ROOT / "data"))
    started = time.monotonic()
    ok = True
    try:
        with tempfile.TemporaryDirectory() as workdir:
            client = NotebookClient(
                nb,
                timeout=timeout,
                kernel_name="python3",
                resources={"metadata": {"path": workdir}},
            )
            client.execute()
    except Exception as error:  # keep going with the other seeds; the notebook is saved below
        ok = False
        print(f"FAILED {label}: {str(error)[-2000:]}")
    finally:
        os.environ.clear()
        os.environ.update(saved)
    nbformat.write(nb, out / f"{label}.ipynb")  # executed copy, with outputs, for the samples
    print(f"{'done' if ok else 'FAILED'}  {label}  {time.monotonic() - started:.0f} s")
    return ok and results.exists()


def cmd_run(args: argparse.Namespace) -> int:
    import torch

    if not torch.cuda.is_available() and not args.allow_cpu:
        print("No GPU: the protocol needs a T4 or another CUDA GPU (--allow-cpu: FAST path only).")
        return 2
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base = {"NLP_LLMS_LAB10_PROTOCOL": "1"}
    if args.overrides:
        json.loads(args.overrides)  # fail early on a typo
        base["NLP_LLMS_LAB10_OVERRIDES"] = args.overrides
    if args.skip_stretch:
        base["NLP_LLMS_LAB10_SKIP_STRETCH"] = "1"
    plan = [(f"seed{s}", s, None) for s in args.seeds]
    if not args.no_repeat and 0 in args.seeds:
        plan.append(("seed0-repeat", 0, None))
    for i, path in enumerate(args.robustness or [], start=1):
        plan.append((f"seed0-rm{i}", 0, str(Path(path).resolve())))
    failures = 0
    for label, seed, reward_model in plan:
        if (out / f"{label}.json").exists() and not args.force:
            print(f"skip  {label} (already done; --force reruns it)")
            continue
        env = {**base, "NLP_LLMS_LAB10_SEED": str(seed)}
        if reward_model:
            env["NLP_LLMS_LAB10_REWARD_MODEL"] = reward_model
        failures += not run_notebook(label, out, env, args.timeout)
    print(f"{len(plan) - failures} of {len(plan)} runs available in {out}")
    return 1 if failures else 0


# ---------------------------------------------------------------------------------------------
# The threshold rule


def comparisons(r: dict) -> dict[str, float]:
    """The quantities the notebook's trained-policy checkpoints assert, for one run."""
    p = r["policies"]
    ref, beta, hack, dpo = p["reference"], p["rlhf_beta"], p["rlhf_beta0"], p["dpo"]
    c = {
        "reward_gain": beta["score"] - ref["score"],  # Checkpoint 3
        "drift_beta": beta["drift"],  # Checkpoint 3 (upper bound)
        "drift_beta0": hack["drift"],
        "hack_score": hack["score"] - beta["score"],  # Checkpoint 4 (a)
        "hack_drift_ratio": hack["drift"] / max(beta["drift"], 1e-9),  # (b)
        "hack_gold": beta["gold"] - hack["gold"],  # (c)
        "hack_distinct": beta["distinct2"] - hack["distinct2"],  # (d)
        "dpo_acc": r["heldout_accuracy"]["dpo"],  # Checkpoint 5
        "dpo_drift": dpo["drift"],
    }
    stretch = r.get("stretch")
    if stretch:
        sweep = sorted(
            (pt for pt in stretch if pt.get("method") != "DPO"), key=lambda pt: pt["beta"]
        )
        drifts = [pt["drift"] for pt in sweep]
        c["sweep_monotone"] = float(all(a >= b for a, b in zip(drifts, drifts[1:], strict=False)))
    return c


def derive_thresholds(runs: dict[str, dict], repeat: dict | None) -> dict:
    """Apply the brief's rule to the five seeds. Returns per-comparison verdicts and thresholds."""
    per_seed = {label: comparisons(r) for label, r in runs.items()}
    noise = {}
    if repeat is not None:
        rep = comparisons(repeat)
        noise = {k: abs(per_seed["seed0"][k] - rep[k]) for k in rep if k in per_seed["seed0"]}

    def values(key):
        return [c[key] for c in per_seed.values()]

    def gap_rule(key, offset=0.0):
        """A comparison that must be positive: gap = value - offset on every seed."""
        gaps = [v - offset for v in values(key)]
        floor = 2 * noise.get(key, float("nan"))
        ok = all(g > 0 for g in gaps) and (not noise or min(gaps) >= floor)
        return {
            "gaps": gaps,
            "min_gap": min(gaps),
            "noise_floor": noise.get(key),
            "assertable": ok,
            "threshold": offset + min(gaps) / 2 if ok else None,
        }

    out = {"per_seed": per_seed, "noise_floor": noise, "rules": {}}
    rules = out["rules"]
    rules["reward_gain_min"] = gap_rule("reward_gain")
    # Drift bound for the BETA policy: between the largest BETA drift and the smallest beta=0 drift.
    hi, lo = max(values("drift_beta")), min(values("drift_beta0"))
    sep = lo - hi
    ok = sep > 0 and (not noise or sep >= 2 * noise.get("drift_beta", 0.0))
    rules["drift_max"] = {
        "max_drift_beta": hi,
        "min_drift_beta0": lo,
        "noise_floor": noise.get("drift_beta"),
        "assertable": ok,
        "threshold": hi + sep / 2 if ok else None,
    }
    # (a) "at least as high, within a tolerance": the tolerance covers the worst seed and the noise.
    worst = min(values("hack_score"))
    tol = max(0.0, -worst) + 2 * noise.get("hack_score", 0.0)
    rules["hack_score_tol"] = {
        "gaps": values("hack_score"),
        "min_gap": worst,
        "noise_floor": noise.get("hack_score"),
        "assertable": True,
        "threshold": tol,
    }
    # (b) ratio and absolute floor.
    ratios = values("hack_drift_ratio")
    rule_b = gap_rule("hack_drift_ratio", offset=1.0)
    rule_b["min_ratio"] = min(ratios)
    rule_b["meets_3x"] = min(ratios) >= 3.0
    rules["hack_drift_ratio"] = rule_b
    rules["hack_drift_floor"] = {
        "min_drift_beta0": lo,
        "assertable": lo > 0,
        "threshold": lo / 2 if lo > 0 else None,
    }
    rules["hack_gold_margin"] = gap_rule("hack_gold")
    rules["hack_distinct_margin"] = gap_rule("hack_distinct")
    rules["dpo_acc_min"] = gap_rule("dpo_acc", offset=0.5)
    if all("sweep_monotone" in c for c in per_seed.values()):
        mono = all(c["sweep_monotone"] == 1.0 for c in per_seed.values())
        rules["sweep_monotone"] = {"assertable": mono, "threshold": mono}
    out["thresholds"] = {k: v["threshold"] for k, v in rules.items()}
    return out


def signature_holds(c: dict[str, float], t: dict) -> dict[str, bool]:
    """Checkpoint 4's four parts for one run, under thresholds `t` (None = not assertable)."""
    return {
        "a": t["hack_score_tol"] is not None and c["hack_score"] >= -t["hack_score_tol"],
        "b": t["hack_drift_ratio"] is not None
        and t["hack_drift_floor"] is not None
        and c["hack_drift_ratio"] >= t["hack_drift_ratio"]
        and c["drift_beta0"] >= t["hack_drift_floor"],
        "c": t["hack_gold_margin"] is not None and c["hack_gold"] >= t["hack_gold_margin"],
        "d": t["hack_distinct_margin"] is not None
        and c["hack_distinct"] >= t["hack_distinct_margin"],
    }


def cmd_summarize(args: argparse.Namespace) -> int:
    out = Path(args.out)
    runs = {}
    for s in SEEDS:
        path = out / f"seed{s}.json"
        if path.exists():
            runs[f"seed{s}"] = json.loads(path.read_text())
    missing = [f"seed{s}" for s in SEEDS if f"seed{s}" not in runs]
    if missing:
        print(f"Missing runs: {', '.join(missing)}. The rule needs all five seeds.")
        return 2
    repeat_path = out / "seed0-repeat.json"
    repeat = json.loads(repeat_path.read_text()) if repeat_path.exists() else None
    everything = {**runs, **({"seed0-repeat": repeat} if repeat else {})}
    if len({json.dumps(r["hp"], sort_keys=True) for r in everything.values()}) != 1:
        print("The runs used different hyperparameters (overrides). Rerun them with one design.")
        return 2
    if any(r["fast"] or r["offline_test_stand_in"] for r in everything.values()):
        print("These are FAST (CPU) or stand-in runs: they set no thresholds. Completed runs:")
        for label, r in everything.items():
            print(f"  {label}: device {r['device']}, {r['seconds'].get('total so far')} s")
        return 0

    result = derive_thresholds(runs, repeat)
    print(f"hyperparameters: {runs['seed0']['hp']}")
    print(f"GPU {runs['seed0']['gpu']} · {runs['seed0']['versions']} · {runs['seed0']['date']}\n")
    head = f"{'run':<14}" + "".join(f"{k:>14}" for k in result["per_seed"]["seed0"])
    print(head)
    for label, c in {
        **result["per_seed"],
        **({"seed0-repeat": comparisons(repeat)} if repeat else {}),
    }.items():
        print(f"{label:<14}" + "".join(f"{v:>14.4f}" for v in c.values()))
    print(
        "\nnoise floor (|seed 0 - repeat|):",
        {k: round(v, 4) for k, v in result["noise_floor"].items()},
    )
    if repeat is None:
        print("WARNING: no repeat of seed 0, so no noise floor; the rule is incomplete.")
    print("\nrule per comparison:")
    for key, rule in result["rules"].items():
        verdict = (
            "assertable" if rule["assertable"] else "NOT ASSERTABLE: change the design and rerun"
        )
        print(f"  {key:<22} threshold {rule['threshold']!s:<24} {verdict}")
    if (
        "hack_drift_ratio" in result["rules"]
        and not result["rules"]["hack_drift_ratio"]["meets_3x"]
    ):
        print("  note: the smallest drift ratio is below 3; the brief says 'at least 3 times'.")

    print("\nseconds per section (seed 0):", runs["seed0"]["seconds"])
    print("GPU peak memory (GB) per seed:", [r["gpu_peak_gb"] for r in runs.values()])
    failures = {
        label: r["check_failures"] for label, r in everything.items() if r["check_failures"]
    }
    print("checks that failed under the notebook's current thresholds:", failures or "none")

    robustness = {}
    for path in sorted(out.glob("seed0-rm*.json")):
        c = comparisons(json.loads(path.read_text()))
        robustness[path.stem] = signature_holds(c, result["thresholds"])
    if robustness:
        print("\nsecondary robustness check (reward models of Lab 9's seeds 1 and 2):", robustness)

    t = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in result["thresholds"].items()}
    print("\nPaste into the notebook's setup cell (only if every comparison is assertable):")
    print(
        f'THRESHOLDS_SOURCE = "seed protocol {runs["seed0"]["date"]}, seeds 0-4, '
        f'{runs["seed0"]["gpu"]}"'
    )
    print("THRESHOLDS = dict(")
    for k, v in t.items():
        print(f"    {k}={v!r},")
    print(")")

    summary = {
        **result,
        "hp": runs["seed0"]["hp"],
        "robustness": robustness,
        "environment": {k: runs["seed0"][k] for k in ("gpu", "versions", "date", "device")},
        "seconds": {label: r["seconds"] for label, r in everything.items()},
        "gpu_peak_gb": {label: r["gpu_peak_gb"] for label, r in everything.items()},
        "final_table_seed0": runs["seed0"]["policies"],
        "heldout_accuracy_seed0": runs["seed0"]["heldout_accuracy"],
        "samples_seed0": runs["seed0"]["samples"],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    if runs["seed0"].get("stretch"):
        points = {"seed": 0, "hp": runs["seed0"]["hp"], "points": runs["seed0"]["stretch"]}
        (out / "reward_drift_points.json").write_text(json.dumps(points, indent=2))
    print(f"\nwrote {out / 'summary.json'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="execute the notebook once per seed")
    run.add_argument("--out", required=True, help="directory for the per-run results")
    run.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    run.add_argument("--no-repeat", action="store_true", help="skip the second run of seed 0")
    run.add_argument("--robustness", nargs="*", help="reward-model files for step 4 (seed 0 only)")
    run.add_argument("--overrides", help="JSON for the notebook's HP, e.g. '{\"LR\": 2e-5}'")
    run.add_argument("--skip-stretch", action="store_true", help="drop the beta sweep")
    run.add_argument("--force", action="store_true", help="rerun runs that already have results")
    run.add_argument("--allow-cpu", action="store_true", help="run on a CPU (FAST path only)")
    run.add_argument("--timeout", type=int, default=3600, help="per-cell limit in seconds")
    summarize = sub.add_parser("summarize", help="apply the threshold rule to finished runs")
    summarize.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    return cmd_run(args) if args.command == "run" else cmd_summarize(args)


if __name__ == "__main__":
    sys.exit(main())
