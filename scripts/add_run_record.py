"""Add a run record printed by a notebook's last code cell (workshop.run_record()) to runs/.

Copy everything between the "run record" marker lines that the cell prints, then:

    pbpaste | uv run --group site python scripts/add_run_record.py
    uv run --group site python scripts/gen_tables.py

The machine is read from the record: a T4 on Colab is colab-t4, Colab without a GPU is
colab-cpu. Anywhere else, name it with --env (a key of readiness.envs in _variables.yml).
The path is offline when an offline flag was set, keyed when the provider was a commercial
API (or --path keyed, for example when only a TypeSafe key was used), and open otherwise.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_records  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def env_of(record: dict, given: str | None) -> str:
    if given:
        return given
    gpu = record.get("gpu") or ""
    if not record.get("colab_release"):
        raise SystemExit("Not a Colab run (no colab_release): name the machine with --env")
    if not gpu:
        return "colab-cpu"
    if "T4" in gpu:
        return "colab-t4"
    raise SystemExit(f"A Colab run on {gpu}, not a T4: name the machine with --env")


def path_of(record: dict, given: str | None) -> str:
    if given:
        return given
    settings = record.get("settings") or {}
    if any(settings.get(flag) for flag in run_records.OFFLINE_FLAGS):
        return "offline"
    return "keyed" if record.get("provider") in ("openai", "anthropic") else "open"


def batch_from(record: dict, env: str, path: str, date: str) -> dict:
    gpu = record.get("gpu") or "no GPU"
    return {
        "schema": run_records.SCHEMA,
        "date": date,
        "source": "colab" if env.startswith("colab") else "notebook",
        "env": env,
        "env_detail": (
            f"Python {record.get('python')}, {gpu}"
            + (f", Colab {record['colab_release']}" if record.get("colab_release") else "")
        ),
        "path": path,
        "mode": record["mode"],
        "settings": {k: str(v) for k, v in (record.get("settings") or {}).items()},
        "packages": record.get("packages", {}),
        "evidence": "workshop.run_record() output, added with scripts/add_run_record.py",
        "runs": [
            {
                "notebook": record["notebook"],
                "scope": "notebook",
                "status": record["status"],
                "seconds": record["seconds"],
                "content_sha": record["content_sha"],
            }
        ],
    }


def extract(text: str) -> dict:
    """The JSON object in pasted text, with or without the marker lines around it."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise SystemExit("No run record found in the input")
    return json.loads(text[start : end + 1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("file", nargs="?", type=Path, help="default: read standard input")
    parser.add_argument("--env", help="a key of readiness.envs, if not a Colab run")
    parser.add_argument("--path", choices=run_records.PATHS)
    parser.add_argument("--date", default=dt.date.today().isoformat())
    args = parser.parse_args()
    text = args.file.read_text(encoding="utf-8") if args.file else sys.stdin.read()
    record = extract(text)
    v = yaml.safe_load((ROOT / "_variables.yml").read_text(encoding="utf-8"))
    envs = set(v["readiness"]["envs"])
    batch = batch_from(record, env_of(record, args.env), path_of(record, args.path), args.date)
    notebooks = {p.stem for p in run_records.NOTEBOOKS.glob("*.ipynb")}
    if record.get("notebook") not in notebooks:
        raise SystemExit(f"Not added: no notebook named {record.get('notebook')!r}")
    if batch["runs"][0]["content_sha"] != run_records.content_sha(record["notebook"]):
        print(
            "Note: the record was made against other code than the committed notebook's;"
            " it is kept, and the readiness page marks it stale."
        )
    stamp = dt.datetime.now().strftime("%H%M%S")
    name = f"{batch['date']}-{batch['env']}-{batch['path']}-{record['notebook']}-{stamp}.json"
    out = run_records.RUNS / name
    body = json.dumps(batch, indent=2) + "\n"
    _, problems = run_records.check_batch(name, body, envs, notebooks)
    if problems:
        raise SystemExit("Not added:\n  " + "\n  ".join(problems))
    out.write_text(body, encoding="utf-8")
    print(f"Wrote {out.relative_to(ROOT)}. Now run scripts/gen_tables.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
