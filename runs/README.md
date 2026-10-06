# Run records

One JSON file per batch of notebook runs: what ran, where, on which path, whether it passed and how long it took. They are the only evidence the site, the facilitator guide and the release check use for "this lab has run". Never state a run time or a verification claim in a page by hand: add a record here and rerun `scripts/gen_tables.py`.

`scripts/run_records.py` validates every file (`tests/test_runs.py`). `scripts/readiness.py` reads them.

## File format

```json
{
  "schema": 1,
  "date": "2026-10-06",
  "source": "test_notebooks",
  "env": "mac-m1pro",
  "env_detail": "macOS 26, Python 3.12, torch 2.14.1, CPU",
  "path": "open",
  "mode": "worked",
  "settings": {"NLP_LLMS_QUICK": "1"},
  "commit": "d462d65",
  "evidence": "where a reader can check this",
  "runs": [
    {"notebook": "14-agents", "scope": "notebook", "status": "pass", "seconds": 263.0}
  ]
}
```

Fields at the top apply to every entry in `runs`; an entry may override any of them (for example its own `date`, `env_detail`, `settings` or `evidence`).

| Field | Values |
|---|---|
| `source` | `test_notebooks` (written by `scripts/test_notebooks.py --record`), `colab` (pasted from a notebook's run record), `backfill` (copied from an earlier report; never satisfies the release check) |
| `env` | a key of `readiness.envs` in `_variables.yml` |
| `path` | `offline` (test doubles and stand-ins: the code runs, nothing about a model), `open` (no API keys and no test doubles: what a participant without keys runs), `keyed` (commercial APIs) |
| `mode` | `worked` (solutions bound, as in CI), `learner`, or `unknown` |
| `scope` | `notebook` (Run all, top to bottom) or `partial` (say what in `scope_note`) |
| `status` | `pass` or `fail` |
| `seconds` | wall time of the scope, or `null` if it was not recorded |
| `note` | optional: anything a reader needs to interpret the entry |
| `content_sha` | hash of the notebook's non-generated code cells at run time (from Phase 2 of the build); a record whose hash no longer matches the notebook is stale |

A record never holds API keys, cell source, outputs or anyone's identity.
