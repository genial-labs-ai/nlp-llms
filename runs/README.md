# Run records

One JSON file per batch of notebook runs: what ran, where, on which path, whether it passed and how long it took. They are the only evidence the site and the facilitator guide use for "this lab has run", and the release check planned for Phase 6 of the five-day revision will use them too. Never state a run time or a verification claim in a page by hand: add a record here and rerun `scripts/gen_tables.py`.

`scripts/run_records.py` validates every file, both when `scripts/gen_tables.py` loads them and in `tests/test_runs.py`. `scripts/readiness.py` turns them into evidence. Until Phase 2 of the five-day revision adds `scripts/test_notebooks.py --record` and a run-record cell in each notebook, records are written by hand from a run's printed times; give the run's real environment, and never label a laptop or CPU-runner time as a Colab or T4 time.

**Rules the readiness page applies.**

- The newest record of a kind wins, so a newer failure replaces an older pass.
- A lab counts as run on Colab only when all of these hold:
  - the run covers the whole notebook (`scope: notebook`);
  - it is a worked run (`mode: worked`, solutions bound);
  - it took the release path, `readiness.release_path` (`open`: no API keys, no test doubles);
  - it used full settings, not QUICK;
  - it ran on the module's own runtime, `modules.mNN.readiness.runtime`;
  - a tool recorded it (it is not a backfill);
  - its `content_sha` still matches the notebook.

  A keyed or learner-mode run is recorded and shown, but does not count.
- On the same date, a run of the current code wins over a stale one, then a failure over a pass.
- The release check will also require that run to be at most `readiness.max_run_age_days` old.

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

Fields at the top apply to every entry in `runs`; an entry may override any of them (for example its own `date`, `env_detail`, `settings` or `evidence`). Only the shared fields may sit at the top: `notebook`, `scope`, `status` and `seconds` belong to each entry.

| Field | Values |
|---|---|
| `source` | `test_notebooks` (a run by `scripts/test_notebooks.py`; its `--record` option arrives in Phase 2), `colab` (a run on Colab, from the notebook's printed times or, after Phase 2, its run-record cell), `backfill` (copied from an earlier report; never counts as Colab evidence) |
| `env` | a key of `readiness.envs` in `_variables.yml` |
| `path` | `offline` (test doubles and stand-ins: the code runs, nothing about a model), `open` (no API keys and no test doubles: what a participant without keys runs), `keyed` (commercial APIs) |
| `mode` | `worked` (solutions bound, as in CI), `learner`, or `unknown` |
| `scope` | `notebook` (Run all, top to bottom) or `partial` (say what in `scope_note`) |
| `status` | `pass` or `fail` |
| `seconds` | wall time of the scope, or `null` if it was not recorded |
| `note` | optional: anything a reader needs to interpret the entry |
| `content_sha` | `run_records.content_sha(slug)` at run time: a hash of the notebook's code cells, generated cells excluded. A record whose hash no longer matches the notebook is stale |

A record never holds API keys, cell source, outputs or anyone's identity.
