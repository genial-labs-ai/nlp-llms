# Data

Fallback copies of the workshop datasets live here once the labs that use them are written. Notebooks fetch data by URL and fall back to these copies.

## Proposed datasets for the running thread

These are proposals. Each license must be confirmed before the lab that uses it is written (PLAN.md, build Day 1).

| Use | Modules | Proposed dataset | License status |
|---|---|---|---|
| Text classification | 1, 2, 6, 11 | AG News, 4-class subset | To confirm |
| Language modeling | 1, 3, 5 | Tiny Shakespeare | Public-domain text; to confirm the packaged source |
| Sequence transduction | 4 | Human-readable dates to ISO format | Generated in the notebook; no license needed |
| Instruction tuning | 7 | A small subset of Databricks Dolly 15k | To confirm (believed CC BY-SA 3.0) |
| Pairwise preferences | 9, 10 | Synthetic, with a known hidden preference | Generated in the notebook |
| Labeled decisions | 11, 12, 14 | Built for the workshop on build Day 8 | Ours |
| RAG documents | 13, 15 | The workshop's own lecture pages and reading list | Ours |
