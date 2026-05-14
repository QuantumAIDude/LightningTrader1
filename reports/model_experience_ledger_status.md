# Model Experience Ledger Status

| Field | Value |
| --- | --- |
| Ledger path | `data/model_experience_ledger_QQQ_sample.jsonl` |
| Exists | True |
| Total records | 500 |
| Experience records | 500 |
| Outcome records | 0 |
| Resolved experience records | 499 |
| Unresolved experience records | 1 |
| Symbols | QQQ |
| Horizons | 1_step, 4min |

## Missing Fields Blocking Meta-Model Training

- unresolved future outcomes

## Recommended Next Step

Run replay/backtest with `experience_ledger_path` enabled, then require enough resolved records before training any meta-gating model.

## Sample Ledger Note

This status reflects the bounded QQQ sample ledger at `data/model_experience_ledger_QQQ_sample.jsonl`. The production default ledger `data/model_experience_ledger.jsonl` was not written by this script.

## Schema Quality

- Valid / warning / invalid: 500 / 0 / 0
- Verdict: research_ready_small_sample
