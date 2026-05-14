# Freshness / Evidence Hardening Handoff

## A. Current Symbol Reality

### Symbols inspected

- `QQQ`
- `AAPL`

### QQQ current reality

- Current deployment stage remains `advisory_only`.
- Current quick system check still returns `status=fail`.
- Current quick system-check blockers remain real and active:
  - `critical_stale=true`
  - `readiness_locked=true`
  - `degenerate=true`
  - `calibration_drift=true`
  - `calibration_evidence_stale=true`
  - `news_status=degraded`
  - `headline_state=stale`
  - `history_state=insufficient_history`
- Current persisted deployment evidence in `data/deployment_evidence_QQQ.json` now shows:
  - checklist summary `5/15`
  - primary blocker `critical stale-data guard is active`
  - calibration evidence age `22.6` days
  - calibration evidence stale `true`
  - `pending_unresolved_rows=2`
  - shadow observations `1`
  - clean shadow observations `0`
  - blocked shadow observations `1`
- Current stage-history file `data/deployment_stage_history_QQQ.jsonl` now has `9` rows.
- Latest persisted QQQ next requirements are now explicit:
  - refresh price data until critical stale clears
  - refresh/extend news until headline state is not stale or missing
  - rebuild live calibration and confirm drift clears
  - resolve fresh intraday predictions so calibration evidence becomes current
  - advance price history enough to resolve the `2` pending calibration snapshots
  - accumulate more passing readiness days
  - improve horizon accuracy above floor
  - collect at least one clean shadow observation cycle

### AAPL current reality

- Current deployment stage remains `supported_not_trained`.
- Current quick system check still returns `status=fail`.
- `AAPL` remains a clean comparison case for onboarding blockers rather than signal blockers.
- Current persisted deployment evidence in `data/deployment_evidence_AAPL.json` shows:
  - checklist summary `4/15`
  - primary blocker `no price data`
  - no calibration file
  - no price history
  - no core 4-minute models
  - no readiness metrics
  - no readiness history
  - no shadow observation cycles
- Current stage-history file `data/deployment_stage_history_AAPL.jsonl` now has `5` rows.

### Did anything advance?

- No symbol advanced a deployment stage in this pass.
- That is the correct outcome from the current evidence.

## B. Freshness / Readiness / Drift Diagnosis

### What was real

- QQQ price freshness failure is real:
  - `data/prices_QQQ.csv` and related cached price artifacts still end on `2026-03-23`.
  - quick system check still marks QQQ as critically stale.
- QQQ readiness lock is real:
  - readiness metrics are weak and the pass streak is not satisfied.
- QQQ signal-quality blockers are still real:
  - weighted horizon score remains far below the deployment floor.
  - hit-rate metrics remain weak.
- QQQ news degradation is real enough to block progression:
  - headline state is still stale.
  - recent news history is still insufficient.
- QQQ shadow evidence problem is real:
  - before this pass there were zero shadow observation entries.
  - after this pass there is now one recorded shadow observation, but it is blocked and not clean.

### What was noisy or too brittle

- The biggest noisy/misleading area was calibration freshness semantics.
- Before this pass, the system could mark calibration state as effectively fresh because:
  - `last_update` on `calibration_live_QQQ.json` was recent
  - while the resolved calibration evidence anchor (`last_processed` / `last_resolved_target_ts`) was still weeks old
- This created a truthfulness problem:
  - quick calibration refresh could touch the file
  - but no new resolved targets had actually arrived
  - and deployment evidence could still understate that the calibration evidence was operationally stale
- The stale threshold was also too loose for deployment evidence:
  - calibration evidence age was being judged with `CALIBRATION_MAX_AGE_DAYS=30`
  - that is acceptable for long-lived file retention, but too permissive for deployment-quality intraday evidence

### What was fixed

- Calibration evidence freshness is now judged separately from calibration file age.
- A new config threshold now exists:
  - `CALIBRATION_EVIDENCE_MAX_AGE_DAYS`, default `7`
- `calibration_tracker.py` now marks evidence stale from resolved evidence age, not from the file update time.
- `system_checkup.py` now reports calibration evidence age/staleness directly.
- `readiness_tracker.py` now recomputes calibration evidence staleness from age even if an older persisted `evidence_stale=false` still exists in stored state.
- This removed the specific inconsistency where:
  - quick system check was right
  - but deployment summary could still inherit an older false stale bit

### What still remains

- QQQ calibration drift remains active.
- QQQ calibration evidence is now represented more honestly, but it is still not good evidence.
- QQQ still has unresolved pending calibration rows because the price history has not advanced far enough to resolve the newest logged predictions.
- The platform still needs genuinely fresh price/news ingestion to see whether QQQ can recover under current-market conditions.
- If fresh data is staged and QQQ still looks weak, the problem is no longer evidence plumbing; it becomes a real signal problem.

## C. Code Changes

### `/Users/quinnesser/IdeaProjects/LightningTrader4000/config.py`

- Added `CALIBRATION_EVIDENCE_MAX_AGE_DAYS` at line `143`.
- Why it matters:
  - separates deployment-evidence freshness from the older 30-day calibration file-aging policy
  - lets the system say “this evidence is operationally stale” without pretending the calibration file itself is missing

### `/Users/quinnesser/IdeaProjects/LightningTrader4000/calibration_tracker.py`

- Added richer calibration state persistence:
  - `last_check`
  - `last_resolved_target_ts`
  - `pending_unresolved_rows`
  - oldest/newest unresolved timestamps
  - `skip_breakdown`
  - recovery/failure streaks
  - drift components
  - evidence age and evidence stale
- Added `_finalize_state()` around line `201`.
- `update_live_calibration()` now:
  - distinguishes unresolved rows from invalid/flat skips
  - writes `calibration_history_<SYMBOL>.jsonl`
  - returns `unresolved` count
  - tracks whether there is any recently resolved calibration evidence at all
- Why it matters:
  - deployment readiness now has auditable calibration evidence instead of a single coarse live state blob

### `/Users/quinnesser/IdeaProjects/LightningTrader4000/system_checkup.py`

- Now imports and uses `CALIBRATION_EVIDENCE_MAX_AGE_DAYS`.
- Calibration section now exposes:
  - `last_check`
  - `last_processed`
  - `last_resolved_target_ts`
  - `evidence_age_days`
  - `evidence_stale`
  - `pending_unresolved_rows`
  - recovery streaks
  - drift components
- Why it matters:
  - quick system check now reflects operational calibration evidence quality, not just file presence

### `/Users/quinnesser/IdeaProjects/LightningTrader4000/readiness_tracker.py`

- Added richer observation summary, stage-history summary, freshness summary, and calibration summary.
- `_calibration_summary()` at line `403` now recomputes stale evidence from age and forces stale when the recomputed age exceeds the new threshold.
- `summarize_deployment_readiness()` at line `1811` now:
  - surfaces calibration evidence stale as a first-class freshness blocker
  - distinguishes clean vs blocked shadow observations
  - emits next requirements for unresolved calibration rows and fresh resolved predictions
- `_persist_deployment_evidence()` at line `746` continues to append stage history and now persists the richer evidence model.
- Why it matters:
  - deployment stage is now backed by concrete freshness/calibration/shadow evidence rather than only status bits

### `/Users/quinnesser/IdeaProjects/LightningTrader4000/autonomy_engine.py`

- Added `_shadow_observation_event()` at line `649`.
- `evaluate_autonomy_cycle()` now records `shadow_observation` events on shadow cycles, including blocked cycles.
- Why it matters:
  - the system can now tell the difference between:
    - no shadow evidence
    - blocked shadow evidence
    - clean shadow evidence

### `/Users/quinnesser/IdeaProjects/LightningTrader4000/tests/test_calibration_integrity.py`

- Added assertions for:
  - unresolved calibration rows
  - `pending_unresolved_rows`
  - `skip_breakdown["unresolved"]`
  - calibration history file creation
  - `last_resolved_target_ts`
  - `evidence_stale` behavior

### `/Users/quinnesser/IdeaProjects/LightningTrader4000/tests/test_readiness_tracker.py`

- Added coverage for:
  - honest `supported_not_trained` classification
  - shadow observation summary behavior
  - clean shadow observation checklist satisfaction
  - recomputing stale calibration evidence from age even when stored stale bits are outdated

### `/Users/quinnesser/IdeaProjects/LightningTrader4000/tests/test_autonomy_engine.py`

- Added/updated coverage so blocked shadow cycles produce `shadow_observation` audit evidence.

## D. Evidence Collection Changes

### New or improved evidence artifacts

- `data/calibration_history_<SYMBOL>.jsonl`
  - append-only calibration evidence history
  - includes drift flags, evidence age, unresolved rows, and recent recovery signals
- `data/autonomy_audit_<SYMBOL>.jsonl`
  - now carries `shadow_observation` entries
  - for QQQ this now contains a blocked shadow observation rather than remaining empty
- `data/deployment_evidence_<SYMBOL>.json`
  - now includes richer freshness, calibration, observation, and next-requirement detail
- `data/deployment_stage_history_<SYMBOL>.jsonl`
  - keeps the stage timeline and checklist snapshots

### How stage advancement is now supported more concretely

- A symbol can no longer rely on vague “looks okay” status.
- The evidence now explicitly answers:
  - whether calibration evidence is current
  - whether unresolved predictions still need realized targets
  - whether shadow observations exist
  - whether any shadow observations were clean
  - what evidence is still missing before a stage can advance

### What still remains weak

- Shadow evidence is still observational, not yet statistically rich.
- There is only one QQQ shadow observation so far, and it is blocked.
- There is still no meaningful paper-autonomy evidence in this pass.
- The biggest missing input is still fresh real market/news data, not code structure.

## E. Validation Details

### What was tested

- Re-ran live calibration update for `QQQ`.
- Re-ran quick system-check inspection for `QQQ` and `AAPL`.
- Re-ran lightweight deployment summaries for `QQQ` and `AAPL`.
- Persisted refreshed deployment evidence for `QQQ` and `AAPL`.
- Verified current persisted evidence/stage-history outputs directly from disk.
- Executed a real blocked `QQQ` shadow cycle earlier in this pass and confirmed a `shadow_observation` audit row was written.

### Automated verification

- `python3 -m py_compile config.py calibration_tracker.py system_checkup.py readiness_tracker.py deployment_gate.py autonomy_engine.py tests/test_readiness_tracker.py`
  - passed
- `python3 -m unittest discover -s tests -p 'test_calibration_integrity.py'`
  - `Ran 6 tests`
  - `OK`
- `python3 -m unittest discover -s tests -p 'test_readiness_tracker.py'`
  - `Ran 10 tests`
  - `OK`
- `python3 -m unittest discover -s tests -p 'test_autonomy_engine.py'`
  - `Ran 10 tests`
  - `OK`
- `python3 -m unittest discover -s tests -p 'test_*.py'`
  - `Ran 159 tests in 6.058s`
  - `OK`

### What still needs soak / live observation

- Fresh market-data recovery for QQQ
- Fresh news refresh for QQQ
- Repeated shadow observations under market-open, non-stale conditions
- Multi-session observation to see whether calibration recovery can actually occur once data starts resolving again
- Real paper-autonomy evidence is still absent

## F. Updated Deployment Judgment

### Is the system ready now?

- No, not for real-world autonomous deployment.
- It remains only suitable for limited advisory deployment.

### Which symbol is closest?

- `QQQ` is still closest only because it already has:
  - actual price history
  - actual models
  - actual readiness history
  - actual calibration state
  - at least one shadow observation entry
- But “closest” should not be confused with “near deployment.”

### What still blocks real advancement?

- The strongest blocker remains QQQ freshness plus weak signal evidence:
  - critical stale market data
  - stale news
  - stale calibration evidence
  - active calibration drift
  - weak readiness metrics
  - no clean shadow evidence
- `AAPL` is not even in the same blocker category yet; it is still onboarding-blocked.

## G. Next-Pass Advice For ChatGPT

### Inspect first next time

- `data/deployment_evidence_QQQ.json`
- `data/deployment_stage_history_QQQ.jsonl`
- `data/calibration_live_QQQ.json`
- `data/calibration_history_QQQ.jsonl`
- `data/autonomy_audit_QQQ.jsonl`
- `data/deployment_evidence_AAPL.json`
- quick system-check output for `QQQ`

### Do not waste time revisiting

- Do not reopen multi-stock platform architecture.
- Do not spend time on new GUI growth.
- Do not revisit options autonomy.
- Do not weaken blocker thresholds just to make stage labels look better.

### What the next smaller pass should target

- Get fresh QQQ price data staged all the way through the latest eligible bars.
- Refresh QQQ news so `headline_state` is not stale.
- Re-run live calibration after fresh price realization so unresolved rows can resolve.
- Inspect whether fresh evidence changes:
  - `critical_stale`
  - `readiness_locked`
  - `calibration_drift`
  - `calibration_evidence_stale`
  - shadow observation quality
- Run additional shadow cycles only under eligible, fresh, non-degenerate conditions.
- If QQQ still fails after fresh data and fresh resolved calibration evidence, stop treating this as an evidence-plumbing issue and treat it as a genuine signal/readiness problem.
