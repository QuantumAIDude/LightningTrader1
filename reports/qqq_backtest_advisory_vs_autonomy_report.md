# QQQ Backtest Report: Advisory vs Autonomous Modes

## 1. Executive Summary

This replay still does not support a profitability claim. On the available local QQQ intraday cache, the advisory layer mostly stood aside, and the autonomous layer took zero trades.

The corrected historical-as-of freshness pass changed the evidence diagnosis, not the trading conclusion. The previous report carried today’s wall-clock stale-data guard into every historical advisory row. The regenerated replay now evaluates price freshness relative to each replay timestamp. All 4,304 replay records were fresh as-of their historical decision time.

Advisory still produced no executable buy/sell payload actions. It emitted 4,178 stand-aside decisions and 126 watchlist decisions. Autonomy still blocked all 4,304 controller cycles across shadow, paper, and live checks. The system remains advisory/filter-first and autonomy dormant.

## 2. Data Used

| Field | Value |
| --- | --- |
| Symbol | QQQ |
| Intraday source | local cache |
| Intraday path | `data/prices_QQQ.csv` |
| Daily context path | `data/prices_QQQ_daily_long.csv` |
| Date range | 2026-01-02 14:30 UTC to 2026-03-23 18:28 UTC |
| Bar interval | 2 minutes, inferred by modal timestamp difference |
| Intraday bars | 10,348 |
| Trading sessions | 55 |
| Missing close values | 0 |
| Duplicate timestamps | 0 |
| Internet/provider access | not used |

Local daily QQQ history was available through 2026-03-23. Historical macro and breadth context were not cached, so the replay used the existing neutral placeholder path for those inputs. Historical news cache logic was available through the existing replay harness, but the report should not be read as a complete macro/news event study.

No standalone `data/validation_evidence_QQQ.json` artifact exists in this repo state. Readiness/deployment evidence still leaves QQQ at `advisory_only`.

## 3. Methodology

The evaluation used the existing `autonomy_backtest.run_stock_autonomy_validation()` path rather than a new parallel backtester.

The replay used five anchored folds over the full local intraday cache. Each fold trained only on earlier sessions and evaluated later sessions. The final fold was treated as holdout by the existing harness. The replay generated 4,304 point-in-time advisory/controller records.

Cost assumptions came from `config.py` and the replay harness:

| Cost input | Value |
| --- | ---: |
| Slippage per side | 0.05% |
| Transaction cost per side | 0.03% |
| Round-trip one-bar diagnostic cost | 0.16% |

Autonomy simulation used the existing shadow controller replay path with bar-close decision price, adverse slippage, and transaction cost assumptions. It did not loosen permissions.

Lookahead controls:

- Fold models were trained on prior sessions only.
- Test sessions were later than train sessions.
- Daily context was truncated to the current replay date.
- Recent news was filtered by `as_of`.
- Next-period advisory diagnostics used the following replay record only.

Exclusions:

- No strategy parameters were tuned.
- No broker integration was exercised.
- No live/paper permissions were changed.
- Options autonomy was excluded because historical contract-level options data was unavailable.
- This is a replay/backtest foundation, not a production-grade purged walk-forward validation package.

## Historical As-Of Freshness Handling

What was wrong before: historical advisory replay loaded the current `data/readiness_state_QQQ.json` and fed its wall-clock stale prediction diagnostics into `summarize_deployment_readiness()`. Because today’s cache is old, every historical advisory row inherited `critical stale-data guard is active`, even when the model/advisory prefix was fresh relative to the replay timestamp.

What is now evaluated as-of: the replay now attaches a `historical_freshness` object to each record with `replay_as_of`, `bars_available_as_of`, `data_last_timestamp_as_of`, `freshness_is_as_of`, `fresh_as_of`, and stale reason/severity fields. For this run, all 4,304 records were fresh as-of their replay timestamp.

What remains current/static: validation evidence, calibration artifact state, model/runtime artifacts, readiness metrics, and deployment-stage evidence remain conservative static/current evidence. Missing validation is not faked. Calibration drift remains visible. Autonomy still consumes the existing readiness/deployment/validation gates.

Effect on counts:

| Metric | Previous report | Corrected as-of replay |
| --- | ---: | ---: |
| Advisory snapshots | 4,304 | 4,304 |
| stand_aside | 4,178 | 4,178 |
| watchlist | 126 | 126 |
| buy | 0 | 0 |
| sell | 0 | 0 |
| Autonomy cycles | 4,304 | 4,304 |
| Autonomy blocked cycles | 4,304 | 4,304 |
| Shadow/paper/live allowed actions | 0 | 0 |
| Historical records fresh as-of replay time | not reported | 4,304 |

The main changed diagnosis is blocker labeling: `critical stale-data guard is active` is no longer a replay-wide advisory blocker. The replay now surfaces `calibration drift flag is active` and missing validation controls as current/static evidence blockers.

## 4. Advisory Mode Results

### Advisory Decision Payload

| Payload action | Count | Share |
| --- | ---: | ---: |
| stand_aside | 4,178 | 97.1% |
| watchlist | 126 | 2.9% |
| buy | 0 | 0.0% |
| sell | 0 | 0.0% |

### Signal Translation

| Signal label/kind | Count |
| --- | ---: |
| Stand Aside | 4,200 |
| Watchlist Long | 97 |
| Watchlist Short | 4 |
| Lean Long | 3 |

| Economic state | Count |
| --- | ---: |
| weak_edge | 3,946 |
| news_conflicted | 254 |
| uncertainty_crushed | 73 |
| cost_crushed | 27 |
| finished_edge | 3 |
| tentative_edge | 1 |

Average finished edge was -4.79 bps. Average raw edge was -3.99 bps. Average advisory confidence was 21.93%.

### Realized Return Diagnostics

The actual advisory payload produced no buy/sell actions, so actionable advisory trade return, win rate, gross return, net return, and drawdown are null rather than positive or negative.

As a diagnostic only, the broader raw-direction/watchlist/lean proxy produced 4,299 one-period directional observations. That proxy is not an execution system. It had:

| Metric | Value |
| --- | ---: |
| Mean next-period gross return | +0.0015% |
| Mean next-period net return after 0.16% round-trip cost | -0.1585% |
| Directional accuracy | 48.69% |
| Win rate after costs | 6.16% |
| Sum of one-period gross returns | +6.66% |
| Sum of one-period net returns | -681.18% |

That diagnostic says the weak/raw directional lean is not economically tradable after costs. It is useful as a cautionary filter, not as a trade generator.

### Failure Modes

Common advisory blockers and evidence flags included:

- calibration drift flag active on 4,304 snapshots;
- validation blockers on every snapshot: `missing_holdout`, `no_strict_oos`, `no_cost_model`, and `drift_detected`;
- weak direction probability;
- predicted move near normal noise;
- raw model move too small;
- uncertainty overwhelming forecast edge;
- cost and uncertainty removing tradable edge;
- fresh/news catalyst conflict.

Two of the three internal `finished_edge` / `Lean Long` candidates lost money on the next replay step after costs. The worst one was 2026-02-04 21:00 UTC, with a +8.217 bps model edge but a -1.2977% net one-period outcome. That remains a strong reason not to treat advisory explanations as permission to trade.

## 5. Autonomous Mode Results

Autonomous mode was dormant and fail-closed.

| Metric | Value |
| --- | ---: |
| Controller cycles evaluated | 4,304 |
| Blocked cycles | 4,304 |
| Shadow entries | 0 |
| Paper/live orders | 0 |
| Allowed trades | 0 |
| Gross PnL | 0.00 |
| Net PnL after costs | 0.00 |
| Fees | 0.00 |
| Win rate | null |
| Max drawdown | 0.00% |

### Mode Eligibility

| Mode | Allowed | Blocked |
| --- | ---: | ---: |
| shadow | 0 | 4,304 |
| paper_auto | 0 | 4,304 |
| live_auto | 0 | 4,304 |

### Block Reasons

| Block reason | Count |
| --- | ---: |
| QQQ is not shadow-ready yet (advisory only) | 4,304 |
| validation evidence does not support shadow mode yet | 4,304 |
| intraday confidence is too low | 4,304 |
| model size signal is too small | 4,304 |
| cost-adjusted edge is too small | 4,302 |
| no directional bias is available | 4,300 |
| quality score is below the autonomy threshold | 4,297 |
| advisory stack marked this setup as no-trade | 4,296 |
| news pipeline is degraded | 4,197 |
| liquidity is too weak for autonomy | 1,281 |
| catalyst risk is elevated | 303 |
| fresh high-impact news conflicts with the model bias | 201 |
| regime is unknown or evidence is insufficient | 78 |

Paper and live modes were also blocked for deployment-stage and validation reasons. Paper mode had `validation evidence blocks paper autonomy: missing_holdout` on every evaluated record. Live mode also required a live Alpaca route and was blocked on `missing_holdout`.

The current autonomy gates look appropriately conservative for this evidence state. Historical freshness now avoids a false stale-data explanation, but it does not and should not override validation, readiness, risk, or deployment blockers.

## 6. Baselines

| Baseline | Result |
| --- | ---: |
| Buy and hold QQQ over same intraday cache | -5.03% |
| SMA 20/50 long-flat, same cost assumptions on position changes | -16.30% |
| SMA 20/50 max drawdown | -17.71% |
| SMA 20/50 position changes | 235 |

A random-entry baseline was not included. Because autonomy allowed zero trades and advisory issued zero executable buy/sell payload actions, matching trade frequency would either be zero or would benchmark a raw diagnostic that is explicitly not an execution policy.

## 7. Interpretation

Advisory mode is useful mainly as a conservative filter and explanation layer right now. It clearly identifies weak edge, news conflict, cost-crushed edge, uncertainty-crushed edge, calibration drift, and validation/evidence blockers. It did not produce executable buy/sell advice in this replay.

Autonomous mode is not doing anything economically meaningful yet. That is correct for this evidence state. Every shadow, paper, and live eligibility check stayed blocked.

The historical-as-of freshness fix improves truthfulness: the replay no longer blames historical decisions on today’s stale cache. But it does not produce a more aggressive result. The system still says: advisory filter only, no executable advisory action, autonomy dormant.

The model does not show a credible actionable edge after costs in this replay. Raw directional diagnostics are effectively untradable after a 16 bps round trip.

Before paper/live autonomy is credible, QQQ needs a real validation artifact, strict out-of-sample controls, explicit transaction-cost/slippage evidence, an untouched holdout or paper ledger, fresh calibration/drift evidence, and a replay path that can persist these reports without expensive ad hoc command invocations.

## 8. Limitations

- The intraday data window is local-only: 2026-01-02 through 2026-03-23.
- No internet or provider refresh was used.
- Historical macro and breadth context were neutral placeholders.
- Historical contract-level options data was unavailable.
- The report uses the existing replay harness, not a full production validation engine.
- The validation evidence layer is present, but QQQ has no standalone validation evidence artifact.
- The replay uses fold-local training, but it is not a complete purged/embargoed validation framework.
- Advisory one-period diagnostics overlap heavily and should not be read as portfolio returns.
- Calibration and validation evidence are still current/static artifacts, not reconstructed as-of each historical timestamp.
- The full 5-fold replay path is slow and needs a dedicated report runner/cache.

## 9. Recommended Next Steps

1. Generate a real `validation_evidence_QQQ.json` artifact from strict walk-forward validation with explicit cost/slippage fields.
2. Add an untouched holdout section and store holdout boundaries in validation evidence.
3. Add calibration evidence as-of handling, or explicitly freeze calibration artifacts used for each replay run.
4. Add purging/embargo controls where overlapping forecast labels can leak information.
5. Add a paper-trading evidence ledger before any paper/live autonomy promotion.
6. Turn this report flow into a repeatable command with progress logging and cached fold artifacts.

## 10. Machine-Readable Summary

```json
{
  "symbol": "QQQ",
  "date_range": "2026-01-02 14:30 UTC to 2026-03-23 18:28 UTC",
  "bar_interval": "2 minutes",
  "data_source": "data/prices_QQQ.csv",
  "historical_as_of_freshness": {
    "fresh_as_of_records": 4304,
    "stale_as_of_records": 0,
    "previous_replay_wide_blocker": "critical stale-data guard is active",
    "corrected_replay_wide_static_blocker": "calibration drift flag is active"
  },
  "advisory": {
    "snapshots": 4304,
    "payload_actions": {
      "stand_aside": 4178,
      "watchlist": 126,
      "buy": 0,
      "sell": 0
    },
    "actionable_signals": 0,
    "win_rate": null,
    "gross_return": null,
    "net_return_after_costs": null,
    "max_drawdown": null,
    "avg_edge_bps": -4.7864,
    "avg_raw_edge_bps": -3.9860,
    "avg_confidence": 21.9254,
    "diagnostic_directional_accuracy": 0.4869,
    "diagnostic_mean_net_return_after_costs": -0.001585,
    "top_blockers": [
      "calibration drift flag is active",
      "missing_holdout",
      "no_strict_oos",
      "no_cost_model",
      "drift_detected",
      "weak direction probability",
      "cost-adjusted edge is too small"
    ]
  },
  "autonomy": {
    "cycles": 4304,
    "blocked_cycles": 4304,
    "allowed_actions": 0,
    "shadow_allowed": 0,
    "paper_allowed": 0,
    "live_allowed": 0,
    "win_rate": null,
    "gross_return": 0.0,
    "net_return_after_costs": 0.0,
    "max_drawdown": 0.0,
    "validation_block_frequency": 4304,
    "readiness_block_frequency": 4304,
    "risk_block_frequency": 78,
    "top_blockers": [
      "advisory only deployment stage",
      "validation evidence does not support shadow mode",
      "missing_holdout",
      "intraday confidence is too low",
      "model size signal is too small",
      "cost-adjusted edge is too small"
    ]
  },
  "baseline": {
    "buy_and_hold_return": -0.050285,
    "sma_20_50_long_flat_return": -0.163032
  },
  "verdict": "Historical as-of freshness improved replay truthfulness but did not change the conclusion: advisory is a conservative filter, autonomy is dormant and fail-closed, and no profitability claim is supported."
}
```
