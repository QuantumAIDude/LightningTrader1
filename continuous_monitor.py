#!/usr/bin/env python3
"""
continuous_monitor.py
Single-cycle operational monitor for LightningTrader4000.

Designed to be scheduled (cron/automation). Writes JSONL events and alert lines.
"""

import argparse
import json
import os
from datetime import datetime, timezone

from config import INSTRUMENTS, DATA_DIR
from instrument_onboarding import ensure_runtime_instruments
from system_checkup import _check_instrument


def _ts_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_instruments(raw: str) -> list[str]:
    val = str(raw or "").strip()
    if not val or val.lower() == "all":
        return list(INSTRUMENTS.keys())
    return [s.strip() for s in val.split(",") if s.strip()]


def _append_jsonl(path: str, payload: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def _append_alert(path: str, msg: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(f"{_ts_now()} {msg}\n")


def main():
    try:
        ensure_runtime_instruments(force=False, auto_onboard=True)
    except Exception:
        pass
    parser = argparse.ArgumentParser(description="Continuous monitor cycle for LightningTrader4000")
    parser.add_argument(
        "--instrument",
        default="all",
        help="Comma-separated instruments or 'all' (default).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick checks only.",
    )
    parser.add_argument(
        "--online",
        action="store_true",
        help="Use live fetch during checks.",
    )
    parser.add_argument(
        "--event-log",
        default=os.path.join(DATA_DIR, "monitoring_events.jsonl"),
        help="JSONL event log path.",
    )
    parser.add_argument(
        "--alert-log",
        default=os.path.join(DATA_DIR, "monitoring_alerts.log"),
        help="Alert log path.",
    )
    parser.add_argument(
        "--fail-on-warn",
        action="store_true",
        help="Exit non-zero on warnings as well as failures.",
    )
    args = parser.parse_args()

    instruments = _parse_instruments(args.instrument)
    run_ts = _ts_now()
    any_fail = False
    any_warn = False

    for instr in instruments:
        if instr not in INSTRUMENTS:
            any_fail = True
            _append_alert(args.alert_log, f"{instr} FAIL unknown instrument")
            _append_jsonl(args.event_log, {
                "ts": run_ts,
                "instrument": instr,
                "status": "fail",
                "error": "Instrument not defined",
            })
            continue

        rep = _check_instrument(instr, quick=args.quick, online=args.online)
        status = str(rep.get("status", "fail")).lower()
        drift_flag = bool(rep.get("calibration", {}).get("drift_flag", False))
        horizon_drift = bool(rep.get("horizon_accuracy", {}).get("drift_flag", False))
        cal_stale = bool(rep.get("calibration", {}).get("stale", True))
        stale = bool(rep.get("data", {}).get("staleness", {}).get("is_stale", True))
        market_state = rep.get("data", {}).get("staleness", {}).get("market_state")
        diag = rep.get("predictions", {}).get("4min", {}).get("diagnostics", {})
        critical_stale = bool(diag.get("critical_stale", False)) if isinstance(diag, dict) else False
        diversity = diag.get("diversity", {}) if isinstance(diag, dict) else {}
        degenerate = bool(diversity.get("degenerate", False)) if isinstance(diversity, dict) else False

        event = {
            "ts": run_ts,
            "instrument": instr,
            "status": status,
            "drift_flag": drift_flag,
            "horizon_drift_flag": horizon_drift,
            "calibration_stale": cal_stale,
            "data_stale": stale,
            "market_state": market_state,
            "critical_stale": critical_stale,
            "degenerate_spread": degenerate,
            "last_ts": rep.get("data", {}).get("last_ts"),
            "latency_4min_s": rep.get("predictions", {}).get("4min_latency_s"),
        }
        _append_jsonl(args.event_log, event)

        if status == "fail":
            any_fail = True
            _append_alert(args.alert_log, f"{instr} FAIL {json.dumps(event, sort_keys=True)}")
        elif status == "warn":
            any_warn = True
            _append_alert(args.alert_log, f"{instr} WARN {json.dumps(event, sort_keys=True)}")

        if drift_flag:
            _append_alert(args.alert_log, f"{instr} DRIFT_FLAG calibration drift detected")
        if horizon_drift:
            _append_alert(args.alert_log, f"{instr} HORIZON_DRIFT daily/weekend drift detected")
        if cal_stale:
            _append_alert(args.alert_log, f"{instr} CALIBRATION_STALE calibration stats are stale")
        if critical_stale:
            _append_alert(args.alert_log, f"{instr} CRITICAL_STALE no-trade guard triggered")
        if degenerate:
            _append_alert(args.alert_log, f"{instr} DEGENERATE_SPREAD near-identical model outputs")

    if any_fail:
        raise SystemExit(2)
    if any_warn and args.fail_on_warn:
        raise SystemExit(1)
    raise SystemExit(0)


if __name__ == "__main__":
    main()
