#!/usr/bin/env python3
# system_checkup.py
# Lightweight production-readiness diagnostic for LightningTrader4000.

import argparse
import json
import time
import os
from datetime import datetime, timezone

import pandas as pd

from config import INSTRUMENTS, DEFAULT_INSTRUMENT, DATA_DIR, CALIBRATION_MAX_AGE_DAYS
from data_fetch import update_price_data
from model_utils import load_models, predict_4min, predict_horizon, _staleness_info
from regime_model import extract_regime_pipeline


def _ts_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def _summarize_prediction_block(res_block: dict) -> dict:
    if not isinstance(res_block, dict):
        return {"error": "invalid prediction output"}
    if res_block.get("error"):
        return {"error": res_block.get("error")}
    # summarize per-model notes and basic outputs
    meta = res_block.get("_meta", {}) if isinstance(res_block, dict) else {}
    summary = {
        "models": {},
        "consensus": res_block.get("consensus"),
        "diagnostics": {
            "critical_stale": bool(meta.get("critical_stale", False)),
            "diversity": meta.get("diversity")
        }
    }
    for key in ["base", "forest", "boost", "extra_trees", "meta"]:
        v = res_block.get(key, {})
        if not isinstance(v, dict):
            continue
        summary["models"][key] = {
            "direction": v.get("direction"),
            "pred_price": v.get("pred_price"),
            "confidence": v.get("confidence"),
            "note": v.get("note"),
        }
    return summary


def _summarize_horizon_block(res_block: dict, mode: str) -> dict:
    if not isinstance(res_block, dict):
        return {"error": "invalid prediction output"}
    if res_block.get("error"):
        return {"error": res_block.get("error")}
    out = {"mode": mode}
    if mode.lower() in ("dailyhigh", "daily_high", "daily-low", "daily_low"):
        out["high"] = _summarize_prediction_block(res_block.get("high", {}))
        out["low"] = _summarize_prediction_block(res_block.get("low", {}))
    elif mode.lower().startswith("week"):
        out["week"] = _summarize_prediction_block(res_block.get("week", {}))
    else:
        out["raw"] = res_block
    return out


def _load_cached_prices(instr: str):
    path = f"{DATA_DIR}/prices_{instr}.csv"
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        df.columns = [c.lower() for c in df.columns]
        return df
    except Exception:
        return None

def _load_calibration_state(instr: str):
    path = os.path.join(DATA_DIR, f"calibration_live_{instr}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _check_instrument(instr: str, quick: bool = False, online: bool = False) -> dict:
    report = {"instrument": instr, "ts": _ts_now(), "status": "ok"}

    # Data health
    if online:
        df = update_price_data(instr)
    else:
        df = _load_cached_prices(instr)
    if df is None or df.empty:
        report["status"] = "fail"
        report["data"] = {"error": "No data returned", "online": online}
        return report

    try:
        last_ts = pd.to_datetime(df.index[-1], utc=True, errors="coerce")
        last_ts_str = last_ts.isoformat() if pd.notna(last_ts) else None
    except Exception:
        last_ts_str = None

    try:
        ohlc_cols = ["open", "high", "low", "close"]
        ohlc_coverage = float(df[ohlc_cols].notna().all(axis=1).mean()) if all(c in df.columns for c in ohlc_cols) else 0.0
    except Exception:
        ohlc_coverage = 0.0

    try:
        dupes = int(df.index.duplicated().sum())
    except Exception:
        dupes = None

    staleness = _staleness_info(df, instrument_key=instr)
    data_meta = {
        "rows": int(len(df)),
        "cols": list(df.columns),
        "last_ts": last_ts_str,
        "ohlc_coverage": round(float(ohlc_coverage), 4),
        "duplicates": dupes,
        "staleness": staleness,
        "data_source": getattr(df, "attrs", {}).get("data_source"),
        "data_error": getattr(df, "attrs", {}).get("data_error"),
        "online": online
    }
    # Futures contract sanity (helps diagnose stale data from expired contract config).
    try:
        instr_meta = INSTRUMENTS.get(instr, {})
        if str(instr_meta.get("type", "")).lower() == "futures":
            attrs = getattr(df, "attrs", {}) if hasattr(df, "attrs") else {}
            cfg_month = str(instr_meta.get("ibkr", {}).get("lastTradeDateOrContractMonth", "") or "")
            eff_month = str(attrs.get("contract_month", "") or cfg_month or "")
            data_meta["contract_month"] = eff_month or None
            data_meta["configured_contract_month"] = cfg_month or None
            data_meta["contract_auto_rolled"] = bool(attrs.get("contract_auto_rolled", False))
            if len(eff_month) >= 6 and eff_month[:6].isdigit():
                now_ym = datetime.now(timezone.utc).strftime("%Y%m")
                expired = eff_month[:6] < now_ym
                data_meta["contract_expired"] = bool(expired)
                if expired and report["status"] == "ok":
                    report["status"] = "warn"
            else:
                data_meta["contract_expired"] = None
    except Exception:
        pass
    report["data"] = data_meta
    # Staleness downgrade: warn or fail if severely stale
    try:
        if staleness.get("is_stale"):
            report["status"] = "warn"
            age = staleness.get("age_sec")
            threshold = staleness.get("threshold_sec") or 0
            market_open = bool(staleness.get("market_open", False))
            severe_mult = 10.0 if market_open else 2.0
            if age is not None and threshold and age > (threshold * severe_mult):
                report["status"] = "fail"
                report["data"]["stale_severity"] = "critical"
    except Exception:
        pass

    # Calibration health
    cal_state = _load_calibration_state(instr)
    if cal_state is None:
        report["calibration"] = {"status": "missing"}
        if report["status"] == "ok":
            report["status"] = "warn"
    else:
        try:
            last_update = pd.to_datetime(cal_state.get("last_update"), utc=True, errors="coerce")
        except Exception:
            last_update = None
        try:
            age_days = (datetime.now(timezone.utc) - last_update).total_seconds() / 86400.0 if last_update else None
        except Exception:
            age_days = None
        cal_stale = False
        try:
            if age_days is None or age_days > float(CALIBRATION_MAX_AGE_DAYS):
                cal_stale = True
        except Exception:
            cal_stale = True
        report["calibration"] = {
            "last_update": last_update.isoformat() if last_update is not None and pd.notna(last_update) else None,
            "age_days": round(float(age_days), 2) if age_days is not None else None,
            "drift_flag": bool(cal_state.get("drift_flag", False)),
            "stale": bool(cal_stale)
        }
        if cal_stale and report["status"] == "ok":
            report["status"] = "warn"

    # Model load
    models = load_models(include_modes=(not quick), instrument_key=instr)
    required = ["base", "forest", "boost", "extra_trees", "regime"]
    missing = [k for k in required if models.get(k) is None]
    report["models"] = {"missing": missing}
    if missing:
        report["status"] = "fail"
        return report

    # Regime pipeline validity
    try:
        scaler, gmm = extract_regime_pipeline(models.get("regime"))
        report["models"]["regime_valid"] = bool(scaler is not None and gmm is not None)
    except Exception:
        report["models"]["regime_valid"] = False

    # Predictions and latency
    report["predictions"] = {}
    t0 = time.perf_counter()
    res4 = predict_4min(models, df, instrument_key=instr)
    report["predictions"]["4min"] = _summarize_prediction_block(res4)
    report["predictions"]["4min_latency_s"] = round(time.perf_counter() - t0, 4)
    try:
        diag = report["predictions"]["4min"].get("diagnostics", {})
        critical_stale = bool(diag.get("critical_stale", False))
        if critical_stale:
            report["status"] = "fail"
        div = diag.get("diversity") if isinstance(diag, dict) else None
        if isinstance(div, dict) and bool(div.get("degenerate", False)) and not critical_stale:
            report["predictions"].setdefault("warnings", []).append("degenerate_model_spread")
            if report["status"] == "ok":
                report["status"] = "warn"
    except Exception:
        pass

    if not quick:
        t1 = time.perf_counter()
        resd = predict_horizon(models, df, instrument_key=instr, mode="DailyHigh")
        report["predictions"]["daily_high_low"] = _summarize_horizon_block(resd, "DailyHigh")
        report["predictions"]["daily_latency_s"] = round(time.perf_counter() - t1, 4)

        t2 = time.perf_counter()
        resw = predict_horizon(models, df, instrument_key=instr, mode="WeekEnd")
        report["predictions"]["weekend"] = _summarize_horizon_block(resw, "WeekEnd")
        report["predictions"]["weekend_latency_s"] = round(time.perf_counter() - t2, 4)

    # Heuristic status downgrade if regime invalid or prediction errors
    if not report["models"].get("regime_valid"):
        report["status"] = "warn"
    pred_errs = []
    for k, v in report["predictions"].items():
        if isinstance(v, dict) and v.get("error"):
            pred_errs.append(k)
    if pred_errs:
        report["status"] = "warn"
        report["predictions"]["errors"] = pred_errs

    return report


def main():
    parser = argparse.ArgumentParser(description="LightningTrader4000 system checkup")
    parser.add_argument("--instrument", default=DEFAULT_INSTRUMENT,
                        help="Comma-separated instruments (e.g., QQQ,GCG26) or 'all'")
    parser.add_argument("--quick", action="store_true",
                        help="Skip daily/weekend checks for faster run.")
    parser.add_argument("--online", action="store_true",
                        help="Use live fetch (may be slow); default uses cached prices.")
    parser.add_argument("--report", default=None,
                        help="Optional JSON output path")
    args = parser.parse_args()

    raw_instr = str(args.instrument).strip()
    if not raw_instr or raw_instr.lower() == "all":
        instr_list = list(INSTRUMENTS.keys())
    else:
        instr_list = [s.strip() for s in raw_instr.split(",") if s.strip()]
        if not instr_list:
            instr_list = [DEFAULT_INSTRUMENT]

    out = {
        "ts": _ts_now(),
        "instruments": {},
        "status": "ok"
    }

    for instr in instr_list:
        if instr not in INSTRUMENTS:
            out["instruments"][instr] = {"status": "fail", "error": "Instrument not defined"}
            out["status"] = "warn"
            continue
        rep = _check_instrument(instr, quick=args.quick, online=args.online)
        out["instruments"][instr] = rep
        if rep.get("status") == "fail":
            out["status"] = "fail"
        elif rep.get("status") == "warn" and out["status"] == "ok":
            out["status"] = "warn"

    if args.report:
        with open(args.report, "w") as f:
            json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))

    if out["status"] == "fail":
        raise SystemExit(2)
    if out["status"] == "warn":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
