#!/usr/bin/env python3
# system_checkup.py
# Lightweight production-readiness diagnostic for LightningTrader4000.

import argparse
import json
import time
import os
from datetime import datetime, timezone

import pandas as pd

from config import (
    INSTRUMENTS,
    DEFAULT_INSTRUMENT,
    DATA_DIR,
    CALIBRATION_MAX_AGE_DAYS,
    CALIBRATION_EVIDENCE_MAX_AGE_DAYS,
    NEWS_PIPELINE_ENABLED,
    NEWS_PIPELINE_SYMBOL,
)
from data_fetch import update_price_data
from instrument_onboarding import ensure_runtime_instruments
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
            "diversity": meta.get("diversity"),
            "readiness_locked": bool(meta.get("readiness_locked", False)),
            "readiness_state": meta.get("readiness_state"),
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
            "pred_range_80_low": v.get("pred_range_80_low"),
            "pred_range_80_high": v.get("pred_range_80_high"),
            "expected_abs_err_bps": v.get("expected_abs_err_bps"),
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


def _load_horizon_accuracy(instr: str):
    path = os.path.join(DATA_DIR, f"horizon_accuracy_{instr}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _load_news_health(instr: str) -> dict:
    out = {
        "enabled": bool(NEWS_PIPELINE_ENABLED),
        "symbol": str(instr or "").upper(),
        "configured_symbol": str(NEWS_PIPELINE_SYMBOL).upper(),
        "status": "not_applicable",
    }
    if not NEWS_PIPELINE_ENABLED:
        out["status"] = "disabled"
        return out
    try:
        from news_pipeline import get_news_health
        health = get_news_health(str(instr or "").upper())
        if isinstance(health, dict):
            out.update(health)
            if "symbol" not in out:
                out["symbol"] = str(instr or "").upper()
            raw_status = str(out.get("status") or "").strip().lower()
            if raw_status in {"stale", "failed", "error"}:
                out["status_detail"] = raw_status
                out["status"] = "degraded"
    except Exception:
        pass
    path = os.path.join(DATA_DIR, f"news_{str(instr).upper()}.csv")
    if not os.path.exists(path):
        out.update({"status": "missing", "rows": 0})
        return out
    try:
        df = pd.read_csv(path)
    except Exception:
        out.update({"status": "degraded", "rows": 0})
        return out
    if df is None or df.empty:
        out.update({"status": "missing", "rows": 0})
        return out
    rows = int(len(df))
    df["published_at"] = pd.to_datetime(df.get("published_at"), utc=True, errors="coerce")
    df = df.dropna(subset=["published_at"]).copy()
    if df.empty:
        out.update({"status": "missing", "rows": rows, "valid_rows": 0})
        return out
    now_utc = pd.Timestamp.now(tz="UTC")
    last_ts = df["published_at"].max()
    age_hours = float((now_utc - last_ts).total_seconds() / 3600.0) if pd.notna(last_ts) else None
    freshness_limit = 96.0 if int(now_utc.dayofweek) >= 5 else 72.0
    day_count = int(df["published_at"].dt.floor("D").nunique())
    sent = pd.to_numeric(df.get("sentiment_score"), errors="coerce").fillna(0.0)
    sent_std = float(sent.std()) if len(sent) else 0.0
    sent_nonzero_frac = float((sent.abs() > 1e-12).mean()) if len(sent) else 0.0
    legacy_signal_ready = bool(
        (age_hours is not None and age_hours <= freshness_limit)
        and day_count >= 5
        and sent_nonzero_frac > 0.02
        and sent_std > 1e-6
    )
    out.update(
        {
            "rows": max(int(out.get("rows") or 0), rows),
            "valid_rows": int(len(df)),
            "last_published_at": out.get("last_published_at") or (last_ts.isoformat() if pd.notna(last_ts) else None),
            "age_hours": out.get("age_hours") if out.get("age_hours") is not None else (round(age_hours, 2) if age_hours is not None else None),
            "unique_days": max(int(out.get("unique_days") or 0), day_count),
            "sentiment_nonzero_frac": round(sent_nonzero_frac, 4),
            "sentiment_std": round(sent_std, 6),
            "signal_ready": bool(out.get("signal_ready") or legacy_signal_ready),
        }
    )
    if bool(out.get("signal_ready")):
        out["status"] = "ok"
    elif str(out.get("status") or "").strip().lower() not in {"missing", "disabled", "not_applicable"}:
        out["status"] = "degraded"
    return out


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
            last_check = pd.to_datetime(cal_state.get("last_check"), utc=True, errors="coerce")
        except Exception:
            last_check = None
        try:
            last_processed = pd.to_datetime(cal_state.get("last_processed"), utc=True, errors="coerce")
        except Exception:
            last_processed = None
        try:
            last_resolved_target = pd.to_datetime(cal_state.get("last_resolved_target_ts"), utc=True, errors="coerce")
        except Exception:
            last_resolved_target = None
        try:
            age_days = (datetime.now(timezone.utc) - last_update).total_seconds() / 86400.0 if last_update else None
        except Exception:
            age_days = None
        try:
            evidence_anchor = last_resolved_target if last_resolved_target is not None and pd.notna(last_resolved_target) else last_processed
            evidence_age_days = (datetime.now(timezone.utc) - evidence_anchor).total_seconds() / 86400.0 if evidence_anchor is not None and pd.notna(evidence_anchor) else None
        except Exception:
            evidence_age_days = None
        cal_stale = False
        try:
            if age_days is None or age_days > float(CALIBRATION_MAX_AGE_DAYS):
                cal_stale = True
        except Exception:
            cal_stale = True
        evidence_stale = bool(
            cal_state.get(
                "evidence_stale",
                evidence_age_days is None or evidence_age_days > float(CALIBRATION_EVIDENCE_MAX_AGE_DAYS),
            )
        )
        report["calibration"] = {
            "last_update": last_update.isoformat() if last_update is not None and pd.notna(last_update) else None,
            "last_check": last_check.isoformat() if last_check is not None and pd.notna(last_check) else None,
            "last_processed": last_processed.isoformat() if last_processed is not None and pd.notna(last_processed) else None,
            "last_resolved_target_ts": last_resolved_target.isoformat() if last_resolved_target is not None and pd.notna(last_resolved_target) else None,
            "age_days": round(float(age_days), 2) if age_days is not None else None,
            "evidence_age_days": round(float(evidence_age_days), 2) if evidence_age_days is not None else None,
            "evidence_stale": bool(evidence_stale),
            "drift_flag": bool(cal_state.get("drift_flag", False)),
            "recent_drift_flag": bool(cal_state.get("recent_drift_flag", False)),
            "recent_price_drift_flag": bool(cal_state.get("recent_price_drift_flag", False)),
            "drift_score": _safe_float(cal_state.get("drift_score")),
            "recent_acc": _safe_float(cal_state.get("recent_acc")),
            "recent_total": int(cal_state.get("recent_total", 0) or 0),
            "recent_price_window_hit_rate": _safe_float(cal_state.get("recent_price_window_hit_rate")),
            "recent_price_window_total": int(cal_state.get("recent_price_window_total", 0) or 0),
            "recent_recovery_streak": int(cal_state.get("recent_recovery_streak", 0) or 0),
            "recent_price_recovery_streak": int(cal_state.get("recent_price_recovery_streak", 0) or 0),
            "pending_unresolved_rows": int(cal_state.get("pending_unresolved_rows", 0) or 0),
            "pending_unresolved_newest_prediction_ts": cal_state.get("pending_unresolved_newest_prediction_ts"),
            "drift_components": cal_state.get("drift_components", {}),
            "stale": bool(cal_stale),
        }
        if cal_stale and report["status"] == "ok":
            report["status"] = "warn"
        if evidence_stale and report["status"] == "ok":
            report["status"] = "warn"

    horizon_acc = _load_horizon_accuracy(instr)
    if horizon_acc is None:
        report["horizon_accuracy"] = {"status": "missing"}
    else:
        drift_flag = bool(horizon_acc.get("global_drift_flag", False))
        report["horizon_accuracy"] = {
            "status": "ok",
            "drift_flag": drift_flag,
            "last_update": horizon_acc.get("last_update"),
            "newly_resolved_rows": horizon_acc.get("newly_resolved_rows"),
            "sample_summary": horizon_acc.get("sample_summary", {}),
            "by_horizon": horizon_acc.get("by_horizon", {}),
        }
        if drift_flag and report["status"] == "ok":
            report["status"] = "warn"

    news_health = _load_news_health(instr)
    report["news_pipeline"] = news_health
    if bool(news_health.get("enabled", NEWS_PIPELINE_ENABLED)):
        if news_health.get("status") in ("missing", "degraded", "error", "failed", "stale") and report["status"] == "ok":
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
    res4 = predict_4min(models, df, instrument_key=instr, log_side_effects=False)
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
        resd = predict_horizon(models, df, instrument_key=instr, mode="DailyHigh", log_side_effects=False)
        report["predictions"]["daily_high_low"] = _summarize_horizon_block(resd, "DailyHigh")
        report["predictions"]["daily_latency_s"] = round(time.perf_counter() - t1, 4)

        t2 = time.perf_counter()
        resw = predict_horizon(models, df, instrument_key=instr, mode="WeekEnd", log_side_effects=False)
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
    try:
        ensure_runtime_instruments(force=False, auto_onboard=True)
    except Exception:
        pass
    parser = argparse.ArgumentParser(description="LightningTrader4000 system checkup")
    parser.add_argument("--instrument", default=DEFAULT_INSTRUMENT,
                        help="Comma-separated instruments (e.g., QQQ,AAPL) or 'all'")
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
            out["status"] = "fail"
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
