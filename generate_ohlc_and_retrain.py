#!/usr/bin/env python3
# generate_ohlc_and_retrain.py
#
# End-to-end utility:
# 1) Acquire OHLC data (from provider or input CSV)
# 2) Persist to data/prices_<instrument>.csv
# 3) Retrain all models
# 4) Run walk-forward validation and operational checks
#
# NOTE: This script will fail fast if real OHLC is unavailable.

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from config import DEFAULT_INSTRUMENT, DATA_DIR, MODEL_DIR, INSTRUMENTS
from data_fetch import (
    fetch_recent_intraday,
    fetch_recent_intraday_alpaca,
    fetch_recent_intraday_yfinance,
)
from features import build_features_for_training, DEFAULT_FEATURES
from instrument_onboarding import ensure_runtime_instruments, register_instrument
from model_trainers import train_all_models, walk_forward_validate, _ridge_pipeline
from model_utils import load_models, predict_4min, predict_horizon
from regime_model import extract_regime_pipeline


def _coerce_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # flatten MultiIndex columns if present
    try:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
    except Exception:
        pass
    df.columns = [c.lower() for c in df.columns]
    # drop duplicate column names to avoid df["close"] returning a DataFrame
    try:
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated()].copy()
    except Exception:
        pass
    # basic column normalization
    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            except Exception:
                pass
    return df


def _ohlc_coverage(df: pd.DataFrame) -> float:
    if df is None or df.empty:
        return 0.0
    cols = ["open", "high", "low", "close"]
    if not all(c in df.columns for c in cols):
        return 0.0
    return float(df[cols].notna().all(axis=1).mean())


def _fetch_ohlc(instr_key: str, provider: str) -> pd.DataFrame:
    instr = INSTRUMENTS.get(instr_key, {})
    provider = (provider or "auto").lower().strip()
    if provider == "auto":
        return fetch_recent_intraday(instr_key)
    if provider == "alpaca":
        sym = instr.get("alpaca", instr_key)
        return fetch_recent_intraday_alpaca(sym)
    if provider == "yfinance":
        sym = instr.get("yahoo", instr_key)
        return fetch_recent_intraday_yfinance(sym, tz=instr.get("tz", "America/New_York"))
    return pd.DataFrame()


def _ridge_wf_eval(df_full: pd.DataFrame, target_col: str, n_splits: int = 5):
    if df_full is None or df_full.empty or target_col not in df_full.columns:
        return None
    y_raw = df_full[target_col].astype(float)
    target_mean = float(np.nanmean(y_raw)) if len(y_raw) else 0.0
    y = y_raw - target_mean
    X = df_full[DEFAULT_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    n = len(X)
    if n < (n_splits + 1) * 50:
        return None
    test_size = max(1, n // (n_splits + 1))
    maes = []
    baselines = []
    dir_accs = []
    for i in range(1, n_splits + 1):
        train_end = test_size * i
        test_start = train_end
        test_end = min(n, test_start + test_size)
        if train_end < 2 or test_start >= n:
            continue
        X_train, X_test = X.iloc[:train_end], X.iloc[test_start:test_end]
        y_train, y_test = y.iloc[:train_end], y.iloc[test_start:test_end]
        model = _ridge_pipeline()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae = float(np.mean(np.abs(preds - y_test)))
        baseline = float(np.mean(np.abs(0.0 - y_test)))
        dir_acc = float(np.mean((preds > 0) == (y_test > 0)))
        maes.append(mae)
        baselines.append(baseline)
        dir_accs.append(dir_acc)
    return {
        "folds": len(maes),
        "mae": maes,
        "baseline_mae": baselines,
        "dir_acc": dir_accs
    }


def _check_predictions(models, df, instr_key):
    out = {"4min": {}, "daily_high": {}, "daily_low": {}, "weekend": {}}
    pred_4 = predict_4min(models, df, instrument_key=instr_key)
    out["4min"] = pred_4
    pred_d = predict_horizon(models, df, instrument_key=instr_key, mode="DailyHigh")
    out["daily_high"] = pred_d
    pred_l = predict_horizon(models, df, instrument_key=instr_key, mode="DailyLow")
    out["daily_low"] = pred_l
    pred_w = predict_horizon(models, df, instrument_key=instr_key, mode="WeekEnd")
    out["weekend"] = pred_w
    return out


def main():
    parser = argparse.ArgumentParser(description="Generate OHLC CSV, retrain models, and validate.")
    parser.add_argument("--instrument", default=DEFAULT_INSTRUMENT,
                        help="Comma-separated list of instruments (e.g., QQQ,AAPL)")
    parser.add_argument("--provider", default="auto",
                        help="Comma-separated list or single value: auto|alpaca|yfinance")
    parser.add_argument("--input-csv", default=None,
                        help="Comma-separated list of local OHLC CSVs instead of fetching.")
    parser.add_argument("--min-ohlc-coverage", type=float, default=0.90)
    parser.add_argument("--max-runtime-sec", type=float, default=None)
    parser.add_argument("--wf-splits", type=int, default=5)
    parser.add_argument("--report", default=None, help="Optional JSON report output.")
    args = parser.parse_args()
    try:
        ensure_runtime_instruments(force=False, auto_onboard=True)
    except Exception:
        pass

    instr_list = [s.strip() for s in str(args.instrument).split(",") if s.strip()]
    if not instr_list:
        print("No instruments provided.", file=sys.stderr)
        sys.exit(2)

    provider_list = [s.strip() for s in str(args.provider).split(",") if s.strip()]
    if len(provider_list) == 1:
        provider_list = provider_list * len(instr_list)
    if len(provider_list) != len(instr_list):
        print("Provider list must match instruments or be a single value.", file=sys.stderr)
        sys.exit(2)

    input_list = []
    if args.input_csv:
        input_list = [s.strip() for s in str(args.input_csv).split(",") if s.strip()]
        if len(input_list) == 1:
            input_list = input_list * len(instr_list)
        if len(input_list) != len(instr_list):
            print("Input CSV list must match instruments or be a single value.", file=sys.stderr)
            sys.exit(2)

    os.makedirs(DATA_DIR, exist_ok=True)
    report = {"ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "instruments": {}}

    for idx, instr in enumerate(instr_list):
        if instr not in INSTRUMENTS:
            try:
                register_instrument(instr, persist=True)
            except Exception:
                pass
        report["instruments"][instr] = {}
        provider = provider_list[idx]
        input_csv = input_list[idx] if input_list else None

        # Load or fetch OHLC data
        if input_csv:
            if not os.path.exists(input_csv):
                report["instruments"][instr]["error"] = f"Input CSV not found: {input_csv}"
                continue
            df = pd.read_csv(input_csv, index_col=0, parse_dates=True)
        else:
            df = _fetch_ohlc(instr, provider)

        if df is None or df.empty:
            report["instruments"][instr]["error"] = "No data returned from provider."
            continue

        df = _coerce_ohlc(df)
        if df.index.tz is None:
            df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        coverage = _ohlc_coverage(df)
        report["instruments"][instr]["ohlc_coverage"] = coverage
        if coverage < args.min_ohlc_coverage:
            report["instruments"][instr]["error"] = f"OHLC coverage too low ({coverage:.2f})."
            continue

        # Persist to standard data file + provider-specific file
        out_path = os.path.join(DATA_DIR, f"prices_{instr}.csv")
        df_sorted = df.sort_index()
        df_sorted.to_csv(out_path)
        report["instruments"][instr]["data_path"] = out_path
        try:
            provider_tag = provider if provider else "auto"
            out_path_provider = os.path.join(DATA_DIR, f"prices_{instr}_{provider_tag}.csv")
            df_sorted.to_csv(out_path_provider)
            report["instruments"][instr]["data_path_provider"] = out_path_provider
        except Exception:
            pass
        report["instruments"][instr]["rows"] = int(len(df))

        # Retrain models
        train_all_models(df, instrument_key=instr, max_runtime_sec=args.max_runtime_sec)

        # Validate artifacts
        model_files = {}
        safe_tag = str(instr).replace("/", "_").replace("\\", "_")
        filenames = [
            f"base_4min_{safe_tag}.pkl", f"forest_4min_{safe_tag}.pkl", f"boost_4min_{safe_tag}.pkl", f"extra_trees_4min_{safe_tag}.pkl",
            f"base_daily_high_{safe_tag}.pkl", f"forest_daily_high_{safe_tag}.pkl", f"boost_daily_high_{safe_tag}.pkl", f"extra_trees_daily_high_{safe_tag}.pkl",
            f"base_daily_low_{safe_tag}.pkl", f"forest_daily_low_{safe_tag}.pkl", f"boost_daily_low_{safe_tag}.pkl", f"extra_trees_daily_low_{safe_tag}.pkl",
            f"base_weekend_{safe_tag}.pkl", f"forest_weekend_{safe_tag}.pkl", f"boost_weekend_{safe_tag}.pkl", f"extra_trees_weekend_{safe_tag}.pkl",
            f"regime_4min_{safe_tag}.pkl", f"meta_4min_{safe_tag}.pkl",
        ]
        # include legacy filenames for default instrument compatibility
        filenames += [
            "base_model.pkl", "forest_model.pkl", "boost_model.pkl", "extra_trees_model.pkl", "regime_model.pkl",
            "base_daily_high.pkl", "forest_daily_high.pkl", "boost_daily_high.pkl", "extra_trees_daily_high.pkl",
            "base_daily_low.pkl", "forest_daily_low.pkl", "boost_daily_low.pkl", "extra_trees_daily_low.pkl",
            "base_weekend.pkl", "forest_weekend.pkl", "boost_weekend.pkl", "extra_trees_weekend.pkl",
        ]
        for fname in filenames:
            model_files[fname] = os.path.exists(os.path.join(MODEL_DIR, fname))
        report["instruments"][instr]["model_files"] = model_files

        models = load_models(instrument_key=instr)
        scaler, gmm = extract_regime_pipeline(models.get("regime")) if models.get("regime") else (None, None)
        report["instruments"][instr]["regime_model_valid"] = bool(gmm is not None)

        # Walk-forward validation
        try:
            wf4 = walk_forward_validate(df, n_splits=max(3, args.wf_splits), instrument_key=instr)
            report["instruments"][instr]["walk_forward_4min"] = {
                "folds": wf4.get("folds"),
                "mae_base": wf4.get("mae_base"),
                "mae_forest": wf4.get("mae_forest"),
                "mae_boost": wf4.get("mae_boost"),
                "mae_extra_trees": wf4.get("mae_extra_trees"),
                "net_returns": wf4.get("net_returns"),
            }
        except Exception as e:
            report["instruments"][instr]["walk_forward_4min"] = {"error": str(e)}

        # Daily/weekend walk-forward (base ridge only)
        mode_wf = {}
        for mode, params in {
            "daily_high": {"daily": True, "weekend": False, "target": "target_high_ret_norm"},
            "daily_low": {"daily": True, "weekend": False, "target": "target_low_ret_norm"},
            "weekend": {"daily": False, "weekend": True, "target": "target_weekend_ret_norm"},
        }.items():
            try:
                df_full = build_features_for_training(
                    df,
                    predict_minutes=4,
                    instrument_key=instr,
                    daily_horizon=params["daily"],
                    weekend_horizon=params["weekend"],
                )
                mode_wf[mode] = _ridge_wf_eval(df_full, params["target"], n_splits=max(3, args.wf_splits))
            except Exception as e:
                mode_wf[mode] = {"error": str(e)}
        report["instruments"][instr]["walk_forward_horizon"] = mode_wf

        # Prediction sanity checks
        try:
            pred_checks = _check_predictions(models, df.tail(2000), instr)
            report["instruments"][instr]["prediction_checks"] = pred_checks
        except Exception as e:
            report["instruments"][instr]["prediction_checks"] = {"error": str(e)}

    # Write report
    if args.report:
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
