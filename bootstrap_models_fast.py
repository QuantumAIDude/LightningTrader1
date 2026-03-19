#!/usr/bin/env python3
"""
bootstrap_models_fast.py
Quickly train lightweight, instrument-scoped model payloads for all modes.

This is a pragmatic fallback when full ensemble retraining is too slow.
"""

import argparse
import os
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, ExtraTreesRegressor

from config import DATA_DIR, DEFAULT_INSTRUMENT, MODEL_DIR
from features import (
    build_features_for_training,
    build_next_session_dataset,
    DEFAULT_FEATURES,
)
from model_trainers import _ridge_pipeline, _model_path
from regime_model import train_regime


def _load_prices(instrument_key: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"prices_{instrument_key}.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    return df.sort_index()


def _fit_model(X: pd.DataFrame, y: pd.Series, kind: str):
    if kind == "base":
        model = _ridge_pipeline()
    elif kind == "forest":
        model = RandomForestRegressor(
            n_estimators=20,
            max_depth=6,
            min_samples_leaf=8,
            random_state=42,
        )
    elif kind == "boost":
        model = HistGradientBoostingRegressor(
            max_depth=3,
            max_iter=120,
            random_state=42,
        )
    else:
        model = ExtraTreesRegressor(
            n_estimators=20,
            max_depth=8,
            min_samples_leaf=10,
            random_state=42,
        )
    model.fit(X, y)
    pred = model.predict(X)
    resid = float(np.std(y - pred)) if len(y) > 1 else 1e-6
    bias = float(np.mean(pred - y)) if len(y) > 1 else 0.0
    return model, max(resid, 1e-6), bias


def _mode_config(instrument_key: str):
    return {
        "4min": {
            "builder": lambda df: build_features_for_training(df, predict_minutes=4, instrument_key=instrument_key),
            "target": "target_ret_norm",
        },
        "daily_high": {
            "builder": lambda df: build_next_session_dataset(df, instrument_key=instrument_key),
            "target": "target_high_ret_norm",
        },
        "daily_low": {
            "builder": lambda df: build_next_session_dataset(df, instrument_key=instrument_key),
            "target": "target_low_ret_norm",
        },
        "weekend": {
            "builder": lambda df: build_features_for_training(
                df,
                predict_minutes=4,
                instrument_key=instrument_key,
                weekend_horizon=True,
            ),
            "target": "target_weekend_ret_norm",
        },
    }


def bootstrap_fast_models(instrument_key: str):
    raw = _load_prices(instrument_key)
    if raw is None or raw.empty:
        raise RuntimeError(f"No cached price data for {instrument_key}")

    forest_feats = DEFAULT_FEATURES + ["ret_2", "ret_4", "mom_3"]
    boost_feats = DEFAULT_FEATURES + ["ret_2", "ret_4"]
    extra_feats = DEFAULT_FEATURES + [
        "ret_2", "ret_4", "mom_3", "ema_5", "ema_20", "trend_strength", "vol_rolling_skew", "vol_rolling_kurt"
    ]
    version = "fastfallback-" + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    os.makedirs(MODEL_DIR, exist_ok=True)

    for mode, cfg in _mode_config(instrument_key).items():
        df = cfg["builder"](raw)
        if df is None or df.empty:
            continue
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        if len(df) < 20:
            continue
        y = df[cfg["target"]].astype(float)
        y = y - float(np.mean(y))

        payloads = {}
        for key, feats in [
            ("base", DEFAULT_FEATURES),
            ("forest", forest_feats),
            ("boost", boost_feats),
            ("extra_trees", extra_feats),
        ]:
            X = df[feats].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            model, resid, bias = _fit_model(X, y, key)
            meta = {
                "resid_std": resid,
                "pred_bias": bias,
                "version": version,
                "target_norm": True,
                "demeaned": True,
                "target_mean": float(np.mean(y)),
                "mode": mode,
            }
            if mode == "weekend":
                meta["weekend_horizon"] = "per_session"
            if mode in ("daily_high", "daily_low"):
                meta["daily_horizon"] = "per_session"
            payload = {
                "model": model if key == "base" else None,
                "fallback_model": model,
                "features": feats,
                "ensemble_models": {},
                "regime_models": {},
                "horizon_models": {},
                "performance": {},
                "performance_mae": {},
                "calibration": {},
                "calibration_sig": {"Weak": 0.5, "Medium": 0.5, "Strong": 0.5},
                "meta": meta,
            }
            payloads[key] = payload

        for key, payload in payloads.items():
            joblib.dump(payload, _model_path(key, mode, instrument_key))

        if mode == "4min":
            try:
                regime_payload = train_regime(df)
                joblib.dump(regime_payload, _model_path("regime", "4min", instrument_key))
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description="Fast fallback model bootstrapper")
    parser.add_argument("--instrument", default=DEFAULT_INSTRUMENT, help="Instrument key")
    args = parser.parse_args()
    bootstrap_fast_models(args.instrument)
    print(f"Fast fallback models built for {args.instrument}")


if __name__ == "__main__":
    main()
