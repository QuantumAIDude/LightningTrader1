#!/usr/bin/env python3
"""
feature_importance_report.py

Lightweight, model-agnostic permutation importance report on 4min targets.
Useful for validating whether news features add signal for QQQ.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from config import DATA_DIR, INSTRUMENTS
from features import build_features_for_training, DEFAULT_FEATURES
from model_utils import load_models


def _load_prices(instrument_key: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"prices_{instrument_key}.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    return df.sort_index()


def _pick_model(models: dict):
    # Use base if available (deterministic, cheap), else fallback to forest.
    for key in ("base", "forest", "boost", "extra_trees"):
        payload = models.get(key)
        if not isinstance(payload, dict):
            continue
        model = payload.get("model") or payload.get("fallback_model")
        feats = payload.get("features")
        if model is not None and feats:
            return key, model, list(feats)
    return None, None, []


def _perm_importance(model, X_test: pd.DataFrame, y_test: pd.Series, max_features: int = 35):
    if X_test.empty or len(y_test) == 0:
        return {}
    baseline = float(np.mean(np.abs(np.asarray(model.predict(X_test), dtype=float) - np.asarray(y_test, dtype=float))))
    cols = list(X_test.columns)
    # prioritize news columns and then first features for runtime control
    news_cols = [c for c in cols if c.startswith("news_")]
    other_cols = [c for c in cols if c not in news_cols]
    keep = news_cols + other_cols[: max(0, max_features - len(news_cols))]
    out = {}
    for c in keep:
        Xp = X_test.copy()
        Xp[c] = np.random.permutation(Xp[c].values)
        mae = float(np.mean(np.abs(np.asarray(model.predict(Xp), dtype=float) - np.asarray(y_test, dtype=float))))
        out[c] = float(mae - baseline)
    return dict(sorted(out.items(), key=lambda kv: kv[1], reverse=True))


def build_feature_importance_report(instrument_key: str) -> dict:
    df = _load_prices(instrument_key)
    if df is None or df.empty:
        return {"instrument": instrument_key, "error": "no data"}
    feat_df = build_features_for_training(df, predict_minutes=4, instrument_key=instrument_key)
    if feat_df is None or feat_df.empty:
        return {"instrument": instrument_key, "error": "feature build failed"}
    feat_df = feat_df.replace([np.inf, -np.inf], np.nan).dropna()
    if len(feat_df) < 400:
        return {"instrument": instrument_key, "error": "not enough rows"}
    models = load_models(instrument_key=instrument_key)
    model_key, model, feats = _pick_model(models)
    if model is None or not feats:
        return {"instrument": instrument_key, "error": "no model available"}
    feats = [f for f in feats if f in feat_df.columns]
    if not feats:
        feats = [f for f in DEFAULT_FEATURES if f in feat_df.columns]
    if not feats:
        return {"instrument": instrument_key, "error": "no usable features"}
    split = int(len(feat_df) * 0.8)
    if split < 100 or (len(feat_df) - split) < 100:
        return {"instrument": instrument_key, "error": "insufficient holdout rows"}
    X_test = feat_df.iloc[split:][feats].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_test = pd.to_numeric(feat_df.iloc[split:]["target_ret_norm"], errors="coerce").fillna(0.0)
    importances = _perm_importance(model, X_test, y_test, max_features=40)
    news = {k: float(v) for k, v in importances.items() if k.startswith("news_")}
    news_activity = {}
    for col in [c for c in feats if c.startswith("news_")]:
        try:
            vals = pd.to_numeric(feat_df[col], errors="coerce").fillna(0.0).values.astype(float)
            news_activity[col] = {
                "nonzero_frac": float(np.mean(np.abs(vals) > 1e-12)),
                "std": float(np.std(vals)),
            }
        except Exception:
            news_activity[col] = {"nonzero_frac": 0.0, "std": 0.0}
    out = {
        "instrument": instrument_key,
        "ts": datetime.now(timezone.utc).isoformat(),
        "model_key": model_key,
        "n_rows": int(len(feat_df)),
        "n_holdout": int(len(X_test)),
        "news_signal_ready": bool(
            any(
                (v.get("nonzero_frac", 0.0) > 0.01) and (v.get("std", 0.0) > 1e-6)
                for v in news_activity.values()
            )
        ),
        "news_activity": news_activity,
        "top_features": list(importances.items())[:20],
        "news_features": news,
    }
    path = os.path.join(DATA_DIR, f"feature_importance_{instrument_key}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    return out


def main():
    parser = argparse.ArgumentParser(description="Generate 4min feature importance report")
    parser.add_argument("--instrument", default="QQQ", help="Ticker or 'all'")
    args = parser.parse_args()
    raw = str(args.instrument).strip()
    if not raw or raw.lower() == "all":
        symbols = sorted(INSTRUMENTS.keys())
    else:
        symbols = [s.strip().upper() for s in raw.split(",") if s.strip()]
    report = {"instruments": {}}
    for sym in symbols:
        report["instruments"][sym] = build_feature_importance_report(sym)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
