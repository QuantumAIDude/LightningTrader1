## model_trainers.py — extended trainer with multi-window ensembles, bootstrap, realistic walk-forward validation,
# regime-specific models, ensemble weighting and triple-barrier labeling improvements.

import os
import time
import joblib
import numpy as np
import math
import logging
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from config import (
    MODEL_DIR, BASE_MODEL_FILE, FOREST_MODEL_FILE, BOOST_MODEL_FILE, REGIME_MODEL_FILE,
    RIDGE_ALPHA, RANDOM_STATE, MAX_HISTORY_DAYS, TRAIN_WINDOWS, ENSEMBLE_BOOTSTRAP_N,
    ENSEMBLE_SUBSAMPLE_FRAC, TRANSACTION_COST_PER_TRADE, SLIPPAGE_PCT, RF_N_ESTIMATORS,
    RF_MAX_DEPTH, RF_MIN_SAMPLES_LEAF, BOOST_MAX_ITER, BOOST_MAX_DEPTH, MODEL_VERSION, USE_CONSTANT_DATA,
    EXTRA_TREES_N_ESTIMATORS, EXTRA_TREES_MAX_DEPTH, EXTRA_TREES_MIN_SAMPLES_LEAF, EXTRA_TREES_MODEL_FILE,
    HORIZON_MINUTES, META_MODEL_FILE, INSTRUMENTS, DEFAULT_INSTRUMENT, SAVE_LEGACY_MODEL_FILES,
    AUTO_WINDOW_SELECTION, WINDOW_SELECTION_SPLITS,
    WINDOW_CANDIDATES_4MIN, WINDOW_CANDIDATES_DAILY, WINDOW_CANDIDATES_WEEKEND,
    TS_SPLIT_GAP_BARS, BACKTEST_SIGNAL_STD_MULT, WINDOW_SELECTION_DEPLOY_METRIC
)
from features import build_features_for_training, build_next_session_dataset, build_next_week_dataset, DEFAULT_FEATURES
from regime_model import train_regime, build_regime_features, extract_regime_pipeline
import pandas as pd

logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)

def _ts_split(n_splits: int):
    try:
        gap = max(0, int(TS_SPLIT_GAP_BARS))
    except Exception:
        gap = 0
    try:
        return TimeSeriesSplit(n_splits=n_splits, gap=gap)
    except TypeError:
        return TimeSeriesSplit(n_splits=n_splits)

def _atomic_joblib_dump(obj, path: str):
    tmp_path = f"{path}.tmp"
    try:
        joblib.dump(obj, tmp_path)
        os.replace(tmp_path, path)
    except Exception:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        raise

def _safe_instrument_tag(instr_key: str) -> str:
    return str(instr_key or DEFAULT_INSTRUMENT).replace("/", "_").replace("\\", "_")

def _model_path(prefix: str, mode: str, instr_key: str) -> str:
    tag = _safe_instrument_tag(instr_key)
    return os.path.join(MODEL_DIR, f"{prefix}_{mode}_{tag}.pkl")

def _compute_recency_weights(index, half_life_days=7.0):
    try:
        now = index.max()
        lambda_decay = np.log(2.0) / (half_life_days * 24.0 * 60.0)
        ages = (now - index).total_seconds() / 60.0
        weights = np.exp(-lambda_decay * ages)
        weights = weights / np.mean(weights)
        return weights
    except Exception:
        return np.ones(len(index), dtype=float)

def _bucket_signal(z):
    if z < 0.5:
        return "Weak"
    elif z < 1.5:
        return "Medium"
    else:
        return "Strong"

def _bucket_vol(v):
    if v < 0.002:
        return "LowVol"
    elif v < 0.005:
        return "MidVol"
    else:
        return "HighVol"

def _build_calibration(model, X, y, df_meta, regimes):
    """ Build a calibration mapping: map (regime, vol_bucket, sig_bucket) -> empirical probability of correct sign.
    Also return a simplified mapping by sig_bucket averaged across regimes/vol.
    Uses TimeSeriesSplit to avoid leakage.
    """
    # Keep calibration computationally bounded for production retrains.
    try:
        max_rows = 5000
        if len(X) > max_rows:
            X = X.iloc[-max_rows:]
            y = y.iloc[-max_rows:]
            df_meta = df_meta.iloc[-max_rows:]
            if regimes is not None:
                regimes = regimes.iloc[-max_rows:]
    except Exception:
        pass
    n_rows = len(X)
    n_splits = 4 if n_rows >= 1200 else 3 if n_rows >= 600 else 2
    tss = _ts_split(max(2, n_splits))
    table = {}
    for train_idx, test_idx in tss.split(X):
        try:
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            preds = model.predict(X.iloc[test_idx])
        except Exception:
            continue
        for i, idx in enumerate(test_idx):
            pred = preds[i]
            actual = y.iloc[idx]
            correct = (pred > 0) == (actual > 0)
            resid_std = np.std(y.iloc[train_idx] - model.predict(X.iloc[train_idx])) if len(train_idx) > 0 else 1e-6
            resid_std = max(resid_std, 1e-6)
            z = abs(pred) / resid_std
            sig_bucket = _bucket_signal(z)
            vol_bucket = _bucket_vol(df_meta["vol_6"].iloc[idx])
            regime = regimes.iloc[idx] if (regimes is not None and idx < len(regimes)) else "Neutral"
            key = (regime, vol_bucket, sig_bucket)
            wins, total = table.get(key, (0, 0))
            table[key] = (wins + int(correct), total + 1)
    calibrated = {}
    sig_agg = {}
    for k, (w, n) in table.items():
        calibrated[k] = (w + 10) / (n + 20)  # pseudocounts for smoothing
        sig = k[2]
        agg_w, agg_n = sig_agg.get(sig, (0.0, 0.0))
        sig_agg[sig] = (agg_w + calibrated[k] * n, agg_n + n)
    sig_map = {}
    for s, (aw, an) in sig_agg.items():
        if an <= 0:
            sig_map[s] = 0.5
        else:
            sig_map[s] = float(aw / an)
    for s in ("Weak","Medium","Strong"):
        sig_map.setdefault(s, 0.5)
    return calibrated, sig_map

# Helper: build model constructors
def _ridge_pipeline():
    return Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=RIDGE_ALPHA))])

def _forest_ctor(seed=RANDOM_STATE):
    return RandomForestRegressor(n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH, random_state=seed)

def _boost_ctor(seed=RANDOM_STATE):
    return HistGradientBoostingRegressor(max_depth=BOOST_MAX_DEPTH, max_iter=BOOST_MAX_ITER, random_state=seed)

def _extra_trees_ctor(seed=RANDOM_STATE):
    return ExtraTreesRegressor(
        n_estimators=EXTRA_TREES_N_ESTIMATORS,
        max_depth=EXTRA_TREES_MAX_DEPTH,
        min_samples_leaf=EXTRA_TREES_MIN_SAMPLES_LEAF,
        random_state=seed
    )

def _forest_ctor_tuned(regime_detail: str, seed=RANDOM_STATE):
    max_depth = RF_MAX_DEPTH
    min_leaf =  max(1, RF_MIN_SAMPLES_LEAF)
    n_estimators = RF_N_ESTIMATORS
    if regime_detail:
        if "HighVol" in regime_detail:
            max_depth = max(4, RF_MAX_DEPTH - 2)
            min_leaf = RF_MIN_SAMPLES_LEAF + 5
            n_estimators = int(RF_N_ESTIMATORS * 1.2)
        elif "LowVol" in regime_detail:
            max_depth = RF_MAX_DEPTH + 2
            min_leaf = max(2, RF_MIN_SAMPLES_LEAF - 2)
    return RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_leaf, random_state=seed)

def _boost_ctor_tuned(regime_detail: str, seed=RANDOM_STATE):
    max_depth = BOOST_MAX_DEPTH
    max_iter = BOOST_MAX_ITER
    if regime_detail:
        if "HighVol" in regime_detail:
            max_depth = max(2, BOOST_MAX_DEPTH - 1)
            max_iter = int(BOOST_MAX_ITER * 1.2)
        elif "LowVol" in regime_detail:
            max_depth = BOOST_MAX_DEPTH + 1
            max_iter = int(BOOST_MAX_ITER * 0.9)
    return HistGradientBoostingRegressor(max_depth=max_depth, max_iter=max_iter, random_state=seed)

def _extra_trees_ctor_tuned(regime_detail: str, seed=RANDOM_STATE):
    max_depth = EXTRA_TREES_MAX_DEPTH
    min_leaf = EXTRA_TREES_MIN_SAMPLES_LEAF
    n_estimators = EXTRA_TREES_N_ESTIMATORS
    if regime_detail:
        if "HighVol" in regime_detail:
            max_depth = max(6, EXTRA_TREES_MAX_DEPTH - 2)
            min_leaf = EXTRA_TREES_MIN_SAMPLES_LEAF + 3
            n_estimators = int(EXTRA_TREES_N_ESTIMATORS * 1.2)
        elif "LowVol" in regime_detail:
            max_depth = EXTRA_TREES_MAX_DEPTH + 2
            min_leaf = max(2, EXTRA_TREES_MIN_SAMPLES_LEAF - 2)
    return ExtraTreesRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_leaf,
        random_state=seed
    )

# ----------------------------
# Train helpers: ensemble across windows and bootstrap
# ----------------------------
def _train_single_model(X, y, model_ctor, sample_weight=None):
    model = model_ctor()
    try:
        model.fit(X, y, sample_weight=sample_weight)
    except (TypeError, ValueError):
        # some sklearn estimators don't accept sample_weight
        try:
            model.fit(X, y)
        except Exception:
            raise
    return model

def _bootstrap_ensemble(X, y, model_ctor, n=12, subsample_frac=0.8, random_state=42, sample_weight=None):
    rng = np.random.RandomState(random_state)
    nrows = len(X)
    members = []
    block_len = int(max(1, nrows * subsample_frac))
    for i in range(n):
        if nrows <= 0:
            members.append(None)
            continue
        start = rng.randint(0, max(1, nrows - block_len + 1))
        idx = np.arange(start, start + block_len)
        Xi = X.iloc[idx]
        yi = y.iloc[idx]
        sw = sample_weight[idx] if (sample_weight is not None and len(sample_weight) == nrows) else None
        try:
            m = _train_single_model(Xi, yi, model_ctor, sample_weight=sw)
            members.append(m)
        except Exception:
            try:
                members.append(_train_single_model(Xi, yi, model_ctor))
            except Exception:
                members.append(None)
    return members

def _apply_triple_barrier(df, profit_k=2.0, stop_k=1.0, max_bars=60):
    vols = df["ewm_vol"].replace(0, 1e-9)
    profit = profit_k * vols
    stop = -stop_k * vols
    labels = []
    n = len(df)
    closes = df["close"].values
    for i in range(n):
        lab = "timeout"
        base = closes[i]
        for j in range(1, min(max_bars, n - i)):
            fut = closes[i + j]
            ret = (fut / base) - 1.0
            if ret >= profit[i]:
                lab = "profit"
                break
            if ret <= stop[i]:
                lab = "stop"
                break
        labels.append(lab)
    return labels

# ----------------------------
# Training (main) — now trains ensembles per window and per-regime
# ----------------------------
def train_all_models(
        df,
        instrument_key: str = None,
        max_runtime_sec: float = None,
        train_horizons: bool = True,
        progress_cb=None,
        modes_filter: list[str] | None = None
):
    """ Train models across configured windows and save payloads. """
    start_ts = None
    try:
        start_ts = time.time()
    except Exception:
        start_ts = None
    # NEW: train for multiple modes
    modes = {
        '4min': {'target_col': 'target_ret_norm', 'daily_horizon': False, 'weekend_horizon': False},
        'daily_high': {'target_col': 'target_high_ret_norm', 'daily_horizon': True, 'weekend_horizon': False},
        'daily_low': {'target_col': 'target_low_ret_norm', 'daily_horizon': True, 'weekend_horizon': False},
        'weekend': {'target_col': 'target_weekend_ret_norm', 'daily_horizon': False, 'weekend_horizon': True},
    }
    models_last = None
    for mode, params in modes.items():
        if modes_filter is not None and mode not in modes_filter:
            continue
        if max_runtime_sec is not None and start_ts is not None:
            if (time.time() - start_ts) > float(max_runtime_sec):
                logger.warning("Training time budget exceeded; stopping early.")
                break
        # enforce using just the last MAX_HISTORY_DAYS for training to match your requirement
        try:
            cutoff = pd.to_datetime(df.index.max()) - np.timedelta64(int(MAX_HISTORY_DAYS), "D")
            df_mode = df[df.index >= cutoff]
        except Exception:
            df_mode = df

        instr_key = instrument_key or DEFAULT_INSTRUMENT
        instr_meta = INSTRUMENTS.get(instr_key, {}) if instr_key else {}
        instr_type = instr_meta.get("type", "equity")
        # detect futures-like series: prefer instrument metadata, fallback to data heuristics
        futures_detected = (instr_type == "futures")
        try:
            if not futures_detected and "overnight_gap" in df_mode.columns:
                if float(np.nanmean(np.abs(df_mode["overnight_gap"]))) > 1e-4:
                    futures_detected = True
        except Exception:
            futures_detected = futures_detected or False

        try:
            if mode in ("daily_high", "daily_low"):
                df_full = build_next_session_dataset(df_mode, instrument_key=instr_key)
            else:
                # For weekend mode, use per-bar features with weekend_horizon targets
                # so predictions can be made from the most recent prior session close.
                df_full = build_features_for_training(
                    df_mode,
                    predict_minutes=4,
                    instrument_key=instr_key,
                    daily_horizon=params['daily_horizon'],
                    weekend_horizon=params.get('weekend_horizon', False)
                )
        except Exception as e:
            logger.warning(f"Skipping mode {mode} due to feature build error: {e}")
            continue
        # Guard against inf values from feature math
        try:
            df_full = df_full.replace([np.inf, -np.inf], np.nan).dropna()
        except Exception:
            pass
        # For futures, reduce roll-gap contamination by dropping extreme gap bars
        if futures_detected:
            try:
                vol_ref = df_full.get("ewm_vol", pd.Series(0.0, index=df_full.index)).replace(0, np.nan).fillna(1e-6)
                gap = df_full.get("overnight_gap", pd.Series(0.0, index=df_full.index)).abs()
                ret1 = df_full.get("ret_1", pd.Series(0.0, index=df_full.index)).abs()
                spike = (gap > (6.0 * vol_ref)) | (ret1 > (8.0 * vol_ref))
                if spike.any():
                    df_full = df_full.loc[~spike].copy()
            except Exception:
                pass
        min_rows = 100
        if mode in ("daily_high", "daily_low"):
            min_rows = 15
        elif mode == "weekend":
            min_rows = 6
        if len(df_full) < min_rows:
            raise RuntimeError("Not enough data to train models")
        # Diagnostic: log target stats
        mean_ret = np.mean(df_full[params['target_col']])
        std_ret = np.std(df_full[params['target_col']])
        logger.info(f"Target ret stats: mean {mean_ret:.8f}, std {std_ret:.8f}")
        if mean_ret < -0.05:
            logger.debug("Negative mean target_ret detected; potential downward bias in data")
        # Use normalized target for training to handle small signals, de-mean to remove directional bias
        try:
            target_mean = float(np.nanmean(df_full[params['target_col']]))
        except Exception:
            target_mean = 0.0
        df_full["target_ret_norm_demean"] = df_full[params['target_col']] - target_mean
        y = df_full["target_ret_norm_demean"]
        try:
            y = y.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        except Exception:
            pass
        # Balanced sample weights (reduce bias toward majority sign)
        try:
            pos = (y > 0).sum()
            neg = (y <= 0).sum()
            if pos > 0 and neg > 0:
                w_pos = (pos + neg) / (2.0 * pos)
                w_neg = (pos + neg) / (2.0 * neg)
                sample_weights_global = np.where(y.values > 0, w_pos, w_neg)
                # scale to mean 1
                sample_weights_global = sample_weights_global / np.mean(sample_weights_global)
            else:
                sample_weights_global = np.ones_like(y.values, dtype=float)
        except Exception:
            sample_weights_global = np.ones_like(y.values, dtype=float)

        # Train regime model once (only for 4min) and keep payload for regime-aware slicing
        if mode == "4min":
            try:
                regime_payload = train_regime(df_full)
            except Exception:
                regime_payload = None
        else:
            regime_payload = None
        # regime labels for slicing (if available from payload)
        def _compute_regimes(df_local):
            try:
                scaler, gmm = extract_regime_pipeline(regime_payload)
                if scaler is not None and gmm is not None:
                    Xr = build_regime_features(df_local).fillna(0.0)
                    Xs = scaler.transform(Xr)
                    comp = gmm.predict(Xs)
                    return pd.Series(comp, index=df_local.index).astype(str)
            except Exception:
                pass
            return (df_local["target_ret"] > 0).map({True: "1", False: "0"})

        regimes_series = _compute_regimes(df_full) if mode == "4min" else None

        sample_weights = _compute_recency_weights(df_full.index, half_life_days=7.0)
        def _resolve_windows(df_in, base_windows, horizon_minutes: int = 4):
            try:
                available_days = int((df_in.index.max() - df_in.index.min()).days)
            except Exception:
                available_days = int(base_windows.get("long_days", 180))
            try:
                h = max(1, int(horizon_minutes))
            except Exception:
                h = 4
            # scale windows mildly for longer horizons
            scale = max(0.8, min(2.5, math.sqrt(h / 4.0)))
            short = int(base_windows.get("short_days", 30) * scale)
            mid = int(base_windows.get("mid_days", 120) * scale)
            long = int(base_windows.get("long_days", 500) * scale)
            if available_days < 60:
                short = max(5, available_days // 2)
                mid = max(short, available_days)
                long = max(mid, available_days)
            short = min(short, max(5, available_days))
            mid = min(max(mid, short), max(5, available_days))
            long = min(max(long, mid), max(5, available_days))
            return {"short": short, "mid": mid, "long": long}

        def slice_by_days(df_in, days):
            cutoff = df_in.index.max() - np.timedelta64(int(days), "D")
            return df_in[df_in.index >= cutoff]

        def _window_mae(df_in, days, target_col):
            df_w = slice_by_days(df_in, int(days))
            if df_w is None or len(df_w) < 300:
                return None
            try:
                Xw = df_w[DEFAULT_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
                yw = df_w[target_col].astype(float)
            except Exception:
                return None
            n = len(Xw)
            if n < 300:
                return None
            n_splits = min(WINDOW_SELECTION_SPLITS, max(2, n // 500))
            if n_splits < 2:
                return None
            tss = _ts_split(n_splits)
            maes = []
            for train_idx, test_idx in tss.split(Xw):
                X_train, X_test = Xw.iloc[train_idx], Xw.iloc[test_idx]
                y_train, y_test = yw.iloc[train_idx], yw.iloc[test_idx]
                try:
                    model = _ridge_pipeline()
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    maes.append(float(mean_absolute_error(y_test, preds)))
                except Exception:
                    continue
            if not maes:
                return None
            return float(np.mean(maes))

        def _opt_signal_threshold(pred_train_raw: np.ndarray, y_train_raw: np.ndarray, costs: float) -> float:
            try:
                p = np.asarray(pred_train_raw, dtype=float)
                yv = np.asarray(y_train_raw, dtype=float)
                mask = np.isfinite(p) & np.isfinite(yv)
                p = p[mask]
                yv = yv[mask]
                if p.size < 40:
                    return 0.0
                abs_p = np.abs(p)
                cands = {0.0}
                for q in (0.10, 0.20, 0.30, 0.45, 0.60, 0.75, 0.85, 0.92, 0.97, 0.985):
                    try:
                        cands.add(float(np.quantile(abs_p, q)))
                    except Exception:
                        continue
                best_thr = 0.0
                best_score = -np.inf
                for thr in sorted([c for c in cands if np.isfinite(c) and c >= 0.0]):
                    pos = np.sign(p)
                    pos[np.abs(p) < thr] = 0.0
                    gross = pos * yv
                    trades = int(np.nansum(np.abs(np.diff(np.concatenate(([0.0], pos))))))
                    net = float(np.nansum(gross) - costs * trades)
                    active = float(np.mean(pos != 0))
                    turnover = float(trades) / max(1.0, float(len(pos)))
                    if np.any(pos != 0):
                        dir_acc = float(np.mean(np.sign(p[pos != 0]) == np.sign(yv[pos != 0])))
                    else:
                        dir_acc = 0.5
                    score = (
                        (net / max(1.0, float(len(yv))))
                        + (0.00045 * (dir_acc - 0.5))
                        - (0.00085 * max(0.0, active - 0.35))
                        - (0.00020 * max(0.0, 0.03 - active))
                        - (0.00035 * turnover)
                    )
                    if score > best_score:
                        best_score = score
                        best_thr = float(thr)
                return max(0.0, float(best_thr))
            except Exception:
                return 0.0

        def _mode_horizon_minutes(mode_label: str) -> int:
            ml = str(mode_label or "").lower()
            if ml == "4min":
                return 4
            if ml in ("daily_high", "daily_low"):
                # Session-level horizon for equities.
                return 390
            if ml == "weekend":
                # Approx. five regular sessions.
                return 1950
            return 4

        def _window_quality(df_in, days, target_col, mode_label: str):
            df_w = slice_by_days(df_in, int(days))
            if df_w is None or len(df_w) < 300:
                return None
            try:
                Xw = df_w[DEFAULT_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
                yw = df_w[target_col].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            except Exception:
                return None
            n = len(Xw)
            if n < 300:
                return None
            n_splits = min(WINDOW_SELECTION_SPLITS, max(2, n // 500))
            if n_splits < 2:
                return None
            tss = _ts_split(n_splits)
            maes = []
            dir_accs = []
            net_per_bar = []
            active_fracs = []
            costs = float(TRANSACTION_COST_PER_TRADE + SLIPPAGE_PCT)
            for train_idx, test_idx in tss.split(Xw):
                X_train, X_test = Xw.iloc[train_idx], Xw.iloc[test_idx]
                y_train, y_test = yw.iloc[train_idx], yw.iloc[test_idx]
                try:
                    model = _ridge_pipeline()
                    model.fit(X_train, y_train)
                    preds = np.asarray(model.predict(X_test), dtype=float)
                    maes.append(float(mean_absolute_error(y_test, preds)))
                except Exception:
                    continue
                if mode_label != "4min":
                    continue
                try:
                    pred_tr = np.asarray(model.predict(X_train), dtype=float)
                    vol_train = np.maximum(
                        pd.to_numeric(df_w.get("ewm_vol"), errors="coerce").iloc[train_idx].fillna(0.001).values.astype(float),
                        1e-6
                    )
                    vol_test = np.maximum(
                        pd.to_numeric(df_w.get("ewm_vol"), errors="coerce").iloc[test_idx].fillna(0.001).values.astype(float),
                        1e-6
                    )
                    y_train_raw = np.asarray(y_train, dtype=float) * vol_train
                    y_test_raw = np.asarray(y_test, dtype=float) * vol_test
                    pred_tr_raw = pred_tr * vol_train
                    pred_test_raw = preds * vol_test
                    thr = _opt_signal_threshold(pred_tr_raw, y_train_raw, costs=costs)
                    pos = np.sign(pred_test_raw)
                    if thr > 0:
                        pos[np.abs(pred_test_raw) < thr] = 0.0
                    gross = pos * y_test_raw
                    trades = int(np.nansum(np.abs(np.diff(np.concatenate(([0.0], pos))))))
                    net = float(np.nansum(gross) - costs * trades)
                    net_per_bar.append(net / max(1.0, float(len(y_test_raw))))
                    active = float(np.mean(pos != 0))
                    active_fracs.append(active)
                    if np.any(pos != 0):
                        dacc = float(np.mean(np.sign(pred_test_raw[pos != 0]) == np.sign(y_test_raw[pos != 0])))
                    else:
                        dacc = float("nan")
                    dir_accs.append(dacc)
                except Exception:
                    continue
            if not maes:
                return None
            out = {
                "mae": float(np.mean(maes)),
                "dir_acc": float(np.nanmean(dir_accs)) if dir_accs else float("nan"),
                "net_per_bar": float(np.nanmean(net_per_bar)) if net_per_bar else float("nan"),
                "active_frac": float(np.nanmean(active_fracs)) if active_fracs else float("nan"),
            }
            # Lower score is better.
            if mode_label == "4min" and WINDOW_SELECTION_DEPLOY_METRIC:
                try:
                    dacc = out["dir_acc"] if np.isfinite(out["dir_acc"]) else 0.5
                    netb = out["net_per_bar"] if np.isfinite(out["net_per_bar"]) else -1e-6
                    active = out["active_frac"] if np.isfinite(out["active_frac"]) else 0.0
                    score = float(out["mae"])
                    score += max(0.0, 0.53 - float(dacc)) * 0.8
                    score += max(0.0, -float(netb)) * 250.0
                    score += max(0.0, 0.04 - float(active)) * 0.6
                    score += max(0.0, float(active) - 0.65) * 0.25
                    out["selection_score"] = float(score)
                except Exception:
                    out["selection_score"] = float(out["mae"])
            else:
                out["selection_score"] = float(out["mae"])
            return out

        def _candidate_windows(mode_label: str, available_days: int):
            if mode_label == "4min":
                base = WINDOW_CANDIDATES_4MIN
            elif mode_label in ("daily_high", "daily_low"):
                base = WINDOW_CANDIDATES_DAILY
            else:
                base = WINDOW_CANDIDATES_WEEKEND
            return [d for d in base if d <= max(10, available_days)]

        def _select_windows_for_mode(df_in, target_col, mode_label: str):
            try:
                available_days = int((df_in.index.max() - df_in.index.min()).days)
            except Exception:
                available_days = 0
            candidates = _candidate_windows(mode_label, available_days)
            scores = []
            for d in candidates:
                if mode_label == "4min" and WINDOW_SELECTION_DEPLOY_METRIC:
                    qual = _window_quality(df_in, d, target_col, mode_label)
                    if not qual:
                        continue
                    score = qual.get("selection_score")
                    if score is None or not np.isfinite(float(score)):
                        continue
                    scores.append((int(d), float(score), qual))
                else:
                    mae = _window_mae(df_in, d, target_col)
                    if mae is not None and np.isfinite(mae):
                        scores.append((int(d), float(mae), {"mae": float(mae), "selection_score": float(mae)}))
            if not scores:
                return _resolve_windows(df_in, TRAIN_WINDOWS, horizon_minutes=_mode_horizon_minutes(mode_label))
            scores.sort(key=lambda x: x[1])
            chosen_days = sorted([d for d, _, _ in scores[:3]])
            windows_sel = {}
            if len(chosen_days) >= 1:
                windows_sel["short"] = chosen_days[0]
            if len(chosen_days) >= 2:
                windows_sel["mid"] = chosen_days[1]
            if len(chosen_days) >= 3:
                windows_sel["long"] = chosen_days[2]
            return windows_sel or _resolve_windows(df_in, TRAIN_WINDOWS, horizon_minutes=_mode_horizon_minutes(mode_label))

        windows = _resolve_windows(df_full, TRAIN_WINDOWS, horizon_minutes=_mode_horizon_minutes(mode))
        if AUTO_WINDOW_SELECTION:
            windows = _select_windows_for_mode(df_full, params['target_col'], mode)
        try:
            if progress_cb:
                progress_cb(f"Training {mode} models… (windows: {windows})")
        except Exception:
            pass
        base_cal = {}
        base_cal_sig = {"Weak": 0.5, "Medium": 0.5, "Strong": 0.5}
        resid_std_base = 1e-6
        pred_bias_base = 0.0
        train_mean_base = 0.0
        models_payload = {}
        # Base (Ridge) as pipeline trained on full window + calibration
        try:
            Xb = df_full[DEFAULT_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            base_pipe = _ridge_pipeline()
            # combine class-balance and recency weights
            combined_sw = sample_weights_global * _compute_recency_weights(df_full.index, half_life_days=7.0)
            base_cal, base_cal_sig = _build_calibration(base_pipe, Xb, y, df_full, regimes_series)
            try:
                base_pipe.fit(Xb, y, sample_weight=combined_sw)
            except Exception:
                try:
                    base_pipe.fit(Xb, y)
                except Exception:
                    pass
            try:
                preds_train = base_pipe.predict(Xb)
                resid_std_base = float(np.std(y - preds_train))
                pred_bias_base = float(np.mean(preds_train - y))
                train_mean_base = float(np.mean(y))
            except Exception:
                resid_std_base = 1e-6
                pred_bias_base = 0.0
                train_mean_base = 0.0
        except Exception:
            logger.exception("Base pipeline training failed")
            base_pipe = _ridge_pipeline()
            base_cal = {}
            base_cal_sig = {"Weak": 0.5, "Medium": 0.5, "Strong": 0.5}
            resid_std_base = 1e-6
            pred_bias_base = 0.0
            train_mean_base = 0.0

        def _train_window_ensembles(df_local, regimes_local, windows_local, feats, model_ctor, label,
                                    ensemble_n_override=None, regime_detail_map=None, model_kind=None,
                                    target_col='target_ret_norm_demean', min_rows: int = 120):
            payload = {"ensemble_models": {}, "features": feats, "regime_models": {}, "calibration": {}, "calibration_sig": {}}
            perf = {}
            perf_mae = {}
            for wname, days in windows_local.items():
                df_w = slice_by_days(df_local, int(days))
                if df_w is None or len(df_w) < min_rows:
                    payload["ensemble_models"][wname] = []
                    perf[wname] = {
                        "mae": float("nan"),
                        "dir_acc": float("nan"),
                        "net_per_bar": float("nan"),
                        "active_frac": float("nan"),
                    }
                    perf_mae[wname] = float("nan")
                    continue
                Xw = df_w[feats].replace([np.inf, -np.inf], np.nan).fillna(0.0)
                yw = df_w[target_col].replace([np.inf, -np.inf], np.nan).fillna(0.0)
                # sample weight: combine recency and global balancing for the window
                sw_full = _compute_recency_weights(df_w.index, half_life_days=7.0)
                # align class-balance for window
                try:
                    pos = (yw > 0).sum()
                    neg = (yw <= 0).sum()
                    if pos > 0 and neg > 0:
                        wp = (pos + neg) / (2.0 * pos)
                        wn = (pos + neg) / (2.0 * neg)
                        class_w = np.where(yw.values > 0, wp, wn)
                        class_w = class_w / np.mean(class_w)
                    else:
                        class_w = np.ones(len(yw))
                except Exception:
                    class_w = np.ones(len(yw))
                sw = sw_full * class_w

                # increase ensemble members slightly for futures to capture volatility
                ensemble_n = ensemble_n_override or ENSEMBLE_BOOTSTRAP_N
                if futures_detected:
                    # Keep production training tractable for futures; avoid oversized ensembles.
                    ensemble_n = min(max(4, ENSEMBLE_BOOTSTRAP_N), 16)

                members = _bootstrap_ensemble(Xw, yw, model_ctor, n=ensemble_n, subsample_frac=ENSEMBLE_SUBSAMPLE_FRAC, random_state=RANDOM_STATE, sample_weight=sw)
                payload["ensemble_models"][wname] = members
                try:
                    split = int(len(Xw) * 0.8)
                    if split < 2:
                        perf[wname] = {
                            "mae": float("nan"),
                            "dir_acc": float("nan"),
                            "net_per_bar": float("nan"),
                            "active_frac": float("nan"),
                        }
                        perf_mae[wname] = float("nan")
                        continue
                    Xtrain, Xtest = Xw.iloc[:split], Xw.iloc[split:]
                    ytrain, ytest = yw.iloc[:split], yw.iloc[split:]
                    m = model_ctor()
                    try:
                        # fit with simple recency weighting on the train segment
                        sw_train = _compute_recency_weights(Xtrain.index, half_life_days=7.0)
                        m.fit(Xtrain, ytrain, sample_weight=sw_train)
                        preds = np.asarray(m.predict(Xtest), dtype=float)
                        mae = float(mean_absolute_error(ytest, preds))
                        # store training bias/resid if single model (for diagnostics later)
                        try:
                            train_bias = float(np.mean(m.predict(Xtrain) - ytrain))
                            train_resid = float(np.std(ytrain - m.predict(Xtrain)))
                        except Exception:
                            train_bias = 0.0
                            train_resid = 1e-6
                    except Exception:
                        mae = 1.0
                        train_bias = 0.0
                        train_resid = 1e-6
                    dir_acc = float("nan")
                    net_pb = float("nan")
                    active_frac = float("nan")
                    if mode == "4min":
                        try:
                            vol_train = np.maximum(
                                pd.to_numeric(df_w.get("ewm_vol"), errors="coerce").iloc[:split].fillna(0.001).values.astype(float),
                                1e-6
                            )
                            vol_test = np.maximum(
                                pd.to_numeric(df_w.get("ewm_vol"), errors="coerce").iloc[split:].fillna(0.001).values.astype(float),
                                1e-6
                            )
                            pred_train = np.asarray(m.predict(Xtrain), dtype=float)
                            pred_train_raw = pred_train * vol_train
                            pred_test_raw = preds * vol_test
                            y_train_raw = np.asarray(ytrain, dtype=float) * vol_train
                            y_test_raw = np.asarray(ytest, dtype=float) * vol_test
                            costs = float(TRANSACTION_COST_PER_TRADE + SLIPPAGE_PCT)
                            thr = _opt_signal_threshold(pred_train_raw, y_train_raw, costs=costs)
                            pos = np.sign(pred_test_raw)
                            if thr > 0:
                                pos[np.abs(pred_test_raw) < thr] = 0.0
                            gross = pos * y_test_raw
                            trades = int(np.nansum(np.abs(np.diff(np.concatenate(([0.0], pos))))))
                            net_pb = float(np.nansum(gross) - costs * trades) / max(1.0, float(len(y_test_raw)))
                            active_frac = float(np.mean(pos != 0))
                            if np.any(pos != 0):
                                dir_acc = float(np.mean(np.sign(pred_test_raw[pos != 0]) == np.sign(y_test_raw[pos != 0])))
                        except Exception:
                            pass
                    perf[wname] = {
                        "mae": float(mae),
                        "dir_acc": float(dir_acc),
                        "net_per_bar": float(net_pb),
                        "active_frac": float(active_frac),
                    }
                    perf_mae[wname] = float(mae)
                except Exception:
                    perf[wname] = {
                        "mae": 1.0,
                        "dir_acc": float("nan"),
                        "net_per_bar": float("nan"),
                        "active_frac": float("nan"),
                    }
                    perf_mae[wname] = 1.0
                try:
                    if regimes_local is not None:
                        for reg in np.unique(regimes_local):
                            reg_mask = (regimes_local == reg)
                            df_reg = df_w[df_w.index.isin(regimes_local[reg_mask].index)]
                            if len(df_reg) < 80:
                                payload["regime_models"].setdefault(str(reg), {})[wname] = []
                                continue
                            Xr = df_reg[feats].replace([np.inf, -np.inf], np.nan).fillna(0.0)
                            yr = df_reg[target_col].replace([np.inf, -np.inf], np.nan).fillna(0.0)
                            ctor = model_ctor
                            try:
                                detail = None
                                if regime_detail_map is not None:
                                    try:
                                        detail = regime_detail_map.get(int(reg), None)
                                    except Exception:
                                        detail = regime_detail_map.get(str(reg), None)
                                if model_kind == "forest":
                                    ctor = lambda: _forest_ctor_tuned(detail)
                                elif model_kind == "boost":
                                    ctor = lambda: _boost_ctor_tuned(detail)
                                elif model_kind == "extra_trees":
                                    ctor = lambda: _extra_trees_ctor_tuned(detail)
                            except Exception:
                                ctor = model_ctor
                            members_r = _bootstrap_ensemble(Xr, yr, ctor, n=min(6, ENSEMBLE_BOOTSTRAP_N), subsample_frac=ENSEMBLE_SUBSAMPLE_FRAC, random_state=RANDOM_STATE)
                            payload["regime_models"].setdefault(str(reg), {})[wname] = members_r
                except Exception:
                    pass
            payload["performance"] = perf
            payload["performance_mae"] = perf_mae
            # compute calibration for this model family across the full df_local slice
            try:
                model_for_cal = model_ctor()
                cal_table, cal_sig = _build_calibration(model_for_cal, df_local[feats], df_local[target_col], df_local, regimes_local)
                payload["calibration"] = cal_table
                payload["calibration_sig"] = cal_sig
            except Exception:
                payload["calibration"] = {}
                payload["calibration_sig"] = base_cal_sig
            return payload

        # Extra small feature augment for forest/boost/extra
        forest_feats = DEFAULT_FEATURES + ["ret_2", "ret_4", "mom_3"]
        regime_detail_map = {}
        try:
            if regime_payload is not None:
                regime_detail_map = regime_payload.get("detail", {})
        except Exception:
            regime_detail_map = {}
        ensemble_min_rows = 120
        if mode in ("daily_high", "daily_low"):
            ensemble_min_rows = 10
        elif mode == "weekend":
            ensemble_min_rows = 4
        ensemble_override = None
        forest_payload = _train_window_ensembles(
            df_full, regimes_series, windows, forest_feats, lambda: _forest_ctor(), "forest",
            regime_detail_map=regime_detail_map, model_kind="forest", target_col=params['target_col'],
            ensemble_n_override=ensemble_override, min_rows=ensemble_min_rows
        )
        boost_feats = DEFAULT_FEATURES + ["ret_2", "ret_4"]
        boost_payload = _train_window_ensembles(
            df_full, regimes_series, windows, boost_feats, lambda: _boost_ctor(), "boost",
            regime_detail_map=regime_detail_map, model_kind="boost", target_col=params['target_col'],
            ensemble_n_override=ensemble_override, min_rows=ensemble_min_rows
        )
        extra_feats = DEFAULT_FEATURES + ["ret_2", "ret_4", "mom_3", "ema_5", "ema_20", "trend_strength", "vol_rolling_skew", "vol_rolling_kurt"]
        extra_payload = _train_window_ensembles(
            df_full, regimes_series, windows, extra_feats, lambda: _extra_trees_ctor(), "extra_trees",
            regime_detail_map=regime_detail_map, model_kind="extra_trees", target_col=params['target_col'],
            ensemble_n_override=ensemble_override, min_rows=ensemble_min_rows
        )
        # Train base ensembles across windows (same as forest/boost) to enable uncertainty and window blending
        base_payload = _train_window_ensembles(
            df_full, regimes_series, windows, DEFAULT_FEATURES, _ridge_pipeline, "base",
            regime_detail_map=regime_detail_map, model_kind="base", target_col=params['target_col'],
            ensemble_n_override=ensemble_override, min_rows=ensemble_min_rows
        )

        # Fallback full-window models to prevent "no model" during sparse windows
        def _fit_fallback(X, y, ctor):
            try:
                m = ctor()
                try:
                    sw_full = _compute_recency_weights(X.index, half_life_days=7.0)
                    m.fit(X, y, sample_weight=sw_full)
                except Exception:
                    m.fit(X, y)
                preds = m.predict(X)
                resid = float(np.std(y - preds))
                bias = float(np.mean(preds - y))
                return m, max(resid, 1e-6), bias
            except Exception:
                return None, 1e-6, 0.0

        y_norm = df_full["target_ret_norm_demean"]
        Xf_full = df_full[forest_feats].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        Xg_full = df_full[boost_feats].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        Xe_full = df_full[extra_feats].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        forest_fallback, forest_resid, forest_bias = _fit_fallback(Xf_full, y_norm, _forest_ctor)
        boost_fallback, boost_resid, boost_bias = _fit_fallback(Xg_full, y_norm, _boost_ctor)
        extra_fallback, extra_resid, extra_bias = _fit_fallback(Xe_full, y_norm, _extra_trees_ctor)
        models_payload["base"] = {
            "model": base_pipe,
            "features": DEFAULT_FEATURES,
            "calibration": base_payload.get("calibration", base_cal),
            "calibration_sig": base_payload.get("calibration_sig", base_cal_sig),
            "meta": {
                "resid_std": max(resid_std_base, 1e-6),
                "pred_bias": pred_bias_base,
                "train_mean": train_mean_base,
                "version": MODEL_VERSION,
                "type": "base",
                # training target metadata
                "target_norm": True,
                "demeaned": True,
                "target_mean": float(target_mean)
            },
            "regime_models": base_payload.get("regime_models", {}),
            "ensemble_models": base_payload.get("ensemble_models", {}),
            "performance": base_payload.get("performance", {}),
            "performance_mae": base_payload.get("performance_mae", {})
        }
        models = {
            "base": models_payload["base"],
            "forest": {
                "model": None,
                "fallback_model": forest_fallback,
                "features": forest_payload["features"],
                "calibration": forest_payload.get("calibration", {}),
                "calibration_sig": forest_payload.get("calibration_sig", models_payload["base"].get("calibration_sig", {"Weak":0.5,"Medium":0.5,"Strong":0.5})),
                "meta": {
                    "resid_std": forest_resid,
                    "pred_bias": forest_bias,
                    "version": MODEL_VERSION,
                    "target_norm": True,
                    "demeaned": True,
                    "target_mean": float(target_mean)
                },
                "regime_models": forest_payload.get("regime_models", {}),
                "ensemble_models": forest_payload["ensemble_models"],
                "performance": forest_payload.get("performance", {}),
                "performance_mae": forest_payload.get("performance_mae", {})
            },
            "boost": {
                "model": None,
                "fallback_model": boost_fallback,
                "features": boost_payload["features"],
                "calibration": boost_payload.get("calibration", {}),
                "calibration_sig": boost_payload.get("calibration_sig", models_payload["base"].get("calibration_sig", {"Weak":0.5,"Medium":0.5,"Strong":0.5})),
                "meta": {
                    "resid_std": boost_resid,
                    "pred_bias": boost_bias,
                    "version": MODEL_VERSION,
                    "target_norm": True,
                    "demeaned": True,
                    "target_mean": float(target_mean)
                },
                "regime_models": boost_payload.get("regime_models", {}),
                "ensemble_models": boost_payload["ensemble_models"],
                "performance": boost_payload.get("performance", {}),
                "performance_mae": boost_payload.get("performance_mae", {})
            },
            "extra_trees": {
                "model": None,
                "fallback_model": extra_fallback,
                "features": extra_payload["features"],
                "calibration": extra_payload.get("calibration", {}),
                "calibration_sig": extra_payload.get("calibration_sig", models_payload["base"].get("calibration_sig", {"Weak":0.5,"Medium":0.5,"Strong":0.5})),
                "meta": {
                    "resid_std": extra_resid,
                    "pred_bias": extra_bias,
                    "version": MODEL_VERSION,
                    "target_norm": True,
                    "demeaned": True,
                    "target_mean": float(target_mean)
                },
                "regime_models": extra_payload.get("regime_models", {}),
                "ensemble_models": extra_payload["ensemble_models"],
                "performance": extra_payload.get("performance", {}),
                "performance_mae": extra_payload.get("performance_mae", {})
            }
        }
        # Multi-horizon training (store per-model horizon payloads)
        # Only needed for 4min models used in predict_4min alignment.
        for k in models.keys():
            models[k].setdefault("horizon_models", {})
        if mode == "4min" and train_horizons:
            try:
                horizons = [int(h) for h in HORIZON_MINUTES if int(h) > 0]
            except Exception:
                horizons = [4]
            for h in horizons:
                if int(h) == 4:
                    continue
                try:
                    df_h = build_features_for_training(df, predict_minutes=int(h), instrument_key=instr_key)
                except Exception:
                    df_h = None
                if df_h is None or len(df_h) < 200:
                    continue
                try:
                    df_h = df_h.replace([np.inf, -np.inf], np.nan).dropna()
                except Exception:
                    pass
                regimes_h = _compute_regimes(df_h)
                windows_h = _resolve_windows(df_h, TRAIN_WINDOWS, horizon_minutes=int(h))
                # scale ensemble size down for longer horizons to keep runtime reasonable
                ensemble_n = ENSEMBLE_BOOTSTRAP_N if int(h) <= 6 else max(6, ENSEMBLE_BOOTSTRAP_N // 2)
                base_h = _train_window_ensembles(df_h, regimes_h, windows_h, DEFAULT_FEATURES, _ridge_pipeline, "base_h", ensemble_n_override=ensemble_n, target_col=params['target_col'])
                forest_h = _train_window_ensembles(df_h, regimes_h, windows_h, forest_feats, lambda: _forest_ctor(), "forest_h", ensemble_n_override=ensemble_n, regime_detail_map=regime_detail_map, model_kind="forest", target_col=params['target_col'])
                boost_h = _train_window_ensembles(df_h, regimes_h, windows_h, boost_feats, lambda: _boost_ctor(), "boost_h", ensemble_n_override=ensemble_n, regime_detail_map=regime_detail_map, model_kind="boost", target_col=params['target_col'])
                extra_h = _train_window_ensembles(df_h, regimes_h, windows_h, extra_feats, lambda: _extra_trees_ctor(), "extra_h", ensemble_n_override=ensemble_n, regime_detail_map=regime_detail_map, model_kind="extra_trees", target_col=params['target_col'])
                models["base"]["horizon_models"][str(h)] = base_h
                models["forest"]["horizon_models"][str(h)] = forest_h
                models["boost"]["horizon_models"][str(h)] = boost_h
                models["extra_trees"]["horizon_models"][str(h)] = extra_h

        # Meta-model (stacked ensemble) trained on out-of-fold base predictions
        meta_payload = None
        try:
            tss = _ts_split(6)
            oof = {k: np.full(len(df_full), np.nan) for k in ["base", "forest", "boost", "extra_trees"]}
            y_meta = df_full["target_ret_norm_demean"]
            for train_idx, test_idx in tss.split(df_full):
                Xb = df_full.loc[df_full.index[train_idx], DEFAULT_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
                Xf = df_full.loc[df_full.index[train_idx], forest_feats].replace([np.inf, -np.inf], np.nan).fillna(0.0)
                Xg = df_full.loc[df_full.index[train_idx], boost_feats].replace([np.inf, -np.inf], np.nan).fillna(0.0)
                Xe = df_full.loc[df_full.index[train_idx], extra_feats].replace([np.inf, -np.inf], np.nan).fillna(0.0)
                y_tr = y_meta.iloc[train_idx]
                # fit base models
                try:
                    m_base = _ridge_pipeline()
                    m_base.fit(Xb, y_tr)
                    Xb_test = df_full.loc[df_full.index[test_idx], DEFAULT_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
                    oof["base"][test_idx] = m_base.predict(Xb_test)
                except Exception:
                    pass
                try:
                    m_f = _forest_ctor(seed=RANDOM_STATE)
                    m_f.fit(Xf, y_tr)
                    Xf_test = df_full.loc[df_full.index[test_idx], forest_feats].replace([np.inf, -np.inf], np.nan).fillna(0.0)
                    oof["forest"][test_idx] = m_f.predict(Xf_test)
                except Exception:
                    pass
                try:
                    m_b = _boost_ctor(seed=RANDOM_STATE)
                    m_b.fit(Xg, y_tr)
                    Xg_test = df_full.loc[df_full.index[test_idx], boost_feats].replace([np.inf, -np.inf], np.nan).fillna(0.0)
                    oof["boost"][test_idx] = m_b.predict(Xg_test)
                except Exception:
                    pass
                try:
                    m_e = _extra_trees_ctor(seed=RANDOM_STATE)
                    m_e.fit(Xe, y_tr)
                    Xe_test = df_full.loc[df_full.index[test_idx], extra_feats].replace([np.inf, -np.inf], np.nan).fillna(0.0)
                    oof["extra_trees"][test_idx] = m_e.predict(Xe_test)
                except Exception:
                    pass
            meta_df = pd.DataFrame(oof, index=df_full.index)
            meta_df["ewm_vol"] = df_full.get("ewm_vol", 0.0)
            meta_df["trend_strength"] = df_full.get("trend_strength", 0.0)
            meta_df = meta_df.dropna()
            y_meta = y_meta.loc[meta_df.index]
            if len(meta_df) > 200:
                meta_model = _ridge_pipeline()
                meta_model.fit(meta_df, y_meta)
                preds_meta = meta_model.predict(meta_df)
                meta_resid = float(np.std(y_meta - preds_meta))
                meta_payload = {
                    "model": meta_model,
                    "features": list(meta_df.columns),
                    "meta": {
                        "resid_std": max(meta_resid, 1e-6),
                        "version": MODEL_VERSION,
                        "target_norm": True,
                        "demeaned": True,
                        "target_mean": float(target_mean)
                    }
                }
        except Exception:
            logger.exception("Meta-model training failed")
        if mode == "4min":
            try:
                if regime_payload is None:
                    regime_payload = train_regime(df_full)
            except Exception:
                regime_payload = None
        # Annotate model metadata with mode info for inference guards
        try:
            for mk in ["base", "forest", "boost", "extra_trees"]:
                meta = models.get(mk, {}).get("meta", {})
                meta["mode"] = mode
                if mode == "weekend":
                    meta["weekend_horizon"] = "per_session"
                if mode in ("daily_high", "daily_low"):
                    meta["daily_horizon"] = "per_session"
        except Exception:
            pass
        # Persist models & regime payload
        try:
            # Instrument-scoped model files (avoid cross-instrument overwrite)
            _atomic_joblib_dump(models["base"], _model_path("base", mode, instr_key))
            _atomic_joblib_dump(models["forest"], _model_path("forest", mode, instr_key))
            _atomic_joblib_dump(models["boost"], _model_path("boost", mode, instr_key))
            _atomic_joblib_dump(models["extra_trees"], _model_path("extra_trees", mode, instr_key))

            # Optional backward-compatible legacy filenames for DEFAULT instrument only
            if instr_key == DEFAULT_INSTRUMENT and SAVE_LEGACY_MODEL_FILES:
                try:
                    _atomic_joblib_dump(models["base"], os.path.join(MODEL_DIR, f"base_{mode}.pkl"))
                    _atomic_joblib_dump(models["forest"], os.path.join(MODEL_DIR, f"forest_{mode}.pkl"))
                    _atomic_joblib_dump(models["boost"], os.path.join(MODEL_DIR, f"boost_{mode}.pkl"))
                    _atomic_joblib_dump(models["extra_trees"], os.path.join(MODEL_DIR, f"extra_trees_{mode}.pkl"))
                    if mode == "4min":
                        _atomic_joblib_dump(models["base"], os.path.join(MODEL_DIR, BASE_MODEL_FILE))
                        _atomic_joblib_dump(models["forest"], os.path.join(MODEL_DIR, FOREST_MODEL_FILE))
                        _atomic_joblib_dump(models["boost"], os.path.join(MODEL_DIR, BOOST_MODEL_FILE))
                        _atomic_joblib_dump(models["extra_trees"], os.path.join(MODEL_DIR, EXTRA_TREES_MODEL_FILE))
                except Exception:
                    logger.exception("Failed to persist legacy model filenames")

            if meta_payload is not None:
                meta_path = _model_path("meta", "4min", instr_key) if mode == "4min" else os.path.join(MODEL_DIR, META_MODEL_FILE)
                _atomic_joblib_dump(meta_payload, meta_path)
            if mode == "4min" and regime_payload is not None:
                regime_path = _model_path("regime", "4min", instr_key)
                _atomic_joblib_dump(regime_payload, regime_path)
                # optional legacy regime path for default instrument
                if instr_key == DEFAULT_INSTRUMENT and SAVE_LEGACY_MODEL_FILES:
                    _atomic_joblib_dump(regime_payload, os.path.join(MODEL_DIR, REGIME_MODEL_FILE))
        except Exception:
            logger.exception("Failed to persist models")

        if meta_payload is not None:
            models["meta"] = meta_payload
        models_last = models
        # continue to next mode
        try:
            if progress_cb:
                progress_cb(f"Finished {mode} models.")
        except Exception:
            pass

    # end for mode loop

    # return last trained models (or empty dict if none)
    return models_last if models_last is not None else {}

# ----------------------------
# Backtesting / validation
# ----------------------------
def walk_forward_validate(df, n_splits=10, include_transaction_costs=True, instrument_key: str = None):
    """ Anchored walk-forward validation that uses expanding training windows and tests successive future windows.
    Operates on the last MAX_HISTORY_DAYS slice of the price series by default. """
    try:
        cutoff = pd.to_datetime(df.index.max()) - np.timedelta64(int(MAX_HISTORY_DAYS), "D")
        df = df[df.index >= cutoff]
    except Exception:
        pass
    DF = build_features_for_training(df, instrument_key=instrument_key or DEFAULT_INSTRUMENT)
    if DF is None or DF.empty:
        return {"folds": 0, "mae_base": [], "mae_forest": [], "mae_boost": [], "mae_extra_trees": [], "net_returns": []}
    y = DF["target_ret_norm_demean"] if "target_ret_norm_demean" in DF.columns else DF["target_ret_norm"]
    y_real = DF["target_ret"] if "target_ret" in DF.columns else y
    feats_base = DEFAULT_FEATURES
    feats_forest = DEFAULT_FEATURES + ["ret_2", "ret_4", "mom_3"]
    feats_boost = DEFAULT_FEATURES + ["ret_2", "ret_4"]
    feats_extra = DEFAULT_FEATURES + ["ret_2", "ret_4", "mom_3", "ema_5", "ema_20", "trend_strength", "vol_rolling_skew", "vol_rolling_kurt"]
    X_base = DF[feats_base].fillna(0.0)
    X_forest = DF[feats_forest].fillna(0.0)
    X_boost = DF[feats_boost].fillna(0.0)
    X_extra = DF[feats_extra].fillna(0.0)
    n = len(X_base)
    if n < (n_splits + 1) * 10:
        return {"folds": 0, "mae_base": [], "mae_forest": [], "mae_boost": [], "mae_extra_trees": [], "net_returns": []}
    test_size = max(1, n // (n_splits + 1))
    gap = max(0, int(TS_SPLIT_GAP_BARS))
    mae_base = []
    mae_forest = []
    mae_boost = []
    mae_extra_trees = []
    net_returns = []
    directional_acc = []
    sharpes = []
    active_fraction = []

    def _optimize_signal_threshold(pred_train, y_train_arr, default_thr: float, costs: float) -> float:
        try:
            p = np.asarray(pred_train, dtype=float)
            yv = np.asarray(y_train_arr, dtype=float)
            mask = np.isfinite(p) & np.isfinite(yv)
            p = p[mask]
            yv = yv[mask]
            if p.size < 40:
                return float(max(0.0, default_thr))
            abs_p = np.abs(p)
            q_levels = [0.0, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.92, 0.97, 0.985, 0.995]
            cand = {float(max(0.0, default_thr))}
            for q in q_levels:
                try:
                    cand.add(float(np.quantile(abs_p, q)))
                except Exception:
                    continue
            cand = sorted([x for x in cand if np.isfinite(x) and x >= 0.0])
            if not cand:
                return float(max(0.0, default_thr))
            best_t = cand[0]
            best_score = -np.inf
            for t in cand:
                pos = np.sign(p)
                pos[np.abs(p) < t] = 0.0
                gross = pos * yv
                trades = int(np.nansum(np.abs(np.diff(np.concatenate(([0.0], pos))))))
                score = float(np.nansum(gross) - costs * trades)
                active = float(np.mean(pos != 0))
                turnover = float(trades) / max(1.0, float(len(pos)))
                if np.any(pos != 0):
                    dir_acc = float(np.mean(np.sign(p[pos != 0]) == np.sign(yv[pos != 0])))
                else:
                    dir_acc = 0.5
                score_per_bar = score / max(1.0, float(len(yv)))
                # Prefer thresholds that are profitable after costs, directionally reliable,
                # and not unrealistically always-on.
                score_adj = (
                    score_per_bar
                    + (0.0005 * (dir_acc - 0.5))
                    - (0.0008 * max(0.0, active - 0.35))
                    - (0.0002 * max(0.0, 0.03 - active))
                    - (0.00035 * turnover)
                )
                if score_adj > best_score:
                    best_score = score_adj
                    best_t = float(t)
            return float(max(0.0, best_t))
        except Exception:
            return float(max(0.0, default_thr))

    def _strategy_stats(pred_vals, y_vals, thr: float, costs: float, flip_sign: bool = False):
        p = np.asarray(pred_vals, dtype=float).copy()
        yv = np.asarray(y_vals, dtype=float).copy()
        if flip_sign:
            p = -p
        pos = np.sign(p)
        if thr > 0:
            pos[np.abs(p) < thr] = 0.0
        gross = pos * yv
        trades = int(np.nansum(np.abs(np.diff(np.concatenate(([0.0], pos))))))
        net = float(np.nansum(gross) - costs * trades)
        active = float(np.mean(pos != 0)) if len(pos) else 0.0
        if np.any(pos != 0):
            dacc = float(np.mean(np.sign(p[pos != 0]) == np.sign(yv[pos != 0])))
        else:
            dacc = 0.5
        score = (
            (net / max(1.0, float(len(yv))))
            + (0.00045 * (dacc - 0.5))
            - (0.00080 * max(0.0, active - 0.35))
            - (0.00020 * max(0.0, 0.03 - active))
        )
        return {
            "pos": pos,
            "net": net,
            "dir_acc": dacc,
            "active": active,
            "score": score,
            "trades": trades,
        }

    for i in range(1, n_splits + 1):
        train_end = test_size * i
        train_end_purged = max(1, train_end - gap)
        test_start = train_end
        test_end = min(n, test_start + test_size)
        if train_end_purged < 2 or test_start >= n:
            continue
        Xb_train, Xb_test = X_base.iloc[:train_end_purged], X_base.iloc[test_start:test_end]
        Xf_train, Xf_test = X_forest.iloc[:train_end_purged], X_forest.iloc[test_start:test_end]
        Xg_train, Xg_test = X_boost.iloc[:train_end_purged], X_boost.iloc[test_start:test_end]
        Xe_train, Xe_test = X_extra.iloc[:train_end_purged], X_extra.iloc[test_start:test_end]
        y_train, y_test = y.iloc[:train_end_purged], y.iloc[test_start:test_end]
        y_train_real, y_test_real = y_real.iloc[:train_end_purged], y_real.iloc[test_start:test_end]
        pipe_base = _ridge_pipeline()
        # use recency weighting in training fold
        try:
            sw_b = _compute_recency_weights(Xb_train.index, half_life_days=7.0)
            pipe_base.fit(Xb_train, y_train, sample_weight=sw_b)
        except Exception:
            pipe_base.fit(Xb_train, y_train)
        pred_base = pipe_base.predict(Xb_test)
        pred_base_train = pipe_base.predict(Xb_train)
        mae_base.append(float(mean_absolute_error(y_test, pred_base)))
        rf = _forest_ctor(seed=RANDOM_STATE)
        try:
            rf.fit(Xf_train, y_train)
            pred_rf = rf.predict(Xf_test)
            pred_rf_train = rf.predict(Xf_train)
            mae_forest.append(float(mean_absolute_error(y_test, pred_rf)))
        except Exception:
            pred_rf = np.zeros(len(Xb_test))
            pred_rf_train = np.zeros(len(Xb_train))
            mae_forest.append(float("nan"))
        boost = _boost_ctor(seed=RANDOM_STATE)
        try:
            boost.fit(Xg_train, y_train)
            pred_boost = boost.predict(Xg_test)
            pred_boost_train = boost.predict(Xg_train)
            mae_boost.append(float(mean_absolute_error(y_test, pred_boost)))
        except Exception:
            pred_boost = np.zeros(len(Xb_test))
            pred_boost_train = np.zeros(len(Xb_train))
            mae_boost.append(float("nan"))
        extra = _extra_trees_ctor(seed=RANDOM_STATE)
        try:
            extra.fit(Xe_train, y_train)
            pred_extra = extra.predict(Xe_test)
            pred_extra_train = extra.predict(Xe_train)
            mae_extra_trees.append(float(mean_absolute_error(y_test, pred_extra)))
        except Exception:
            pred_extra = np.zeros(len(Xb_test))
            pred_extra_train = np.zeros(len(Xb_train))
            mae_extra_trees.append(float("nan"))
        # Convert normalized predictions back to raw returns for PnL/cost evaluation.
        try:
            vol_series = pd.to_numeric(DF.get("ewm_vol", pd.Series(0.001, index=DF.index)), errors="coerce").fillna(0.001)
            vol_train = np.maximum(vol_series.iloc[:train_end_purged].values.astype(float), 1e-6)
            vol_test = np.maximum(vol_series.iloc[test_start:test_end].values.astype(float), 1e-6)
        except Exception:
            vol_train = np.ones(len(Xb_train), dtype=float)
            vol_test = np.ones(len(Xb_test), dtype=float)
        pred_base_raw = np.asarray(pred_base, dtype=float) * vol_test
        pred_rf_raw = np.asarray(pred_rf, dtype=float) * vol_test
        pred_boost_raw = np.asarray(pred_boost, dtype=float) * vol_test
        pred_extra_raw = np.asarray(pred_extra, dtype=float) * vol_test
        pred_base_train_raw = np.asarray(pred_base_train, dtype=float) * vol_train
        pred_rf_train_raw = np.asarray(pred_rf_train, dtype=float) * vol_train
        pred_boost_train_raw = np.asarray(pred_boost_train, dtype=float) * vol_train
        pred_extra_train_raw = np.asarray(pred_extra_train, dtype=float) * vol_train

        stack_test = np.vstack([pred_base_raw, pred_rf_raw, pred_boost_raw, pred_extra_raw])
        stack_train = np.vstack([pred_base_train_raw, pred_rf_train_raw, pred_boost_train_raw, pred_extra_train_raw])
        preds_mean = (0.7 * np.mean(stack_test, axis=0)) + (0.3 * np.median(stack_test, axis=0))
        preds_train_mean = (0.7 * np.mean(stack_train, axis=0)) + (0.3 * np.median(stack_train, axis=0))
        realized = y_test_real.values
        if include_transaction_costs:
            costs = TRANSACTION_COST_PER_TRADE + SLIPPAGE_PCT
        else:
            costs = 0.0
        try:
            sig_ref = float(np.std(y_train_real))
        except Exception:
            sig_ref = 0.0
        try:
            sig_mult = float(BACKTEST_SIGNAL_STD_MULT)
        except Exception:
            sig_mult = 0.0
        sig_thr_default = max(0.0, sig_mult * max(0.0, sig_ref))
        # Meta-selection across signal families (ensemble/momentum/contrarian) on train fold only.
        mom1 = pd.to_numeric(DF.get("ret_1", pd.Series(0.0, index=DF.index)), errors="coerce").fillna(0.0)
        mom6 = pd.to_numeric(DF.get("ret_6", pd.Series(0.0, index=DF.index)), errors="coerce").fillna(0.0)
        cand_train = {
            "ensemble": np.asarray(preds_train_mean, dtype=float),
            "ensemble_inv": -np.asarray(preds_train_mean, dtype=float),
            "mom1": np.asarray(mom1.iloc[:train_end_purged].values, dtype=float),
            "mom1_inv": -np.asarray(mom1.iloc[:train_end_purged].values, dtype=float),
            "mom6": np.asarray(mom6.iloc[:train_end_purged].values, dtype=float),
            "mom6_inv": -np.asarray(mom6.iloc[:train_end_purged].values, dtype=float),
        }
        cand_test = {
            "ensemble": np.asarray(preds_mean, dtype=float),
            "ensemble_inv": -np.asarray(preds_mean, dtype=float),
            "mom1": np.asarray(mom1.iloc[test_start:test_end].values, dtype=float),
            "mom1_inv": -np.asarray(mom1.iloc[test_start:test_end].values, dtype=float),
            "mom6": np.asarray(mom6.iloc[test_start:test_end].values, dtype=float),
            "mom6_inv": -np.asarray(mom6.iloc[test_start:test_end].values, dtype=float),
        }
        best_name = "ensemble"
        best_thr = sig_thr_default
        best_score = -np.inf
        for name, tr_sig in cand_train.items():
            try:
                tr_sig = np.asarray(tr_sig, dtype=float)
                if tr_sig.size != len(y_train_real):
                    continue
                thr = _optimize_signal_threshold(tr_sig, y_train_real.values, sig_thr_default, costs)
                st = _strategy_stats(tr_sig, y_train_real.values, thr, costs, flip_sign=False)
                if st["score"] > best_score:
                    best_score = float(st["score"])
                    best_name = name
                    best_thr = float(thr)
            except Exception:
                continue
        st_test = _strategy_stats(cand_test.get(best_name, preds_mean), realized, best_thr, costs, flip_sign=False)
        pos = st_test["pos"]
        try:
            active_fraction.append(float(st_test["active"]))
        except Exception:
            active_fraction.append(float("nan"))
        # directional accuracy on active (non-zero) positions
        try:
            dir_acc = float(st_test["dir_acc"])
            directional_acc.append(dir_acc)
        except Exception:
            directional_acc.append(float("nan"))
        net_return = float(st_test["net"])
        net_returns.append(net_return)
        try:
            gross = pos * realized
            if np.std(gross) > 0:
                sharpes.append(float(np.mean(gross) / np.std(gross)))
            else:
                sharpes.append(float("nan"))
        except Exception:
            sharpes.append(float("nan"))
    return {
        "folds": len(mae_base),
        "mae_base": mae_base,
        "mae_forest": mae_forest,
        "mae_boost": mae_boost,
        "mae_extra_trees": mae_extra_trees,
        "net_returns": net_returns,
        "directional_acc": directional_acc,
        "sharpe_like": sharpes,
        "active_fraction": active_fraction
    }
