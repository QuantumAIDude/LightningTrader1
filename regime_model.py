# regime_model.py — improved regime detection using GaussianMixture clustering
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import logging

logger = logging.getLogger(__name__)

# configuration
REGIME_N_COMPONENTS = 4  # default number of regimes to discover
REGIME_FEATURES = [
    "ret_1", "ret_3", "ret_6", "ret_12", "ret_24",
    "vol_6", "vol_24",
    "mom_6", "mom_24", "mom_48",
    "kf_slope", "trend_strength", "ewm_vol",
    "trend_z_24", "ret_accel_6_24", "vol_ratio_6_24",
    "vol_rolling_skew", "vol_rolling_kurt",
    "session_volatility", "session_ret_from_open", "atr_6",
    "tod_ret_mean", "wday_tod_ret_mean",
    "range_1", "roll_max_dist", "roll_min_dist", "dollar_vol_z", "cusum_intensity"
]

# helper to parse interval (avoid importing features to prevent circular imports)
def _interval_minutes(interval: str) -> int:
    interval = str(interval).strip().lower()
    if interval.endswith("m"):
        return int(interval[:-1])
    if interval.endswith("h"):
        return int(interval[:-1]) * 60
    try:
        return int(interval)
    except Exception:
        return 1

# Build regime features from price data (keeps same features as before but robustifies missing cols)
def build_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_index()
    if "ret_1" not in df.columns:
        df["ret_1"] = df["close"].pct_change()
    if "ret_3" not in df.columns:
        df["ret_3"] = df["close"].pct_change(3).fillna(0.0)
    if "vol_6" not in df.columns:
        df["vol_6"] = df["ret_1"].rolling(6).std().fillna(0.0)
    if "ret_6" not in df.columns:
        df["ret_6"] = df["close"].pct_change(6).fillna(0.0)
    if "ret_12" not in df.columns:
        df["ret_12"] = df["close"].pct_change(12).fillna(0.0)
    if "ret_24" not in df.columns:
        df["ret_24"] = df["close"].pct_change(24).fillna(0.0)
    if "vol_24" not in df.columns:
        df["vol_24"] = df["ret_1"].rolling(24).std().fillna(0.0)
    if "mom_6" not in df.columns:
        df["mom_6"] = df["close"] - df["close"].shift(6)
    if "mom_24" not in df.columns:
        df["mom_24"] = df["close"] - df["close"].shift(24)
    if "mom_48" not in df.columns:
        df["mom_48"] = df["close"] - df["close"].shift(48)

    df["kf_slope"] = df.get("kf_slope", df["ret_1"] * 0.0).fillna(0.0)
    df["trend_strength"] = df.get("trend_strength", df["ret_1"] * 0.0).fillna(0.0)
    df["ewm_vol"] = df.get("ewm_vol", df["ret_1"].rolling(20).std()).fillna(0.0)
    df["trend_z_24"] = (
        df["ret_1"].rolling(24, min_periods=4).mean() /
        (df["ret_1"].rolling(24, min_periods=4).std().replace(0, np.nan))
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["ret_accel_6_24"] = (df.get("ret_6", 0.0) - df.get("ret_24", 0.0)).fillna(0.0)
    df["vol_ratio_6_24"] = (df.get("vol_6", 0.0) / (df.get("vol_24", 0.0).replace(0, np.nan))).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    df["vol_rolling_skew"] = df.get("vol_rolling_skew", df["ret_1"].rolling(24).skew()).fillna(0.0)
    df["vol_rolling_kurt"] = df.get("vol_rolling_kurt", df["ret_1"].rolling(24).kurt()).fillna(0.0)
    df["atr_6"] = df.get("atr_6", df["close"].diff().abs().rolling(6).mean()).fillna(0.0)
    if "session_volatility" not in df.columns:
        df["session_volatility"] = df["ret_1"].rolling(20).std().fillna(0.0)
    if "session_ret_from_open" not in df.columns:
        df["session_ret_from_open"] = 0.0
    if "tod_ret_mean" not in df.columns:
        df["tod_ret_mean"] = 0.0
    if "wday_tod_ret_mean" not in df.columns:
        df["wday_tod_ret_mean"] = 0.0
    if "range_1" not in df.columns:
        high = df["high"] if "high" in df.columns else df["close"]
        low = df["low"] if "low" in df.columns else df["close"]
        df["range_1"] = ((high.shift(1) - low.shift(1)) / (df["close"].shift(1).replace(0, 1e-9))).fillna(0.0)
    if "roll_max_dist" not in df.columns or "roll_min_dist" not in df.columns:
        try:
            from config import INTERVAL
            bar_min = _interval_minutes(INTERVAL)
        except Exception:
            bar_min = 2
        roll_window = max(12, int(120 / max(1, bar_min)))
        close_shift = df["close"].shift(1)
        roll_max = close_shift.rolling(roll_window, min_periods=1).max()
        roll_min = close_shift.rolling(roll_window, min_periods=1).min()
        df["roll_max_dist"] = (close_shift - roll_max) / (roll_max.replace(0, 1e-9))
        df["roll_min_dist"] = (close_shift - roll_min) / (roll_min.replace(0, 1e-9))
    if "dollar_vol_z" not in df.columns:
        try:
            dv = (df["close"].shift(1).astype(float) * df.get("volume", pd.Series(0, index=df.index)).shift(1).astype(float))
            dv = pd.to_numeric(dv, errors="coerce")
            dv_mean = dv.rolling(20, min_periods=1).mean()
            dv_std = dv.rolling(20, min_periods=1).std().replace(0, pd.NA)
            df["dollar_vol_z"] = (dv - dv_mean) / dv_std
        except Exception:
            df["dollar_vol_z"] = 0.0
        df["dollar_vol_z"] = pd.to_numeric(df["dollar_vol_z"], errors="coerce").fillna(0.0)
    if "cusum_intensity" not in df.columns:
        df["cusum_intensity"] = df["ret_1"].abs().rolling(24, min_periods=1).mean().fillna(0.0)

    X = df[REGIME_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X

def extract_regime_pipeline(payload):
    """Return (scaler, gmm) from a payload that may store a dict or sklearn Pipeline."""
    if payload is None:
        return None, None
    p = payload.get("pipeline", payload) if isinstance(payload, dict) else payload
    if isinstance(p, dict):
        return p.get("scaler"), p.get("gmm")
    # sklearn Pipeline
    scaler = None
    gmm = None
    try:
        if hasattr(p, "named_steps"):
            scaler = p.named_steps.get("scaler")
            gmm = p.named_steps.get("gmm")
            if gmm is None or scaler is None:
                for _, step in p.named_steps.items():
                    if scaler is None and isinstance(step, StandardScaler):
                        scaler = step
                    if gmm is None and isinstance(step, GaussianMixture):
                        gmm = step
    except Exception:
        pass
    try:
        if hasattr(p, "steps"):
            for _, step in p.steps:
                if scaler is None and isinstance(step, StandardScaler):
                    scaler = step
                if gmm is None and isinstance(step, GaussianMixture):
                    gmm = step
    except Exception:
        pass
    return scaler, gmm

def _select_gmm_components(X, max_components=6, random_state=42):
    """Try fit GMM with REGIME_N_COMPONENTS. If it fails or BIC suggests a different number, pick best by BIC 2..max_components."""
    try:
        # first try default
        gmm = GaussianMixture(n_components=REGIME_N_COMPONENTS, covariance_type="full", random_state=random_state)
        gmm.fit(X)
        bic0 = gmm.bic(X)
    except Exception:
        gmm = None
        bic0 = np.inf
    best = {"gmm": gmm, "bic": bic0, "n": REGIME_N_COMPONENTS}
    # try alternatives if fit failed or to improve BIC
    for k in range(2, min(max_components, X.shape[0]) + 1):
        try:
            gm = GaussianMixture(n_components=k, covariance_type="full", random_state=random_state)
            gm.fit(X)
            bic = gm.bic(X)
            if bic < best["bic"]:
                best = {"gmm": gm, "bic": bic, "n": k}
        except Exception:
            continue
    return best["gmm"], best["n"]

def train_regime(df: pd.DataFrame, lookback_days: int = None):
    """
    Train a GaussianMixture to discover latent regimes.
    By default we use the last MAX_HISTORY_DAYS slice if available, otherwise whole df.
    Returns payload: {'pipeline': {'scaler': scaler, 'gmm': gmm}, 'map': mapping, 'gmm': gmm}
    """
    from config import MAX_HISTORY_DAYS
    try:
        if lookback_days is None:
            lookback_days = int(MAX_HISTORY_DAYS)
        cutoff = pd.to_datetime(df.index.max()) - np.timedelta64(int(lookback_days), "D")
        df_train = df[df.index >= cutoff].copy()
    except Exception:
        df_train = df.copy()

    X = build_regime_features(df_train)
    if X is None or X.empty:
        return {"pipeline": None, "map": {}, "gmm": None}

    scaler = StandardScaler()
    try:
        Xs = scaler.fit_transform(X)
    except Exception:
        # fallback: fit scaler on zeros
        scaler = StandardScaler()
        Xs = scaler.fit_transform(np.zeros((len(X), len(X.columns))))

    # Fit GMM with BIC-based fallback
    try:
        gmm, ncomp = _select_gmm_components(Xs, max_components=min(6, max(2, Xs.shape[0] // 10)), random_state=42)
        if gmm is None:
            # last-resort simple 2-component
            gmm = GaussianMixture(n_components=2, covariance_type="diag", random_state=42)
            gmm.fit(Xs)
    except Exception:
        try:
            gmm = GaussianMixture(n_components=2, covariance_type="diag", random_state=42)
            gmm.fit(Xs)
        except Exception:
            logger.exception("GMM training failed entirely")
            return {"pipeline": None, "map": {}, "gmm": None}

    # Build mapping from component -> readable label (Bullish/Neutral/Bearish) using original (unscaled) short-return means
    try:
        comp_assign = gmm.predict(Xs)
        # compute mean ret_1 + volatility per component on original data (unscaled)
        comp_means = []
        comp_vols = []
        for i in range(gmm.n_components):
            mask = (comp_assign == i)
            if mask.sum() == 0:
                comp_means.append(0.0)
                comp_vols.append(0.0)
            else:
                comp_means.append(float(np.nanmean(df_train["ret_1"].values[mask])))
                comp_vols.append(float(np.nanstd(df_train["ret_1"].values[mask])))
        overall_vol = float(np.nanstd(df_train["ret_1"].values)) if len(df_train) > 5 else 1e-6
        overall_vol = max(overall_vol, 1e-6)
        comp_map = {}
        comp_detail = {}
        # First-pass directional label with magnitude threshold tied to overall vol
        dir_labels = {}
        for i in range(gmm.n_components):
            mean_ret = comp_means[i] if i < len(comp_means) else 0.0
            if mean_ret > 0.25 * overall_vol:
                dir_labels[i] = "Bullish"
            elif mean_ret < -0.25 * overall_vol:
                dir_labels[i] = "Bearish"
            else:
                dir_labels[i] = "Neutral"
        # If all components are neutral, force a ranked mapping
        if all(lbl == "Neutral" for lbl in dir_labels.values()):
            order = sorted(range(gmm.n_components), key=lambda i: comp_means[i] if i < len(comp_means) else 0.0)
            if len(order) >= 2:
                dir_labels[order[0]] = "Bearish"
                dir_labels[order[-1]] = "Bullish"
            else:
                dir_labels[order[0]] = "Neutral"
        # Ensure at least one bullish and one bearish label
        if "Bullish" not in dir_labels.values():
            try:
                best = int(np.argmax(comp_means))
                dir_labels[best] = "Bullish"
            except Exception:
                pass
        if "Bearish" not in dir_labels.values():
            try:
                worst = int(np.argmin(comp_means))
                dir_labels[worst] = "Bearish"
            except Exception:
                pass
        for i in range(gmm.n_components):
            mean_ret = comp_means[i] if i < len(comp_means) else 0.0
            vol = comp_vols[i] if i < len(comp_vols) else 0.0
            label = dir_labels.get(i, "Neutral")
            # volatility bucket
            if vol > 1.3 * overall_vol:
                vol_lab = "HighVol"
            elif vol < 0.7 * overall_vol:
                vol_lab = "LowVol"
            else:
                vol_lab = "MidVol"
            comp_map[i] = label
            comp_detail[i] = f"{label}-{vol_lab}"
    except Exception:
        # default simple mapping
        comp_map = {i: f"R{i}" for i in range(gmm.n_components)}
        comp_detail = {i: f"R{i}" for i in range(gmm.n_components)}

    # store component stats for debugging and future calibration
    comp_stats = {}
    try:
        for i in range(gmm.n_components):
            comp_stats[i] = {
                "mean_ret_1": float(comp_means[i]) if i < len(comp_means) else 0.0,
                "vol_ret_1": float(comp_vols[i]) if i < len(comp_vols) else 0.0
            }
    except Exception:
        comp_stats = {}

    # Transition matrix for regime persistence
    transition = None
    try:
        ncomp = gmm.n_components
        counts = np.ones((ncomp, ncomp)) * 1e-3
        for i in range(1, len(comp_assign)):
            a = int(comp_assign[i - 1])
            b = int(comp_assign[i])
            if 0 <= a < ncomp and 0 <= b < ncomp:
                counts[a, b] += 1.0
        transition = counts / counts.sum(axis=1, keepdims=True)
    except Exception:
        transition = None

    pipeline = {
        "scaler": scaler,
        "gmm": gmm
    }
    return {
        "pipeline": pipeline,
        "map": comp_map,
        "detail": comp_detail,
        "stats": comp_stats,
        "transition": transition,
        "gmm": gmm,
        "features": list(X.columns),
    }

def predict_regime_probs(payload, df: pd.DataFrame, recent_look_hours: int = 6):
    """
    Given a regime payload and a dataframe, returns a dict of regime probabilities percentages.
    The payload is trained on the longer history; for the *current* regime probabilities we focus on a
    recent window (default ~recent_look_hours of trading). This makes the regime signal local and responsive.
    Output format:
      {"Bullish": float_pct, "Neutral": float_pct, "Bearish": float_pct,
       "dominant_component": "<idx>", "component_probs": [p0,p1,...]}
    """
    if df is None or df.empty:
        return {"Bullish": 0.0, "Neutral": 100.0, "Bearish": 0.0, "dominant_component": "0", "component_probs": [1.0] + [0.0]*3, "regime_conf": 0.0}

    # fallback heuristic if payload missing
    if payload is None:
        return fallback_regime_probs(df)

    try:
        scaler, gmm = extract_regime_pipeline(payload)
        if scaler is None or gmm is None:
            return fallback_regime_probs(df)
        # compute bars_per_hour from INTERVAL if available
        try:
            from config import INTERVAL
            bars_per_hour = max(1, int(round(60.0 / float(_interval_minutes(INTERVAL)))))
        except Exception:
            bars_per_hour = max(1, 60 // 1)
        recent_bars = int(max(1, recent_look_hours * bars_per_hour))

        df_recent = df.tail(recent_bars)
        if df_recent is None or df_recent.empty:
            df_recent = df.copy().tail(24)

        # Adaptive lookback: respond faster when short-horizon volatility/shift rises.
        try:
            ret_probe = pd.to_numeric(df_recent["close"], errors="coerce").pct_change().fillna(0.0)
            rv_short = float(ret_probe.tail(max(6, bars_per_hour)).std())
            rv_long = float(ret_probe.tail(max(24, bars_per_hour * 6)).std())
            if np.isfinite(rv_short) and np.isfinite(rv_long) and rv_short > 1.35 * max(rv_long, 1e-8):
                df_recent = df.tail(max(8, int(recent_bars * 0.55)))
        except Exception:
            pass

        Xr = build_regime_features(df_recent).fillna(0.0)
        # Backward/forward compatibility: align inference features with scaler fit schema.
        fit_features = None
        try:
            p_feats = payload.get("features") if isinstance(payload, dict) else None
            if isinstance(p_feats, (list, tuple)) and len(p_feats) > 0:
                fit_features = [str(c) for c in p_feats]
        except Exception:
            fit_features = None
        if fit_features is None:
            try:
                s_feats = getattr(scaler, "feature_names_in_", None)
                if s_feats is not None and len(s_feats) > 0:
                    fit_features = [str(c) for c in list(s_feats)]
            except Exception:
                fit_features = None
        if fit_features:
            for c in fit_features:
                if c not in Xr.columns:
                    Xr[c] = 0.0
            Xr = Xr[fit_features]
        else:
            try:
                n_expected = int(getattr(scaler, "n_features_in_", 0) or 0)
            except Exception:
                n_expected = 0
            if n_expected > 0 and Xr.shape[1] != n_expected:
                if Xr.shape[1] > n_expected:
                    Xr = Xr.iloc[:, :n_expected]
                else:
                    for i in range(n_expected - Xr.shape[1]):
                        Xr[f"_pad_{i}"] = 0.0
        Xs = scaler.transform(Xr)
        post = gmm.predict_proba(Xs)
        # Exponentially weighted posterior (recent bars matter more) for faster adaptation.
        try:
            n = int(post.shape[0])
            if n <= 1:
                probs = post.mean(axis=0)
                shock_score = 0.0
            else:
                close_probe = pd.to_numeric(df_recent["close"], errors="coerce").pct_change().fillna(0.0).values[-n:]
                base_scale = float(np.nanstd(close_probe)) + 1e-9
                recent_abs = float(np.nanmean(np.abs(close_probe[-max(3, n // 4):]))) if len(close_probe) else 0.0
                shock_score = max(0.0, recent_abs / base_scale)
                half_life = max(2.0, (0.28 * n) if shock_score > 1.35 else (0.55 * n))
                idx = np.arange(n - 1, -1, -1, dtype=float)
                w = np.exp(-np.log(2.0) * (idx / half_life))
                w = np.where(np.isfinite(w), w, 0.0)
                probs = np.average(post, axis=0, weights=w)
                # Blend with very latest posterior; increase blend during shocks.
                latest_w = 0.45 if shock_score > 1.35 else 0.25
                probs = (1.0 - latest_w) * probs + latest_w * post[-1]
        except Exception:
            probs = post.mean(axis=0)
            shock_score = 0.0
        # Transition smoothing based on last assignment
        try:
            last_comp = int(gmm.predict(Xs)[-1])
            transition = payload.get("transition", None)
            if transition is not None and last_comp < transition.shape[0]:
                trans_row = np.array(transition[last_comp], dtype=float)
                if np.isfinite(trans_row).all():
                    trans_w = 0.20 if shock_score > 1.35 else 0.30
                    probs = (1.0 - trans_w) * probs + trans_w * trans_row
        except Exception:
            pass

        # map mixture components into Bullish/Neutral/Bearish using payload map if available
        comp_map = payload.get("map", None)
        comp_detail = payload.get("detail", None)
        def _needs_relabel(cm: dict | None) -> bool:
            if not cm:
                return True
            vals = list(cm.values())
            valid = {"Bullish", "Neutral", "Bearish"}
            has_bull = "Bullish" in vals
            has_bear = "Bearish" in vals
            has_invalid = any(v not in valid for v in vals)
            return (not has_bull) or (not has_bear) or has_invalid

        if comp_map is None or _needs_relabel(comp_map):
            # build a mapping based on short-return means in recent window
            try:
                assignments = gmm.predict(Xs)
                # compute ret_1 on the fly if missing
                try:
                    ret_1 = df_recent["ret_1"]
                except Exception:
                    ret_1 = df_recent["close"].pct_change().fillna(0.0)
                comp_means = []
                comp_vols = []
                for i in range(gmm.n_components):
                    mask = (assignments == i)
                    if mask.sum() == 0:
                        comp_means.append(0.0)
                        comp_vols.append(0.0)
                    else:
                        comp_means.append(float(np.nanmean(ret_1.values[mask])))
                        comp_vols.append(float(np.nanstd(ret_1.values[mask])))
                overall_vol = float(np.nanstd(ret_1.values)) if len(ret_1) > 5 else 1e-6
                overall_vol = max(overall_vol, 1e-6)
                dir_labels = {}
                for i in range(gmm.n_components):
                    mean_ret = comp_means[i] if i < len(comp_means) else 0.0
                    if mean_ret > 0.25 * overall_vol:
                        dir_labels[i] = "Bullish"
                    elif mean_ret < -0.25 * overall_vol:
                        dir_labels[i] = "Bearish"
                    else:
                        dir_labels[i] = "Neutral"
                if all(lbl == "Neutral" for lbl in dir_labels.values()):
                    order = sorted(range(gmm.n_components), key=lambda i: comp_means[i] if i < len(comp_means) else 0.0)
                    if len(order) >= 2:
                        dir_labels[order[0]] = "Bearish"
                        dir_labels[order[-1]] = "Bullish"
                    elif len(order) == 1:
                        dir_labels[order[0]] = "Neutral"
                if "Bullish" not in dir_labels.values():
                    try:
                        best = int(np.argmax(comp_means))
                        dir_labels[best] = "Bullish"
                    except Exception:
                        pass
                if "Bearish" not in dir_labels.values():
                    try:
                        worst = int(np.argmin(comp_means))
                        dir_labels[worst] = "Bearish"
                    except Exception:
                        pass
                comp_labels = {}
                for i in range(gmm.n_components):
                    comp_labels[i] = dir_labels.get(i, "Neutral")
                comp_map = comp_labels
            except Exception:
                comp_map = {i: "Neutral" for i in range(gmm.n_components)}

        bullish_pct = 0.0
        neutral_pct = 0.0
        bearish_pct = 0.0
        for i, p in enumerate(probs):
            label = comp_map.get(i, "Neutral")
            if label == "Bullish":
                bullish_pct += p
            elif label == "Bearish":
                bearish_pct += p
            else:
                neutral_pct += p

        out = {
            "Bullish": round(float(bullish_pct * 100.0), 1),
            "Neutral": round(float(neutral_pct * 100.0), 1),
            "Bearish": round(float(bearish_pct * 100.0), 1)
        }

        # ensure sums to 100 (numerical safety)
        total = out["Bullish"] + out["Neutral"] + out["Bearish"]
        if abs(total - 100.0) > 0.1 and total > 0:
            s = total
            out = {k: round(float(v / s * 100.0), 1) for k, v in out.items()}

        dominant_comp = int(np.argmax(probs))
        out["dominant_component"] = str(int(dominant_comp))
        out["component_probs"] = [float(p) for p in probs]
        out["recent_window_bars"] = int(len(Xs))
        out["shock_score"] = float(round(shock_score, 4))
        try:
            p = np.clip(probs, 1e-9, 1.0)
            entropy = float(-np.sum(p * np.log(p)) / np.log(len(p)))
            out["regime_conf"] = float(max(0.0, min(1.0, 1.0 - entropy)))
        except Exception:
            out["regime_conf"] = 0.5
        if comp_map is not None:
            try:
                out["component_labels"] = [comp_map.get(i, "Neutral") for i in range(gmm.n_components)]
            except Exception:
                out["component_labels"] = []
        if comp_detail is not None:
            try:
                out["component_details"] = [comp_detail.get(i, "Neutral") for i in range(gmm.n_components)]
            except Exception:
                out["component_details"] = []

        return out
    except Exception:
        logger.exception("Regime predict failed")
        return fallback_regime_probs(df)

def fallback_regime_probs(df: pd.DataFrame) -> dict:
    out = {"Bullish": 0.0, "Neutral": 0.0, "Bearish": 0.0}
    if df is None or df.empty:
        return {"Bullish": 0.0, "Neutral": 100.0, "Bearish": 0.0, "dominant_component": "0", "component_probs": [1.0] + [0.0]*3, "regime_conf": 0.0}
    recent = df["close"].iloc[-24:] if len(df) >= 24 else df["close"]
    if recent.empty:
        return {"Bullish": 0.0, "Neutral": 100.0, "Bearish": 0.0, "dominant_component": "0", "component_probs": [1.0] + [0.0]*3, "regime_conf": 0.0}
    ret_6 = recent.pct_change(6).dropna()
    mean_short = float(ret_6.mean()) if not ret_6.empty else 0.0
    vol_short = float(ret_6.std()) if not ret_6.empty else 0.0
    score = mean_short / (vol_short + 1e-9)
    if score > 0.2:
        out["Bullish"] = 0.7
        out["Neutral"] = 0.25
        out["Bearish"] = 0.05
        dominant = "3"  # assume higher component
        comp_probs = [0.05, 0.1, 0.1, 0.75]
    elif score < -0.2:
        out["Bullish"] = 0.05
        out["Neutral"] = 0.25
        out["Bearish"] = 0.7
        dominant = "0"
        comp_probs = [0.75, 0.1, 0.1, 0.05]
    else:
        out["Bullish"] = 0.2
        out["Neutral"] = 0.6
        out["Bearish"] = 0.2
        dominant = "1"
        comp_probs = [0.2, 0.4, 0.3, 0.1]
    for k in out:
        out[k] = round(float(out[k] * 100.0), 1)
    out["dominant_component"] = dominant
    out["component_probs"] = comp_probs
    out["regime_conf"] = 0.3
    return out
