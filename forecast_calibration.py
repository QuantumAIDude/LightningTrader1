from __future__ import annotations

import numpy as np

from accuracy_windows import min_window_hit_rate, price_window_threshold_dollars, summarize_price_window_errors


def _to_float_array(values) -> np.ndarray:
    if values is None:
        return np.asarray([], dtype=float)
    try:
        arr = np.asarray(values, dtype=float)
    except Exception:
        try:
            arr = np.asarray(list(values), dtype=float)
        except Exception:
            return np.asarray([], dtype=float)
    return np.ravel(arr).astype(float, copy=False)


def _coerce_triplet(pred_returns, actual_returns, close_prices):
    pred = _to_float_array(pred_returns)
    actual = _to_float_array(actual_returns)
    close = _to_float_array(close_prices)
    n = min(pred.size, actual.size, close.size)
    if n <= 0:
        return (
            np.asarray([], dtype=float),
            np.asarray([], dtype=float),
            np.asarray([], dtype=float),
        )
    pred = pred[:n]
    actual = actual[:n]
    close = close[:n]
    mask = np.isfinite(pred) & np.isfinite(actual) & np.isfinite(close) & (close > 0.0)
    return pred[mask], actual[mask], close[mask]


def _cap_bias(bias, actual_returns):
    try:
        sigma = float(np.nanstd(actual_returns))
    except Exception:
        sigma = 0.0
    cap = max(0.00025, 0.50 * sigma)
    try:
        return float(max(-cap, min(cap, float(bias))))
    except Exception:
        return 0.0


def score_return_forecast(pred_returns, actual_returns, close_prices, horizon: str = "intraday",
                          instrument_key: str = None) -> dict:
    pred, actual, close = _coerce_triplet(pred_returns, actual_returns, close_prices)
    min_hit = float(min_window_hit_rate(horizon, instrument_key=instrument_key))
    if pred.size == 0:
        return {
            "n": 0,
            "mae_dollars": None,
            "rmse_dollars": None,
            "window_hit_rate": None,
            "window_ratio_mean": None,
            "window_threshold_mean_dollars": None,
            "min_window_hit_rate": min_hit,
            "objective_score": None,
        }
    actual_prices = close * (1.0 + actual)
    pred_prices = close * (1.0 + pred)
    metrics = summarize_price_window_errors(
        np.abs(pred_prices - actual_prices),
        actual_prices=actual_prices,
        horizon=horizon,
        instrument_key=instrument_key,
    )
    ratio = float(metrics.get("window_ratio_mean")) if metrics.get("window_ratio_mean") is not None else 9.0
    hit = float(metrics.get("window_hit_rate")) if metrics.get("window_hit_rate") is not None else 0.0
    mae = float(metrics.get("mae_dollars")) if metrics.get("mae_dollars") is not None else ratio
    threshold = float(metrics.get("window_threshold_mean_dollars")) if metrics.get("window_threshold_mean_dollars") is not None else 0.01
    hit_penalty = max(0.0, min_hit - hit)
    score = ratio + (1.50 * hit_penalty) + (0.15 * (mae / max(0.01, threshold)))
    metrics["min_window_hit_rate"] = min_hit
    metrics["objective_score"] = float(score)
    return metrics


def apply_return_calibration(pred_returns, calibration: dict | None):
    pred = _to_float_array(pred_returns)
    if pred.size == 0:
        return pred
    if not isinstance(calibration, dict):
        return pred
    try:
        scale = float(calibration.get("scale", 1.0))
    except Exception:
        scale = 1.0
    try:
        bias = float(calibration.get("bias", 0.0))
    except Exception:
        bias = 0.0
    if not np.isfinite(scale):
        scale = 1.0
    if not np.isfinite(bias):
        bias = 0.0
    return (pred * scale) + bias


def fit_return_calibration(pred_returns, actual_returns, close_prices, horizon: str = "intraday",
                           instrument_key: str = None, allow_flip: bool = True,
                           min_samples: int = 48) -> dict:
    pred, actual, close = _coerce_triplet(pred_returns, actual_returns, close_prices)
    baseline = score_return_forecast(
        np.zeros_like(actual),
        actual,
        close,
        horizon=horizon,
        instrument_key=instrument_key,
    )
    baseline_out = {
        "scale": 0.0,
        "bias": 0.0,
        "selected": "anchor_only",
        **baseline,
    }
    if pred.size < int(min_samples):
        baseline_out["reason"] = "insufficient_samples"
        return baseline_out

    try:
        design = np.column_stack([pred, np.ones_like(pred)])
        ls_scale, ls_bias = np.linalg.lstsq(design, actual, rcond=None)[0]
    except Exception:
        ls_scale, ls_bias = 1.0, 0.0
    scale_cap_lo = -1.50 if allow_flip else 0.0
    scale_cap_hi = 1.50
    try:
        ls_scale = float(max(scale_cap_lo, min(scale_cap_hi, float(ls_scale))))
    except Exception:
        ls_scale = 1.0
    ls_bias = _cap_bias(ls_bias, actual)
    scale_candidates = {
        0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.65, 0.80, 1.00, 1.20, ls_scale,
        0.5 * ls_scale, 0.75 * ls_scale, 1.25 * ls_scale,
    }
    if allow_flip:
        scale_candidates.update({-1.0, -0.75, -0.50, -0.35, -0.20, -0.10})
    scale_candidates = sorted(
        float(max(scale_cap_lo, min(scale_cap_hi, s)))
        for s in scale_candidates
        if np.isfinite(s)
    )

    best = dict(baseline_out)
    for scale in scale_candidates:
        scaled = pred * float(scale)
        bias_candidates = {
            0.0,
            ls_bias,
            _cap_bias(float(np.nanmedian(actual - scaled)), actual),
            _cap_bias(float(np.nanmean(actual - scaled)), actual),
        }
        for bias in sorted(bias_candidates):
            pred_adj = scaled + float(bias)
            metrics = score_return_forecast(
                pred_adj,
                actual,
                close,
                horizon=horizon,
                instrument_key=instrument_key,
            )
            score = metrics.get("objective_score")
            if score is None:
                continue
            if best.get("objective_score") is None or float(score) + 1e-9 < float(best["objective_score"]):
                best = {
                    "scale": float(scale),
                    "bias": float(bias),
                    "selected": "grid",
                    **metrics,
                }
    try:
        baseline_score = float(baseline_out.get("objective_score"))
        best_score = float(best.get("objective_score"))
    except Exception:
        baseline_score = None
        best_score = None
    if baseline_score is not None and best_score is not None and best.get("selected") != "anchor_only":
        if best_score > (baseline_score * 0.98):
            baseline_out["reason"] = "no_material_edge"
            return baseline_out
    return best


def apply_intraday_activity_gate(reference_prices, proxy_values, gate: dict | None, instrument_key: str = None):
    ref = _to_float_array(reference_prices)
    proxy = _to_float_array(proxy_values)
    n = min(ref.size, proxy.size)
    if n <= 0:
        return np.asarray([], dtype=bool)
    ref = ref[:n]
    proxy = np.maximum(np.abs(proxy[:n]), 1e-9)
    if not isinstance(gate, dict):
        return np.ones(n, dtype=bool)
    try:
        scale = float(gate.get("scale", 0.0))
    except Exception:
        scale = 0.0
    if not np.isfinite(scale) or scale <= 0.0:
        return np.ones(n, dtype=bool)
    base = np.maximum(np.abs(ref) * proxy, 1e-6)
    windows = np.asarray(
        [price_window_threshold_dollars("intraday", instrument_key=instrument_key, reference_price=px) for px in ref],
        dtype=float,
    )
    windows = np.where(np.isfinite(windows) & (windows > 0.0), windows, 0.01)
    pred_abs_move = scale * base
    return np.asarray(pred_abs_move <= windows, dtype=bool)


def fit_intraday_activity_gate(reference_prices, actual_prices, proxy_values, instrument_key: str = None,
                               min_active_floor: float = 0.05,
                               quantiles: tuple[float, ...] = (0.55, 0.60, 0.65, 0.70)) -> dict:
    ref = _to_float_array(reference_prices)
    actual = _to_float_array(actual_prices)
    proxy = _to_float_array(proxy_values)
    n = min(ref.size, actual.size, proxy.size)
    if n <= 0:
        return {"scale": None, "quantile": None, "selected": "disabled", "role": "quiet_bar_filter"}
    ref = ref[:n]
    actual = actual[:n]
    proxy = np.maximum(np.abs(proxy[:n]), 1e-9)
    mask = np.isfinite(ref) & np.isfinite(actual) & np.isfinite(proxy) & (ref > 0.0)
    ref = ref[mask]
    actual = actual[mask]
    proxy = proxy[mask]
    if ref.size < 40:
        return {"scale": None, "quantile": None, "selected": "insufficient_samples", "role": "quiet_bar_filter"}
    abs_move = np.abs(actual - ref)
    base = np.maximum(np.abs(ref) * proxy, 1e-6)
    ratio = abs_move / base
    best = None
    min_hit = float(min_window_hit_rate("intraday", instrument_key=instrument_key))
    for q in quantiles:
        try:
            scale = float(np.quantile(ratio, float(q)))
        except Exception:
            continue
        gate = {"scale": scale}
        active_mask = apply_intraday_activity_gate(ref, proxy, gate, instrument_key=instrument_key)
        active_frac = float(np.mean(active_mask)) if active_mask.size else 0.0
        if active_mask.sum() <= 0:
            continue
        metrics = summarize_price_window_errors(
            abs_move[active_mask],
            actual_prices=actual[active_mask],
            horizon="intraday",
            instrument_key=instrument_key,
        )
        score = float(metrics.get("window_ratio_mean") or 9.0)
        score += 1.5 * max(0.0, min_hit - float(metrics.get("window_hit_rate") or 0.0))
        score += 2.0 * max(0.0, float(min_active_floor) - active_frac)
        if best is None or score < best["objective_score"]:
            best = {
                "scale": float(scale),
                "quantile": float(q),
                "selected": "quantile",
                "role": "quiet_bar_filter",
                "objective_score": float(score),
                "active_fraction_train": float(active_frac),
                "window_ratio_mean": metrics.get("window_ratio_mean"),
                "window_hit_rate": metrics.get("window_hit_rate"),
            }
    return best or {"scale": None, "quantile": None, "selected": "disabled", "role": "quiet_bar_filter"}
