# config.py — central configuration for LightningTrader4000
# Extended with training windows, ensemble & backtest params, transaction-cost modeling,
# and model versioning. Use environment variables to override defaults.

import os
from pathlib import Path
from datetime import datetime

# ---------------------------
# Basic paths & instrument defaults
# ---------------------------
DATA_DIR = os.environ.get("DATA_DIR", "data")
MODEL_DIR = os.environ.get("MODEL_DIR", "models")
Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

# Legacy single-symbol fallback (used by some modules when INSTRUMENTS key missing)
SYMBOL = os.environ.get("SYMBOL", "QQQ")
DEFAULT_INSTRUMENT = os.environ.get("DEFAULT_INSTRUMENT", SYMBOL)

# Frequency for bars (examples: "1m", "2m", "5m", "1h")
INTERVAL = os.environ.get("INTERVAL", "2m")

# Prediction / training windows
TRAIN_WINDOW_DAYS = int(os.environ.get("TRAIN_WINDOW_DAYS", "14"))  # legacy small window
MAX_HISTORY_DAYS = int(os.environ.get("MAX_HISTORY_DAYS", "720"))  # retention window used by data_fetch (aligns w/ long window)

# Extended "ensemble" windows (days) — we train short/mid/long models and ensemble them.
TRAIN_WINDOWS = {
    "short_days": int(os.environ.get("SHORT_TRAIN_DAYS", "45")),     # ~1.5 months
    "mid_days": int(os.environ.get("MID_TRAIN_DAYS", "180")),        # ~6 months
    "long_days": int(os.environ.get("LONG_TRAIN_DAYS", "720")),      # ~2 years
}

# Model filenames (kept legacy names that other modules import)
BASE_MODEL_FILE = os.environ.get("BASE_MODEL_FILE", "base_model.pkl")
FOREST_MODEL_FILE = os.environ.get("FOREST_MODEL_FILE", "forest_model.pkl")
BOOST_MODEL_FILE = os.environ.get("BOOST_MODEL_FILE", "boost_model.pkl")
EXTRA_TREES_MODEL_FILE = os.environ.get(
    "EXTRA_TREES_MODEL_FILE",
    os.environ.get("EXTRA_MODEL_FILE", "extra_trees_model.pkl")
)
REGIME_MODEL_FILE = os.environ.get("REGIME_MODEL_FILE", "regime_model.pkl")
META_MODEL_FILE = os.environ.get("META_MODEL_FILE", "meta_model.pkl")

# New horizon model filenames
DAILY_MODEL_FILE = os.environ.get("DAILY_MODEL_FILE", "daily_model.pkl")
WEEK_MODEL_FILE = os.environ.get("WEEK_MODEL_FILE", "week_model.pkl")

AUTO_PREDICT_SECONDS = int(os.environ.get("AUTO_PREDICT_SECONDS", "240"))

# Window selection (data-driven training windows)
AUTO_WINDOW_SELECTION = os.environ.get("AUTO_WINDOW_SELECTION", "true").lower() in ("1", "true", "yes")
WINDOW_SELECTION_SPLITS = int(os.environ.get("WINDOW_SELECTION_SPLITS", "4"))
# Purged/gapped split to reduce short-horizon leakage between train/test slices.
TS_SPLIT_GAP_BARS = int(os.environ.get("TS_SPLIT_GAP_BARS", "2"))

def _parse_days_list(val: str, default: list[int]) -> list[int]:
    try:
        parts = [p.strip() for p in str(val).split(",") if p.strip()]
        out = [int(p) for p in parts]
        return [d for d in out if d > 0] or default
    except Exception:
        return default

WINDOW_CANDIDATES_4MIN = _parse_days_list(os.environ.get("WINDOW_CANDIDATES_4MIN", "20,45,90,180,360"), [20, 45, 90, 180, 360])
WINDOW_CANDIDATES_DAILY = _parse_days_list(os.environ.get("WINDOW_CANDIDATES_DAILY", "30,60,90,180,360,720"), [30, 60, 90, 180, 360, 720])
WINDOW_CANDIDATES_WEEKEND = _parse_days_list(os.environ.get("WINDOW_CANDIDATES_WEEKEND", "60,120,180,360,720,1080"), [60, 120, 180, 360, 720, 1080])

# Randomness / model hyperparams
RANDOM_STATE = int(os.environ.get("RANDOM_STATE", "42"))
RIDGE_ALPHA = float(os.environ.get("RIDGE_ALPHA", "1.0"))
RF_N_ESTIMATORS = int(os.environ.get("RF_N_ESTIMATORS", "80"))
RF_MAX_DEPTH = int(os.environ.get("RF_MAX_DEPTH", "8"))
RF_MIN_SAMPLES_LEAF = int(os.environ.get("RF_MIN_SAMPLES_LEAF", "10"))
BOOST_MAX_ITER = int(os.environ.get("BOOST_MAX_ITER", "300"))
BOOST_MAX_DEPTH = int(os.environ.get("BOOST_MAX_DEPTH", "3"))
EXTRA_TREES_N_ESTIMATORS = int(os.environ.get("EXTRA_TREES_N_ESTIMATORS", os.environ.get("EXTRA_N_ESTIMATORS", "80")))
EXTRA_TREES_MAX_DEPTH = int(os.environ.get("EXTRA_TREES_MAX_DEPTH", os.environ.get("EXTRA_MAX_DEPTH", "8")))
EXTRA_TREES_MIN_SAMPLES_LEAF = int(os.environ.get("EXTRA_TREES_MIN_SAMPLES_LEAF", os.environ.get("EXTRA_MIN_SAMPLES_LEAF", "12")))

# Ensemble / bootstrap config for uncertainty estimation
ENSEMBLE_BOOTSTRAP_N = int(os.environ.get("ENSEMBLE_BOOTSTRAP_N", "6"))  # number of bootstrap members
ENSEMBLE_SUBSAMPLE_FRAC = float(os.environ.get("ENSEMBLE_SUBSAMPLE_FRAC", "0.8"))
# Inference-time cap for ensemble members (limits latency on large legacy artifacts).
MAX_ENSEMBLE_MEMBERS_PREDICT = int(os.environ.get("MAX_ENSEMBLE_MEMBERS_PREDICT", "8"))

# Data source choice
# NOTE: Synthetic data is disabled to avoid fake predictions.
USE_CONSTANT_DATA = False

# Synthetic constants (deprecated; retained for backward compatibility)
CONSTANT_PRICE = float(os.environ.get("CONSTANT_PRICE", "622.00"))
CONSTANT_RET_STD = float(os.environ.get("CONSTANT_RET_STD", "0.0005"))
CONSTANT_BOOTSTRAP_BARS = int(os.environ.get("CONSTANT_BOOTSTRAP_BARS", "60"))

# Alpaca (read from env)
ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")
ALPACA_PAPER = os.environ.get("ALPACA_PAPER", "true").lower() in ("1", "true", "yes")
ALPACA_LOOKBACK_DAYS = int(os.environ.get("ALPACA_LOOKBACK_DAYS", "60"))
YFINANCE_LOOKBACK_DAYS = int(os.environ.get("YFINANCE_LOOKBACK_DAYS", "30"))
YFINANCE_REFRESH_BUFFER_DAYS = int(os.environ.get("YFINANCE_REFRESH_BUFFER_DAYS", "3"))
DATA_PROVIDER = os.environ.get("DATA_PROVIDER", "alpaca")

# IBKR (futures)
IB_GATEWAY_HOST = os.environ.get("IB_GATEWAY_HOST", "127.0.0.1")
IB_GATEWAY_PORT = int(os.environ.get("IB_GATEWAY_PORT", "7497"))
IB_CLIENT_ID = int(os.environ.get("IB_CLIENT_ID", "1"))

# Transaction cost / slippage defaults used by backtests and simulated live trading
TRANSACTION_COST_PER_TRADE = float(os.environ.get("TRANSACTION_COST_PER_TRADE", "0.0003"))  # fraction (30bp)
SLIPPAGE_PCT = float(os.environ.get("SLIPPAGE_PCT", "0.0005"))  # fraction
TARGET_VOL = float(os.environ.get("TARGET_VOL", "0.004"))
# Backtest/live risk-off threshold: do not trade predictions below this multiple of train target std.
BACKTEST_SIGNAL_STD_MULT = float(os.environ.get("BACKTEST_SIGNAL_STD_MULT", "0.6"))

# Prediction logging (optional)
LOG_PREDICTIONS = os.environ.get("LOG_PREDICTIONS", "false").lower() in ("1", "true", "yes")
LIVE_CALIBRATION = os.environ.get("LIVE_CALIBRATION", "true").lower() in ("1", "true", "yes")
CALIBRATION_MAX_AGE_DAYS = int(os.environ.get("CALIBRATION_MAX_AGE_DAYS", "30"))
CALIBRATION_HALF_LIFE_DAYS = float(os.environ.get("CALIBRATION_HALF_LIFE_DAYS", "7"))
DRIFT_ACC_THRESHOLD = float(os.environ.get("DRIFT_ACC_THRESHOLD", "0.48"))
DRIFT_MIN_SAMPLES = int(os.environ.get("DRIFT_MIN_SAMPLES", "60"))
# Daily/weekly horizon fetch (disable remote to keep UI fast/offline)
ALLOW_REMOTE_DAILY = os.environ.get("ALLOW_REMOTE_DAILY", "false").lower() in ("1", "true", "yes")

# Retrain schedule (strings: 'daily', 'weekly', 'monthly', or integer days)
RETRAIN_SHORT = os.environ.get("RETRAIN_SHORT", "1")   # daily
RETRAIN_MID = os.environ.get("RETRAIN_MID", "7")       # weekly
RETRAIN_LONG = os.environ.get("RETRAIN_LONG", "30")    # monthly-ish

# Instruments registry (expand as needed)
INSTRUMENTS = {
    "QQQ": {
        "type": "equity",
        "alpaca": "QQQ",
        "yahoo": "QQQ",
        "tz": "America/New_York",
        # equity session (NY time)
        "session_open_minute": 9 * 60 + 30,
        "session_close_minute": 16 * 60,
        "tick_size": 0.01,
        # guardrails for short-horizon signals
        "vol_floor": 0.00025,
        "max_abs_ret_mult": 6.0,
        "stale_multiplier": 3.5,
        # During market-closed periods (overnight/weekend), allow older last bars without false stale alarms.
        "closed_max_stale_hours": 80.0,
        "target_vol": 0.004
    },
    "GCG26": {
        "type": "futures",
        "ibkr": {
            "symbol": "GC",
            "secType": "FUT",
            "exchange": "COMEX",
            "currency": "USD",
            # Seed month; auto-roll logic in data_fetch resolves this forward as needed.
            "lastTradeDateOrContractMonth": "202604"
        },
        # Auto-roll delivery month for live IBKR fetches.
        "auto_roll_contract": True,
        # Gold contract cycle months: Feb/Apr/Jun/Aug/Oct/Dec.
        "contract_month_cycle": [2, 4, 6, 8, 10, 12],
        # If already in the delivery month, roll to the next cycle month after this day.
        "roll_day_of_month": 20,
        "yahoo": "GC=F",
        "tz": "America/New_York",
        # CME Globex session (approx): 6:00pm -> 5:00pm NY, with a 1-hour maintenance break
        "session_open_minute": 18 * 60,
        "session_close_minute": 17 * 60,
        # gold futures tick is usually $0.10 (check contract specs if you change instruments)
        "tick_size": 0.1,
        # futures are more volatile; floor + risk guard adjusted
        "vol_floor": 0.00045,
        "max_abs_ret_mult": 7.5,
        "stale_multiplier": 2.0,
        "closed_max_stale_hours": 60.0,
        "target_vol": 0.0032,
        # week starts Sunday evening for Globex
        "week_start_day": 6
    }
}

# Model versioning
MODEL_VERSION = os.environ.get("MODEL_VERSION", datetime.utcnow().strftime("%Y%m%d%H%M%S"))
SAVE_LEGACY_MODEL_FILES = os.environ.get("SAVE_LEGACY_MODEL_FILES", "false").lower() in ("1", "true", "yes")

# Multi-horizon training (minutes)
def _parse_horizons(val: str):
    try:
        parts = [p.strip() for p in val.split(",") if p.strip()]
        out = [int(p) for p in parts]
        return [h for h in out if h > 0] or [4]
    except Exception:
        return [4]

HORIZON_MINUTES = _parse_horizons(os.environ.get("HORIZON_MINUTES", "4,10,30"))
