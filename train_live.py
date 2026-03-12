#train_live.py
import argparse
import logging
import os

import pandas as pd

from config import DEFAULT_INSTRUMENT, INSTRUMENTS, DATA_DIR
from data_fetch import update_price_data
from instrument_onboarding import ensure_runtime_instruments
from model_trainers import train_all_models

logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)

def _load_cached_prices(instrument_key: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"prices_{instrument_key}.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        return df.sort_index()
    except Exception:
        logger.exception("Failed to read cached prices")
        return pd.DataFrame()

def train_missing_models(instrument_key: str = None, online: bool = True, max_runtime_sec: float | None = None):
    """
    Load data, train all models, save to disk. Returns True if successful.
    """
    instrument = instrument_key or DEFAULT_INSTRUMENT
    try:
        ensure_runtime_instruments(force=False, auto_onboard=True)
        if online:
            df = update_price_data(instrument)
        else:
            df = _load_cached_prices(instrument)
        if df is None or df.empty:
            raise RuntimeError(f"No data available for {instrument}")
        # train_all_models expects raw price df; it will call feature builders internally.
        train_all_models(
            df,
            instrument_key=instrument,
            train_horizons=True,
            max_runtime_sec=max_runtime_sec
        )
        print(f"All models trained for {instrument} and saved to 'models/' folder.")
        return True
    except Exception as e:
        logger.exception("Error training models")
        print(f"Error training models for {instrument}: {e}")
        return False

def train_all_configured_instruments(online: bool = True, max_runtime_sec: float | None = None):
    ensure_runtime_instruments(force=False, auto_onboard=True)
    ok = True
    for instrument in INSTRUMENTS.keys():
        logger.info("Training instrument %s", instrument)
        success = train_missing_models(instrument, online=online, max_runtime_sec=max_runtime_sec)
        ok = ok and success
    return ok

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LightningTrader models")
    parser.add_argument(
        "--instrument",
        default="all",
        help="Instrument key, comma-separated list, or 'all' (default)."
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use cached CSV price data only (no live fetch)."
    )
    parser.add_argument(
        "--max-runtime-sec",
        type=float,
        default=None,
        help="Optional training time budget per mode."
    )
    args = parser.parse_args()

    online = not args.offline
    raw = str(args.instrument).strip()
    if raw.lower() == "all":
        success = train_all_configured_instruments(online=online, max_runtime_sec=args.max_runtime_sec)
    else:
        keys = [k.strip() for k in raw.split(",") if k.strip()]
        success = True
        for key in keys:
            if key not in INSTRUMENTS:
                print(f"Unknown instrument: {key}")
                success = False
                continue
            success = train_missing_models(key, online=online, max_runtime_sec=args.max_runtime_sec) and success
    raise SystemExit(0 if success else 2)
