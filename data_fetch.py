# data_fetch.py — robust fetcher with logging and IB util safety
import os
import time
import logging
import threading
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

from config import (
    DATA_DIR, SYMBOL, INTERVAL, MAX_HISTORY_DAYS,
    ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_PAPER, DATA_PROVIDER,
    IB_GATEWAY_HOST, IB_GATEWAY_PORT, IB_CLIENT_ID, INSTRUMENTS,
    YFINANCE_LOOKBACK_DAYS, YFINANCE_REFRESH_BUFFER_DAYS
)

logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)

_DATA_LOCK = threading.RLock()

def _ensure_utc_index(df: pd.DataFrame, tz: str = "America/New_York") -> pd.DataFrame:
    if df is None or df.empty:
        return df
    try:
        idx = pd.to_datetime(df.index)
        if idx.tz is None:
            try:
                idx = idx.tz_localize(tz, ambiguous="infer", nonexistent="shift_forward")
            except Exception:
                idx = idx.tz_localize(tz)
        idx = idx.tz_convert("UTC")
        df.index = idx
    except Exception:
        logger.exception("Failed to localize index; leaving as-is")
    return df

def _data_file_for_instrument(instr_key: str, source: str | None = None) -> str:
    safe = instr_key.replace("/", "_").replace("\\", "_")
    os.makedirs(DATA_DIR, exist_ok=True)
    suffix = f"_{source}" if source else ""
    return os.path.join(DATA_DIR, f"prices_{safe}{suffix}.csv")

# Attempt to import Alpaca (non-fatal)
try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.historical import StockLatestQuote
    _ALPACA_AVAILABLE = True
except Exception:
    _ALPACA_AVAILABLE = False
    logger.debug("Alpaca SDK not available; falling back to yfinance where applicable.")

# Attempt to import ib_insync for IBKR futures (non-fatal)
_ib_util = None
try:
    from ib_insync import IB, Contract, util as ib_util
    _IB_AVAILABLE = True
    _ib_util = ib_util
except Exception:
    _IB_AVAILABLE = False
    logger.debug("ib_insync not available; IBKR paths will be disabled.")

def _interval_to_pandas_freq(interval: str) -> str:
    if interval.endswith("m"):
        minutes = int(interval[:-1])
        return f"{minutes}min"
    if interval.endswith("h"):
        hours = int(interval[:-1])
        return f"{hours}H"
    return interval

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

def _parse_contract_ym(value: str | None):
    s = str(value or "").strip()
    if len(s) < 6 or (not s[:6].isdigit()):
        return None
    y = int(s[:4])
    m = int(s[4:6])
    if m < 1 or m > 12:
        return None
    return y, m

def _next_cycle_contract_ym(year: int, month: int, cycle_months: list[int], strictly_after: bool = False):
    cycle = sorted({int(m) for m in cycle_months if 1 <= int(m) <= 12})
    if not cycle:
        cycle = [3, 6, 9, 12]
    for m in cycle:
        if (m > month) if strictly_after else (m >= month):
            return year, m
    return year + 1, cycle[0]

def _resolve_ibkr_contract_meta(ib_meta: dict, instr_meta: dict | None = None):
    """
    Resolve a usable IBKR futures contract month, optionally auto-rolling stale config.
    Adds lightweight metadata keys prefixed with '_' for diagnostics.
    """
    resolved = dict(ib_meta or {})
    instr_meta = instr_meta or {}
    configured = str(resolved.get("lastTradeDateOrContractMonth", "") or "")
    now = datetime.now(timezone.utc)
    auto_roll = bool(instr_meta.get("auto_roll_contract", True))
    cycle = instr_meta.get("contract_month_cycle", [3, 6, 9, 12])
    try:
        cycle = [int(x) for x in cycle]
    except Exception:
        cycle = [3, 6, 9, 12]
    roll_day = int(instr_meta.get("roll_day_of_month", 20))

    parsed = _parse_contract_ym(configured)
    if parsed is None:
        y, m = _next_cycle_contract_ym(now.year, now.month, cycle, strictly_after=False)
        resolved_month = f"{y}{m:02d}"
    else:
        y_cfg, m_cfg = parsed
        ym_cfg = int(f"{y_cfg}{m_cfg:02d}")
        ym_now = int(now.strftime("%Y%m"))
        y_res, m_res = y_cfg, m_cfg
        if auto_roll:
            if ym_cfg < ym_now:
                y_res, m_res = _next_cycle_contract_ym(now.year, now.month, cycle, strictly_after=False)
            elif ym_cfg == ym_now and int(now.day) >= roll_day:
                y_res, m_res = _next_cycle_contract_ym(y_cfg, m_cfg, cycle, strictly_after=True)
        resolved_month = f"{y_res}{m_res:02d}"

    resolved["lastTradeDateOrContractMonth"] = resolved_month
    resolved["_configured_contract_month"] = configured or None
    resolved["_resolved_contract_month"] = resolved_month
    resolved["_contract_auto_rolled"] = bool(configured and configured != resolved_month)
    return resolved

def fetch_recent_intraday_yfinance(
        symbol: str,
        retries=3,
        pause=3,
        tz: str = "America/New_York",
        lookback_days: int = None,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None
):
    import yfinance as yf
    df = pd.DataFrame()
    period = f"{int(lookback_days) if lookback_days else int(YFINANCE_LOOKBACK_DAYS)}d"
    for _ in range(retries):
        try:
            if start is not None:
                df = yf.download(
                    symbol,
                    start=start,
                    end=end,
                    interval=INTERVAL,
                    progress=False,
                    auto_adjust=False
                )
            else:
                df = yf.download(
                    symbol,
                    period=period,
                    interval=INTERVAL,
                    progress=False,
                    auto_adjust=False
                )
            if not df.empty:
                break
        except Exception:
            df = pd.DataFrame()
            logger.exception("yfinance download attempt failed")
        time.sleep(pause)

    if df.empty:
        return pd.DataFrame()

    # normalize column names and keep open/high/low/close/volume if present
    # yfinance sometimes returns MultiIndex with (field, ticker)
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df.columns = df.columns.get_level_values(0)
        except Exception:
            df.columns = ["_".join([str(p) for p in c if p is not None]) for c in df.columns]
    new_cols = []
    for c in df.columns:
        colname = str(c).lower().replace(" ", "_")
        # strip common ticker suffixes like _qqq
        if colname.endswith(f"_{symbol.lower()}"):
            colname = colname[: -(len(symbol) + 1)]
        # map any column containing known field to canonical name
        for field in ("open", "high", "low", "close", "adj_close", "volume"):
            parts = colname.split("_")
            if field in parts:
                colname = field
                break
        new_cols.append(colname)
    df.columns = new_cols

    # prefer 'close' column; but also try to keep open/high/low/volume
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        return pd.DataFrame()
    adj = next((c for c in numeric_cols if "adj" in c and "close" in c), None)
    if adj is not None and "close" not in df.columns:
        df = df.rename(columns={adj: "close"})
    if "close" not in df.columns:
        chosen = numeric_cols[0]
        df = df.rename(columns={chosen: "close"})

    # keep high/low/open if present; otherwise leave them absent
    cols_to_keep = ["open", "high", "low", "close", "volume"]
    present = [c for c in cols_to_keep if c in df.columns]
    df = df.loc[:, present]
    df = _ensure_utc_index(df, tz=tz)
    return df

def _try_fetch_latest_quotes_alpaca(client, symbol):
    try:
        from alpaca.data import StockDataClient
        from alpaca.data.requests import StockLatestQuoteRequest
        if hasattr(client, "get_stock_latest_quote"):
            q = client.get_stock_latest_quote(symbol)
            ts = pd.to_datetime(datetime.utcnow(), utc=True)
            return pd.DataFrame([{
                "timestamp": ts,
                "bid": q.bp,
                "ask": q.ap,
                "bid_size": getattr(q, "bs", np.nan),
                "ask_size": getattr(q, "as", np.nan)
            }]).set_index("timestamp")
    except Exception:
        logger.debug("Alpaca latest quote fetch not available or failed", exc_info=True)
    return pd.DataFrame()

def fetch_recent_intraday_alpaca(symbol: str, retries=2):
    if not _ALPACA_AVAILABLE:
        return pd.DataFrame()
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        return pd.DataFrame()
    try:
        client = StockHistoricalDataClient(api_key=ALPACA_API_KEY, secret_key=ALPACA_SECRET_KEY)
    except Exception:
        logger.exception("Alpaca client init failed")
        try:
            client = StockHistoricalDataClient(api_key=ALPACA_API_KEY, secret_key=ALPACA_SECRET_KEY)
        except Exception:
            logger.exception("Alpaca client retry failed")
            return pd.DataFrame()

    from config import ALPACA_LOOKBACK_DAYS
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=int(ALPACA_LOOKBACK_DAYS))
    try:
        req = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame.Minute,
            start=start,
            end=end
        )
        bars = client.get_stock_bars(req).df
    except Exception:
        logger.exception("Alpaca get_stock_bars failed")
        return pd.DataFrame()

    if bars is None or bars.empty:
        return pd.DataFrame()

    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.reset_index()
    else:
        bars = bars.reset_index().rename(columns={"index": "timestamp"})

    bars.columns = [c.lower() if isinstance(c, str) else c for c in bars.columns]
    if "symbol" in bars.columns:
        bars = bars[bars["symbol"] == symbol]
    # keep open/high/low/close/volume if present
    keep = [c for c in ["timestamp", "open", "high", "low", "close", "volume"] if c in bars.columns]
    if "timestamp" not in bars.columns or "close" not in bars.columns:
        return pd.DataFrame()

    df = bars[keep].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp")

    target_freq = _interval_to_pandas_freq(INTERVAL)
    if target_freq != "1min":
        try:
            # resample preserving high/low/open/close semantics
            agg_map = {}
            if "open" in df.columns: agg_map["open"] = "first"
            if "high" in df.columns: agg_map["high"] = "max"
            if "low" in df.columns: agg_map["low"] = "min"
            agg_map["close"] = "last"
            agg_map["volume"] = "sum"
            df = df.resample(target_freq, label="right", closed="right").agg(agg_map)
            df = df.dropna(subset=["close"])
        except Exception:
            logger.exception("Alpaca resample failed")
            pass

    df.index = pd.to_datetime(df.index, utc=True)

    # attach latest quote microstructure if available
    try:
        quote_df = _try_fetch_latest_quotes_alpaca(client, symbol)
        if not quote_df.empty:
            last_idx = df.index.max()
            q_ts = quote_df.index.max()
            if q_ts >= last_idx - pd.Timedelta(target_freq):
                df = df.reindex(df.index.union([q_ts]))
                df.loc[q_ts, "close"] = df["close"].iloc[-1]
                df.loc[q_ts, "volume"] = df["volume"].iloc[-1]
                df.loc[q_ts, "bid"] = quote_df["bid"].iloc[0]
                df.loc[q_ts, "ask"] = quote_df["ask"].iloc[0]
    except Exception:
        logger.debug("Alpaca quote attach failed", exc_info=True)

    return df

def fetch_recent_intraday_ibkr(instr_meta: dict):
    if not _IB_AVAILABLE:
        return pd.DataFrame()

    try:
        ib = IB()
        ib.connect(host=IB_GATEWAY_HOST, port=IB_GATEWAY_PORT, clientId=IB_CLIENT_ID, timeout=2)
    except Exception:
        logger.exception("IB connect failed (primary params)")
        try:
            ib = IB()
            ib.connect(host='127.0.0.1', port=7497, clientId=IB_CLIENT_ID, timeout=2)
        except Exception:
            logger.exception("IB connect failed (fallback params)")
            return pd.DataFrame()

    try:
        c = Contract(
            symbol=instr_meta.get("symbol"),
            secType=instr_meta.get("secType", "FUT"),
            exchange=instr_meta.get("exchange"),
            currency=instr_meta.get("currency", "USD"),
            lastTradeDateOrContractMonth=instr_meta.get("lastTradeDateOrContractMonth")
        )
    except Exception:
        logger.exception("IB Contract build failed")
        ib.disconnect()
        return pd.DataFrame()

    end_dt = datetime.now(timezone.utc)
    try:
        bars = ib.reqHistoricalData(
            c,
            endDateTime=end_dt,
            durationStr="120 M",
            barSizeSetting="1 min",
            whatToShow="TRADES",
            useRTH=False,
            formatDate=1
        )
    except Exception:
        logger.exception("IB historical data request failed")
        ib.disconnect()
        return pd.DataFrame()

    if not bars:
        ib.disconnect()
        return pd.DataFrame()

    try:
        if _ib_util is not None:
            df = _ib_util.df(bars)
        else:
            data = []
            for b in bars:
                data.append({"date": pd.to_datetime(b.date), "open": getattr(b, "open", None), "high": getattr(b, "high", None), "low": getattr(b, "low", None), "close": getattr(b, "close", None), "volume": getattr(b, "volume", None)})
            df = pd.DataFrame(data)
    except Exception:
        logger.exception("Failed to convert IB bars to DataFrame")
        ib.disconnect()
        return pd.DataFrame()

    ib.disconnect()

    if df.empty or "close" not in df.columns:
        return pd.DataFrame()

    if "date" in df.columns:
        df = df.rename(columns={"date": "timestamp"})
    try:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    except Exception:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC", ambiguous="NaT", nonexistent="NaT")
    df = df.set_index("timestamp")
    target_freq = _interval_to_pandas_freq(INTERVAL)
    if target_freq != "1min":
        try:
            agg_map = {}
            if "open" in df.columns: agg_map["open"] = "first"
            if "high" in df.columns: agg_map["high"] = "max"
            if "low" in df.columns: agg_map["low"] = "min"
            agg_map["close"] = "last"
            if "volume" in df.columns: agg_map["volume"] = "sum"
            df = df.resample(target_freq, label="right", closed="right").agg(agg_map)
            df = df.dropna(subset=["close"])
        except Exception:
            logger.exception("IB resample failed")
            pass
    df.index = pd.to_datetime(df.index, utc=True)
    return df

def fetch_recent_intraday(symbol_key: str, retries=3, pause=3):
    if symbol_key not in INSTRUMENTS:
        symbol_key = SYMBOL

    instr = INSTRUMENTS[symbol_key]
    instr_type = instr.get("type", "equity")

    if instr_type == "equity":
        alp_sym = instr.get("alpaca", symbol_key)
        df_alp = fetch_recent_intraday_alpaca(alp_sym) if _ALPACA_AVAILABLE else pd.DataFrame()
        if not df_alp.empty:
            return df_alp
        y_sym = instr.get("yahoo", symbol_key)
        df_y = fetch_recent_intraday_yfinance(y_sym, tz=instr.get("tz", "America/New_York"))
        if not df_y.empty:
            return df_y
        return pd.DataFrame()

    if instr_type == "futures":
        ib_meta = instr.get("ibkr")
        if ib_meta and _IB_AVAILABLE:
            ib_meta_res = _resolve_ibkr_contract_meta(ib_meta, instr_meta=instr)
            df_ib = fetch_recent_intraday_ibkr(ib_meta_res)
            if not df_ib.empty:
                try:
                    df_ib.attrs["contract_month"] = ib_meta_res.get("_resolved_contract_month")
                    df_ib.attrs["configured_contract_month"] = ib_meta_res.get("_configured_contract_month")
                    df_ib.attrs["contract_auto_rolled"] = bool(ib_meta_res.get("_contract_auto_rolled", False))
                except Exception:
                    pass
                return df_ib
        y_sym = instr.get("yahoo")
        if y_sym:
            df_y = fetch_recent_intraday_yfinance(y_sym, tz=instr.get("tz", "America/New_York"))
            if not df_y.empty:
                try:
                    df_y.attrs["data_error"] = "no_live_futures_data"
                except Exception:
                    pass
                return df_y
        # annotate for UI so user knows why it's empty
        try:
            df_err = pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC"))
            df_err.attrs["data_error"] = "no_live_futures_data"
            return df_err
        except Exception:
            return pd.DataFrame()

    df_generic = fetch_recent_intraday_yfinance(symbol_key)
    return df_generic

def update_price_data(instr_key: str = None):
    with _DATA_LOCK:
        if instr_key is None:
            instr_key = SYMBOL

        instr = INSTRUMENTS.get(instr_key, {})
        instr_type = instr.get("type", "equity")

        data_file_active = _data_file_for_instrument(instr_key)
        data_file_alp = _data_file_for_instrument(instr_key, "alpaca")
        data_file_yf = _data_file_for_instrument(instr_key, "yfinance")

        def _dedupe_frame(df: pd.DataFrame) -> pd.DataFrame:
            if df is None or df.empty:
                return df
            # Ensure unique column names to avoid concat reindex errors
            try:
                if df.columns.duplicated().any():
                    df = df.loc[:, ~df.columns.duplicated()].copy()
            except Exception:
                pass
            # Ensure unique index values
            try:
                if df.index.duplicated().any():
                    df = df[~df.index.duplicated(keep="last")].copy()
            except Exception:
                pass
            return df

        def _read_csv(path: str) -> pd.DataFrame:
            if not os.path.exists(path):
                return pd.DataFrame()
            try:
                df = pd.read_csv(path, index_col=0, parse_dates=True)
                df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
                df.columns = [c.lower() for c in df.columns]
                if "close" not in df.columns:
                    return pd.DataFrame()
                return _dedupe_frame(df)
            except Exception:
                logger.exception("Failed to read CSV in update_price_data")
                return pd.DataFrame()

        def _merge(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
            existing = _dedupe_frame(existing)
            new = _dedupe_frame(new)
            if existing.empty:
                return new
            if new.empty:
                return existing
            try:
                freq = _interval_to_pandas_freq(INTERVAL)
                new.index = pd.to_datetime(new.index, utc=True)
                try:
                    new.index = new.index.floor(freq)
                except Exception:
                    pass
            except Exception:
                logger.exception("Failed to normalize new df index to freq")
            df = pd.concat([existing, new], axis=0, sort=False)
            df = df[~df.index.duplicated(keep="last")]
            return df

        def _write_csv(path: str, df: pd.DataFrame):
            if df is None or df.empty:
                return
            try:
                df.to_csv(path, date_format="%Y-%m-%dT%H:%M:%SZ")
            except Exception:
                logger.exception("Failed to write CSV with ISO format; retrying without date_format")
                try:
                    df.to_csv(path)
                except Exception:
                    logger.exception("Failed to persist price CSV at all")

        df_alp_existing = _read_csv(data_file_alp)
        df_y_existing = _read_csv(data_file_yf)
        df_active_existing = _read_csv(data_file_active)

        df_active = pd.DataFrame()
        data_error = None
        source_used = None

        if instr_type == "equity":
            df_alp_new = pd.DataFrame()
            if _ALPACA_AVAILABLE:
                alp_sym = instr.get("alpaca", instr_key)
                df_alp_new = fetch_recent_intraday_alpaca(alp_sym)
            if not df_alp_new.empty:
                source_used = "alpaca"
                df_alp = _merge(df_alp_existing, df_alp_new)
                _write_csv(data_file_alp, df_alp)
                df_active = df_alp
            else:
                y_sym = instr.get("yahoo", instr_key)
                start = None
                if not df_y_existing.empty:
                    try:
                        start = df_y_existing.index.max() - pd.Timedelta(days=YFINANCE_REFRESH_BUFFER_DAYS)
                    except Exception:
                        start = None
                df_y_new = fetch_recent_intraday_yfinance(
                    y_sym,
                    tz=instr.get("tz", "America/New_York"),
                    start=start,
                    end=datetime.now(timezone.utc)
                )
                if not df_y_new.empty:
                    source_used = "yfinance"
                    df_y = _merge(df_y_existing, df_y_new)
                    _write_csv(data_file_yf, df_y)
                    df_active = df_y
        else:
            # futures: try IBKR, otherwise yfinance (with no-live-futures flag)
            ib_meta = instr.get("ibkr")
            if ib_meta and _IB_AVAILABLE:
                ib_meta_res = _resolve_ibkr_contract_meta(ib_meta, instr_meta=instr)
                df_ib = fetch_recent_intraday_ibkr(ib_meta_res)
                if not df_ib.empty:
                    resolved_month = ib_meta_res.get("_resolved_contract_month")
                    source_used = f"ibkr:{resolved_month}" if resolved_month else "ibkr"
                    try:
                        df_ib.attrs["contract_month"] = resolved_month
                        df_ib.attrs["configured_contract_month"] = ib_meta_res.get("_configured_contract_month")
                        df_ib.attrs["contract_auto_rolled"] = bool(ib_meta_res.get("_contract_auto_rolled", False))
                    except Exception:
                        pass
                    df_active = df_ib
            if df_active.empty:
                y_sym = instr.get("yahoo")
                if y_sym:
                    start = None
                    if not df_y_existing.empty:
                        try:
                            start = df_y_existing.index.max() - pd.Timedelta(days=YFINANCE_REFRESH_BUFFER_DAYS)
                        except Exception:
                            start = None
                    df_y_new = fetch_recent_intraday_yfinance(
                        y_sym,
                        tz=instr.get("tz", "America/New_York"),
                        start=start,
                        end=datetime.now(timezone.utc)
                    )
                    if not df_y_new.empty:
                        source_used = "yfinance"
                        df_y = _merge(df_y_existing, df_y_new)
                        _write_csv(data_file_yf, df_y)
                        df_active = df_y
                        data_error = "no_live_futures_data"
                        # keep contract metadata visible even when falling back to delayed data
                        try:
                            if ib_meta:
                                ib_meta_res = _resolve_ibkr_contract_meta(ib_meta, instr_meta=instr)
                                df_active.attrs["contract_month"] = ib_meta_res.get("_resolved_contract_month")
                                df_active.attrs["configured_contract_month"] = ib_meta_res.get("_configured_contract_month")
                                df_active.attrs["contract_auto_rolled"] = bool(ib_meta_res.get("_contract_auto_rolled", False))
                        except Exception:
                            pass

        # If no new data, fall back to most recent cached provider data
        if df_active.empty:
            if not df_alp_existing.empty:
                source_used = "cache:alpaca"
                df_active = df_alp_existing
            elif not df_y_existing.empty:
                source_used = "cache:yfinance"
                df_active = df_y_existing
            elif not df_active_existing.empty:
                source_used = "cache:active"
                df_active = df_active_existing

        if df_active is None or df_active.empty:
            logger.warning("No data available; returning empty dataframe (synthetic data disabled).")
            return df_active

        try:
            cutoff = datetime.now(timezone.utc) - timedelta(days=MAX_HISTORY_DAYS)
            df_active = df_active[df_active.index >= cutoff]
        except Exception:
            logger.exception("Failed to trim df by MAX_HISTORY_DAYS")
            pass
        df_active = _dedupe_frame(df_active)
        df_active = df_active.sort_index()

        try:
            if data_error:
                df_active.attrs["data_error"] = data_error
            if source_used:
                df_active.attrs["data_source"] = source_used
            # Preserve effective futures contract metadata even when using cache fallback.
            if instr_type == "futures":
                ib_meta = instr.get("ibkr")
                if ib_meta:
                    ib_meta_res = _resolve_ibkr_contract_meta(ib_meta, instr_meta=instr)
                    df_active.attrs.setdefault("contract_month", ib_meta_res.get("_resolved_contract_month"))
                    df_active.attrs.setdefault("configured_contract_month", ib_meta_res.get("_configured_contract_month"))
                    df_active.attrs.setdefault("contract_auto_rolled", bool(ib_meta_res.get("_contract_auto_rolled", False)))
        except Exception:
            pass

        # Persist active dataset
        _write_csv(data_file_active, df_active)

        return df_active
