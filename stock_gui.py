#stock_gui.py
import json
import threading
import tkinter as tk
from tkinter import ttk
from tkinter import font as tkfont
import pandas as pd
from model_utils import load_models, predict_4min, predict_horizon
from data_fetch import update_price_data
from config import AUTO_PREDICT_SECONDS, INSTRUMENTS, DEFAULT_INSTRUMENT
from features import LIVE_MAX_BARS, BAR_MINUTES
from model_trainers import train_all_models
from instrument_onboarding import ensure_runtime_instruments, list_runtime_instruments
from readiness_tracker import evaluate_readiness_metrics
from trader_analytics import build_trader_dashboard
import datetime
import time
import logging
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(filename="lightning_gui.log", level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


def _dashboard_prediction_signature(predictions: dict | None) -> str:
    if not isinstance(predictions, dict):
        return "none"

    def _avg_price(block: dict | None, key: str = "pred_price"):
        if not isinstance(block, dict):
            return None
        vals = []
        for model_key in ("base", "forest", "boost", "extra_trees"):
            item = block.get(model_key)
            if not isinstance(item, dict):
                continue
            val = item.get(key)
            if isinstance(val, (int, float)):
                vals.append(round(float(val), 6))
        if not vals:
            return None
        return round(float(sum(vals) / len(vals)), 6)

    summary = {}
    intraday = predictions.get("intraday")
    if isinstance(intraday, dict):
        cons = intraday.get("consensus", {})
        if isinstance(cons, dict):
            summary["intraday"] = {
                "direction": str(cons.get("direction") or "-"),
                "price": round(float(cons.get("pred_price")), 6) if isinstance(cons.get("pred_price"), (int, float)) else None,
                "confidence": round(float(cons.get("confidence")), 4) if isinstance(cons.get("confidence"), (int, float)) else None,
            }
    daily = predictions.get("daily")
    if isinstance(daily, dict):
        summary["daily"] = {
            "high_avg": _avg_price(daily.get("high")),
            "low_avg": _avg_price(daily.get("low")),
        }
    weekend = predictions.get("weekend")
    if isinstance(weekend, dict):
        summary["weekend"] = {
            "close_avg": _avg_price(weekend.get("week")),
            "high_avg": _avg_price(weekend.get("week"), key="pred_high_price"),
            "low_avg": _avg_price(weekend.get("week"), key="pred_low_price"),
        }
    try:
        return json.dumps(summary, sort_keys=True, separators=(",", ":"))
    except Exception:
        return repr(summary)

class LightningGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Lightning Trader Terminal")
        self.root.geometry("1460x920")
        self.root.configure(bg="#081018")

        self.header_font = tkfont.Font(family="Avenir Next", size=19, weight="bold")
        self.big_font = tkfont.Font(family="Avenir Next", size=15, weight="bold")
        self.cell_font = tkfont.Font(family="Avenir Next", size=12)
        self.small_font = tkfont.Font(family="Avenir Next", size=11)

        self.style = ttk.Style(root)
        self.style.theme_use("clam")
        self.style.configure("TButton", font=self.small_font, padding=8, foreground="white", background="#162130")
        self.style.configure("Treeview", rowheight=30, font=self.cell_font, background="#0f1724", fieldbackground="#0f1724", foreground="#e5eefb")
        self.style.configure("Treeview.Heading", font=self.big_font, background="#152033", foreground="#dbeafe")

        self.models = {}
        self.df = None
        self._lock = threading.Lock()
        self._pred_lock = threading.Lock()
        self._pred_in_flight = False
        self._pred_started_at = None
        self._pred_timeout_sec = 30.0
        self._pred_seq = 0
        self._acc_lock = threading.Lock()
        self._acc_in_flight = False
        self._dashboard_cache = {}
        self._dashboard_cache_ttl_sec = max(60, int(AUTO_PREDICT_SECONDS))

        self.executor = ThreadPoolExecutor(max_workers=2)
        self._overview_last_refresh = 0.0
        self._overview_refresh_sec = max(120, int(AUTO_PREDICT_SECONDS))

        # Top frame
        self.top_frame = tk.Frame(self.root, bg="#081018")
        self.top_frame.pack(fill="x", padx=12, pady=8)

        self.reload_btn = ttk.Button(self.top_frame, text="Reload Data & Models", command=self.reload_once_threaded)
        self.reload_btn.pack(side="left", padx=6)

        self.retrain_btn = ttk.Button(self.top_frame, text="Retrain Models", command=self.retrain_once)
        self.retrain_btn.pack(side="left", padx=6)

        self.auto_var = tk.BooleanVar(value=True)
        self.auto_cb = tk.Checkbutton(self.top_frame, text="Auto-refresh", variable=self.auto_var, bg="#081018", fg="white", selectcolor="#081018", font=self.small_font, command=self.toggle_auto)
        self.auto_cb.pack(side="left", padx=12)

        # Instrument selection
        self.instrument_var = tk.StringVar(value=DEFAULT_INSTRUMENT)
        try:
            ensure_runtime_instruments(force=True, auto_onboard=True)
        except Exception:
            pass
        instrument_frame = tk.Frame(self.top_frame, bg="#081018")
        instrument_frame.pack(side="left", padx=12)
        tk.Label(instrument_frame, text="Instrument:", bg="#081018", fg="white", font=self.small_font).pack(side="left", padx=(0,6))
        choices = list_runtime_instruments() or list(INSTRUMENTS.keys())
        if choices and self.instrument_var.get() not in choices:
            self.instrument_var.set(choices[0])
        self.instrument_menu = ttk.Combobox(instrument_frame, values=choices, textvariable=self.instrument_var, state="readonly", width=12)
        self.instrument_menu.pack(side="left")
        self.instrument_menu.bind("<<ComboboxSelected>>", self._on_instrument_change)

        # Mode selection: 4min, DailyHigh, WeekEnd
        self.mode_var = tk.StringVar(value="4min")
        mode_frame = tk.Frame(self.top_frame, bg="#081018")
        mode_frame.pack(side="left", padx=12)
        tk.Label(mode_frame, text="Mode:", bg="#081018", fg="white", font=self.small_font).pack(side="left", padx=(0,6))
        self.mode_menu = ttk.Combobox(mode_frame, values=["4min", "DailyHigh", "WeekEnd"], textvariable=self.mode_var, state="readonly", width=12)
        self.mode_menu.pack(side="left")
        self.mode_apply_btn = ttk.Button(mode_frame, text="Apply Mode", command=self.mode_apply)
        self.mode_apply_btn.pack(side="left", padx=(6,0))

        # Current price display
        self.current_price_var = tk.StringVar(value="Current Price: -")
        self.current_price_label = tk.Label(self.top_frame, textvariable=self.current_price_var, font=self.header_font, fg="#7dd3fc", bg="#081018")
        self.current_price_label.pack(side="right", padx=8)

        self.status_label = tk.Label(self.top_frame, text="Status: initializing...", fg="white", bg="#081018", font=self.small_font)
        self.status_label.pack(side="right", padx=12)

        # Summary / avg prediction
        self.summary_frame = tk.Frame(self.root, bg="#081018")
        self.summary_frame.pack(fill="x", padx=12, pady=(0, 6))
        self.avg_pred_var = tk.StringVar(value="Avg Prediction: -")
        self.avg_pred_label = tk.Label(self.summary_frame, textvariable=self.avg_pred_var, font=self.big_font, fg="#fbbf24", bg="#081018")
        self.avg_pred_label.pack(side="left", padx=6)
        self.consensus_var = tk.StringVar(value="Consensus: -")
        self.consensus_label = tk.Label(self.summary_frame, textvariable=self.consensus_var, font=self.big_font, fg="#34d399", bg="#081018")
        self.consensus_label.pack(side="left", padx=12)
        self.accuracy_var = tk.StringVar(value="Avg Discrepancy: loading…")
        self.accuracy_label = tk.Label(self.summary_frame, textvariable=self.accuracy_var, font=self.small_font, fg="#93c5fd", bg="#081018")
        self.accuracy_label.pack(side="left", padx=12)
        self.data_meta_var = tk.StringVar(value="Data: - | Last Bar: -")
        self.data_meta_label = tk.Label(self.summary_frame, textvariable=self.data_meta_var, font=self.small_font, fg="#cbd5f5", bg="#081018")
        self.data_meta_label.pack(side="right", padx=6)

        # Instrument overview section: one row per symbol with all horizons.
        self.overview_frame = tk.Frame(self.root, bg="#081018")
        self.overview_frame.pack(fill="x", padx=12, pady=(0, 8))
        self.overview_label = tk.Label(
            self.overview_frame,
            text="Instrument Overview (4min / Daily / WeekEnd)",
            font=self.small_font,
            fg="#e2e8f0",
            bg="#081018",
        )
        self.overview_label.pack(anchor="w", padx=2, pady=(0, 3))
        overview_cols = ("instrument", "four_min", "daily", "weekend", "status")
        self.overview_tree = ttk.Treeview(self.overview_frame, columns=overview_cols, show="headings", height=5)
        labels = {
            "instrument": "Instrument",
            "four_min": "4min",
            "daily": "Daily High/Low",
            "weekend": "WeekEnd",
            "status": "Status",
        }
        for col in overview_cols:
            self.overview_tree.heading(col, text=labels[col])
            width = 190 if col == "instrument" else 240 if col in ("daily", "weekend") else 230
            self.overview_tree.column(col, anchor=tk.CENTER, width=width)
        self.overview_tree.pack(fill="x")

        self.main_pane = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, bg="#081018", sashwidth=8, relief=tk.FLAT, bd=0)
        self.main_pane.pack(fill="both", expand=True, padx=12, pady=(0, 10))
        self.left_panel = tk.Frame(self.main_pane, bg="#081018")
        self.right_panel = tk.Frame(self.main_pane, bg="#081018", width=430)
        self.main_pane.add(self.left_panel, stretch="always")
        self.main_pane.add(self.right_panel, minsize=380)

        # Treeview for 4min outputs
        columns_4min = ("model", "direction", "predicted_price", "confidence", "note")
        self.tree_4min = ttk.Treeview(self.left_panel, columns=columns_4min, show="headings", height=14)
        for col in columns_4min:
            label = col.replace("_", " ").title()
            if col == "confidence":
                label = "Confidence Score"
            self.tree_4min.heading(col, text=label)
            width = 320 if col == "note" else 180
            self.tree_4min.column(col, anchor=tk.CENTER, width=width)
        self.tree_4min.pack(padx=0, pady=8, fill="both", expand=True)

        # Horizon frame (DailyHigh / WeekEnd)
        self.horizon_frame = tk.Frame(self.left_panel, bg="#081018")
        self.horizon_high_label = tk.Label(self.horizon_frame, text="Daily High Forecast", font=self.big_font, fg="#93c5fd", bg="#081018")
        self.horizon_high_label.pack(anchor="w", padx=12, pady=(4, 2))
        columns_h = ("model", "predicted_price", "confidence", "note")
        self.tree_high = ttk.Treeview(self.horizon_frame, columns=columns_h, show="headings", height=6)
        for col in columns_h:
            label = col.replace("_", " ").title()
            if col == "confidence":
                label = "Confidence Score"
            self.tree_high.heading(col, text=label)
            width = 420 if col == "note" else 220
            self.tree_high.column(col, anchor=tk.CENTER, width=width)
        self.tree_high.pack(padx=12, pady=(0, 8), fill="both", expand=True)
        self.week_extremes_var = tk.StringVar(value="Projected Week High/Low: -")
        self.week_extremes_label = tk.Label(
            self.horizon_frame,
            textvariable=self.week_extremes_var,
            font=self.small_font,
            fg="#fcd34d",
            bg="#081018",
        )
        self.week_extremes_label.pack(anchor="w", padx=12, pady=(0, 6))

        self.horizon_low_label = tk.Label(self.horizon_frame, text="Daily Low Forecast", font=self.big_font, fg="#fca5a5", bg="#081018")
        self.horizon_low_label.pack(anchor="w", padx=12, pady=(4, 2))
        self.tree_low = ttk.Treeview(self.horizon_frame, columns=columns_h, show="headings", height=6)
        for col in columns_h:
            label = col.replace("_", " ").title()
            if col == "confidence":
                label = "Confidence Score"
            self.tree_low.heading(col, text=label)
            width = 420 if col == "note" else 220
            self.tree_low.column(col, anchor=tk.CENTER, width=width)
        self.tree_low.pack(padx=12, pady=(0, 8), fill="both", expand=True)

        self.regime_label = tk.Label(self.left_panel, text="Market Regime: Unknown", font=self.header_font, fg="#60a5fa", bg="#081018")
        self.regime_label.pack(pady=6)

        self.cards_frame = tk.Frame(self.right_panel, bg="#081018")
        self.cards_frame.pack(fill="x", pady=(0, 8))
        self.setup_card_var = tk.StringVar(value="Alignment: loading…")
        self.setup_detail_var = tk.StringVar(value="-")
        self.vol_card_var = tk.StringVar(value="Volatility: loading…")
        self.vol_detail_var = tk.StringVar(value="-")
        self.options_card_var = tk.StringVar(value="Options: loading…")
        self.news_card_var = tk.StringVar(value="News: loading…")

        def _make_card(parent, title: str, main_var, detail_var=None, accent="#93c5fd"):
            frame = tk.Frame(parent, bg="#0f1724", highlightbackground="#1f2a44", highlightthickness=1, bd=0)
            frame.pack(fill="x", pady=4)
            tk.Label(frame, text=title, font=self.small_font, fg=accent, bg="#0f1724").pack(anchor="w", padx=10, pady=(8, 0))
            tk.Label(frame, textvariable=main_var, font=self.big_font, fg="#f8fafc", bg="#0f1724", wraplength=360, justify="left").pack(anchor="w", padx=10, pady=(2, 0))
            if detail_var is not None:
                tk.Label(frame, textvariable=detail_var, font=self.small_font, fg="#cbd5e1", bg="#0f1724", wraplength=360, justify="left").pack(anchor="w", padx=10, pady=(2, 8))
            else:
                tk.Label(frame, text="", font=self.small_font, bg="#0f1724").pack(anchor="w", padx=10, pady=(0, 8))

        _make_card(self.cards_frame, "Setup", self.setup_card_var, self.setup_detail_var, accent="#34d399")
        _make_card(self.cards_frame, "Volatility", self.vol_card_var, self.vol_detail_var, accent="#fbbf24")
        _make_card(self.cards_frame, "Options", self.options_card_var, None, accent="#c084fc")
        _make_card(self.cards_frame, "News", self.news_card_var, None, accent="#7dd3fc")

        self.market_label = tk.Label(self.right_panel, text="Market Context", font=self.big_font, fg="#e2e8f0", bg="#081018")
        self.market_label.pack(anchor="w", pady=(2, 2))
        self.market_tree = ttk.Treeview(self.right_panel, columns=("item", "value", "detail"), show="headings", height=5)
        for col, width in (("item", 120), ("value", 90), ("detail", 170)):
            self.market_tree.heading(col, text=col.title())
            self.market_tree.column(col, anchor=tk.CENTER if col != "detail" else tk.W, width=width)
        self.market_tree.pack(fill="x", pady=(0, 8))

        self.levels_label = tk.Label(self.right_panel, text="Key Levels", font=self.big_font, fg="#e2e8f0", bg="#081018")
        self.levels_label.pack(anchor="w", pady=(2, 2))
        self.levels_tree = ttk.Treeview(self.right_panel, columns=("item", "value", "detail"), show="headings", height=6)
        for col, width in (("item", 120), ("value", 110), ("detail", 150)):
            self.levels_tree.heading(col, text=col.title())
            self.levels_tree.column(col, anchor=tk.CENTER if col != "detail" else tk.W, width=width)
        self.levels_tree.pack(fill="x", pady=(0, 8))

        self.options_label = tk.Label(self.right_panel, text="Options Positioning", font=self.big_font, fg="#e2e8f0", bg="#081018")
        self.options_label.pack(anchor="w", pady=(2, 2))
        self.options_tree = ttk.Treeview(self.right_panel, columns=("item", "value", "detail"), show="headings", height=5)
        for col, width in (("item", 120), ("value", 110), ("detail", 150)):
            self.options_tree.heading(col, text=col.title())
            self.options_tree.column(col, anchor=tk.CENTER if col != "detail" else tk.W, width=width)
        self.options_tree.pack(fill="x", pady=(0, 8))

        self.news_label = tk.Label(self.right_panel, text="Recent Catalysts", font=self.big_font, fg="#e2e8f0", bg="#081018")
        self.news_label.pack(anchor="w", pady=(2, 2))
        self.news_tree = ttk.Treeview(self.right_panel, columns=("age", "tone", "headline"), show="headings", height=7)
        for col, width in (("age", 60), ("tone", 80), ("headline", 270)):
            self.news_tree.heading(col, text=col.title())
            self.news_tree.column(col, anchor=tk.CENTER if col != "headline" else tk.W, width=width)
        self.news_tree.pack(fill="both", expand=False, pady=(0, 4))

        self._stop_event = threading.Event()
        self._retrain_in_progress = False
        self._last_retrain_attempt = 0.0
        self._training_stage = None
        self._worker_thread = threading.Thread(target=self.background_loop, daemon=True)
        self._worker_thread.start()

        self._ui_max_bars = self._compute_ui_max_bars()
        self.status("Initializing…")
        self._show_mode_ui()
        self.refresh_overview_threaded()
        self.refresh_accuracy_summary_threaded()
        self._initial_loader_thread = threading.Thread(target=self._initial_model_waiter, daemon=True)
        self._initial_loader_thread.start()

    def _compute_ui_max_bars(self) -> int:
        try:
            bars_per_day = max(1, int(round(1440 / float(BAR_MINUTES))))
        except Exception:
            bars_per_day = 720
        # Keep ~3 weeks of bars for horizon features without loading full history
        return max(LIVE_MAX_BARS, bars_per_day * 21)

    def _trim_df_for_ui(self, df: pd.DataFrame | None) -> pd.DataFrame | None:
        if df is None or getattr(df, "empty", True):
            return df
        max_bars = getattr(self, "_ui_max_bars", LIVE_MAX_BARS)
        try:
            if len(df) > max_bars:
                return df.iloc[-max_bars:].copy()
        except Exception:
            pass
        return df

    def mode_apply(self):
        self.status(f"Mode set to {self.mode_var.get()}; refreshing predictions...")
        self._show_mode_ui()
        # immediate refresh without reloading data/models (fast switch)
        self.run_prediction_and_refresh_ui()

    def _show_mode_ui(self):
        mode = self.mode_var.get()
        if mode == "4min":
            try:
                if self.horizon_frame.winfo_ismapped():
                    self.horizon_frame.pack_forget()
            except Exception:
                pass
            if not self.tree_4min.winfo_ismapped():
                self.tree_4min.pack(padx=12, pady=8, fill="both", expand=True)
        else:
            try:
                if self.tree_4min.winfo_ismapped():
                    self.tree_4min.pack_forget()
            except Exception:
                pass
            if not self.horizon_frame.winfo_ismapped():
                self.horizon_frame.pack(padx=12, pady=8, fill="both", expand=True)
            if mode == "DailyHigh":
                self.horizon_high_label.config(text="Daily High Forecast")
                self.week_extremes_label.pack_forget()
                self.horizon_low_label.pack(anchor="w", padx=12, pady=(4, 2))
                self.tree_low.pack(padx=12, pady=(0, 8), fill="both", expand=True)
            else:
                self.horizon_high_label.config(text="Week-End Close Forecast")
                if not self.week_extremes_label.winfo_ismapped():
                    self.week_extremes_label.pack(anchor="w", padx=12, pady=(0, 6), before=self.horizon_low_label)
                self.horizon_low_label.pack_forget()
                self.tree_low.pack_forget()

    def _on_instrument_change(self, event=None):
        self.status(f"Instrument changed to {self.instrument_var.get()}; reloading…")
        self._dashboard_cache.pop(str(self.instrument_var.get() or "").upper(), None)
        self.reload_once_threaded()

    def _format_accuracy_summary(self, report: dict, selected: str) -> str:
        if not isinstance(report, dict) or report.get("error"):
            return "Avg Discrepancy: unavailable"
        metrics = report.get("metrics", {}) if isinstance(report.get("metrics"), dict) else {}

        def _block(label: str, mae_key: str, target_key: str, ok_key: str, dir_key: str) -> str:
            try:
                mae = metrics.get(mae_key)
                target = metrics.get(target_key)
                ok = bool(metrics.get(ok_key))
                dir_acc = metrics.get(dir_key)
                if mae is None or target is None:
                    return f"{label} -"
                dir_part = f" Dir {100.0 * float(dir_acc):.0f}%" if dir_acc is not None else " Dir -"
                return f"{label} ${float(mae):.2f}/${float(target):.2f} {'OK' if ok else 'MISS'}{dir_part}"
            except Exception:
                return f"{label} -"

        parts = [
            _block("4m", "intraday_avg_discrepancy_dollars", "intraday_target_window_dollars", "intraday_success", "intraday_updown_acc"),
            _block("1D", "daily_avg_discrepancy_dollars", "daily_target_window_dollars", "daily_success", "daily_updown_acc"),
            _block("1W", "weekend_avg_discrepancy_dollars", "weekend_target_window_dollars", "weekend_success", "weekend_updown_acc"),
        ]
        return f"Avg Discrepancy ({selected}): " + " | ".join(parts)

    def refresh_accuracy_summary_threaded(self):
        with self._acc_lock:
            if self._acc_in_flight:
                return
            self._acc_in_flight = True
        selected = self.instrument_var.get()

        def _job():
            try:
                report = evaluate_readiness_metrics(selected, online=False)
                text = self._format_accuracy_summary(report, selected)
            except Exception:
                logger.exception("Accuracy summary refresh failed")
                text = f"Avg Discrepancy ({selected}): unavailable"
            finally:
                with self._acc_lock:
                    self._acc_in_flight = False
            try:
                self.root.after(0, lambda: self.accuracy_var.set(text))
            except Exception:
                pass

        self.executor.submit(_job)

    def _refresh_instrument_choices(self):
        try:
            ensure_runtime_instruments(force=False, auto_onboard=True)
            choices = list_runtime_instruments()
            if not choices:
                choices = list(INSTRUMENTS.keys())
        except Exception:
            choices = list(INSTRUMENTS.keys())

        def _apply():
            try:
                current = self.instrument_var.get()
                self.instrument_menu.configure(values=choices)
                if choices and current not in choices:
                    self.instrument_var.set(choices[0])
            except Exception:
                pass

        try:
            self.root.after(0, _apply)
        except Exception:
            pass

    @staticmethod
    def _fmt_price(v):
        if isinstance(v, (int, float)):
            return f"${float(v):,.2f}"
        return "-"

    @staticmethod
    def _append_band_note(note: str, item: dict) -> str:
        if not isinstance(item, dict):
            return note or ""
        parts = []
        lo = item.get("pred_range_80_low")
        hi = item.get("pred_range_80_high")
        err = item.get("expected_abs_err_bps")
        if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
            parts.append(f"80% {float(lo):,.2f}..{float(hi):,.2f}")
        if isinstance(err, (int, float)):
            parts.append(f"Err~{float(err):.1f}bps")
        if not parts:
            return note or ""
        suffix = " | ".join(parts)
        return f"{note} | {suffix}" if note else suffix

    def _compute_overview_rows(self):
        rows = []
        try:
            ensure_runtime_instruments(force=False, auto_onboard=True)
            symbols = list_runtime_instruments()
        except Exception:
            symbols = list(INSTRUMENTS.keys())
        if not symbols:
            return rows
        for sym in symbols:
            try:
                models = load_models(instrument_key=sym)
                df = update_price_data(sym)
                if df is None or getattr(df, "empty", True):
                    rows.append((sym, "-", "-", "-", "No data"))
                    continue
                if not models:
                    rows.append((sym, "-", "-", "-", "No models"))
                    continue

                p4 = predict_4min(models, df, instrument_key=sym)
                if isinstance(p4, dict) and p4.get("error"):
                    four = "-"
                    status = str(p4.get("error"))
                else:
                    c = p4.get("consensus", {}) if isinstance(p4, dict) else {}
                    four = f"{self._fmt_price(c.get('pred_price'))} ({c.get('direction', '-')})"
                    status = "OK"

                pdaily = predict_horizon(models, df, instrument_key=sym, mode="DailyHigh")
                daily = "-"
                if isinstance(pdaily, dict) and not pdaily.get("error"):
                    high_block = pdaily.get("high", {})
                    low_block = pdaily.get("low", {})
                    highs = [v.get("pred_price") for v in high_block.values() if isinstance(v, dict) and isinstance(v.get("pred_price"), (int, float))]
                    lows = [v.get("pred_price") for v in low_block.values() if isinstance(v, dict) and isinstance(v.get("pred_price"), (int, float))]
                    if highs or lows:
                        high_avg = (sum(highs) / len(highs)) if highs else None
                        low_avg = (sum(lows) / len(lows)) if lows else None
                        if high_avg is not None and low_avg is not None:
                            daily = f"H {self._fmt_price(high_avg)} / L {self._fmt_price(low_avg)}"
                        elif high_avg is not None:
                            daily = f"H {self._fmt_price(high_avg)}"
                        elif low_avg is not None:
                            daily = f"L {self._fmt_price(low_avg)}"

                pweek = predict_horizon(models, df, instrument_key=sym, mode="WeekEnd")
                week = "-"
                if isinstance(pweek, dict) and not pweek.get("error"):
                    block = pweek.get("week", {})
                    vals = [v.get("pred_price") for v in block.values() if isinstance(v, dict) and isinstance(v.get("pred_price"), (int, float))]
                    if vals:
                        week = self._fmt_price(sum(vals) / len(vals))

                rows.append((sym, four, daily, week, status))
            except Exception:
                rows.append((sym, "-", "-", "-", "Error"))
        return rows

    def _apply_overview_rows(self, rows):
        try:
            for i in self.overview_tree.get_children():
                self.overview_tree.delete(i)
            for row in rows:
                self.overview_tree.insert("", "end", values=row)
        except Exception:
            pass

    def refresh_overview_threaded(self):
        def _job():
            rows = self._compute_overview_rows()
            self._overview_last_refresh = time.time()
            try:
                self.root.after(0, lambda: self._apply_overview_rows(rows))
            except Exception:
                pass
        try:
            self.executor.submit(_job)
        except Exception:
            logger.debug("Failed to schedule overview refresh", exc_info=True)

    def toggle_auto(self):
        self.status("Auto-refresh enabled" if self.auto_var.get() else "Auto-refresh disabled")

    def status(self, text):
        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        try:
            self.status_label.config(text=f"Status: {text} — {ts} UTC")
        except Exception:
            pass
        logger.info(text)

    def _set_training_stage(self, msg: str | None):
        self._training_stage = msg
        if msg:
            try:
                self.root.after(0, lambda: self.status(msg))
            except Exception:
                pass

    def _initial_model_waiter(self):
        self.status("Checking models…")
        while not self._stop_event.is_set():
            try:
                selected = self.instrument_var.get()
                models = load_models(instrument_key=selected)
                missing = [k for k in ["base","forest","boost","extra_trees"] if k not in models]
                mode = self.mode_var.get()
                needs_retrain = self._needs_retrain(models, require_horizons=(mode != "4min"))
                if (missing or needs_retrain):
                    self.status("Models missing or legacy. Click 'Retrain Models' to build them.")
                    with self._lock:
                        self.models = models
                        self.df = self._trim_df_for_ui(update_price_data(selected))
                    self.root.after(0, self.run_prediction_and_refresh_ui)
                    self.root.after(0, self.refresh_overview_threaded)
                    self.root.after(0, self.refresh_accuracy_summary_threaded)
                    return
                with self._lock:
                    self.models = models
                    self.df = self._trim_df_for_ui(update_price_data(selected))
                self.root.after(0, self.run_prediction_and_refresh_ui)
                self.root.after(0, self.refresh_overview_threaded)
                self.root.after(0, self.refresh_accuracy_summary_threaded)
                self.status("Models loaded — predictions available")
                return
            except Exception as e:
                logger.exception("Model init loop error")
                self.status(f"Model init error: {e}")
                time.sleep(5)

    def _needs_retrain(self, models: dict, require_horizons: bool = True) -> bool:
        required = ["base", "forest", "boost", "extra_trees"]
        for k in required:
            payload = models.get(k)
            if payload is None:
                return True
            meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
            if not meta or not meta.get("version"):
                return True
            # if we have a version marker, treat as modern even if ensemble is sparse
        if require_horizons:
            # Horizon modes should also be present (daily_high, daily_low, weekend)
            modes = models.get("modes", {}) if isinstance(models, dict) else {}
            for mode in ("daily_high", "daily_low", "weekend"):
                mode_payloads = modes.get(mode, {})
                if not mode_payloads:
                    return True
                # Require at least one valid model per horizon mode
                any_valid = False
                for k in required:
                    payload = mode_payloads.get(k)
                    if payload is None:
                        continue
                    meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
                    if meta and meta.get("version"):
                        any_valid = True
                        break
                if not any_valid:
                    return True
        # Regime model must be present (non-legacy)
        if models.get("regime") is None:
            return True
        return False

    def _models_ready(self, models: dict) -> bool:
        if not models or self._needs_retrain(models):
            return False
        required = ["base", "forest", "boost", "extra_trees"]

        def _has_ensemble(payload):
            try:
                ens = payload.get("ensemble_models", {}) if isinstance(payload, dict) else {}
                for v in ens.values():
                    if any(m is not None for m in (v or [])):
                        return True
                # fallback if single model exists
                if isinstance(payload, dict):
                    if payload.get("fallback_model") is not None:
                        return True
                    if payload.get("model") is not None:
                        return True
                return False
            except Exception:
                return False

        for k in required:
            if not _has_ensemble(models.get(k, {})):
                return False

        modes = models.get("modes", {}) if isinstance(models, dict) else {}
        for mode in ("daily_high", "daily_low", "weekend"):
            mode_payloads = modes.get(mode, {})
            for k in required:
                if not _has_ensemble(mode_payloads.get(k, {})):
                    return False
        return True

    def _mode_ready(self, models: dict, mode: str) -> bool:
        if not models:
            return False
        required = ["base", "forest", "boost", "extra_trees"]

        def _has_ensemble(payload):
            try:
                ens = payload.get("ensemble_models", {}) if isinstance(payload, dict) else {}
                for v in ens.values():
                    if any(m is not None for m in (v or [])):
                        return True
                return False
            except Exception:
                return False

        # Regime model is mandatory for predictions (strict ML-only mode)
        if models.get("regime") is None:
            return False

        if mode == "4min":
            return any(_has_ensemble(models.get(k, {})) for k in required)
        if mode == "DailyHigh":
            modes = models.get("modes", {}) if isinstance(models, dict) else {}
            dh = modes.get("daily_high", {})
            dl = modes.get("daily_low", {})
            has_high = any(_has_ensemble(dh.get(k, {})) for k in required)
            has_low = any(_has_ensemble(dl.get(k, {})) for k in required)
            return has_high and has_low
        if mode == "WeekEnd":
            modes = models.get("modes", {}) if isinstance(models, dict) else {}
            wk = modes.get("weekend", {})
            return any(_has_ensemble(wk.get(k, {})) for k in required)
        return False

    def _train_and_reload(self, selected):
        try:
            self.status("Background training started...")
            df = update_price_data(selected)
            if df is None or getattr(df, "empty", True):
                self.status("No data available for training")
                return
            train_all_models(
                df,
                instrument_key=selected,
                max_runtime_sec=None,
                train_horizons=True,
                progress_cb=self._set_training_stage
            )
            models = load_models(instrument_key=selected)
            df2 = update_price_data(selected)
            with self._lock:
                self.models = models
                self.df = self._trim_df_for_ui(df2)
            self._dashboard_cache.pop(str(selected or "").upper(), None)
            self.root.after(0, self.run_prediction_and_refresh_ui)
            self.root.after(0, self.refresh_overview_threaded)
            self.root.after(0, self.refresh_accuracy_summary_threaded)
            self.status("Background training complete")
        except Exception:
            logger.exception("Background training failed")
            self.status("Background training failed; see logs")
        finally:
            self._retrain_in_progress = False

    def reload_once_threaded(self):
        def _job():
            try:
                self.status("Reloading data and models…")
                selected = self.instrument_var.get()
                models = load_models(instrument_key=selected)
                df = update_price_data(selected)
                mode = self.mode_var.get()
                if self._needs_retrain(models, require_horizons=(mode != "4min")):
                    self.status("Models missing or legacy. Click 'Retrain Models' to build them.")
                with self._lock:
                    self.models = models
                    self.df = self._trim_df_for_ui(df)
                self._dashboard_cache.pop(str(selected or "").upper(), None)
                self.root.after(0, self.run_prediction_and_refresh_ui)
                self.root.after(0, self.refresh_overview_threaded)
                self.root.after(0, self.refresh_accuracy_summary_threaded)
                self.status("Reload complete")
            except Exception:
                logger.exception("Reload failed")
                self.status("Reload failed; see logs")
        self.executor.submit(_job)

    def retrain_once(self):
        def _tr():
            try:
                self._retrain_in_progress = True
                self.status("Retraining models…")
                selected = self.instrument_var.get()
                df = update_price_data(selected)
                train_all_models(
                    df,
                    instrument_key=selected,
                    max_runtime_sec=None,
                    train_horizons=True,
                    progress_cb=self._set_training_stage
                )
                models = load_models(instrument_key=selected)
                df2 = update_price_data(selected)
                with self._lock:
                    self.models = models
                    self.df = self._trim_df_for_ui(df2)
                self.root.after(0, self.run_prediction_and_refresh_ui)
                self.root.after(0, self.refresh_overview_threaded)
                self.root.after(0, self.refresh_accuracy_summary_threaded)
                self.status("Retrain complete")
            except Exception:
                logger.exception("Retrain failed")
                self.status("Retrain failed; see logs")
            finally:
                self._retrain_in_progress = False
        self.executor.submit(_tr)

    def _base_payload(self, mode: str) -> dict:
        return {
            "mode": mode,
            "tree_4min": [],
            "tree_high": [],
            "tree_low": [],
            "market_rows": [],
            "levels_rows": [],
            "options_rows": [],
            "news_rows": [],
            "current_price": "Current Price: -",
            "avg_pred": "Avg Prediction: -",
            "consensus": "Consensus: -",
            "data_meta": "Data: - | Last Bar: -",
            "week_extremes": "Projected Week High/Low: -",
            "regime_text": "Market Regime: Unknown",
            "setup_card": "Alignment: unavailable",
            "setup_detail": "-",
            "vol_card": "Volatility: unavailable",
            "vol_detail": "-",
            "options_card": "Options: unavailable",
            "news_card": "News: unavailable",
            "status_messages": []
        }

    def _build_waiting_payload(self, mode: str, msg: str, status: str | None = None) -> dict:
        payload = self._base_payload(mode)
        if mode == "4min":
            for m in ["Base", "Forest", "Boost", "Extra Trees"]:
                payload["tree_4min"].append(((m, "-", "-", "-", msg), None))
        else:
            for m in ["Base", "Forest", "Boost", "Extra Trees"]:
                payload["tree_high"].append(((m, "-", "-", msg), None))
                if mode == "DailyHigh":
                    payload["tree_low"].append(((m, "-", "-", msg), None))
        if status:
            payload["status_messages"].append(status)
        return payload

    def _build_error_payload(self, mode: str, err: str) -> dict:
        payload = self._build_waiting_payload(mode, "Prediction error")
        payload["status_messages"].append("Prediction error — fallback displayed")
        payload["status_messages"].append(f"Prediction error: {err}")
        return payload

    def _apply_prediction_payload(self, payload: dict):
        def _clear(tree):
            for i in tree.get_children():
                tree.delete(i)

        _clear(self.tree_4min)
        _clear(self.tree_high)
        _clear(self.tree_low)
        _clear(self.market_tree)
        _clear(self.levels_tree)
        _clear(self.options_tree)
        _clear(self.news_tree)

        for values, tag in payload.get("tree_4min", []):
            row_id = self.tree_4min.insert("", "end", values=values)
            if tag == "up":
                self.tree_4min.tag_configure("up", foreground="#10b981")
                self.tree_4min.item(row_id, tags=("up",))
            elif tag == "down":
                self.tree_4min.tag_configure("down", foreground="#ef4444")
                self.tree_4min.item(row_id, tags=("down",))

        for values, tag in payload.get("tree_high", []):
            row_id = self.tree_high.insert("", "end", values=values)
            if tag == "up":
                self.tree_high.tag_configure("up", foreground="#10b981")
                self.tree_high.item(row_id, tags=("up",))
            elif tag == "down":
                self.tree_high.tag_configure("down", foreground="#ef4444")
                self.tree_high.item(row_id, tags=("down",))

        for values, tag in payload.get("tree_low", []):
            row_id = self.tree_low.insert("", "end", values=values)
            if tag == "up":
                self.tree_low.tag_configure("up", foreground="#10b981")
                self.tree_low.item(row_id, tags=("up",))
            elif tag == "down":
                self.tree_low.tag_configure("down", foreground="#ef4444")
                self.tree_low.item(row_id, tags=("down",))

        for values in payload.get("market_rows", []):
            self.market_tree.insert("", "end", values=values)
        for values in payload.get("levels_rows", []):
            self.levels_tree.insert("", "end", values=values)
        for values in payload.get("options_rows", []):
            self.options_tree.insert("", "end", values=values)
        for values in payload.get("news_rows", []):
            row_id = self.news_tree.insert("", "end", values=values)
            tone = str(values[1]).lower() if len(values) > 1 else ""
            if "pos" in tone or "bull" in tone:
                self.news_tree.tag_configure("pos", foreground="#10b981")
                self.news_tree.item(row_id, tags=("pos",))
            elif "neg" in tone or "bear" in tone:
                self.news_tree.tag_configure("neg", foreground="#ef4444")
                self.news_tree.item(row_id, tags=("neg",))

        self.current_price_var.set(payload.get("current_price", "Current Price: -"))
        self.avg_pred_var.set(payload.get("avg_pred", "Avg Prediction: -"))
        self.consensus_var.set(payload.get("consensus", "Consensus: -"))
        self.data_meta_var.set(payload.get("data_meta", "Data: - | Last Bar: -"))
        self.week_extremes_var.set(payload.get("week_extremes", "Projected Week High/Low: -"))
        self.regime_label.config(text=payload.get("regime_text", "Market Regime: Unknown"))
        self.setup_card_var.set(payload.get("setup_card", "Alignment: unavailable"))
        self.setup_detail_var.set(payload.get("setup_detail", "-"))
        self.vol_card_var.set(payload.get("vol_card", "Volatility: unavailable"))
        self.vol_detail_var.set(payload.get("vol_detail", "-"))
        self.options_card_var.set(payload.get("options_card", "Options: unavailable"))
        self.news_card_var.set(payload.get("news_card", "News: unavailable"))
        for msg in payload.get("status_messages", []):
            self.status(msg)

    def _compute_prediction_payload(self, df_copy, models_copy, mode: str, selected: str) -> dict:
        payload = self._base_payload(mode)
        model_keys = ["base", "forest", "boost", "extra_trees"]
        dashboard_predictions = {}

        data_error = None
        try:
            data_error = getattr(df_copy, "attrs", {}).get("data_error")
        except Exception:
            data_error = None

        if not self._mode_ready(models_copy, mode):
            return self._build_waiting_payload(mode, "Models missing. Click 'Retrain Models'.", status="Models missing. Click 'Retrain Models'.")

        if self._retrain_in_progress:
            if self._training_stage:
                payload["status_messages"].append(self._training_stage)
            else:
                payload["status_messages"].append("Training other models in background…")

        last_price = float(df_copy["close"].iloc[-1])
        payload["current_price"] = f"Current Price: ${last_price:,.2f}"
        try:
            last_ts = pd.to_datetime(df_copy.index[-1], utc=True, errors="coerce")
            if pd.isna(last_ts):
                last_ts_str = "-"
            else:
                last_ts_str = last_ts.strftime("%Y-%m-%d %H:%M:%S UTC")
        except Exception:
            last_ts_str = "-"
        try:
            source = getattr(df_copy, "attrs", {}).get("data_source", "-")
        except Exception:
            source = "-"
        if data_error:
            source = f"{source} (delayed)" if source else "delayed"
        payload["data_meta"] = f"Data: {source} | Last Bar: {last_ts_str}"

        if mode == "4min":
            results = predict_4min(models_copy, df_copy, instrument_key=selected)
            if isinstance(results, dict) and results.get("error"):
                return self._build_waiting_payload(mode, results.get("error"), status=results.get("error"))
            dashboard_predictions["intraday"] = results
            try:
                meta_diag = results.get("_meta", {}) if isinstance(results, dict) else {}
                if bool(meta_diag.get("critical_stale", False)):
                    payload["status_messages"].append("No-trade guard active: critical stale data")
                else:
                    div = meta_diag.get("diversity", {})
                    if isinstance(div, dict) and bool(div.get("degenerate", False)):
                        payload["status_messages"].append("Risk-off: model outputs are near-identical")
            except Exception:
                pass
            prices = []
            for model_name in model_keys:
                label = model_name.replace("_", " ").title()
                item = results.get(model_name, {})
                note = item.get("note", "")
                pred_ret = item.get("pred_ret")
                if isinstance(pred_ret, (float, int)):
                    move_note = f"Move {pred_ret * 10000.0:+.2f} bps"
                    note = f"{note} | {move_note}" if note else move_note
                note = self._append_band_note(note, item)
                if item.get("pred_price") is None:
                    payload["tree_4min"].append(((label, "-", "-", "-", note or "Waiting…"), None))
                    continue
                direction = item.get("direction", "-")
                price = item.get("pred_price", "-")
                conf = item.get("confidence", 0.0)
                if isinstance(price, (float, int)):
                    prices.append(float(price))
                price_str = f"${price:,.2f}" if isinstance(price, (float, int)) else price
                conf_str = f"{conf:.2f}%" if isinstance(conf, (float, int)) else conf
                tag = "up" if direction == "UP" else "down" if direction == "DOWN" else None
                payload["tree_4min"].append(((label, direction, price_str, conf_str, note or ""), tag))
            if prices:
                avg_price = sum(prices) / len(prices)
                payload["avg_pred"] = f"Avg Prediction: ${avg_price:,.2f}"
            else:
                payload["avg_pred"] = "Avg Prediction: -"
            cons = results.get("consensus") if isinstance(results, dict) else None
            if isinstance(cons, dict) and cons:
                c_dir = cons.get("direction", "-")
                c_conf = cons.get("confidence", "-")
                c_size = cons.get("position_size", None)
                if isinstance(c_conf, (int, float)):
                    c_conf_str = f"{c_conf:.2f}%"
                else:
                    c_conf_str = str(c_conf)
                if isinstance(c_size, (int, float)):
                    c_size_str = f"{c_size:.2f}"
                else:
                    c_size_str = "-"
                c_note = cons.get("note", "")
                if c_note:
                    payload["consensus"] = f"Consensus: {c_dir} ({c_conf_str}) | Size: {c_size_str} | {c_note}"
                else:
                    payload["consensus"] = f"Consensus: {c_dir} ({c_conf_str}) | Size: {c_size_str}"
            else:
                payload["consensus"] = "Consensus: -"
        else:
            results = predict_horizon(models_copy, df_copy, instrument_key=selected, mode=mode)
            if not results or (isinstance(results, dict) and results.get("error")):
                msg = results.get("error") if isinstance(results, dict) else "Missing horizon results"
                return self._build_waiting_payload(mode, msg, status=msg)
            if mode == "DailyHigh":
                dashboard_predictions["daily"] = results
                high = results.get("high", {})
                low = results.get("low", {})
                if not high or not low:
                    raise RuntimeError("Missing daily high/low results")
                high_prices = []
                low_prices = []
                for model_name in model_keys:
                    label = model_name.replace("_", " ").title()
                    item = high.get(model_name, {})
                    note = item.get("note", "")
                    pred_ret = item.get("pred_ret")
                    if isinstance(pred_ret, (float, int)):
                        move_note = f"Move {pred_ret * 10000.0:+.2f} bps"
                        note = f"{note} | {move_note}" if note else move_note
                    note = self._append_band_note(note, item)
                    price = item.get("pred_price", "-")
                    conf = item.get("confidence", 0.0)
                    if isinstance(price, (float, int)):
                        high_prices.append(float(price))
                    price_str = f"${price:,.2f}" if isinstance(price, (float, int)) else price
                    conf_str = f"{conf:.2f}%" if isinstance(conf, (float, int)) else conf
                    payload["tree_high"].append(((label, price_str, conf_str, note or ""), None))

                    item_l = low.get(model_name, {})
                    note_l = item_l.get("note", "")
                    pred_ret_l = item_l.get("pred_ret")
                    if isinstance(pred_ret_l, (float, int)):
                        move_note_l = f"Move {pred_ret_l * 10000.0:+.2f} bps"
                        note_l = f"{note_l} | {move_note_l}" if note_l else move_note_l
                    note_l = self._append_band_note(note_l, item_l)
                    price_l = item_l.get("pred_price", "-")
                    conf_l = item_l.get("confidence", 0.0)
                    if isinstance(price_l, (float, int)):
                        low_prices.append(float(price_l))
                    price_l_str = f"${price_l:,.2f}" if isinstance(price_l, (float, int)) else price_l
                    conf_l_str = f"{conf_l:.2f}%" if isinstance(conf_l, (float, int)) else conf_l
                    payload["tree_low"].append(((label, price_l_str, conf_l_str, note_l or ""), None))
                if high_prices or low_prices:
                    high_avg = (sum(high_prices) / len(high_prices)) if high_prices else None
                    low_avg = (sum(low_prices) / len(low_prices)) if low_prices else None
                    if high_avg is not None and low_avg is not None:
                        payload["avg_pred"] = f"Avg High: ${high_avg:,.2f} | Avg Low: ${low_avg:,.2f}"
                    elif high_avg is not None:
                        payload["avg_pred"] = f"Avg High: ${high_avg:,.2f}"
                    elif low_avg is not None:
                        payload["avg_pred"] = f"Avg Low: ${low_avg:,.2f}"
                    else:
                        payload["avg_pred"] = "Avg Prediction: -"
                else:
                    payload["avg_pred"] = "Avg Prediction: -"
            else:
                dashboard_predictions["weekend"] = results
                week = results.get("week", {})
                if not week:
                    raise RuntimeError("Missing weekend results")
                week_prices = []
                week_highs = []
                week_lows = []
                for model_name in model_keys:
                    label = model_name.replace("_", " ").title()
                    item = week.get(model_name, {})
                    note = item.get("note", "")
                    pred_ret = item.get("pred_ret")
                    if isinstance(pred_ret, (float, int)):
                        move_note = f"Move {pred_ret * 10000.0:+.2f} bps"
                        note = f"{note} | {move_note}" if note else move_note
                    note = self._append_band_note(note, item)
                    price = item.get("pred_price", "-")
                    conf = item.get("confidence", 0.0)
                    if isinstance(price, (float, int)):
                        week_prices.append(float(price))
                    high_price = item.get("pred_high_price")
                    low_price = item.get("pred_low_price")
                    if isinstance(high_price, (float, int)):
                        week_highs.append(float(high_price))
                    if isinstance(low_price, (float, int)):
                        week_lows.append(float(low_price))
                    price_str = f"${price:,.2f}" if isinstance(price, (float, int)) else price
                    conf_str = f"{conf:.2f}%" if isinstance(conf, (float, int)) else conf
                    payload["tree_high"].append(((label, price_str, conf_str, note or ""), None))
                if week_prices:
                    avg_price = sum(week_prices) / len(week_prices)
                    payload["avg_pred"] = f"Avg Week Close: ${avg_price:,.2f}"
                else:
                    payload["avg_pred"] = "Avg Prediction: -"
                if week_highs or week_lows:
                    high_avg = (sum(week_highs) / len(week_highs)) if week_highs else None
                    low_avg = (sum(week_lows) / len(week_lows)) if week_lows else None
                    if high_avg is not None and low_avg is not None:
                        payload["week_extremes"] = f"Projected Week High: ${high_avg:,.2f} | Projected Week Low: ${low_avg:,.2f}"
                    elif high_avg is not None:
                        payload["week_extremes"] = f"Projected Week High: ${high_avg:,.2f}"
                    elif low_avg is not None:
                        payload["week_extremes"] = f"Projected Week Low: ${low_avg:,.2f}"
            payload["consensus"] = "Consensus: -"

        regime = results.get("regime", {"Neutral": 100.0})
        parts = [f"{int(round(regime.get(l, 0)))}% {l}" for l in ["Bullish", "Neutral", "Bearish"]]
        payload["regime_text"] = f"Market Regime: {' / '.join(parts)}"

        dashboard = None
        try:
            last_bar_key = None
            try:
                last_bar = pd.to_datetime(df_copy.index[-1], utc=True, errors="coerce")
                if pd.notna(last_bar):
                    last_bar_key = str(last_bar.isoformat())
            except Exception:
                last_bar_key = None
            prediction_sig = _dashboard_prediction_signature(dashboard_predictions)
            cached = self._dashboard_cache.get(str(selected).upper())
            if isinstance(cached, dict):
                age = time.time() - float(cached.get("ts", 0.0) or 0.0)
                if (
                    cached.get("bar_key") == last_bar_key
                    and cached.get("prediction_sig") == prediction_sig
                    and age <= float(self._dashboard_cache_ttl_sec)
                ):
                    dashboard = cached.get("dashboard")
            if dashboard is None:
                try:
                    if "intraday" not in dashboard_predictions:
                        aux = predict_4min(models_copy, df_copy, instrument_key=selected, log_side_effects=False)
                        if isinstance(aux, dict) and not aux.get("error"):
                            dashboard_predictions["intraday"] = aux
                    if "daily" not in dashboard_predictions:
                        aux = predict_horizon(models_copy, df_copy, instrument_key=selected, mode="DailyHigh", log_side_effects=False)
                        if isinstance(aux, dict) and not aux.get("error"):
                            dashboard_predictions["daily"] = aux
                    if "weekend" not in dashboard_predictions:
                        aux = predict_horizon(models_copy, df_copy, instrument_key=selected, mode="WeekEnd", log_side_effects=False)
                        if isinstance(aux, dict) and not aux.get("error"):
                            dashboard_predictions["weekend"] = aux
                except Exception:
                    logger.debug("Aux dashboard prediction refresh failed", exc_info=True)
                dashboard = build_trader_dashboard(df_copy, selected, predictions=dashboard_predictions)
                self._dashboard_cache[str(selected).upper()] = {
                    "ts": time.time(),
                    "bar_key": last_bar_key,
                    "prediction_sig": _dashboard_prediction_signature(dashboard_predictions),
                    "dashboard": dashboard,
                }
        except Exception:
            logger.exception("Dashboard analytics build failed")
            dashboard = None
        try:
            if isinstance(dashboard, dict):
                payload["setup_card"] = dashboard.get("setup_card", payload["setup_card"])
                payload["setup_detail"] = dashboard.get("setup_detail", payload["setup_detail"])
                payload["vol_card"] = dashboard.get("vol_card", payload["vol_card"])
                payload["vol_detail"] = dashboard.get("vol_detail", payload["vol_detail"])
                payload["options_card"] = dashboard.get("options_card", payload["options_card"])
                payload["news_card"] = dashboard.get("news_card", payload["news_card"])
                payload["market_rows"] = dashboard.get("market_rows", [])
                payload["levels_rows"] = dashboard.get("levels_rows", [])
                payload["options_rows"] = dashboard.get("options_rows", [])
                payload["news_rows"] = dashboard.get("news_rows", [])
        except Exception:
            logger.exception("Dashboard analytics build failed")
        payload["status_messages"].append("Predicted")
        return payload

    def run_prediction_and_refresh_ui(self):
        mode = self.mode_var.get()
        with self._lock:
            df_copy = self.df if self.df is not None else None
            models_copy = dict(self.models) if self.models else {}

        if df_copy is None or getattr(df_copy, "empty", True) or not models_copy:
            payload = self._build_waiting_payload(mode, "Waiting…")
            self.root.after(0, lambda: self._apply_prediction_payload(payload))
            return

        with self._pred_lock:
            if self._pred_in_flight:
                if self._pred_started_at is not None:
                    try:
                        if (time.time() - self._pred_started_at) > self._pred_timeout_sec:
                            logger.warning("Prediction job timed out; allowing a new run")
                            self._pred_in_flight = False
                    except Exception:
                        pass
            if self._pred_in_flight:
                return
            self._pred_in_flight = True
            self._pred_started_at = time.time()
            self._pred_seq += 1
            seq = self._pred_seq

        selected = self.instrument_var.get()

        def _job():
            try:
                payload = self._compute_prediction_payload(df_copy, models_copy, mode, selected)
            except Exception as e:
                logger.exception("Prediction UI update failed")
                payload = self._build_error_payload(mode, str(e))
            finally:
                with self._pred_lock:
                    self._pred_in_flight = False
                    self._pred_started_at = None
            try:
                def _apply_if_latest():
                    with self._pred_lock:
                        if seq != self._pred_seq:
                            return
                    self._apply_prediction_payload(payload)
                self.root.after(0, _apply_if_latest)
            except Exception:
                pass

        try:
            self.executor.submit(_job)
        except Exception:
            with self._pred_lock:
                self._pred_in_flight = False
                self._pred_started_at = None
            logger.exception("Failed to schedule prediction job")

    def background_loop(self):
        while not self._stop_event.is_set():
            if self.auto_var.get():
                try:
                    self._refresh_instrument_choices()
                    selected = self.instrument_var.get()
                    df = update_price_data(selected)
                    models = load_models(instrument_key=selected)
                    with self._lock:
                        self.df = self._trim_df_for_ui(df)
                        self.models = models
                    self.root.after(0, self.run_prediction_and_refresh_ui)
                    if (time.time() - self._overview_last_refresh) >= float(self._overview_refresh_sec):
                        self.root.after(0, self.refresh_overview_threaded)
                except Exception:
                    logger.exception("Background update error")
                    self.status("Background update error; see logs")
            time.sleep(AUTO_PREDICT_SECONDS)

    def shutdown(self):
        self._stop_event.set()
        try:
            self.executor.shutdown(wait=False)
        except Exception:
            pass

def launch_gui():
    root = tk.Tk()
    app = LightningGUI(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.shutdown(), root.destroy()))
    root.mainloop()
