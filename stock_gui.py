#stock_gui.py
import json
import os
import threading
import tkinter as tk
from tkinter import ttk
from tkinter import font as tkfont
from tkinter import messagebox
import webbrowser
import pandas as pd
from app_meta import APP_DISPLAY_NAME, APP_VERSION, GITHUB_RELEASES_REPO, GITHUB_RELEASES_URL, app_title
from app_paths import get_runtime_logs_dir, get_window_icon_path
from model_utils import load_models, predict_4min, predict_horizon
from data_fetch import update_price_data
from config import AUTO_PREDICT_SECONDS, INSTRUMENTS, DEFAULT_INSTRUMENT
from execution_engine import (
    clear_local_broker_credentials,
    cancel_routed_order,
    flatten_paper_position,
    refresh_broker_connection,
    run_broker_verification,
    save_local_broker_credentials,
    place_paper_order,
    place_routed_order,
)
from features import LIVE_MAX_BARS, BAR_MINUTES
from model_trainers import train_all_models
from instrument_onboarding import ensure_runtime_instruments, list_runtime_instruments
from readiness_tracker import evaluate_readiness_metrics
from replay_engine import (
    flatten_replay_position,
    get_active_replay_frame,
    get_replay_dashboard_snapshot,
    place_replay_order,
    reset_active_replay,
    set_replay_paused,
    start_active_replay,
    step_active_replay,
    stop_active_replay,
)
from trader_analytics import build_trader_dashboard
from trader_profiles import get_trader_profile, list_trader_profiles
from update_checker import get_update_status
import datetime
import time
import logging
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    try:
        log_dir = get_runtime_logs_dir()
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(str(log_dir), "lightning_gui.log")
    except Exception:
        log_path = "lightning_gui.log"
    logging.basicConfig(filename=log_path, level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


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

    def _avg_metric(block: dict | None, key: str):
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
            "high_p_up": _avg_metric(daily.get("high"), "p_up"),
            "low_p_up": _avg_metric(daily.get("low"), "p_up"),
            "high_conf": _avg_metric(daily.get("high"), "confidence"),
            "low_conf": _avg_metric(daily.get("low"), "confidence"),
        }
    weekend = predictions.get("weekend")
    if isinstance(weekend, dict):
        summary["weekend"] = {
            "close_avg": _avg_price(weekend.get("week")),
            "high_avg": _avg_price(weekend.get("week"), key="pred_high_price"),
            "low_avg": _avg_price(weekend.get("week"), key="pred_low_price"),
            "close_p_up": _avg_metric(weekend.get("week"), "p_up"),
            "close_conf": _avg_metric(weekend.get("week"), "confidence"),
        }
    try:
        return json.dumps(summary, sort_keys=True, separators=(",", ":"))
    except Exception:
        return repr(summary)

class LightningGUI:
    def __init__(self, root):
        self.root = root
        self.root.title(app_title())
        self.root.geometry("1540x980")
        self.root.minsize(1320, 860)
        self.root.configure(bg="#081018")
        self._window_icon = None
        self._settings_window = None
        self._about_window = None
        self._help_window = None

        self.header_font = tkfont.Font(family="Avenir Next", size=19, weight="bold")
        self.big_font = tkfont.Font(family="Avenir Next", size=15, weight="bold")
        self.cell_font = tkfont.Font(family="Avenir Next", size=12)
        self.small_font = tkfont.Font(family="Avenir Next", size=11)
        self.mono_font = tkfont.Font(family="Menlo", size=10)

        self._apply_window_icon()
        self._build_menu_bar()

        self.style = ttk.Style(root)
        self.style.theme_use("clam")
        self.style.configure("TButton", font=self.small_font, padding=8, foreground="white", background="#162130", borderwidth=0)
        self.style.map("TButton", background=[("active", "#22304a"), ("pressed", "#1b263b")])
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
        self._latest_advisory_meta = {}
        self._latest_execution_meta = {}
        self._latest_replay_meta = {}
        self._latest_price = None
        self._alpaca_setup_window = None
        self._alpaca_setup_prompted = False

        self.executor = ThreadPoolExecutor(max_workers=2)
        self._overview_last_refresh = 0.0
        self._overview_refresh_sec = max(120, int(AUTO_PREDICT_SECONDS))
        self._update_check_in_flight = False
        self._latest_update_meta = {}

        # Top frame
        self.top_frame = tk.Frame(self.root, bg="#081018")
        self.top_frame.pack(fill="x", padx=12, pady=8)

        self.brand_frame = tk.Frame(self.top_frame, bg="#081018")
        self.brand_frame.pack(side="left", padx=(0, 18))
        self.brand_var = tk.StringVar(value=APP_DISPLAY_NAME)
        self.brand_subtitle_var = tk.StringVar(value="Trading workstation | paper-safe by default")
        self.version_badge_var = tk.StringVar(value=f"v{APP_VERSION}")
        tk.Label(self.brand_frame, textvariable=self.brand_var, font=self.header_font, fg="#f8fafc", bg="#081018").pack(anchor="w")
        sub_row = tk.Frame(self.brand_frame, bg="#081018")
        sub_row.pack(anchor="w")
        tk.Label(sub_row, textvariable=self.brand_subtitle_var, font=self.small_font, fg="#8fb4d9", bg="#081018").pack(side="left")
        tk.Label(sub_row, textvariable=self.version_badge_var, font=self.small_font, fg="#fbbf24", bg="#10203a", padx=8, pady=2).pack(side="left", padx=(10, 0))

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

        self.profile_var = tk.StringVar(value=get_trader_profile().get("key", "balanced"))
        profile_frame = tk.Frame(self.top_frame, bg="#081018")
        profile_frame.pack(side="left", padx=12)
        tk.Label(profile_frame, text="Trader Style:", bg="#081018", fg="white", font=self.small_font).pack(side="left", padx=(0, 6))
        self.profile_menu = ttk.Combobox(
            profile_frame,
            values=list_trader_profiles(),
            textvariable=self.profile_var,
            state="readonly",
            width=13,
        )
        self.profile_menu.pack(side="left")
        self.profile_menu.bind("<<ComboboxSelected>>", self._on_profile_change)

        # Mode selection: 4min, DailyHigh, WeekEnd
        self.mode_var = tk.StringVar(value="4min")
        mode_frame = tk.Frame(self.top_frame, bg="#081018")
        mode_frame.pack(side="left", padx=12)
        tk.Label(mode_frame, text="Mode:", bg="#081018", fg="white", font=self.small_font).pack(side="left", padx=(0,6))
        self.mode_menu = ttk.Combobox(mode_frame, values=["4min", "DailyHigh", "WeekEnd"], textvariable=self.mode_var, state="readonly", width=12)
        self.mode_menu.pack(side="left")
        self.mode_apply_btn = ttk.Button(mode_frame, text="Apply Mode", command=self.mode_apply)
        self.mode_apply_btn.pack(side="left", padx=(6,0))

        trade_frame = tk.Frame(self.top_frame, bg="#081018")
        trade_frame.pack(side="left", padx=12)
        self.paper_buy_btn = ttk.Button(trade_frame, text="Local Buy", command=lambda: self._place_paper_order("buy"))
        self.paper_buy_btn.pack(side="left", padx=(0, 6))
        self.paper_sell_btn = ttk.Button(trade_frame, text="Local Sell", command=lambda: self._place_paper_order("sell"))
        self.paper_sell_btn.pack(side="left", padx=(0, 6))
        self.paper_flatten_btn = ttk.Button(trade_frame, text="Local Flat", command=self._flatten_paper_position)
        self.paper_flatten_btn.pack(side="left")

        routed_frame = tk.Frame(self.top_frame, bg="#081018")
        routed_frame.pack(side="left", padx=12)
        self.route_buy_var = tk.StringVar(value="Buy (Local Paper)")
        self.route_sell_var = tk.StringVar(value="Sell (Local Paper)")
        self.route_cancel_var = tk.StringVar(value="Cancel Routed")
        self.route_setup_var = tk.StringVar(value="Alpaca Setup")
        self.route_refresh_var = tk.StringVar(value="Refresh Broker")
        self.route_verify_var = tk.StringVar(value="Verify Alpaca Paper")
        self.route_buy_btn = ttk.Button(routed_frame, textvariable=self.route_buy_var, command=lambda: self._place_routed_order("buy"))
        self.route_buy_btn.pack(side="left", padx=(0, 6))
        self.route_sell_btn = ttk.Button(routed_frame, textvariable=self.route_sell_var, command=lambda: self._place_routed_order("sell"))
        self.route_sell_btn.pack(side="left", padx=(0, 6))
        self.route_cancel_btn = ttk.Button(routed_frame, textvariable=self.route_cancel_var, command=self._cancel_routed_order)
        self.route_cancel_btn.pack(side="left")
        self.route_setup_btn = ttk.Button(routed_frame, textvariable=self.route_setup_var, command=self._open_alpaca_setup)
        self.route_setup_btn.pack(side="left", padx=(6, 6))
        self.route_refresh_btn = ttk.Button(routed_frame, textvariable=self.route_refresh_var, command=self._refresh_broker_status)
        self.route_refresh_btn.pack(side="left", padx=(6, 6))
        self.route_verify_btn = ttk.Button(routed_frame, textvariable=self.route_verify_var, command=self._verify_broker)
        self.route_verify_btn.pack(side="left")

        replay_frame = tk.Frame(self.top_frame, bg="#081018")
        replay_frame.pack(side="left", padx=12)
        self.replay_toggle_var = tk.StringVar(value="Replay Resume")
        self.replay_start_btn = ttk.Button(replay_frame, text="Replay Start", command=self._start_replay)
        self.replay_start_btn.pack(side="left", padx=(0, 6))
        self.replay_step_btn = ttk.Button(replay_frame, text="Replay Step", command=self._step_replay)
        self.replay_step_btn.pack(side="left", padx=(0, 6))
        self.replay_toggle_btn = ttk.Button(replay_frame, textvariable=self.replay_toggle_var, command=self._toggle_replay)
        self.replay_toggle_btn.pack(side="left", padx=(0, 6))
        self.replay_reset_btn = ttk.Button(replay_frame, text="Replay Reset", command=self._reset_replay)
        self.replay_reset_btn.pack(side="left", padx=(0, 6))
        self.replay_stop_btn = ttk.Button(replay_frame, text="Replay Stop", command=self._stop_replay)
        self.replay_stop_btn.pack(side="left", padx=(0, 6))
        self.replay_buy_btn = ttk.Button(replay_frame, text="Replay Buy", command=lambda: self._place_replay_order("buy"))
        self.replay_buy_btn.pack(side="left", padx=(0, 6))
        self.replay_sell_btn = ttk.Button(replay_frame, text="Replay Sell", command=lambda: self._place_replay_order("sell"))
        self.replay_sell_btn.pack(side="left", padx=(0, 6))
        self.replay_flat_btn = ttk.Button(replay_frame, text="Replay Flat", command=self._flatten_replay_position)
        self.replay_flat_btn.pack(side="left")

        app_actions_frame = tk.Frame(self.top_frame, bg="#081018")
        app_actions_frame.pack(side="right", padx=(12, 0))
        ttk.Button(app_actions_frame, text="Settings", command=self._open_settings_dialog).pack(side="left", padx=(0, 6))
        ttk.Button(app_actions_frame, text="Updates", command=lambda: self._check_for_updates(background=False)).pack(side="left", padx=(0, 6))
        ttk.Button(app_actions_frame, text="Help", command=self._open_help_dialog).pack(side="left")

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
        self.profile_summary_var = tk.StringVar(value="Profile: -")
        self.profile_summary_label = tk.Label(self.summary_frame, textvariable=self.profile_summary_var, font=self.small_font, fg="#f5d0fe", bg="#081018")
        self.profile_summary_label.pack(side="left", padx=12)
        self.profile_summary_var.set(f"Profile: {get_trader_profile(self.profile_var.get()).get('label', 'Balanced')}")
        self.data_meta_var = tk.StringVar(value="Data: - | Last Bar: -")
        self.data_meta_label = tk.Label(self.summary_frame, textvariable=self.data_meta_var, font=self.small_font, fg="#cbd5f5", bg="#081018")
        self.data_meta_label.pack(side="right", padx=6)
        self.update_status_var = tk.StringVar(value=f"Version {APP_VERSION} | Updates not checked")
        self.update_status_label = tk.Label(self.summary_frame, textvariable=self.update_status_var, font=self.small_font, fg="#86efac", bg="#081018")
        self.update_status_label.pack(side="right", padx=10)

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

        self.right_canvas = tk.Canvas(self.right_panel, bg="#081018", highlightthickness=0, bd=0)
        self.right_scrollbar = ttk.Scrollbar(self.right_panel, orient="vertical", command=self.right_canvas.yview)
        self.right_canvas.configure(yscrollcommand=self.right_scrollbar.set)
        self.right_scrollbar.pack(side="right", fill="y")
        self.right_canvas.pack(side="left", fill="both", expand=True)
        self.right_body = tk.Frame(self.right_canvas, bg="#081018")
        self._right_body_window = self.right_canvas.create_window((0, 0), window=self.right_body, anchor="nw")
        self.right_body.bind(
            "<Configure>",
            lambda e: self.right_canvas.configure(scrollregion=self.right_canvas.bbox("all")),
        )
        self.right_canvas.bind(
            "<Configure>",
            lambda e: self.right_canvas.itemconfigure(self._right_body_window, width=e.width),
        )

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

        self.cards_frame = tk.Frame(self.right_body, bg="#081018")
        self.cards_frame.pack(fill="x", pady=(0, 8))
        self.setup_card_var = tk.StringVar(value="Alignment: loading…")
        self.setup_detail_var = tk.StringVar(value="-")
        self.action_card_var = tk.StringVar(value="Action: loading…")
        self.action_detail_var = tk.StringVar(value="-")
        self.quality_card_var = tk.StringVar(value="Quality: loading…")
        self.quality_detail_var = tk.StringVar(value="-")
        self.vol_card_var = tk.StringVar(value="Volatility: loading…")
        self.vol_detail_var = tk.StringVar(value="-")
        self.options_card_var = tk.StringVar(value="Options: loading…")
        self.news_card_var = tk.StringVar(value="News: loading…")
        self.profile_card_var = tk.StringVar(value="Profile: loading…")
        self.profile_detail_var = tk.StringVar(value="-")
        self.broker_card_var = tk.StringVar(value="Broker: loading…")
        self.broker_detail_var = tk.StringVar(value="-")
        self.replay_card_var = tk.StringVar(value="Replay: loading…")
        self.replay_detail_var = tk.StringVar(value="-")

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
        _make_card(self.cards_frame, "Action", self.action_card_var, self.action_detail_var, accent="#fbbf24")
        _make_card(self.cards_frame, "Signal Quality", self.quality_card_var, self.quality_detail_var, accent="#c084fc")
        _make_card(self.cards_frame, "Volatility", self.vol_card_var, self.vol_detail_var, accent="#fbbf24")
        _make_card(self.cards_frame, "Options", self.options_card_var, None, accent="#c084fc")
        _make_card(self.cards_frame, "News", self.news_card_var, None, accent="#7dd3fc")
        _make_card(self.cards_frame, "Trader Profile", self.profile_card_var, self.profile_detail_var, accent="#f5d0fe")
        _make_card(self.cards_frame, "Broker / Forward Test", self.broker_card_var, self.broker_detail_var, accent="#fdba74")
        _make_card(self.cards_frame, "Replay / Simulation", self.replay_card_var, self.replay_detail_var, accent="#67e8f9")

        self.market_label = tk.Label(self.right_body, text="Market Context", font=self.big_font, fg="#e2e8f0", bg="#081018")
        self.market_label.pack(anchor="w", pady=(2, 2))
        self.market_tree = ttk.Treeview(self.right_body, columns=("item", "value", "detail"), show="headings", height=5)
        for col, width in (("item", 120), ("value", 90), ("detail", 170)):
            self.market_tree.heading(col, text=col.title())
            self.market_tree.column(col, anchor=tk.CENTER if col != "detail" else tk.W, width=width)
        self.market_tree.pack(fill="x", pady=(0, 8))

        self.signal_label = tk.Label(self.right_body, text="Signal Context", font=self.big_font, fg="#e2e8f0", bg="#081018")
        self.signal_label.pack(anchor="w", pady=(2, 2))
        self.signal_tree = ttk.Treeview(self.right_body, columns=("item", "value", "detail"), show="headings", height=6)
        for col, width in (("item", 120), ("value", 100), ("detail", 160)):
            self.signal_tree.heading(col, text=col.title())
            self.signal_tree.column(col, anchor=tk.CENTER if col != "detail" else tk.W, width=width)
        self.signal_tree.pack(fill="x", pady=(0, 8))

        self.breadth_label = tk.Label(self.right_body, text="Breadth / Internals", font=self.big_font, fg="#e2e8f0", bg="#081018")
        self.breadth_label.pack(anchor="w", pady=(2, 2))
        self.breadth_tree = ttk.Treeview(self.right_body, columns=("item", "value", "detail"), show="headings", height=5)
        for col, width in (("item", 120), ("value", 100), ("detail", 170)):
            self.breadth_tree.heading(col, text=col.title())
            self.breadth_tree.column(col, anchor=tk.CENTER if col != "detail" else tk.W, width=width)
        self.breadth_tree.pack(fill="x", pady=(0, 8))

        self.macro_label = tk.Label(self.right_body, text="Macro / Event Risk", font=self.big_font, fg="#e2e8f0", bg="#081018")
        self.macro_label.pack(anchor="w", pady=(2, 2))
        self.macro_tree = ttk.Treeview(self.right_body, columns=("item", "value", "detail"), show="headings", height=5)
        for col, width in (("item", 120), ("value", 110), ("detail", 160)):
            self.macro_tree.heading(col, text=col.title())
            self.macro_tree.column(col, anchor=tk.CENTER if col != "detail" else tk.W, width=width)
        self.macro_tree.pack(fill="x", pady=(0, 8))

        self.stack_label = tk.Label(self.right_body, text="Advisory Stack", font=self.big_font, fg="#e2e8f0", bg="#081018")
        self.stack_label.pack(anchor="w", pady=(2, 2))
        self.stack_tree = ttk.Treeview(self.right_body, columns=("item", "value", "detail"), show="headings", height=6)
        for col, width in (("item", 120), ("value", 100), ("detail", 160)):
            self.stack_tree.heading(col, text=col.title())
            self.stack_tree.column(col, anchor=tk.CENTER if col != "detail" else tk.W, width=width)
        self.stack_tree.pack(fill="x", pady=(0, 8))

        self.probability_label = tk.Label(self.right_body, text="Probability Map", font=self.big_font, fg="#e2e8f0", bg="#081018")
        self.probability_label.pack(anchor="w", pady=(2, 2))
        self.probability_tree = ttk.Treeview(self.right_body, columns=("item", "value", "detail"), show="headings", height=6)
        for col, width in (("item", 125), ("value", 105), ("detail", 190)):
            self.probability_tree.heading(col, text=col.title())
            self.probability_tree.column(col, anchor=tk.CENTER if col != "detail" else tk.W, width=width)
        self.probability_tree.pack(fill="x", pady=(0, 8))

        self.quality_label = tk.Label(self.right_body, text="Signal Quality", font=self.big_font, fg="#e2e8f0", bg="#081018")
        self.quality_label.pack(anchor="w", pady=(2, 2))
        self.quality_tree = ttk.Treeview(self.right_body, columns=("item", "value", "detail"), show="headings", height=6)
        for col, width in (("item", 125), ("value", 105), ("detail", 190)):
            self.quality_tree.heading(col, text=col.title())
            self.quality_tree.column(col, anchor=tk.CENTER if col != "detail" else tk.W, width=width)
        self.quality_tree.pack(fill="x", pady=(0, 8))

        self.liquidity_label = tk.Label(self.right_body, text="Liquidity / Execution", font=self.big_font, fg="#e2e8f0", bg="#081018")
        self.liquidity_label.pack(anchor="w", pady=(2, 2))
        self.liquidity_tree = ttk.Treeview(self.right_body, columns=("item", "value", "detail"), show="headings", height=6)
        for col, width in (("item", 125), ("value", 105), ("detail", 190)):
            self.liquidity_tree.heading(col, text=col.title())
            self.liquidity_tree.column(col, anchor=tk.CENTER if col != "detail" else tk.W, width=width)
        self.liquidity_tree.pack(fill="x", pady=(0, 8))

        self.catalyst_label = tk.Label(self.right_body, text="Catalyst Context", font=self.big_font, fg="#e2e8f0", bg="#081018")
        self.catalyst_label.pack(anchor="w", pady=(2, 2))
        self.catalyst_tree = ttk.Treeview(self.right_body, columns=("item", "value", "detail"), show="headings", height=5)
        for col, width in (("item", 125), ("value", 105), ("detail", 190)):
            self.catalyst_tree.heading(col, text=col.title())
            self.catalyst_tree.column(col, anchor=tk.CENTER if col != "detail" else tk.W, width=width)
        self.catalyst_tree.pack(fill="x", pady=(0, 8))

        self.alerts_label = tk.Label(self.right_body, text="Trader Alerts", font=self.big_font, fg="#e2e8f0", bg="#081018")
        self.alerts_label.pack(anchor="w", pady=(2, 2))
        self.alerts_tree = ttk.Treeview(self.right_body, columns=("item", "value", "detail"), show="headings", height=5)
        for col, width in (("item", 125), ("value", 95), ("detail", 200)):
            self.alerts_tree.heading(col, text=col.title())
            self.alerts_tree.column(col, anchor=tk.CENTER if col != "detail" else tk.W, width=width)
        self.alerts_tree.pack(fill="x", pady=(0, 8))

        self.execution_label = tk.Label(self.right_body, text="Execution", font=self.big_font, fg="#e2e8f0", bg="#081018")
        self.execution_label.pack(anchor="w", pady=(2, 2))
        self.execution_tree = ttk.Treeview(self.right_body, columns=("item", "value", "detail"), show="headings", height=5)
        for col, width in (("item", 125), ("value", 105), ("detail", 190)):
            self.execution_tree.heading(col, text=col.title())
            self.execution_tree.column(col, anchor=tk.CENTER if col != "detail" else tk.W, width=width)
        self.execution_tree.pack(fill="x", pady=(0, 8))

        self.replay_label = tk.Label(self.right_body, text="Replay State", font=self.big_font, fg="#e2e8f0", bg="#081018")
        self.replay_label.pack(anchor="w", pady=(2, 2))
        self.replay_tree = ttk.Treeview(self.right_body, columns=("item", "value", "detail"), show="headings", height=5)
        for col, width in (("item", 125), ("value", 105), ("detail", 190)):
            self.replay_tree.heading(col, text=col.title())
            self.replay_tree.column(col, anchor=tk.CENTER if col != "detail" else tk.W, width=width)
        self.replay_tree.pack(fill="x", pady=(0, 8))

        self.portfolio_label = tk.Label(self.right_body, text="Portfolio", font=self.big_font, fg="#e2e8f0", bg="#081018")
        self.portfolio_label.pack(anchor="w", pady=(2, 2))
        self.portfolio_tree = ttk.Treeview(self.right_body, columns=("item", "value", "detail"), show="headings", height=4)
        for col, width in (("item", 125), ("value", 105), ("detail", 190)):
            self.portfolio_tree.heading(col, text=col.title())
            self.portfolio_tree.column(col, anchor=tk.CENTER if col != "detail" else tk.W, width=width)
        self.portfolio_tree.pack(fill="x", pady=(0, 8))

        self.performance_label = tk.Label(self.right_body, text="Paper Performance", font=self.big_font, fg="#e2e8f0", bg="#081018")
        self.performance_label.pack(anchor="w", pady=(2, 2))
        self.performance_tree = ttk.Treeview(self.right_body, columns=("item", "value", "detail"), show="headings", height=6)
        for col, width in (("item", 125), ("value", 105), ("detail", 190)):
            self.performance_tree.heading(col, text=col.title())
            self.performance_tree.column(col, anchor=tk.CENTER if col != "detail" else tk.W, width=width)
        self.performance_tree.pack(fill="x", pady=(0, 8))

        self.opportunity_label = tk.Label(self.right_body, text="Opportunity Scan", font=self.big_font, fg="#e2e8f0", bg="#081018")
        self.opportunity_label.pack(anchor="w", pady=(2, 2))
        self.opportunity_tree = ttk.Treeview(self.right_body, columns=("item", "value", "detail"), show="headings", height=5)
        for col, width in (("item", 125), ("value", 95), ("detail", 200)):
            self.opportunity_tree.heading(col, text=col.title())
            self.opportunity_tree.column(col, anchor=tk.CENTER if col != "detail" else tk.W, width=width)
        self.opportunity_tree.pack(fill="x", pady=(0, 8))

        self.setups_label = tk.Label(self.right_body, text="Trade Plan", font=self.big_font, fg="#e2e8f0", bg="#081018")
        self.setups_label.pack(anchor="w", pady=(2, 2))
        self.setup_tree = ttk.Treeview(self.right_body, columns=("item", "value", "detail"), show="headings", height=5)
        for col, width in (("item", 125), ("value", 105), ("detail", 190)):
            self.setup_tree.heading(col, text=col.title())
            self.setup_tree.column(col, anchor=tk.CENTER if col != "detail" else tk.W, width=width)
        self.setup_tree.pack(fill="x", pady=(0, 8))

        self.scenario_label = tk.Label(self.right_body, text="Scenario Map", font=self.big_font, fg="#e2e8f0", bg="#081018")
        self.scenario_label.pack(anchor="w", pady=(2, 2))
        self.scenario_tree = ttk.Treeview(self.right_body, columns=("item", "value", "detail"), show="headings", height=4)
        for col, width in (("item", 125), ("value", 140), ("detail", 160)):
            self.scenario_tree.heading(col, text=col.title())
            self.scenario_tree.column(col, anchor=tk.CENTER if col != "detail" else tk.W, width=width)
        self.scenario_tree.pack(fill="x", pady=(0, 8))

        self.invalidate_label = tk.Label(self.right_body, text="Invalidation / No-Trade", font=self.big_font, fg="#e2e8f0", bg="#081018")
        self.invalidate_label.pack(anchor="w", pady=(2, 2))
        self.invalidate_tree = ttk.Treeview(self.right_body, columns=("item", "value", "detail"), show="headings", height=5)
        for col, width in (("item", 125), ("value", 105), ("detail", 190)):
            self.invalidate_tree.heading(col, text=col.title())
            self.invalidate_tree.column(col, anchor=tk.CENTER if col != "detail" else tk.W, width=width)
        self.invalidate_tree.pack(fill="x", pady=(0, 8))

        self.reliability_label = tk.Label(self.right_body, text="Historical Reliability", font=self.big_font, fg="#e2e8f0", bg="#081018")
        self.reliability_label.pack(anchor="w", pady=(2, 2))
        self.reliability_tree = ttk.Treeview(self.right_body, columns=("item", "value", "detail"), show="headings", height=6)
        for col, width in (("item", 125), ("value", 105), ("detail", 190)):
            self.reliability_tree.heading(col, text=col.title())
            self.reliability_tree.column(col, anchor=tk.CENTER if col != "detail" else tk.W, width=width)
        self.reliability_tree.pack(fill="x", pady=(0, 8))

        self.levels_label = tk.Label(self.right_body, text="Key Levels", font=self.big_font, fg="#e2e8f0", bg="#081018")
        self.levels_label.pack(anchor="w", pady=(2, 2))
        self.levels_tree = ttk.Treeview(self.right_body, columns=("item", "value", "detail"), show="headings", height=8)
        for col, width in (("item", 120), ("value", 110), ("detail", 150)):
            self.levels_tree.heading(col, text=col.title())
            self.levels_tree.column(col, anchor=tk.CENTER if col != "detail" else tk.W, width=width)
        self.levels_tree.pack(fill="x", pady=(0, 8))

        self.options_label = tk.Label(self.right_body, text="Options Positioning", font=self.big_font, fg="#e2e8f0", bg="#081018")
        self.options_label.pack(anchor="w", pady=(2, 2))
        self.options_tree = ttk.Treeview(self.right_body, columns=("item", "value", "detail"), show="headings", height=5)
        for col, width in (("item", 120), ("value", 110), ("detail", 150)):
            self.options_tree.heading(col, text=col.title())
            self.options_tree.column(col, anchor=tk.CENTER if col != "detail" else tk.W, width=width)
        self.options_tree.pack(fill="x", pady=(0, 8))

        self.news_label = tk.Label(self.right_body, text="Recent Catalysts", font=self.big_font, fg="#e2e8f0", bg="#081018")
        self.news_label.pack(anchor="w", pady=(2, 2))
        self.news_tree = ttk.Treeview(self.right_body, columns=("age", "tone", "headline"), show="headings", height=7)
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
        self._sync_action_buttons()
        try:
            self.root.after(900, self._maybe_prompt_alpaca_setup)
            self.root.after(1500, lambda: self._check_for_updates(background=True))
        except Exception:
            pass

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

    def _on_profile_change(self, event=None):
        profile = get_trader_profile(self.profile_var.get())
        self._dashboard_cache.clear()
        self.profile_summary_var.set(f"Profile: {profile.get('label', 'Balanced')}")
        self.status(f"Trader style set to {profile.get('label', 'Balanced')}; refreshing advisory context…")
        self.run_prediction_and_refresh_ui()

    def _current_replay_snapshot(self):
        try:
            return get_replay_dashboard_snapshot(str(self.instrument_var.get() or "").upper())
        except Exception:
            logger.debug("Replay snapshot lookup failed", exc_info=True)
            return {"active": False, "status": "inactive"}

    def _current_prediction_inputs(self):
        selected = str(self.instrument_var.get() or "").upper()
        replay_snapshot = self._current_replay_snapshot()
        with self._lock:
            df_copy = self.df if self.df is not None else None
            models_copy = dict(self.models) if self.models else {}
        if replay_snapshot.get("active"):
            try:
                replay_df = get_active_replay_frame(selected)
                if replay_df is not None and not getattr(replay_df, "empty", True):
                    df_copy = self._trim_df_for_ui(replay_df)
            except Exception:
                logger.debug("Replay frame build failed", exc_info=True)
        return df_copy, models_copy, selected, replay_snapshot

    def _place_paper_order(self, side: str):
        selected = str(self.instrument_var.get() or "").upper()
        current_price = self._latest_price
        result = place_paper_order(
            selected,
            side,
            current_price=current_price,
            advisory=self._latest_advisory_meta,
            profile_name=self.profile_var.get(),
        )
        status = str(result.get("status") or "-")
        if status == "filled":
            fill = result.get("fill") or {}
            self.status(
                f"Paper {str(side).upper()} filled: {selected} {int(float(fill.get('qty') or 0))} @ ${float(fill.get('fill_price') or 0.0):.2f}"
            )
        else:
            self.status(f"Paper {str(side).upper()} not sent: {result.get('reason') or result.get('status') or 'blocked'}")
        self._dashboard_cache.clear()
        self.run_prediction_and_refresh_ui()

    def _place_routed_order(self, side: str):
        selected = str(self.instrument_var.get() or "").upper()
        result = place_routed_order(
            selected,
            side,
            current_price=self._latest_price,
            advisory=self._latest_advisory_meta,
            profile_name=self.profile_var.get(),
        )
        route = str(result.get("routing_label") or result.get("routing_mode") or "route")
        status = str(result.get("status") or "-")
        if status in {"filled", "submitted"}:
            self.status(f"{route} {str(side).upper()} sent for {selected}: {status}")
        else:
            self.status(f"{route} {str(side).upper()} not sent: {result.get('reason') or status}")
        self._dashboard_cache.clear()
        self.run_prediction_and_refresh_ui()

    def _cancel_routed_order(self):
        selected = str(self.instrument_var.get() or "").upper()
        result = cancel_routed_order(symbol=selected)
        if str(result.get("status")) == "canceled":
            self.status(f"Canceled routed order for {selected}")
        else:
            self.status(f"Cancel skipped: {result.get('reason') or result.get('status') or 'blocked'}")
        self._dashboard_cache.clear()
        self.run_prediction_and_refresh_ui()

    def _current_broker_diagnostics(self, settings_override: dict | None = None) -> dict:
        selected = str(self.instrument_var.get() or "").upper()
        try:
            return refresh_broker_connection(selected, settings_override=settings_override)
        except Exception:
            logger.exception("Broker diagnostics lookup failed")
            return {
                "status": "error",
                "config_state": "error",
                "credentials_present": False,
                "sdk_installed": False,
                "operator_hint": "Broker diagnostics unavailable right now.",
                "route_label": "Local Paper",
                "requested_label": "Local Paper",
            }

    @staticmethod
    def _friendly_broker_setup_message(diagnostics: dict | None) -> str:
        diagnostics = dict(diagnostics or {})
        config_state = str(diagnostics.get("config_state") or "")
        operator_hint = str(diagnostics.get("operator_hint") or diagnostics.get("detail") or "").strip()
        mapping = {
            "sdk_missing": "This copy of the app is missing the Alpaca connection component.",
            "credentials_missing": "Enter your Alpaca API key and secret to enable broker-routed paper trading.",
            "invalid_credentials": "Alpaca rejected the key or secret. Please check them and try again.",
            "network_error": "The app could not reach Alpaca just now. Check your internet connection and try again.",
            "paper_flag_off": "Paper mode is off. Turn it on unless you intentionally want live routing.",
            "live_flag_mismatch": "Live mode is selected while Alpaca is still in paper-safe mode.",
            "mode_default_local_paper": "The app is still using Local Paper. Save Alpaca Paper settings to switch the broker route.",
        }
        return mapping.get(config_state, operator_hint or "Open Alpaca Setup to connect the broker.")

    def _maybe_prompt_alpaca_setup(self):
        if self._alpaca_setup_prompted:
            return
        diagnostics = self._current_broker_diagnostics()
        config_state = str(diagnostics.get("config_state") or "")
        should_prompt = config_state in {"credentials_missing", "invalid_credentials"}
        if should_prompt:
            self._alpaca_setup_prompted = True
            self._open_alpaca_setup(
                diagnostics=diagnostics,
                headline="Set up Alpaca Paper Trading",
                message=self._friendly_broker_setup_message(diagnostics),
            )

    def _close_alpaca_setup(self):
        win = getattr(self, "_alpaca_setup_window", None)
        if win is not None:
            try:
                win.destroy()
            except Exception:
                pass
        self._alpaca_setup_window = None

    def _open_alpaca_setup(self, diagnostics: dict | None = None, headline: str | None = None, message: str | None = None):
        if self._alpaca_setup_window is not None:
            try:
                self._alpaca_setup_window.lift()
                self._alpaca_setup_window.focus_force()
                return
            except Exception:
                self._close_alpaca_setup()

        diagnostics = diagnostics or self._current_broker_diagnostics()
        headline = headline or "Set up Alpaca Paper Trading"
        message = message or self._friendly_broker_setup_message(diagnostics)
        win = tk.Toplevel(self.root)
        win.title("Alpaca Setup")
        win.geometry("560x360")
        win.configure(bg="#081018")
        win.transient(self.root)
        win.grab_set()
        self._apply_dialog_icon(win)
        self._alpaca_setup_window = win
        win.protocol("WM_DELETE_WINDOW", self._close_alpaca_setup)

        saved_locally = str(diagnostics.get("credentials_source") or "") == "local_store"
        tk.Label(win, text=headline, font=self.big_font, fg="#f8fafc", bg="#081018").pack(anchor="w", padx=18, pady=(16, 6))
        copy = (
            "Alpaca lets the app talk to your broker account. "
            "Enter your Alpaca paper trading keys below. Paper mode is the safe default."
        )
        tk.Label(win, text=copy, font=self.small_font, fg="#cbd5e1", bg="#081018", wraplength=520, justify="left").pack(anchor="w", padx=18)
        tk.Label(win, text=message, font=self.small_font, fg="#fbbf24", bg="#081018", wraplength=520, justify="left").pack(anchor="w", padx=18, pady=(10, 8))

        form = tk.Frame(win, bg="#081018")
        form.pack(fill="x", padx=18, pady=(4, 10))
        tk.Label(form, text="API Key", font=self.small_font, fg="#e2e8f0", bg="#081018").grid(row=0, column=0, sticky="w", pady=(0, 6))
        tk.Label(form, text="Secret Key", font=self.small_font, fg="#e2e8f0", bg="#081018").grid(row=1, column=0, sticky="w", pady=(0, 6))
        api_var = tk.StringVar(value="")
        secret_var = tk.StringVar(value="")
        paper_var = tk.BooleanVar(value=bool((diagnostics.get("settings") or {}).get("paper", True)))
        status_var = tk.StringVar(value="Saved locally on this computer after a successful verification." if not saved_locally else "Saved keys already exist on this computer. Enter new ones only if you want to replace them.")

        api_entry = ttk.Entry(form, textvariable=api_var, width=42)
        api_entry.grid(row=0, column=1, sticky="ew", pady=(0, 6))
        secret_entry = ttk.Entry(form, textvariable=secret_var, width=42, show="*")
        secret_entry.grid(row=1, column=1, sticky="ew", pady=(0, 6))
        form.columnconfigure(1, weight=1)

        paper_row = tk.Frame(win, bg="#081018")
        paper_row.pack(fill="x", padx=18, pady=(0, 10))
        tk.Checkbutton(
            paper_row,
            text="Use paper mode (recommended)",
            variable=paper_var,
            bg="#081018",
            fg="white",
            selectcolor="#081018",
            font=self.small_font,
        ).pack(anchor="w")
        tk.Label(
            paper_row,
            text="These keys are stored locally on this computer for future launches. You can change or remove them later.",
            font=self.small_font,
            fg="#93c5fd",
            bg="#081018",
            wraplength=520,
            justify="left",
        ).pack(anchor="w", pady=(4, 0))

        status_label = tk.Label(win, textvariable=status_var, font=self.small_font, fg="#cbd5e1", bg="#081018", wraplength=520, justify="left")
        status_label.pack(anchor="w", padx=18, pady=(4, 12))

        def _entered_settings() -> dict:
            api_key = str(api_var.get() or "").strip()
            secret_key = str(secret_var.get() or "").strip()
            return {
                "provider": "alpaca",
                "broker_mode": "alpaca_paper" if paper_var.get() else "alpaca_live",
                "paper": bool(paper_var.get()),
                "api_key": api_key,
                "secret_key": secret_key,
            }

        def _validate_and_optionally_save(save_on_success: bool):
            settings_override = _entered_settings()
            if not settings_override.get("api_key") or not settings_override.get("secret_key"):
                status_var.set("Please enter both the Alpaca API key and secret key.")
                return
            diag = self._current_broker_diagnostics(settings_override=settings_override)
            if not diag.get("sdk_installed"):
                status_var.set(self._friendly_broker_setup_message(diag))
                return
            if not diag.get("connected"):
                status_var.set(self._friendly_broker_setup_message(diag))
                return
            if settings_override.get("paper"):
                verify = run_broker_verification(
                    str(self.instrument_var.get() or "").upper(),
                    current_price=self._latest_price,
                    advisory=self._latest_advisory_meta,
                    profile_name=self.profile_var.get(),
                    settings_override=settings_override,
                )
                if str(verify.get("status")) != "verified":
                    status_var.set(str(verify.get("detail") or "Verification failed."))
                    return
                success_text = "Success. Alpaca Paper verified."
            else:
                success_text = "Connection worked. Live mode is selected, so no paper test order was sent."
            if save_on_success:
                save_local_broker_credentials(
                    settings_override.get("api_key") or "",
                    settings_override.get("secret_key") or "",
                    paper=bool(settings_override.get("paper")),
                    broker_mode=str(settings_override.get("broker_mode") or "alpaca_paper"),
                )
                self._dashboard_cache.clear()
                self._refresh_broker_status()
                self.run_prediction_and_refresh_ui()
                status_var.set(f"{success_text} Saved locally on this computer.")
                messagebox.showinfo("Alpaca Setup", f"{success_text}\n\nThe app will reuse these keys on future launches.")
                self._close_alpaca_setup()
            else:
                status_var.set(success_text)

        def _clear_saved():
            removed = clear_local_broker_credentials()
            self._dashboard_cache.clear()
            self._refresh_broker_status()
            self.run_prediction_and_refresh_ui()
            if removed:
                status_var.set("Saved Alpaca keys were removed from this computer.")
                messagebox.showinfo("Alpaca Setup", "Saved Alpaca keys were removed from this computer.")
            else:
                status_var.set("There were no saved Alpaca keys to remove.")

        button_row = tk.Frame(win, bg="#081018")
        button_row.pack(fill="x", padx=18, pady=(0, 18))
        ttk.Button(button_row, text="Test Connection", command=lambda: _validate_and_optionally_save(False)).pack(side="left")
        ttk.Button(button_row, text="Save and Verify", command=lambda: _validate_and_optionally_save(True)).pack(side="left", padx=(8, 0))
        ttk.Button(button_row, text="Remove Saved Keys", command=_clear_saved).pack(side="left", padx=(8, 0))
        ttk.Button(button_row, text="Close", command=self._close_alpaca_setup).pack(side="right")

    def _refresh_broker_status(self):
        selected = str(self.instrument_var.get() or "").upper()
        result = refresh_broker_connection(selected)
        route = str(result.get("route_label") or "Unavailable")
        state = "connected" if result.get("connected") else str(result.get("status") or "unavailable").replace("_", " ")
        hint = self._friendly_broker_setup_message(result)
        self.status(f"Broker refresh: {route} | {state} | {hint}")
        self._dashboard_cache.clear()
        self.run_prediction_and_refresh_ui()

    def _verify_broker(self):
        selected = str(self.instrument_var.get() or "").upper()
        diagnostics = self._current_broker_diagnostics()
        if not diagnostics.get("credentials_present"):
            self._open_alpaca_setup(
                diagnostics=diagnostics,
                headline="Add your Alpaca keys first",
                message=self._friendly_broker_setup_message(diagnostics),
            )
            self.status("Alpaca verification needs an API key and secret first.")
            return
        result = run_broker_verification(
            selected,
            current_price=self._latest_price,
            advisory=self._latest_advisory_meta,
            profile_name=self.profile_var.get(),
        )
        status = str(result.get("status") or "unknown").replace("_", " ")
        self.status(f"Broker verify: {status} | {result.get('detail') or '-'}")
        self._dashboard_cache.clear()
        self.run_prediction_and_refresh_ui()

    def _flatten_paper_position(self):
        selected = str(self.instrument_var.get() or "").upper()
        result = flatten_paper_position(selected, current_price=self._latest_price, profile_name=self.profile_var.get())
        if str(result.get("status")) == "filled":
            fill = result.get("fill") or {}
            self.status(
                f"Flattened {selected}: {int(float(fill.get('qty') or 0))} @ ${float(fill.get('fill_price') or 0.0):.2f}"
            )
        else:
            self.status(f"Flatten skipped: {result.get('reason') or result.get('status') or 'blocked'}")
        self._dashboard_cache.clear()
        self.run_prediction_and_refresh_ui()

    def _start_replay(self):
        selected = str(self.instrument_var.get() or "").upper()
        with self._lock:
            df_copy = self.df.copy() if self.df is not None else None
        if df_copy is None or getattr(df_copy, "empty", True):
            self.status("Replay start skipped: no local data loaded")
            return
        snapshot = start_active_replay(df_copy, selected, profile_name=self.profile_var.get())
        self._dashboard_cache.clear()
        self.status(f"Replay simulation started: {snapshot.get('detail') or selected}")
        self.run_prediction_and_refresh_ui()

    def _step_replay(self):
        snapshot = step_active_replay()
        self._dashboard_cache.clear()
        self.status(f"Replay step: {snapshot.get('detail') or snapshot.get('status') or 'updated'}")
        self.run_prediction_and_refresh_ui()

    def _toggle_replay(self):
        current = self._current_replay_snapshot()
        if not current.get("active"):
            self.status("Replay toggle skipped: no active replay session")
            return
        snapshot = set_replay_paused(not bool(current.get("paused", True)))
        self._dashboard_cache.clear()
        state = "paused" if snapshot.get("paused", True) else "running"
        self.status(f"Replay {state}: {snapshot.get('detail') or '-'}")
        self.run_prediction_and_refresh_ui()

    def _reset_replay(self):
        snapshot = reset_active_replay()
        self._dashboard_cache.clear()
        self.status(f"Replay reset: {snapshot.get('detail') or '-'}")
        self.run_prediction_and_refresh_ui()

    def _stop_replay(self):
        snapshot = stop_active_replay()
        self._dashboard_cache.clear()
        self.status(snapshot.get("detail") or "Replay stopped; live data view restored")
        self.run_prediction_and_refresh_ui()

    def _place_replay_order(self, side: str):
        selected = str(self.instrument_var.get() or "").upper()
        result = place_replay_order(side, advisory=self._latest_advisory_meta)
        if str(result.get("status")) == "filled":
            order = result.get("order") or {}
            self.status(f"Replay {str(side).upper()} filled: {selected} {int(float(order.get('qty') or 0))} @ ${float(order.get('fill_price') or 0.0):.2f}")
        else:
            self.status(f"Replay {str(side).upper()} blocked: {result.get('reason') or result.get('status') or 'blocked'}")
        self._dashboard_cache.clear()
        self.run_prediction_and_refresh_ui()

    def _flatten_replay_position(self):
        selected = str(self.instrument_var.get() or "").upper()
        result = flatten_replay_position(advisory=self._latest_advisory_meta)
        if str(result.get("status")) == "filled":
            order = result.get("order") or {}
            self.status(f"Replay flat: {selected} {int(float(order.get('qty') or 0))} @ ${float(order.get('fill_price') or 0.0):.2f}")
        else:
            self.status(f"Replay flat skipped: {result.get('reason') or result.get('status') or 'blocked'}")
        self._dashboard_cache.clear()
        self.run_prediction_and_refresh_ui()

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

    def _sync_action_buttons(self):
        runtime = dict((self._latest_execution_meta or {}).get("runtime") or {})
        route_label = str((self._latest_execution_meta or {}).get("route_label") or runtime.get("route_label") or "Local Paper")
        self.route_buy_var.set(f"Buy ({route_label})")
        self.route_sell_var.set(f"Sell ({route_label})")
        self.route_cancel_var.set("Cancel Routed")
        self.route_setup_var.set(str((self._latest_execution_meta or {}).get("route_setup_label") or "Alpaca Setup"))
        self.route_refresh_var.set(str((self._latest_execution_meta or {}).get("route_refresh_label") or "Refresh Broker"))
        self.route_verify_var.set(str((self._latest_execution_meta or {}).get("route_verify_label") or "Verify Alpaca Paper"))
        can_place = bool(runtime.get("can_place_orders"))
        can_cancel = bool(runtime.get("can_cancel"))
        self.route_buy_btn.configure(state=("normal" if can_place else "disabled"))
        self.route_sell_btn.configure(state=("normal" if can_place else "disabled"))
        self.route_cancel_btn.configure(state=("normal" if can_cancel else "disabled"))
        self.route_setup_btn.configure(state="normal")
        self.route_refresh_btn.configure(state="normal")
        self.route_verify_btn.configure(state="normal")

        replay_active = bool((self._latest_replay_meta or {}).get("active"))
        replay_paused = bool((self._latest_replay_meta or {}).get("paused", True))
        replay_done = bool((self._latest_replay_meta or {}).get("completed"))
        self.replay_toggle_var.set("Replay Resume" if replay_paused else "Replay Pause")
        self.replay_step_btn.configure(state=("normal" if replay_active and not replay_done else "disabled"))
        self.replay_toggle_btn.configure(state=("normal" if replay_active and not replay_done else "disabled"))
        self.replay_reset_btn.configure(state=("normal" if replay_active else "disabled"))
        self.replay_stop_btn.configure(state=("normal" if replay_active else "disabled"))
        self.replay_buy_btn.configure(state=("normal" if replay_active and not replay_done else "disabled"))
        self.replay_sell_btn.configure(state=("normal" if replay_active and not replay_done else "disabled"))
        self.replay_flat_btn.configure(state=("normal" if replay_active and not replay_done else "disabled"))

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

    def _apply_window_icon(self):
        try:
            icon_path = get_window_icon_path()
            if icon_path and icon_path.exists():
                self._window_icon = tk.PhotoImage(file=str(icon_path))
                self.root.iconphoto(True, self._window_icon)
        except Exception:
            logger.debug("Window icon load failed", exc_info=True)

    def _apply_dialog_icon(self, window):
        if window is None:
            return
        try:
            if self._window_icon is not None:
                window.iconphoto(True, self._window_icon)
        except Exception:
            logger.debug("Dialog icon load failed", exc_info=True)

    def _build_menu_bar(self):
        try:
            menu = tk.Menu(self.root)
            app_menu = tk.Menu(menu, tearoff=0)
            app_menu.add_command(label="Settings", command=self._open_settings_dialog)
            app_menu.add_command(label="Check for Updates", command=lambda: self._check_for_updates(background=False))
            app_menu.add_separator()
            app_menu.add_command(label="About", command=self._open_about_dialog)
            app_menu.add_command(label="Quit", command=self._quit_app)
            menu.add_cascade(label="App", menu=app_menu)

            broker_menu = tk.Menu(menu, tearoff=0)
            broker_menu.add_command(label="Alpaca Setup", command=self._open_alpaca_setup)
            broker_menu.add_command(label="Refresh Broker", command=self._refresh_broker_status)
            broker_menu.add_command(label="Verify Alpaca Paper", command=self._verify_broker)
            broker_menu.add_command(label="Remove Saved Keys", command=self._clear_saved_keys_from_menu)
            menu.add_cascade(label="Broker", menu=broker_menu)

            help_menu = tk.Menu(menu, tearoff=0)
            help_menu.add_command(label="Quick Start", command=self._open_help_dialog)
            help_menu.add_command(label="Open Releases Page", command=self._open_release_page)
            menu.add_cascade(label="Help", menu=help_menu)
            self.root.config(menu=menu)
            self._menu_bar = menu
        except Exception:
            logger.debug("Menu bar init failed", exc_info=True)

    def _format_update_status(self, meta: dict | None) -> str:
        meta = dict(meta or {})
        status = str(meta.get("status") or "unknown")
        latest = str(meta.get("latest_version") or "").strip()
        if status == "update_available" and latest:
            return f"Update available: v{latest}"
        if status == "up_to_date":
            return f"Version {APP_VERSION} | Up to date"
        if status == "cached":
            return f"Update check cached | {meta.get('message') or 'Using cached result'}"
        if status == "error":
            return f"Update check failed | {meta.get('message') or '-'}"
        if status == "unavailable":
            return f"Update check unavailable | {meta.get('message') or '-'}"
        return f"Version {APP_VERSION} | Updates not checked"

    @staticmethod
    def _update_status_color(meta: dict | None) -> str:
        status = str((meta or {}).get("status") or "unknown")
        mapping = {
            "update_available": "#fbbf24",
            "up_to_date": "#86efac",
            "cached": "#93c5fd",
            "unavailable": "#fca5a5",
            "error": "#fda4af",
        }
        return mapping.get(status, "#86efac")

    def _set_update_status(self, meta: dict | None):
        self._latest_update_meta = dict(meta or {})
        try:
            self.update_status_var.set(self._format_update_status(meta))
            self.update_status_label.config(fg=self._update_status_color(meta))
        except Exception:
            pass
        self._refresh_support_dialogs()

    def _check_for_updates(self, background: bool = True):
        if self._update_check_in_flight:
            return
        self._update_check_in_flight = True

        def _job():
            try:
                meta = get_update_status(force=not background)
            except Exception as exc:
                meta = {"status": "error", "message": str(exc)}
            finally:
                self._update_check_in_flight = False

            def _apply():
                self._set_update_status(meta)
                if not background:
                    status = str(meta.get("status") or "unknown")
                    if status == "update_available":
                        if messagebox.askyesno(
                            "Update Available",
                            f"{meta.get('message')}\n\nOpen the GitHub Releases page now?",
                        ):
                            self._open_release_page()
                    elif status == "up_to_date":
                        messagebox.showinfo("Check for Updates", meta.get("message") or "You are up to date.")
                    else:
                        messagebox.showinfo("Check for Updates", meta.get("message") or "Update check unavailable.")
            try:
                self.root.after(0, _apply)
            except Exception:
                pass

        try:
            self.executor.submit(_job)
        except Exception:
            self._update_check_in_flight = False
            logger.debug("Update check scheduling failed", exc_info=True)

    def _open_release_page(self):
        if not GITHUB_RELEASES_URL:
            messagebox.showinfo("Releases", "No GitHub Releases page is configured yet.")
            return
        try:
            webbrowser.open(GITHUB_RELEASES_URL)
            self.status("Opened GitHub Releases page")
        except Exception:
            logger.debug("Open release page failed", exc_info=True)
            messagebox.showinfo("Releases", GITHUB_RELEASES_URL)

    def _clear_saved_keys_from_menu(self):
        removed = clear_local_broker_credentials()
        self._dashboard_cache.clear()
        self._refresh_broker_status()
        self.run_prediction_and_refresh_ui()
        if removed:
            messagebox.showinfo("Saved Keys Removed", "Saved Alpaca keys were removed from this computer.")
        else:
            messagebox.showinfo("Saved Keys Removed", "There were no saved Alpaca keys to remove.")

    def _refresh_support_dialogs(self):
        if self._settings_window is not None:
            try:
                self._settings_broker_var.set(self.broker_card_var.get())
                self._settings_broker_detail_var.set(self.broker_detail_var.get())
                self._settings_update_var.set(self.update_status_var.get())
            except Exception:
                pass
        if self._about_window is not None:
            try:
                self._about_update_var.set(self.update_status_var.get())
            except Exception:
                pass

    def _open_settings_dialog(self):
        if self._settings_window is not None:
            try:
                self._settings_window.lift()
                self._settings_window.focus_force()
                self._refresh_support_dialogs()
                return
            except Exception:
                self._settings_window = None
        win = tk.Toplevel(self.root)
        win.title("Settings")
        win.geometry("560x360")
        win.configure(bg="#081018")
        win.transient(self.root)
        self._apply_dialog_icon(win)
        self._settings_window = win
        win.protocol("WM_DELETE_WINDOW", lambda: setattr(self, "_settings_window", None) or win.destroy())
        tk.Label(win, text="Settings", font=self.header_font, fg="#f8fafc", bg="#081018").pack(anchor="w", padx=18, pady=(16, 6))
        tk.Label(win, text=f"{APP_DISPLAY_NAME} v{APP_VERSION}", font=self.small_font, fg="#93c5fd", bg="#081018").pack(anchor="w", padx=18)
        self._settings_broker_var = tk.StringVar(value=self.broker_card_var.get())
        self._settings_broker_detail_var = tk.StringVar(value=self.broker_detail_var.get())
        self._settings_update_var = tk.StringVar(value=self.update_status_var.get())
        broker_box = tk.Frame(win, bg="#0f1724", highlightbackground="#1f2a44", highlightthickness=1)
        broker_box.pack(fill="x", padx=18, pady=(14, 8))
        tk.Label(broker_box, text="Broker", font=self.small_font, fg="#fdba74", bg="#0f1724").pack(anchor="w", padx=10, pady=(8, 0))
        tk.Label(broker_box, textvariable=self._settings_broker_var, font=self.big_font, fg="#f8fafc", bg="#0f1724", wraplength=500, justify="left").pack(anchor="w", padx=10, pady=(2, 0))
        tk.Label(broker_box, textvariable=self._settings_broker_detail_var, font=self.small_font, fg="#cbd5e1", bg="#0f1724", wraplength=500, justify="left").pack(anchor="w", padx=10, pady=(2, 8))
        update_box = tk.Frame(win, bg="#0f1724", highlightbackground="#1f2a44", highlightthickness=1)
        update_box.pack(fill="x", padx=18, pady=(0, 8))
        tk.Label(update_box, text="Updates", font=self.small_font, fg="#67e8f9", bg="#0f1724").pack(anchor="w", padx=10, pady=(8, 0))
        tk.Label(update_box, textvariable=self._settings_update_var, font=self.small_font, fg="#f8fafc", bg="#0f1724", wraplength=500, justify="left").pack(anchor="w", padx=10, pady=(4, 8))
        actions = tk.Frame(win, bg="#081018")
        actions.pack(fill="x", padx=18, pady=(8, 18))
        ttk.Button(actions, text="Alpaca Setup", command=self._open_alpaca_setup).pack(side="left")
        ttk.Button(actions, text="Refresh Broker", command=self._refresh_broker_status).pack(side="left", padx=(8, 0))
        ttk.Button(actions, text="Verify Alpaca Paper", command=self._verify_broker).pack(side="left", padx=(8, 0))
        ttk.Button(actions, text="Check for Updates", command=lambda: self._check_for_updates(background=False)).pack(side="left", padx=(8, 0))
        ttk.Button(actions, text="Quick Start", command=self._open_help_dialog).pack(side="left", padx=(8, 0))
        ttk.Button(actions, text="Remove Saved Keys", command=self._clear_saved_keys_from_menu).pack(side="left", padx=(8, 0))
        ttk.Button(actions, text="Close", command=lambda: setattr(self, "_settings_window", None) or win.destroy()).pack(side="right")

    def _open_about_dialog(self):
        if self._about_window is not None:
            try:
                self._about_window.lift()
                self._about_window.focus_force()
                return
            except Exception:
                self._about_window = None
        win = tk.Toplevel(self.root)
        win.title("About")
        win.geometry("500x300")
        win.configure(bg="#081018")
        win.transient(self.root)
        self._apply_dialog_icon(win)
        self._about_window = win
        win.protocol("WM_DELETE_WINDOW", lambda: setattr(self, "_about_window", None) or win.destroy())
        tk.Label(win, text=APP_DISPLAY_NAME, font=self.header_font, fg="#f8fafc", bg="#081018").pack(anchor="w", padx=18, pady=(18, 4))
        tk.Label(win, text=f"Version {APP_VERSION}", font=self.small_font, fg="#fbbf24", bg="#081018").pack(anchor="w", padx=18)
        tk.Label(win, text=f"GitHub Releases: {GITHUB_RELEASES_REPO}", font=self.small_font, fg="#93c5fd", bg="#081018", wraplength=460, justify="left").pack(anchor="w", padx=18, pady=(8, 0))
        tk.Label(win, text="A desktop trading workstation with paper-safe broker defaults, replay, advisory context, and installable-app packaging support.", font=self.small_font, fg="#cbd5e1", bg="#081018", wraplength=460, justify="left").pack(anchor="w", padx=18, pady=(12, 0))
        self._about_update_var = tk.StringVar(value=self.update_status_var.get())
        tk.Label(win, textvariable=self._about_update_var, font=self.small_font, fg="#86efac", bg="#081018", wraplength=460, justify="left").pack(anchor="w", padx=18, pady=(12, 0))
        buttons = tk.Frame(win, bg="#081018")
        buttons.pack(fill="x", padx=18, pady=(18, 18))
        ttk.Button(buttons, text="Check for Updates", command=lambda: self._check_for_updates(background=False)).pack(side="left")
        ttk.Button(buttons, text="Open Releases Page", command=self._open_release_page).pack(side="left", padx=(8, 0))
        ttk.Button(buttons, text="Close", command=lambda: setattr(self, "_about_window", None) or win.destroy()).pack(side="right")

    def _open_help_dialog(self):
        if self._help_window is not None:
            try:
                self._help_window.lift()
                self._help_window.focus_force()
                return
            except Exception:
                self._help_window = None
        win = tk.Toplevel(self.root)
        win.title("Quick Start")
        win.geometry("560x360")
        win.configure(bg="#081018")
        win.transient(self.root)
        self._apply_dialog_icon(win)
        self._help_window = win
        win.protocol("WM_DELETE_WINDOW", lambda: setattr(self, "_help_window", None) or win.destroy())
        tk.Label(win, text="Quick Start", font=self.header_font, fg="#f8fafc", bg="#081018").pack(anchor="w", padx=18, pady=(18, 6))
        copy = (
            "1. Use Alpaca Setup to add your paper trading keys.\n"
            "2. Keep paper mode on unless you intentionally want live routing.\n"
            "3. Watch the Broker card to confirm whether orders go to Local Paper or Alpaca Paper.\n"
            "4. Use Replay to practice bar-by-bar without touching live markets.\n"
            "5. Use Check for Updates before internal rollout or release testing.\n"
            "6. Open Settings any time to update broker setup, saved keys, or release status."
        )
        tk.Label(win, text=copy, font=self.small_font, fg="#cbd5e1", bg="#081018", justify="left", wraplength=500).pack(anchor="w", padx=18, pady=(0, 18))
        ttk.Button(win, text="Alpaca Setup", command=self._open_alpaca_setup).pack(anchor="w", padx=18)
        buttons = tk.Frame(win, bg="#081018")
        buttons.pack(fill="x", padx=18, pady=(18, 18))
        ttk.Button(buttons, text="Settings", command=self._open_settings_dialog).pack(side="left")
        ttk.Button(buttons, text="Close", command=lambda: setattr(self, "_help_window", None) or win.destroy()).pack(side="right")

    def _quit_app(self):
        try:
            self.shutdown()
        finally:
            try:
                self.root.destroy()
            except Exception:
                pass

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
            "breadth_rows": [],
            "macro_rows": [],
            "signal_rows": [],
            "stack_rows": [],
            "probability_rows": [],
            "quality_rows": [],
            "liquidity_rows": [],
            "catalyst_rows": [],
            "alerts_rows": [],
            "execution_rows": [],
            "replay_rows": [],
            "portfolio_rows": [],
            "performance_rows": [],
            "opportunity_rows": [],
            "setup_rows": [],
            "scenario_rows": [],
            "invalidation_rows": [],
            "reliability_rows": [],
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
            "action_card": "Action: unavailable",
            "action_detail": "-",
            "quality_card": "Quality: unavailable",
            "quality_detail": "-",
            "vol_card": "Volatility: unavailable",
            "vol_detail": "-",
            "options_card": "Options: unavailable",
            "news_card": "News: unavailable",
            "profile_card": "Profile: unavailable",
            "profile_detail": "-",
            "broker_card": "Broker: unavailable",
            "broker_detail": "-",
            "replay_card": "Replay: unavailable",
            "replay_detail": "-",
            "profile_summary": "Profile: -",
            "advisory_meta": {},
            "execution_meta": {},
            "replay_meta": {},
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
        _clear(self.breadth_tree)
        _clear(self.macro_tree)
        _clear(self.signal_tree)
        _clear(self.stack_tree)
        _clear(self.probability_tree)
        _clear(self.quality_tree)
        _clear(self.liquidity_tree)
        _clear(self.catalyst_tree)
        _clear(self.alerts_tree)
        _clear(self.execution_tree)
        _clear(self.replay_tree)
        _clear(self.portfolio_tree)
        _clear(self.performance_tree)
        _clear(self.opportunity_tree)
        _clear(self.setup_tree)
        _clear(self.scenario_tree)
        _clear(self.invalidate_tree)
        _clear(self.reliability_tree)
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
        for values in payload.get("breadth_rows", []):
            self.breadth_tree.insert("", "end", values=values)
        for values in payload.get("macro_rows", []):
            self.macro_tree.insert("", "end", values=values)
        for values in payload.get("signal_rows", []):
            self.signal_tree.insert("", "end", values=values)
        for values in payload.get("stack_rows", []):
            self.stack_tree.insert("", "end", values=values)
        for values in payload.get("probability_rows", []):
            self.probability_tree.insert("", "end", values=values)
        for values in payload.get("quality_rows", []):
            self.quality_tree.insert("", "end", values=values)
        for values in payload.get("liquidity_rows", []):
            self.liquidity_tree.insert("", "end", values=values)
        for values in payload.get("catalyst_rows", []):
            self.catalyst_tree.insert("", "end", values=values)
        for values in payload.get("alerts_rows", []):
            self.alerts_tree.insert("", "end", values=values)
        for values in payload.get("execution_rows", []):
            self.execution_tree.insert("", "end", values=values)
        for values in payload.get("replay_rows", []):
            self.replay_tree.insert("", "end", values=values)
        for values in payload.get("portfolio_rows", []):
            self.portfolio_tree.insert("", "end", values=values)
        for values in payload.get("performance_rows", []):
            self.performance_tree.insert("", "end", values=values)
        for values in payload.get("opportunity_rows", []):
            self.opportunity_tree.insert("", "end", values=values)
        for values in payload.get("setup_rows", []):
            self.setup_tree.insert("", "end", values=values)
        for values in payload.get("scenario_rows", []):
            self.scenario_tree.insert("", "end", values=values)
        for values in payload.get("invalidation_rows", []):
            self.invalidate_tree.insert("", "end", values=values)
        for values in payload.get("reliability_rows", []):
            self.reliability_tree.insert("", "end", values=values)
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
        self.profile_summary_var.set(payload.get("profile_summary", "Profile: -"))
        self.week_extremes_var.set(payload.get("week_extremes", "Projected Week High/Low: -"))
        self.regime_label.config(text=payload.get("regime_text", "Market Regime: Unknown"))
        self.setup_card_var.set(payload.get("setup_card", "Alignment: unavailable"))
        self.setup_detail_var.set(payload.get("setup_detail", "-"))
        self.action_card_var.set(payload.get("action_card", "Action: unavailable"))
        self.action_detail_var.set(payload.get("action_detail", "-"))
        self.quality_card_var.set(payload.get("quality_card", "Quality: unavailable"))
        self.quality_detail_var.set(payload.get("quality_detail", "-"))
        self.vol_card_var.set(payload.get("vol_card", "Volatility: unavailable"))
        self.vol_detail_var.set(payload.get("vol_detail", "-"))
        self.options_card_var.set(payload.get("options_card", "Options: unavailable"))
        self.news_card_var.set(payload.get("news_card", "News: unavailable"))
        self.profile_card_var.set(payload.get("profile_card", "Profile: unavailable"))
        self.profile_detail_var.set(payload.get("profile_detail", "-"))
        self.broker_card_var.set(payload.get("broker_card", "Broker: unavailable"))
        self.broker_detail_var.set(payload.get("broker_detail", "-"))
        self.replay_card_var.set(payload.get("replay_card", "Replay: unavailable"))
        self.replay_detail_var.set(payload.get("replay_detail", "-"))
        self._latest_advisory_meta = dict(payload.get("advisory_meta") or {})
        self._latest_execution_meta = dict(payload.get("execution_meta") or {})
        self._latest_replay_meta = dict(payload.get("replay_meta") or {})
        try:
            current_label = str(payload.get("current_price", ""))
            self._latest_price = float(current_label.split("$", 1)[1].replace(",", "")) if "$" in current_label else None
        except Exception:
            self._latest_price = None
        self._sync_action_buttons()
        self._refresh_support_dialogs()
        for msg in payload.get("status_messages", []):
            self.status(msg)

    def _compute_prediction_payload(self, df_copy, models_copy, mode: str, selected: str, replay_snapshot: dict | None = None) -> dict:
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
        if isinstance(replay_snapshot, dict) and replay_snapshot.get("active"):
            payload["data_meta"] = f"Data: replay simulation | Last Bar: {last_ts_str}"
            payload["status_messages"].append(f"Replay {str(replay_snapshot.get('status') or 'paused')}")
        else:
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
                dashboard = build_trader_dashboard(
                    df_copy,
                    selected,
                    predictions=dashboard_predictions,
                    profile_name=self.profile_var.get(),
                )
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
                payload["action_card"] = dashboard.get("action_card", payload["action_card"])
                payload["action_detail"] = dashboard.get("action_detail", payload["action_detail"])
                payload["quality_card"] = dashboard.get("quality_card", payload["quality_card"])
                payload["quality_detail"] = dashboard.get("quality_detail", payload["quality_detail"])
                payload["vol_card"] = dashboard.get("vol_card", payload["vol_card"])
                payload["vol_detail"] = dashboard.get("vol_detail", payload["vol_detail"])
                payload["options_card"] = dashboard.get("options_card", payload["options_card"])
                payload["news_card"] = dashboard.get("news_card", payload["news_card"])
                payload["profile_card"] = dashboard.get("profile_card", payload["profile_card"])
                payload["profile_detail"] = dashboard.get("profile_detail", payload["profile_detail"])
                payload["broker_card"] = dashboard.get("broker_card", payload["broker_card"])
                payload["broker_detail"] = dashboard.get("broker_detail", payload["broker_detail"])
                payload["replay_card"] = dashboard.get("replay_card", payload["replay_card"])
                payload["replay_detail"] = dashboard.get("replay_detail", payload["replay_detail"])
                payload["profile_summary"] = dashboard.get("profile_card", payload["profile_summary"])
                payload["market_rows"] = dashboard.get("market_rows", [])
                payload["breadth_rows"] = dashboard.get("breadth_rows", [])
                payload["macro_rows"] = dashboard.get("macro_rows", [])
                payload["signal_rows"] = dashboard.get("signal_rows", [])
                payload["stack_rows"] = dashboard.get("stack_rows", [])
                payload["probability_rows"] = dashboard.get("probability_rows", [])
                payload["quality_rows"] = dashboard.get("quality_rows", [])
                payload["liquidity_rows"] = dashboard.get("liquidity_rows", [])
                payload["catalyst_rows"] = dashboard.get("catalyst_rows", [])
                payload["alerts_rows"] = dashboard.get("alerts_rows", [])
                payload["execution_rows"] = dashboard.get("execution_rows", [])
                payload["replay_rows"] = dashboard.get("replay_rows", [])
                payload["portfolio_rows"] = dashboard.get("portfolio_rows", [])
                payload["performance_rows"] = dashboard.get("performance_rows", [])
                payload["opportunity_rows"] = dashboard.get("opportunity_rows", [])
                payload["setup_rows"] = dashboard.get("setup_rows", [])
                payload["scenario_rows"] = dashboard.get("scenario_rows", [])
                payload["invalidation_rows"] = dashboard.get("invalidation_rows", [])
                payload["reliability_rows"] = dashboard.get("reliability_rows", [])
                payload["levels_rows"] = dashboard.get("levels_rows", [])
                payload["options_rows"] = dashboard.get("options_rows", [])
                payload["news_rows"] = dashboard.get("news_rows", [])
                payload["advisory_meta"] = dashboard.get("advisory_meta", {})
                payload["execution_meta"] = dashboard.get("execution_meta", {})
                payload["replay_meta"] = dashboard.get("replay_meta", {})
        except Exception:
            logger.exception("Dashboard analytics build failed")
        payload["status_messages"].append("Predicted")
        return payload

    def run_prediction_and_refresh_ui(self):
        mode = self.mode_var.get()
        df_copy, models_copy, selected, replay_snapshot = self._current_prediction_inputs()

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

        def _job():
            try:
                payload = self._compute_prediction_payload(df_copy, models_copy, mode, selected, replay_snapshot=replay_snapshot)
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
                    replay_snapshot = self._current_replay_snapshot()
                    if replay_snapshot.get("active"):
                        if not replay_snapshot.get("paused", True) and not replay_snapshot.get("completed", False):
                            step_active_replay()
                        self.root.after(0, self.run_prediction_and_refresh_ui)
                        time.sleep(AUTO_PREDICT_SECONDS)
                        continue
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
