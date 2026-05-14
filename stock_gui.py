#stock_gui.py
import json
import os
import threading
import tkinter as tk
from collections import OrderedDict
from tkinter import ttk
from tkinter import font as tkfont
from tkinter import messagebox
import webbrowser
import pandas as pd
from app_meta import APP_DISPLAY_NAME, APP_VERSION, GITHUB_RELEASES_REPO, GITHUB_RELEASES_URL, app_title
from app_paths import get_runtime_logs_dir, get_window_icon_path
from data_hub import get_data_hub
from data_fetch import update_price_data
from config import AUTO_PREDICT_SECONDS, INSTRUMENTS, DEFAULT_INSTRUMENT
from model_zoo import (
    build_inference_payload,
    load_model_stack as load_models,
    predict_intraday as predict_4min,
    predict_mode as predict_horizon,
    train_baseline_models as train_all_models,
)
from onboarding_jobs import run_onboarding_job, summarize_onboarding_state
from autonomy_engine import (
    AUTONOMY_DECISION_STYLE_LABELS,
    AUTONOMY_MODE_LABELS,
    evaluate_autonomy_cycle,
    get_autonomy_snapshot,
    list_autonomy_strategies,
    set_autonomy_decision_style,
    set_autonomy_kill_switch,
    set_autonomy_loss_limit,
    set_autonomy_mode,
)
from execution_engine import (
    clear_local_broker_credentials,
    cancel_routed_order,
    flatten_routed_position,
    flatten_paper_position,
    refresh_broker_connection,
    run_broker_verification,
    save_local_broker_credentials,
    place_paper_order,
    place_routed_order,
)
from features import LIVE_MAX_BARS, BAR_MINUTES
from instrument_onboarding import ensure_runtime_instruments, list_runtime_instruments, register_instrument
from instrument_profile import build_instrument_meta, summarize_instrument_meta
from readiness_tracker import evaluate_readiness_metrics, load_readiness_state, summarize_deployment_readiness
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
from symbol_lifecycle import build_symbol_lifecycle_snapshot, lifecycle_rows
from trader_analytics import build_trader_dashboard
from trader_profiles import get_trader_profile, list_trader_profiles
from update_checker import download_latest_release, get_update_status
from workspace_state import (
    activate_workspace,
    close_workspace,
    get_workspace,
    load_workspace_state,
    list_workspaces,
    list_workspace_symbols,
    open_workspace,
    remember_workspace_view,
)
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

MODEL_STACK_LOADER = load_models


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


def _loading_charge_palette(step: int) -> tuple[str, str]:
    palette = [
        ("#2563eb", "#93c5fd"),
        ("#1d4ed8", "#7dd3fc"),
        ("#0ea5e9", "#bfdbfe"),
        ("#2563eb", "#dbeafe"),
    ]
    return palette[int(step) % len(palette)]


class LightningGUI:
    def __init__(self, root):
        self.root = root
        self.root.title(app_title())
        self.root.geometry("1660x1020")
        self.root.minsize(1420, 900)
        self.root.configure(bg="#081018")
        self._window_icon = None
        self._settings_window = None
        self._about_window = None
        self._help_window = None
        self.data_hub = get_data_hub()
        self._data_hub_subscription_symbol = None
        self._data_hub_subscription_token = None

        self.header_font = tkfont.Font(family="Avenir Next", size=19, weight="bold")
        self.big_font = tkfont.Font(family="Avenir Next", size=15, weight="bold")
        self.cell_font = tkfont.Font(family="Avenir Next", size=12)
        self.small_font = tkfont.Font(family="Avenir Next", size=11)
        self.mono_font = tkfont.Font(family="Menlo", size=10)

        self._apply_window_icon()
        self._build_menu_bar()

        self.style = ttk.Style(root)
        self.style.theme_use("clam")
        self.style.configure("TButton", font=self.small_font, padding=9, foreground="white", background="#162130", borderwidth=0)
        self.style.map("TButton", background=[("active", "#22304a"), ("pressed", "#1b263b")])
        self.style.configure("Secondary.TButton", font=self.small_font, padding=9, foreground="#dbeafe", background="#0f1724", borderwidth=0)
        self.style.map("Secondary.TButton", background=[("active", "#172338"), ("pressed", "#0f1724")], foreground=[("active", "#f8fafc")])
        self.style.configure("Treeview", rowheight=28, font=self.cell_font, background="#0f1724", fieldbackground="#0f1724", foreground="#e5eefb")
        self.style.configure("Treeview.Heading", font=self.small_font, background="#152033", foreground="#dbeafe", padding=(8, 6))
        self.style.configure(
            "Busy.Horizontal.TProgressbar",
            troughcolor="#122030",
            background="#60a5fa",
            lightcolor="#93c5fd",
            darkcolor="#2563eb",
            bordercolor="#122030",
        )

        self.models = {}
        self.df = None
        self._lock = threading.Lock()
        self._pred_lock = threading.Lock()
        self._pred_in_flight = False
        self._pred_started_at = None
        self._pred_timeout_sec = 30.0
        self._pred_seq = 0
        self._pending_prediction_request = False
        self._pending_prediction_reason = None
        self._prediction_payload_cache = {}
        self._acc_lock = threading.Lock()
        self._acc_in_flight = False
        self._dashboard_cache = {}
        self._dashboard_cache_ttl_sec = max(60, int(AUTO_PREDICT_SECONDS))
        self._latest_advisory_meta = {}
        self._latest_execution_meta = {}
        self._latest_replay_meta = {}
        self._latest_autonomy_meta = {}
        self._latest_price = None
        self._workspace_buttons = {}
        self._workspace_runtime_cache = OrderedDict()
        self._workspace_runtime_cache_limit = 2
        self._workspace_runtime_cache_ttl_sec = max(300, int(AUTO_PREDICT_SECONDS) * 3)
        self._alpaca_setup_window = None
        self._alpaca_setup_prompted = False
        self._card_labels = []
        self._right_item_width = 138
        self._right_value_width = 118
        self._right_detail_width = 250
        self._right_news_width = 360
        self._activity_running = False
        self._loading_overlay_visible = False
        self._loading_anim_step = 0
        self._loading_after_id = None

        self.executor = ThreadPoolExecutor(max_workers=2)
        self._overview_last_refresh = 0.0
        self._overview_refresh_sec = max(120, int(AUTO_PREDICT_SECONDS))
        self._update_check_in_flight = False
        self._update_download_in_flight = False
        self._latest_update_meta = {}

        try:
            ensure_runtime_instruments(force=False, auto_onboard=True)
        except Exception:
            pass
        initial_symbol = self._bootstrap_workspace_state()

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
        self.instrument_var = tk.StringVar(value=initial_symbol or DEFAULT_INSTRUMENT)
        instrument_frame = tk.Frame(self.top_frame, bg="#081018")
        instrument_frame.pack(side="left", padx=12)
        tk.Label(instrument_frame, text="Stock:", bg="#081018", fg="white", font=self.small_font).pack(side="left", padx=(0,6))
        choices = list_runtime_instruments() or list(INSTRUMENTS.keys())
        if choices and self.instrument_var.get() not in choices:
            self.instrument_var.set(choices[0])
        self.instrument_menu = ttk.Combobox(instrument_frame, values=choices, textvariable=self.instrument_var, state="readonly", width=12)
        self.instrument_menu.pack(side="left")
        self.instrument_menu.bind("<<ComboboxSelected>>", self._on_instrument_change)
        self.workspace_open_btn = ttk.Button(instrument_frame, text="Open Workspace", command=self._open_selected_workspace)
        self.workspace_open_btn.pack(side="left", padx=(6, 6))
        self.workspace_add_btn = ttk.Button(instrument_frame, text="Add Stock", command=self._open_add_stock_dialog)
        self.workspace_add_btn.pack(side="left")

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

        autonomy_frame = tk.Frame(self.top_frame, bg="#081018")
        autonomy_frame.pack(side="left", padx=12)
        self.autonomy_mode_var = tk.StringVar(value="OFF")
        self.autonomy_decision_var = tk.StringVar(value=AUTONOMY_DECISION_STYLE_LABELS.get("full_autonomy", "Full Autonomy"))
        self.autonomy_strategy_choices = {
            str(item.get("label") or item.get("key")): str(item.get("key"))
            for item in list_autonomy_strategies()
            if item.get("key")
        }
        self.autonomy_strategy_var = tk.StringVar(value=next(iter(self.autonomy_strategy_choices.keys()), "Trend Following"))
        self.autonomy_halt_var = tk.StringVar(value="Halt Auto")
        tk.Label(autonomy_frame, text="Auto Route:", bg="#081018", fg="#fca5a5", font=self.small_font).pack(side="left", padx=(0, 6))
        self.autonomy_mode_menu = ttk.Combobox(
            autonomy_frame,
            values=[AUTONOMY_MODE_LABELS[k] for k in ("off", "shadow", "paper_auto", "live_auto")],
            textvariable=self.autonomy_mode_var,
            state="readonly",
            width=12,
        )
        self.autonomy_mode_menu.pack(side="left")
        tk.Label(autonomy_frame, text="Decision:", bg="#081018", fg="#fca5a5", font=self.small_font).pack(side="left", padx=(8, 6))
        self.autonomy_decision_menu = ttk.Combobox(
            autonomy_frame,
            values=[AUTONOMY_DECISION_STYLE_LABELS[k] for k in ("full_autonomy", "strategy_autonomy")],
            textvariable=self.autonomy_decision_var,
            state="readonly",
            width=17,
        )
        self.autonomy_decision_menu.pack(side="left")
        self.autonomy_decision_menu.bind(
            "<<ComboboxSelected>>",
            lambda _event=None: self.autonomy_strategy_menu.configure(
                state=("readonly" if self._decision_style_from_label(self.autonomy_decision_var.get()) == "strategy_autonomy" else "disabled")
            ),
        )
        self.autonomy_strategy_menu = ttk.Combobox(
            autonomy_frame,
            values=list(self.autonomy_strategy_choices.keys()),
            textvariable=self.autonomy_strategy_var,
            state="readonly",
            width=18,
        )
        self.autonomy_strategy_menu.pack(side="left", padx=(6, 0))
        self.autonomy_apply_btn = ttk.Button(autonomy_frame, text="Apply Auto", command=self._apply_autonomy_mode)
        self.autonomy_apply_btn.pack(side="left", padx=(6, 6))
        self.autonomy_risk_btn = ttk.Button(autonomy_frame, text="Auto Risk", command=self._open_autonomy_settings_dialog)
        self.autonomy_risk_btn.pack(side="left", padx=(0, 6))
        self.autonomy_halt_btn = ttk.Button(autonomy_frame, textvariable=self.autonomy_halt_var, command=self._toggle_autonomy_halt)
        self.autonomy_halt_btn.pack(side="left")

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
        self.route_flat_var = tk.StringVar(value="Flat (Local Paper)")
        self.route_cancel_var = tk.StringVar(value="Cancel Routed")
        self.route_setup_var = tk.StringVar(value="Alpaca Setup")
        self.route_refresh_var = tk.StringVar(value="Refresh Broker")
        self.route_verify_var = tk.StringVar(value="Verify Alpaca Paper")
        self.route_buy_btn = ttk.Button(routed_frame, textvariable=self.route_buy_var, command=lambda: self._place_routed_order("buy"))
        self.route_buy_btn.pack(side="left", padx=(0, 6))
        self.route_sell_btn = ttk.Button(routed_frame, textvariable=self.route_sell_var, command=lambda: self._place_routed_order("sell"))
        self.route_sell_btn.pack(side="left", padx=(0, 6))
        self.route_flat_btn = ttk.Button(routed_frame, textvariable=self.route_flat_var, command=self._flatten_routed_position)
        self.route_flat_btn.pack(side="left", padx=(0, 6))
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
        self.autonomy_badge_var = tk.StringVar(value="Operator: Advisory Only")
        self.autonomy_badge_label = tk.Label(self.summary_frame, textvariable=self.autonomy_badge_var, font=self.small_font, fg="#081018", bg="#94a3b8", padx=10, pady=3)
        self.autonomy_badge_label.pack(side="left", padx=12)
        self.accuracy_var = tk.StringVar(value="Avg Discrepancy: loading…")
        self.accuracy_label = tk.Label(self.summary_frame, textvariable=self.accuracy_var, font=self.small_font, fg="#93c5fd", bg="#081018")
        self.accuracy_label.pack(side="left", padx=12)
        self.profile_summary_var = tk.StringVar(value="Profile: -")
        self.profile_summary_label = tk.Label(self.summary_frame, textvariable=self.profile_summary_var, font=self.small_font, fg="#f5d0fe", bg="#081018")
        self.profile_summary_label.pack(side="left", padx=12)
        self.profile_summary_var.set(f"Profile: {get_trader_profile(self.profile_var.get()).get('label', 'Balanced')}")
        self.activity_var = tk.StringVar(value="Activity: startup")
        self.activity_frame = tk.Frame(self.summary_frame, bg="#081018")
        self.activity_frame.pack(side="right", padx=10)
        self.activity_label = tk.Label(self.activity_frame, textvariable=self.activity_var, font=self.small_font, fg="#60a5fa", bg="#081018")
        self.activity_label.pack(side="left", padx=(0, 8))
        self.activity_bar = ttk.Progressbar(self.activity_frame, style="Busy.Horizontal.TProgressbar", mode="indeterminate", length=108)
        self.activity_bar.pack(side="left")
        self.data_meta_var = tk.StringVar(value="Data: - | Last Bar: -")
        self.data_meta_label = tk.Label(self.summary_frame, textvariable=self.data_meta_var, font=self.small_font, fg="#cbd5f5", bg="#081018")
        self.data_meta_label.pack(side="right", padx=6)
        self.update_status_var = tk.StringVar(value=f"Version {APP_VERSION} | Updates not checked")
        self.update_status_label = tk.Label(self.summary_frame, textvariable=self.update_status_var, font=self.small_font, fg="#86efac", bg="#081018")
        self.update_status_label.pack(side="right", padx=10)

        self.workspace_frame = tk.Frame(self.root, bg="#081018")
        self.workspace_frame.pack(fill="x", padx=12, pady=(0, 6))
        self.workspace_label = tk.Label(self.workspace_frame, text="Workspaces", font=self.small_font, fg="#cbd5e1", bg="#081018")
        self.workspace_label.pack(side="left", padx=(2, 8))
        self.workspace_tabs_frame = tk.Frame(self.workspace_frame, bg="#081018")
        self.workspace_tabs_frame.pack(side="left", fill="x", expand=True)
        self.workspace_close_btn = ttk.Button(self.workspace_frame, text="Close Active", style="Secondary.TButton", command=self._close_active_workspace)
        self.workspace_close_btn.pack(side="right", padx=(8, 0))
        self.workspace_status_var = tk.StringVar(value="Workspace: loading…")
        self.workspace_status_label = tk.Label(self.workspace_frame, textvariable=self.workspace_status_var, font=self.small_font, fg="#7dd3fc", bg="#081018")
        self.workspace_status_label.pack(side="right", padx=(8, 0))

        # Instrument overview section: one row per symbol with maturity and current operating posture.
        self.overview_frame = tk.Frame(self.root, bg="#081018")
        self.overview_frame.pack(fill="x", padx=12, pady=(0, 8))
        self.overview_label = tk.Label(
            self.overview_frame,
            text="Workspace Overview",
            font=self.small_font,
            fg="#e2e8f0",
            bg="#081018",
        )
        self.overview_label.pack(anchor="w", padx=2, pady=(0, 3))
        self.overview_hint_label = tk.Label(
            self.overview_frame,
            text="Compare symbol maturity, horizon coverage, and operator posture. Double-click a row to focus that workspace.",
            font=self.small_font,
            fg="#94a3b8",
            bg="#081018",
        )
        self.overview_hint_label.pack(anchor="w", padx=2, pady=(0, 5))
        overview_cols = ("instrument", "coverage", "four_min", "daily", "weekend", "status")
        self.overview_tree = ttk.Treeview(self.overview_frame, columns=overview_cols, show="headings", height=5)
        labels = {
            "instrument": "Instrument",
            "coverage": "Coverage",
            "four_min": "4min",
            "daily": "Daily High/Low",
            "weekend": "WeekEnd",
            "status": "Operator Status",
        }
        for col in overview_cols:
            self.overview_tree.heading(col, text=labels[col])
            width = 160 if col == "instrument" else 170 if col == "coverage" else 200 if col in ("daily", "weekend") else 240
            self.overview_tree.column(col, anchor=tk.CENTER, width=width)
        self.overview_tree.pack(fill="x")
        self.overview_tree.bind("<Double-1>", self._open_overview_workspace)

        self.main_pane = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, bg="#081018", sashwidth=8, relief=tk.FLAT, bd=0)
        self.main_pane.pack(fill="both", expand=True, padx=12, pady=(0, 10))
        self.left_panel = tk.Frame(self.main_pane, bg="#081018")
        self.right_panel = tk.Frame(self.main_pane, bg="#081018", width=560)
        self.main_pane.add(self.left_panel, stretch="always")
        self.main_pane.add(self.right_panel, minsize=500)

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
            lambda e: (self.right_canvas.itemconfigure(self._right_body_window, width=e.width), self._refresh_card_wraplengths(e.width)),
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

        self.cards_section = tk.Frame(self.right_body, bg="#081018")
        self.cards_section.pack(fill="x", pady=(0, 8))
        self.cards_label = tk.Label(self.cards_section, text="Operator Summary", font=self.big_font, fg="#f8fafc", bg="#081018")
        self.cards_label.pack(anchor="w", pady=(2, 1))
        self.cards_hint = tk.Label(
            self.cards_section,
            text="The top row should answer what matters now: trade posture, evidence quality, autonomy mode, and platform state.",
            font=self.small_font,
            fg="#94a3b8",
            bg="#081018",
            wraplength=560,
            justify="left",
        )
        self.cards_hint.pack(anchor="w", pady=(0, 6))
        self.cards_frame = tk.Frame(self.cards_section, bg="#081018")
        self.cards_frame.pack(fill="x", pady=(0, 8))
        self.setup_card_var = tk.StringVar(value="Alignment: loading…")
        self.setup_detail_var = tk.StringVar(value="-")
        self.action_card_var = tk.StringVar(value="Action: loading…")
        self.action_detail_var = tk.StringVar(value="-")
        self.autonomy_card_var = tk.StringVar(value="Autonomy: loading…")
        self.autonomy_detail_var = tk.StringVar(value="-")
        self.quality_card_var = tk.StringVar(value="Quality: loading…")
        self.quality_detail_var = tk.StringVar(value="-")
        self.vol_card_var = tk.StringVar(value="Volatility: loading…")
        self.vol_detail_var = tk.StringVar(value="-")
        self.options_card_var = tk.StringVar(value="Options: loading…")
        self.options_detail_var = tk.StringVar(value="-")
        self.news_card_var = tk.StringVar(value="News: loading…")
        self.news_detail_var = tk.StringVar(value="-")
        self.profile_card_var = tk.StringVar(value="Profile: loading…")
        self.profile_detail_var = tk.StringVar(value="-")
        self.broker_card_var = tk.StringVar(value="Broker: loading…")
        self.broker_detail_var = tk.StringVar(value="-")
        self.replay_card_var = tk.StringVar(value="Replay: loading…")
        self.replay_detail_var = tk.StringVar(value="-")

        self.cards_frame.columnconfigure(0, weight=1)
        self.cards_frame.columnconfigure(1, weight=1)

        def _make_card(parent, title: str, main_var, detail_var=None, accent="#93c5fd", row: int = 0, column: int = 0):
            frame = tk.Frame(parent, bg="#0f1724", highlightbackground="#1f2a44", highlightthickness=1, bd=0)
            frame.grid(row=row, column=column, sticky="nsew", padx=(0 if column == 0 else 6, 6 if column == 0 else 0), pady=4)
            tk.Frame(frame, bg=accent, height=3).pack(fill="x")
            tk.Label(frame, text=title, font=self.small_font, fg=accent, bg="#0f1724").pack(anchor="w", padx=12, pady=(8, 0))
            main_label = tk.Label(frame, textvariable=main_var, font=self.big_font, fg="#f8fafc", bg="#0f1724", wraplength=500, justify="left")
            main_label.pack(anchor="w", padx=12, pady=(4, 0))
            self._card_labels.append(main_label)
            if detail_var is not None:
                detail_label = tk.Label(frame, textvariable=detail_var, font=self.small_font, fg="#cbd5e1", bg="#0f1724", wraplength=500, justify="left")
                detail_label.pack(anchor="w", padx=12, pady=(4, 10))
                self._card_labels.append(detail_label)
            else:
                tk.Label(frame, text="", font=self.small_font, bg="#0f1724").pack(anchor="w", padx=12, pady=(0, 10))

        _make_card(self.cards_frame, "Setup", self.setup_card_var, self.setup_detail_var, accent="#34d399", row=0, column=0)
        _make_card(self.cards_frame, "Action", self.action_card_var, self.action_detail_var, accent="#fbbf24", row=0, column=1)
        _make_card(self.cards_frame, "Signal Quality", self.quality_card_var, self.quality_detail_var, accent="#c084fc", row=1, column=0)
        _make_card(self.cards_frame, "Autonomy Controller", self.autonomy_card_var, self.autonomy_detail_var, accent="#fca5a5", row=1, column=1)
        _make_card(self.cards_frame, "News", self.news_card_var, self.news_detail_var, accent="#7dd3fc", row=2, column=0)
        _make_card(self.cards_frame, "Volatility", self.vol_card_var, self.vol_detail_var, accent="#fbbf24", row=2, column=1)
        _make_card(self.cards_frame, "Broker / Forward Test", self.broker_card_var, self.broker_detail_var, accent="#fdba74", row=3, column=0)
        _make_card(self.cards_frame, "Replay / Simulation", self.replay_card_var, self.replay_detail_var, accent="#67e8f9", row=3, column=1)
        _make_card(self.cards_frame, "Trader Profile", self.profile_card_var, self.profile_detail_var, accent="#f5d0fe", row=4, column=0)
        _make_card(self.cards_frame, "Options", self.options_card_var, self.options_detail_var, accent="#c084fc", row=4, column=1)

        self.context_section_label = tk.Label(self.right_body, text="Context and Support Feeds", font=self.big_font, fg="#f8fafc", bg="#081018")
        self.context_section_label.pack(anchor="w", pady=(2, 1))
        self.context_section_hint = tk.Label(
            self.right_body,
            text="Market, breadth, macro, and news context explain why a setup exists and whether the tape supports it.",
            font=self.small_font,
            fg="#94a3b8",
            bg="#081018",
            wraplength=560,
            justify="left",
        )
        self.context_section_hint.pack(anchor="w", pady=(0, 6))
        self.market_label = tk.Label(self.right_body, text="Market Context", font=self.big_font, fg="#e2e8f0", bg="#081018")
        self.market_label.pack(anchor="w", pady=(2, 2))
        self.market_tree = ttk.Treeview(self.right_body, columns=("item", "value", "detail"), show="headings", height=5)
        for col, width in (("item", self._right_item_width), ("value", self._right_value_width), ("detail", self._right_detail_width)):
            self.market_tree.heading(col, text=col.title())
            self.market_tree.column(col, anchor=tk.CENTER if col != "detail" else tk.W, width=width)
        self.market_tree.pack(fill="x", pady=(0, 8))

        self.signal_label = tk.Label(self.right_body, text="Signal Context", font=self.big_font, fg="#e2e8f0", bg="#081018")
        self.signal_label.pack(anchor="w", pady=(2, 2))
        self.signal_tree = ttk.Treeview(self.right_body, columns=("item", "value", "detail"), show="headings", height=6)
        for col, width in (("item", self._right_item_width), ("value", self._right_value_width), ("detail", self._right_detail_width)):
            self.signal_tree.heading(col, text=col.title())
            self.signal_tree.column(col, anchor=tk.CENTER if col != "detail" else tk.W, width=width)
        self.signal_tree.pack(fill="x", pady=(0, 8))

        self.breadth_label = tk.Label(self.right_body, text="Breadth / Internals", font=self.big_font, fg="#e2e8f0", bg="#081018")
        self.breadth_label.pack(anchor="w", pady=(2, 2))
        self.breadth_tree = ttk.Treeview(self.right_body, columns=("item", "value", "detail"), show="headings", height=5)
        for col, width in (("item", self._right_item_width), ("value", self._right_value_width), ("detail", self._right_detail_width)):
            self.breadth_tree.heading(col, text=col.title())
            self.breadth_tree.column(col, anchor=tk.CENTER if col != "detail" else tk.W, width=width)
        self.breadth_tree.pack(fill="x", pady=(0, 8))

        self.macro_label = tk.Label(self.right_body, text="Macro / Event Risk", font=self.big_font, fg="#e2e8f0", bg="#081018")
        self.macro_label.pack(anchor="w", pady=(2, 2))
        self.macro_tree = ttk.Treeview(self.right_body, columns=("item", "value", "detail"), show="headings", height=5)
        for col, width in (("item", self._right_item_width), ("value", self._right_value_width), ("detail", self._right_detail_width)):
            self.macro_tree.heading(col, text=col.title())
            self.macro_tree.column(col, anchor=tk.CENTER if col != "detail" else tk.W, width=width)
        self.macro_tree.pack(fill="x", pady=(0, 8))

        self.decision_section_label = tk.Label(self.right_body, text="Decision Stack", font=self.big_font, fg="#f8fafc", bg="#081018")
        self.decision_section_label.pack(anchor="w", pady=(8, 1))
        self.decision_section_hint = tk.Label(
            self.right_body,
            text="These panels should tell the operator whether the symbol is mature enough, what the signal means, and why it is or is not tradeable.",
            font=self.small_font,
            fg="#94a3b8",
            bg="#081018",
            wraplength=560,
            justify="left",
        )
        self.decision_section_hint.pack(anchor="w", pady=(0, 6))
        self.stack_label = tk.Label(self.right_body, text="Advisory Stack", font=self.big_font, fg="#e2e8f0", bg="#081018")
        self.stack_label.pack(anchor="w", pady=(2, 2))
        self.stack_tree = ttk.Treeview(self.right_body, columns=("item", "value", "detail"), show="headings", height=6)
        for col, width in (("item", self._right_item_width), ("value", self._right_value_width), ("detail", self._right_detail_width)):
            self.stack_tree.heading(col, text=col.title())
            self.stack_tree.column(col, anchor=tk.CENTER if col != "detail" else tk.W, width=width)
        self.stack_tree.pack(fill="x", pady=(0, 8))

        self.probability_label = tk.Label(self.right_body, text="Probability Map", font=self.big_font, fg="#e2e8f0", bg="#081018")
        self.probability_label.pack(anchor="w", pady=(2, 2))
        self.probability_tree = ttk.Treeview(self.right_body, columns=("item", "value", "detail"), show="headings", height=6)
        for col, width in (("item", self._right_item_width), ("value", self._right_value_width), ("detail", self._right_detail_width + 20)):
            self.probability_tree.heading(col, text=col.title())
            self.probability_tree.column(col, anchor=tk.CENTER if col != "detail" else tk.W, width=width)
        self.probability_tree.pack(fill="x", pady=(0, 8))

        self.quality_label = tk.Label(self.right_body, text="Signal Quality", font=self.big_font, fg="#e2e8f0", bg="#081018")
        self.quality_label.pack(anchor="w", pady=(2, 2))
        self.quality_tree = ttk.Treeview(self.right_body, columns=("item", "value", "detail"), show="headings", height=6)
        for col, width in (("item", self._right_item_width), ("value", self._right_value_width), ("detail", self._right_detail_width + 20)):
            self.quality_tree.heading(col, text=col.title())
            self.quality_tree.column(col, anchor=tk.CENTER if col != "detail" else tk.W, width=width)
        self.quality_tree.pack(fill="x", pady=(0, 8))

        self.liquidity_label = tk.Label(self.right_body, text="Liquidity / Execution", font=self.big_font, fg="#e2e8f0", bg="#081018")
        self.liquidity_label.pack(anchor="w", pady=(2, 2))
        self.liquidity_tree = ttk.Treeview(self.right_body, columns=("item", "value", "detail"), show="headings", height=6)
        for col, width in (("item", self._right_item_width), ("value", self._right_value_width), ("detail", self._right_detail_width + 20)):
            self.liquidity_tree.heading(col, text=col.title())
            self.liquidity_tree.column(col, anchor=tk.CENTER if col != "detail" else tk.W, width=width)
        self.liquidity_tree.pack(fill="x", pady=(0, 8))

        self.catalyst_label = tk.Label(self.right_body, text="Catalyst Context", font=self.big_font, fg="#e2e8f0", bg="#081018")
        self.catalyst_label.pack(anchor="w", pady=(2, 2))
        self.catalyst_tree = ttk.Treeview(self.right_body, columns=("item", "value", "detail"), show="headings", height=5)
        for col, width in (("item", self._right_item_width), ("value", self._right_value_width), ("detail", self._right_detail_width + 20)):
            self.catalyst_tree.heading(col, text=col.title())
            self.catalyst_tree.column(col, anchor=tk.CENTER if col != "detail" else tk.W, width=width)
        self.catalyst_tree.pack(fill="x", pady=(0, 8))

        self.alerts_label = tk.Label(self.right_body, text="Trader Alerts", font=self.big_font, fg="#e2e8f0", bg="#081018")
        self.alerts_label.pack(anchor="w", pady=(2, 2))
        self.alerts_tree = ttk.Treeview(self.right_body, columns=("item", "value", "detail"), show="headings", height=5)
        for col, width in (("item", self._right_item_width), ("value", self._right_value_width), ("detail", self._right_detail_width + 20)):
            self.alerts_tree.heading(col, text=col.title())
            self.alerts_tree.column(col, anchor=tk.CENTER if col != "detail" else tk.W, width=width)
        self.alerts_tree.pack(fill="x", pady=(0, 8))

        self.operations_section_label = tk.Label(self.right_body, text="Execution and Operations", font=self.big_font, fg="#f8fafc", bg="#081018")
        self.operations_section_label.pack(anchor="w", pady=(8, 1))
        self.operations_section_hint = tk.Label(
            self.right_body,
            text="Execution posture, replay state, portfolio context, and paper performance live here so operational state stays separate from market context.",
            font=self.small_font,
            fg="#94a3b8",
            bg="#081018",
            wraplength=560,
            justify="left",
        )
        self.operations_section_hint.pack(anchor="w", pady=(0, 6))
        self.execution_label = tk.Label(self.right_body, text="Execution", font=self.big_font, fg="#e2e8f0", bg="#081018")
        self.execution_label.pack(anchor="w", pady=(2, 2))
        self.execution_tree = ttk.Treeview(self.right_body, columns=("item", "value", "detail"), show="headings", height=5)
        for col, width in (("item", self._right_item_width), ("value", self._right_value_width), ("detail", self._right_detail_width + 20)):
            self.execution_tree.heading(col, text=col.title())
            self.execution_tree.column(col, anchor=tk.CENTER if col != "detail" else tk.W, width=width)
        self.execution_tree.pack(fill="x", pady=(0, 8))

        self.replay_label = tk.Label(self.right_body, text="Replay State", font=self.big_font, fg="#e2e8f0", bg="#081018")
        self.replay_label.pack(anchor="w", pady=(2, 2))
        self.replay_tree = ttk.Treeview(self.right_body, columns=("item", "value", "detail"), show="headings", height=5)
        for col, width in (("item", self._right_item_width), ("value", self._right_value_width), ("detail", self._right_detail_width + 20)):
            self.replay_tree.heading(col, text=col.title())
            self.replay_tree.column(col, anchor=tk.CENTER if col != "detail" else tk.W, width=width)
        self.replay_tree.pack(fill="x", pady=(0, 8))

        self.portfolio_label = tk.Label(self.right_body, text="Portfolio", font=self.big_font, fg="#e2e8f0", bg="#081018")
        self.portfolio_label.pack(anchor="w", pady=(2, 2))
        self.portfolio_tree = ttk.Treeview(self.right_body, columns=("item", "value", "detail"), show="headings", height=4)
        for col, width in (("item", self._right_item_width), ("value", self._right_value_width), ("detail", self._right_detail_width + 20)):
            self.portfolio_tree.heading(col, text=col.title())
            self.portfolio_tree.column(col, anchor=tk.CENTER if col != "detail" else tk.W, width=width)
        self.portfolio_tree.pack(fill="x", pady=(0, 8))

        self.performance_label = tk.Label(self.right_body, text="Paper Performance", font=self.big_font, fg="#e2e8f0", bg="#081018")
        self.performance_label.pack(anchor="w", pady=(2, 2))
        self.performance_tree = ttk.Treeview(self.right_body, columns=("item", "value", "detail"), show="headings", height=6)
        for col, width in (("item", self._right_item_width), ("value", self._right_value_width), ("detail", self._right_detail_width + 20)):
            self.performance_tree.heading(col, text=col.title())
            self.performance_tree.column(col, anchor=tk.CENTER if col != "detail" else tk.W, width=width)
        self.performance_tree.pack(fill="x", pady=(0, 8))

        self.opportunity_label = tk.Label(self.right_body, text="Opportunity Scan", font=self.big_font, fg="#e2e8f0", bg="#081018")
        self.opportunity_label.pack(anchor="w", pady=(2, 2))
        self.opportunity_tree = ttk.Treeview(self.right_body, columns=("item", "value", "detail"), show="headings", height=5)
        for col, width in (("item", self._right_item_width), ("value", self._right_value_width), ("detail", self._right_detail_width + 20)):
            self.opportunity_tree.heading(col, text=col.title())
            self.opportunity_tree.column(col, anchor=tk.CENTER if col != "detail" else tk.W, width=width)
        self.opportunity_tree.pack(fill="x", pady=(0, 8))

        self.plan_section_label = tk.Label(self.right_body, text="Trade Plan and Risk Framing", font=self.big_font, fg="#f8fafc", bg="#081018")
        self.plan_section_label.pack(anchor="w", pady=(8, 1))
        self.plan_section_hint = tk.Label(
            self.right_body,
            text="These sections stay dense, but the goal is clear planning: probabilities, plan shape, invalidation, reliability, and key levels.",
            font=self.small_font,
            fg="#94a3b8",
            bg="#081018",
            wraplength=560,
            justify="left",
        )
        self.plan_section_hint.pack(anchor="w", pady=(0, 6))
        self.setups_label = tk.Label(self.right_body, text="Trade Plan", font=self.big_font, fg="#e2e8f0", bg="#081018")
        self.setups_label.pack(anchor="w", pady=(2, 2))
        self.setup_tree = ttk.Treeview(self.right_body, columns=("item", "value", "detail"), show="headings", height=5)
        for col, width in (("item", self._right_item_width), ("value", self._right_value_width), ("detail", self._right_detail_width + 20)):
            self.setup_tree.heading(col, text=col.title())
            self.setup_tree.column(col, anchor=tk.CENTER if col != "detail" else tk.W, width=width)
        self.setup_tree.pack(fill="x", pady=(0, 8))

        self.scenario_label = tk.Label(self.right_body, text="Scenario Map", font=self.big_font, fg="#e2e8f0", bg="#081018")
        self.scenario_label.pack(anchor="w", pady=(2, 2))
        self.scenario_tree = ttk.Treeview(self.right_body, columns=("item", "value", "detail"), show="headings", height=4)
        for col, width in (("item", self._right_item_width), ("value", 148), ("detail", self._right_detail_width)):
            self.scenario_tree.heading(col, text=col.title())
            self.scenario_tree.column(col, anchor=tk.CENTER if col != "detail" else tk.W, width=width)
        self.scenario_tree.pack(fill="x", pady=(0, 8))

        self.invalidate_label = tk.Label(self.right_body, text="Invalidation / No-Trade", font=self.big_font, fg="#e2e8f0", bg="#081018")
        self.invalidate_label.pack(anchor="w", pady=(2, 2))
        self.invalidate_tree = ttk.Treeview(self.right_body, columns=("item", "value", "detail"), show="headings", height=5)
        for col, width in (("item", self._right_item_width), ("value", self._right_value_width), ("detail", self._right_detail_width + 20)):
            self.invalidate_tree.heading(col, text=col.title())
            self.invalidate_tree.column(col, anchor=tk.CENTER if col != "detail" else tk.W, width=width)
        self.invalidate_tree.pack(fill="x", pady=(0, 8))

        self.reliability_label = tk.Label(self.right_body, text="Historical Reliability", font=self.big_font, fg="#e2e8f0", bg="#081018")
        self.reliability_label.pack(anchor="w", pady=(2, 2))
        self.reliability_tree = ttk.Treeview(self.right_body, columns=("item", "value", "detail"), show="headings", height=6)
        for col, width in (("item", self._right_item_width), ("value", self._right_value_width), ("detail", self._right_detail_width + 20)):
            self.reliability_tree.heading(col, text=col.title())
            self.reliability_tree.column(col, anchor=tk.CENTER if col != "detail" else tk.W, width=width)
        self.reliability_tree.pack(fill="x", pady=(0, 8))

        self.levels_label = tk.Label(self.right_body, text="Key Levels", font=self.big_font, fg="#e2e8f0", bg="#081018")
        self.levels_label.pack(anchor="w", pady=(2, 2))
        self.levels_tree = ttk.Treeview(self.right_body, columns=("item", "value", "detail"), show="headings", height=8)
        for col, width in (("item", self._right_item_width), ("value", self._right_value_width), ("detail", self._right_detail_width)):
            self.levels_tree.heading(col, text=col.title())
            self.levels_tree.column(col, anchor=tk.CENTER if col != "detail" else tk.W, width=width)
        self.levels_tree.pack(fill="x", pady=(0, 8))

        self.options_label = tk.Label(self.right_body, text="Options Positioning", font=self.big_font, fg="#e2e8f0", bg="#081018")
        self.options_label.pack(anchor="w", pady=(2, 2))
        self.options_tree = ttk.Treeview(self.right_body, columns=("item", "value", "detail"), show="headings", height=5)
        for col, width in (("item", self._right_item_width), ("value", self._right_value_width), ("detail", self._right_detail_width)):
            self.options_tree.heading(col, text=col.title())
            self.options_tree.column(col, anchor=tk.CENTER if col != "detail" else tk.W, width=width)
        self.options_tree.pack(fill="x", pady=(0, 8))

        self.news_label = tk.Label(self.right_body, text="Recent Catalysts", font=self.big_font, fg="#e2e8f0", bg="#081018")
        self.news_label.pack(anchor="w", pady=(2, 2))
        self.news_tree = ttk.Treeview(self.right_body, columns=("age", "tone", "headline"), show="headings", height=7)
        for col, width in (("age", 64), ("tone", 90), ("headline", self._right_news_width)):
            self.news_tree.heading(col, text=col.title())
            self.news_tree.column(col, anchor=tk.CENTER if col != "headline" else tk.W, width=width)
        self.news_tree.pack(fill="both", expand=False, pady=(0, 4))
        self._build_loading_overlay()

        self._stop_event = threading.Event()
        self._retrain_in_progress = False
        self._last_retrain_attempt = 0.0
        self._training_stage = None
        self._worker_thread = threading.Thread(target=self.background_loop, daemon=True)
        self._worker_thread.start()

        self._ui_max_bars = self._compute_ui_max_bars()
        self._set_activity("Booting workspace", active=True, accent="#60a5fa")
        self.status("Initializing…")
        self._show_mode_ui()
        self.refresh_overview_threaded()
        self.refresh_accuracy_summary_threaded()
        self._initial_loader_thread = threading.Thread(target=self._initial_model_waiter, daemon=True)
        self._initial_loader_thread.start()
        self._load_autonomy_ui_state()
        self._refresh_workspace_tabs()
        self._sync_action_buttons()
        self._bind_runtime_subscription(self.instrument_var.get())
        try:
            self.root.after(900, self._maybe_prompt_alpaca_setup)
            self.root.after(1500, lambda: self._check_for_updates(background=True))
        except Exception:
            pass

    def _bootstrap_workspace_state(self) -> str:
        try:
            state = load_workspace_state()
        except Exception:
            state = {}
        choices = list_runtime_instruments() or list(INSTRUMENTS.keys()) or [DEFAULT_INSTRUMENT]
        default_symbol = str(DEFAULT_INSTRUMENT or (choices[0] if choices else "QQQ")).upper()
        active_symbol = str((state or {}).get("active_symbol") or default_symbol).upper()
        if active_symbol not in choices and choices:
            active_symbol = str(choices[0]).upper()
        open_workspace(active_symbol, meta=INSTRUMENTS.get(active_symbol, {}), activate=True, pinned=True)
        return active_symbol

    @staticmethod
    def _symbol_key(symbol: str | None) -> str:
        return str(symbol or "").upper().strip()

    def _workspace_meta(self, symbol: str | None = None) -> dict:
        current_symbol = symbol
        if current_symbol is None:
            try:
                current_symbol = self.instrument_var.get()
            except Exception:
                current_symbol = None
        sym = self._symbol_key(current_symbol)
        return build_instrument_meta(sym, overrides=INSTRUMENTS.get(sym, {}))

    def _workspace_is_active(self, symbol: str | None) -> bool:
        return self._symbol_key(symbol) == self._symbol_key(self.instrument_var.get())

    def _workspace_label(self, symbol: str) -> str:
        ws = get_workspace(symbol) or {}
        mode = str(ws.get("view_mode") or "4min")
        return f"{self._symbol_key(symbol)} | {mode}"

    def _lifecycle_snapshot(self, symbol: str | None = None, runtime: dict | None = None) -> dict:
        current_symbol = symbol
        if current_symbol is None:
            try:
                current_symbol = self.instrument_var.get()
            except Exception:
                current_symbol = None
        sym = self._symbol_key(current_symbol)
        ws = get_workspace(sym) or {}
        bundle = runtime if isinstance(runtime, dict) else (self._runtime_bundle(sym) or {})
        deployment = dict((bundle or {}).get("deployment_meta") or {})
        if not deployment:
            deployment = summarize_deployment_readiness(sym, state=load_readiness_state(sym) or {})
        return build_symbol_lifecycle_snapshot(
            sym,
            runtime_bundle=bundle,
            workspace=ws,
            onboarding_summary=summarize_onboarding_state(sym),
            deployment_summary=deployment,
            news_health=dict((bundle or {}).get("news_health") or {}),
            model_summary=dict((bundle or {}).get("model_summary") or {}),
        )

    def _workspace_summary_text(self, symbol: str | None = None) -> str:
        meta = self._workspace_meta(symbol)
        current_symbol = symbol
        if current_symbol is None:
            try:
                current_symbol = self.instrument_var.get()
            except Exception:
                current_symbol = None
        sym = self._symbol_key(current_symbol)
        ws = get_workspace(sym) or {}
        lifecycle = self._lifecycle_snapshot(sym)
        note = str(ws.get("last_note") or "").strip()
        base = " | ".join(
            bit
            for bit in (
                f"{sym} | {summarize_instrument_meta(meta, sym)}",
                str(lifecycle.get("lifecycle_label") or ""),
                str(lifecycle.get("operator_state_label") or ""),
                str(lifecycle.get("evidence_state_label") or ""),
                str(lifecycle.get("model_coverage_label") or ""),
            )
            if bit
        )
        onboarding_status = str(lifecycle.get("onboarding_status") or "").lower()
        if onboarding_status not in {"", "not started"}:
            base = f"{base} | Onboarding {str(lifecycle.get('onboarding_status') or '-')} ({str(lifecycle.get('onboarding_progress') or '-')})"
        if str(lifecycle.get("primary_next_requirement") or "").strip():
            base = f"{base} | Next: {str(lifecycle.get('primary_next_requirement'))}"
        if note:
            return f"{base} | {note}"
        return base

    def _apply_lifecycle_placeholders(self, payload: dict, symbol: str, reason: str):
        lifecycle = self._lifecycle_snapshot(symbol)
        detail = str(reason or lifecycle.get("empty_state_detail") or "Waiting…").strip() or "Waiting…"
        payload["signal_rows"] = [("Ticker Lifecycle", lifecycle.get("lifecycle_label") or "-", lifecycle.get("workspace_summary") or detail)]
        payload["stack_rows"] = lifecycle_rows(lifecycle)
        payload["quality_rows"] = [
            ("Operator State", lifecycle.get("operator_state_label") or "-", lifecycle.get("operator_state_detail") or detail),
            ("Evidence State", lifecycle.get("evidence_state_label") or "-", lifecycle.get("evidence_state_detail") or detail),
            ("Coverage", lifecycle.get("model_coverage_label") or "-", lifecycle.get("model_state_detail") or detail),
        ]
        payload["setup_rows"] = [
            ("Next Requirement", lifecycle.get("next_requirement_category") or "-", lifecycle.get("primary_next_requirement") or lifecycle.get("onboarding_summary") or detail),
        ]
        payload["reliability_rows"] = [
            ("Deployment", lifecycle.get("stage_label") or "-", f"{lifecycle.get('rollout_label') or '-'} | {lifecycle.get('symbol_maturity_label') or '-'}"),
        ]
        payload["market_rows"] = [("Data State", lifecycle.get("data_state_label") or "-", lifecycle.get("data_state_detail") or detail)]
        payload["breadth_rows"] = [("Onboarding", lifecycle.get("onboarding_status") or "-", lifecycle.get("onboarding_summary") or detail)]
        payload["macro_rows"] = [("Feature State", lifecycle.get("feature_state_label") or "-", lifecycle.get("feature_state_detail") or detail)]
        payload["liquidity_rows"] = [("Model State", lifecycle.get("model_state_label") or "-", lifecycle.get("model_state_detail") or detail)]
        payload["catalyst_rows"] = [("News State", lifecycle.get("news_state_label") or "-", lifecycle.get("news_state_detail") or detail)]
        payload["levels_rows"] = [("Lifecycle", lifecycle.get("lifecycle_label") or "-", lifecycle.get("lifecycle_detail") or detail)]
        payload["options_rows"] = [("Calibration", lifecycle.get("calibration_state_label") or "-", lifecycle.get("calibration_state_detail") or detail)]
        payload["news_rows"] = [("-", "info", lifecycle.get("news_state_detail") or detail)]
        payload["setup_card"] = f"Lifecycle: {lifecycle.get('lifecycle_label') or 'Unavailable'}"
        payload["setup_detail"] = lifecycle.get("lifecycle_detail") or detail
        payload["action_card"] = f"Action: {lifecycle.get('operator_state_label') or 'Unavailable'}"
        payload["action_detail"] = lifecycle.get("operator_state_detail") or detail
        payload["quality_card"] = f"Quality: {lifecycle.get('evidence_state_label') or 'Unavailable'}"
        payload["quality_detail"] = lifecycle.get("evidence_state_detail") or detail
        payload["vol_detail"] = lifecycle.get("data_state_detail") or detail
        payload["options_detail"] = lifecycle.get("model_state_detail") or detail
        payload["news_card"] = f"News: {lifecycle.get('news_state_label') or 'Unavailable'}"
        payload["news_detail"] = lifecycle.get("news_state_detail") or detail
        payload["profile_detail"] = lifecycle.get("workspace_summary") or detail
        payload["broker_detail"] = lifecycle.get("operator_state_detail") or detail

    def _refresh_workspace_tabs(self):
        frame = getattr(self, "workspace_tabs_frame", None)
        if frame is None:
            return
        for child in list(frame.winfo_children()):
            try:
                child.destroy()
            except Exception:
                pass
        active = self._symbol_key(self.instrument_var.get())
        workspaces = list_workspaces(open_only=True) or []
        if not workspaces and active:
            workspaces = [get_workspace(active) or {"symbol": active, "title": active}]
        for item in workspaces:
            sym = self._symbol_key(item.get("symbol"))
            label = self._workspace_label(sym)
            is_active = sym == active
            btn = tk.Button(
                frame,
                text=label,
                command=lambda target=sym: self._activate_workspace(target),
                font=self.small_font,
                relief=tk.FLAT,
                bd=0,
                padx=10,
                pady=4,
                bg="#2563eb" if is_active else "#162130",
                fg="#f8fafc" if is_active else "#cbd5e1",
                activebackground="#1d4ed8" if is_active else "#22304a",
                activeforeground="#ffffff",
            )
            btn.pack(side="left", padx=(0, 6))
            self._workspace_buttons[sym] = btn
        if hasattr(self, "workspace_status_var"):
            self.workspace_status_var.set(self._workspace_summary_text(active))
        if hasattr(self, "workspace_close_btn"):
            self.workspace_close_btn.configure(state="normal" if len(workspaces) > 1 else "disabled")

    def _store_runtime_bundle(self, symbol: str, models: dict, df: pd.DataFrame | None):
        sym = self._symbol_key(symbol)
        self._workspace_runtime_cache[sym] = {
            "models": dict(models or {}),
            "df": self._trim_df_for_ui(df.copy()) if isinstance(df, pd.DataFrame) else df,
            "loaded_at": time.time(),
        }
        self._workspace_runtime_cache.move_to_end(sym)
        while len(self._workspace_runtime_cache) > int(self._workspace_runtime_cache_limit):
            self._workspace_runtime_cache.popitem(last=False)

    def _unbind_runtime_subscription(self):
        hub = getattr(self, "data_hub", None)
        token = getattr(self, "_data_hub_subscription_token", None)
        sym = getattr(self, "_data_hub_subscription_symbol", None)
        if hub is not None and token and sym:
            try:
                hub.unsubscribe(sym, token)
            except Exception:
                logger.debug("DataHub unsubscribe failed", exc_info=True)
        self._data_hub_subscription_symbol = None
        self._data_hub_subscription_token = None

    def _bind_runtime_subscription(self, symbol: str | None):
        sym = self._symbol_key(symbol)
        if not sym:
            return
        if sym == getattr(self, "_data_hub_subscription_symbol", None) and getattr(self, "_data_hub_subscription_token", None):
            return
        self._unbind_runtime_subscription()
        hub = getattr(self, "data_hub", None)
        if hub is None:
            return

        def _on_bundle(payload: dict):
            def _apply():
                bundle = dict(payload or {})
                if self._symbol_key(bundle.get("symbol")) != sym:
                    return
                models = dict(bundle.get("models") or {})
                latest_df = self._trim_df_for_ui(bundle.get("df"))
                self._store_runtime_bundle(sym, models, latest_df)
                if self._workspace_is_active(sym):
                    with self._lock:
                        self.models = models
                        self.df = latest_df
                    if hasattr(self, "workspace_status_var"):
                        self.workspace_status_var.set(self._workspace_summary_text(sym))

            try:
                self.root.after(0, _apply)
            except Exception:
                try:
                    _apply()
                except Exception:
                    logger.debug("DataHub bundle apply failed", exc_info=True)

        try:
            self._data_hub_subscription_token = hub.subscribe(sym, _on_bundle)
            self._data_hub_subscription_symbol = sym
        except Exception:
            logger.debug("DataHub subscribe failed", exc_info=True)
            self._data_hub_subscription_symbol = None
            self._data_hub_subscription_token = None

    def _runtime_bundle(self, symbol: str, allow_stale: bool = True) -> dict | None:
        sym = self._symbol_key(symbol)
        hub = getattr(self, "data_hub", None)
        if hub is not None:
            try:
                bundle = hub.get_runtime_bundle(sym, allow_stale=allow_stale)
                if isinstance(bundle, dict):
                    return bundle
            except Exception:
                logger.debug("DataHub runtime bundle lookup failed", exc_info=True)
        bundle = self._workspace_runtime_cache.get(sym)
        if not isinstance(bundle, dict):
            return None
        age = time.time() - float(bundle.get("loaded_at") or 0.0)
        if not allow_stale and age > float(self._workspace_runtime_cache_ttl_sec):
            return None
        self._workspace_runtime_cache.move_to_end(sym)
        return bundle

    def _refresh_symbol_runtime(self, symbol: str, *, force: bool = False, reason: str | None = None, load_models: bool = True) -> dict:
        sym = self._symbol_key(symbol)
        hub = getattr(self, "data_hub", None)
        if hub is not None:
            return dict(hub.refresh_symbol(sym, force=force, reason=reason, load_models=load_models) or {})
        models = MODEL_STACK_LOADER(instrument_key=sym) if load_models else {}
        df = update_price_data(sym)
        return {"symbol": sym, "models": models, "df": df}

    def _apply_runtime_bundle(self, symbol: str, bundle: dict | None) -> bool:
        if not isinstance(bundle, dict):
            return False
        with self._lock:
            self.models = dict(bundle.get("models") or {})
            self.df = self._trim_df_for_ui(bundle.get("df"))
        remember_workspace_view(symbol, note="Using cached workspace context")
        self._refresh_workspace_tabs()
        return True

    def _invalidate_workspace_caches(self, symbol: str | None = None, clear_prediction: bool = True, clear_dashboard: bool = True):
        sym = self._symbol_key(symbol or self.instrument_var.get())
        if clear_dashboard:
            self._dashboard_cache.pop(sym, None)
        if clear_prediction:
            for key in list(self._prediction_payload_cache.keys()):
                if isinstance(key, tuple) and self._symbol_key(key[0]) == sym:
                    self._prediction_payload_cache.pop(key, None)

    def _activate_workspace(self, symbol: str, force_reload: bool = False):
        sym = self._symbol_key(symbol)
        if not sym:
            return
        open_workspace(sym, meta=INSTRUMENTS.get(sym, {}), activate=True)
        self._bind_runtime_subscription(sym)
        self.instrument_var.set(sym)
        if hasattr(self, "accuracy_var"):
            self.accuracy_var.set(f"Avg Discrepancy ({sym}): loading…")
        ws = get_workspace(sym) or {}
        if hasattr(self, "mode_var"):
            self.mode_var.set(str(ws.get("view_mode") or self.mode_var.get() or "4min"))
        if hasattr(self, "profile_var"):
            self.profile_var.set(str(ws.get("profile_name") or self.profile_var.get() or get_trader_profile().get("key", "balanced")))
        self.profile_summary_var.set(f"Profile: {get_trader_profile(self.profile_var.get()).get('label', 'Balanced')}")
        self._show_mode_ui()
        self._load_autonomy_ui_state()
        self._refresh_workspace_tabs()
        cached = None if force_reload else self._runtime_bundle(sym)
        if self._apply_runtime_bundle(sym, cached):
            self.run_prediction_and_refresh_ui(reason=f"Opened {sym} workspace from cache…")
            self.reload_once_threaded()
            return
        waiting = self._build_waiting_payload(self.mode_var.get(), f"Opening {sym} workspace…", status=f"Loading {sym} workspace…", symbol=sym)
        self._apply_prediction_payload(waiting)
        self.reload_once_threaded()

    def _open_selected_workspace(self):
        self._activate_workspace(self.instrument_var.get())

    def _close_active_workspace(self):
        sym = self._symbol_key(self.instrument_var.get())
        state = close_workspace(sym)
        next_symbol = self._symbol_key((state or {}).get("active_symbol") or DEFAULT_INSTRUMENT)
        if next_symbol and next_symbol != sym:
            self._activate_workspace(next_symbol)
        else:
            self._refresh_workspace_tabs()

    def _open_overview_workspace(self, event=None):
        try:
            selection = self.overview_tree.selection()
            if not selection:
                return
            values = self.overview_tree.item(selection[0], "values")
            if not values:
                return
            self._activate_workspace(values[0])
        except Exception:
            logger.debug("Workspace overview selection failed", exc_info=True)

    def _open_add_stock_dialog(self):
        win = tk.Toplevel(self.root)
        win.title("Add Stock Workspace")
        win.geometry("620x500")
        win.configure(bg="#081018")
        win.transient(self.root)
        win.grab_set()
        self._apply_dialog_icon(win)

        tk.Label(win, text="Add Stock Workspace", font=self.big_font, fg="#f8fafc", bg="#081018").pack(anchor="w", padx=18, pady=(16, 8))
        tk.Label(
            win,
            text="Create a new stock workspace, register its profile, and optionally start loading or training it right away.",
            font=self.small_font,
            fg="#cbd5e1",
            bg="#081018",
            wraplength=480,
            justify="left",
        ).pack(anchor="w", padx=18)

        form = tk.Frame(win, bg="#081018")
        form.pack(fill="x", padx=18, pady=(14, 10))
        symbol_var = tk.StringVar(value="")
        display_var = tk.StringVar(value="")
        company_var = tk.StringVar(value="")
        sector_var = tk.StringVar(value="Technology")
        type_var = tk.StringVar(value="equity")
        aliases_var = tk.StringVar(value="")

        rows = [
            ("Symbol", symbol_var, ttk.Entry(form, textvariable=symbol_var, width=22)),
            ("Display", display_var, ttk.Entry(form, textvariable=display_var, width=28)),
            ("Company", company_var, ttk.Entry(form, textvariable=company_var, width=32)),
            ("News Aliases", aliases_var, ttk.Entry(form, textvariable=aliases_var, width=36)),
            ("Sector", sector_var, ttk.Combobox(form, textvariable=sector_var, values=[
                "Technology",
                "Communication Services",
                "Consumer Discretionary",
                "Consumer Staples",
                "Healthcare",
                "Financials",
                "Industrials",
                "Energy",
                "Materials",
                "Real Estate",
                "Utilities",
                "Broad Market",
            ], state="readonly", width=24)),
            ("Type", type_var, ttk.Combobox(form, textvariable=type_var, values=["equity", "etf"], state="readonly", width=16)),
        ]
        for idx, (label, _, widget) in enumerate(rows):
            tk.Label(form, text=label, font=self.small_font, fg="#e2e8f0", bg="#081018").grid(row=idx, column=0, sticky="w", pady=(0, 8))
            widget.grid(row=idx, column=1, sticky="ew", pady=(0, 8))
        form.columnconfigure(1, weight=1)

        preview_frame = tk.Frame(win, bg="#0f1724", highlightbackground="#1f2a44", highlightthickness=1, bd=0)
        preview_frame.pack(fill="x", padx=18, pady=(0, 10))
        tk.Label(preview_frame, text="Lifecycle Preview", font=self.small_font, fg="#7dd3fc", bg="#0f1724").pack(anchor="w", padx=12, pady=(10, 0))
        preview_var = tk.StringVar(
            value="New stock workspaces begin as supported-only. Price history staging, feature readiness, models, calibration, and readiness evidence are earned symbol by symbol."
        )
        tk.Label(
            preview_frame,
            textvariable=preview_var,
            font=self.small_font,
            fg="#cbd5e1",
            bg="#0f1724",
            wraplength=500,
            justify="left",
        ).pack(anchor="w", padx=12, pady=(4, 10))

        status_var = tk.StringVar(value="Create a workspace to register the symbol. Training is optional and does not fake maturity if prerequisites are missing.")
        tk.Label(win, textvariable=status_var, font=self.small_font, fg="#93c5fd", bg="#081018", wraplength=480, justify="left").pack(anchor="w", padx=18, pady=(0, 12))

        def _refresh_preview(*_args):
            sym = self._symbol_key(symbol_var.get())
            if not sym:
                preview_var.set("Enter a stock ticker to preview how LightningTrader4000 will register and stage it.")
                return
            aliases = [item.strip() for item in str(aliases_var.get() or "").split(",") if item.strip()]
            preview_meta = build_instrument_meta(
                sym,
                overrides={
                    "display_name": str(display_var.get() or sym).strip() or sym,
                    "company_name": str(company_var.get() or display_var.get() or sym).strip() or sym,
                    "sector": str(sector_var.get() or "").strip() or "Broad Market",
                    "type": str(type_var.get() or "equity").strip().lower(),
                    "news_aliases": aliases or [sym, str(company_var.get() or display_var.get() or sym).strip() or sym],
                },
            )
            kind = str(preview_meta.get("instrument_kind") or preview_meta.get("type") or "equity").title()
            sector = str(preview_meta.get("sector") or "Broad Market")
            stage_hint = "Existing symbol: workspace opens with current artifacts." if sym in INSTRUMENTS else "New symbol: profile resolves first, then price data, features, models, calibration, and readiness evidence."
            preview_var.set(
                f"{sym} | {kind} | {sector} | "
                f"Context {', '.join((preview_meta.get('context_daily_symbols') or {}).keys()) or 'default'} | "
                f"{stage_hint}"
            )

        for var in (symbol_var, display_var, company_var, aliases_var, sector_var, type_var):
            try:
                var.trace_add("write", _refresh_preview)
            except Exception:
                pass
        _refresh_preview()

        def _create(start_training: bool):
            sym = self._symbol_key(symbol_var.get())
            if not sym:
                status_var.set("Enter a stock symbol first.")
                return
            aliases = [item.strip() for item in str(aliases_var.get() or "").split(",") if item.strip()]
            meta = build_instrument_meta(
                sym,
                overrides={
                    "display_name": str(display_var.get() or sym).strip() or sym,
                    "company_name": str(company_var.get() or display_var.get() or sym).strip() or sym,
                    "sector": str(sector_var.get() or "").strip() or "Broad Market",
                    "type": str(type_var.get() or "equity").strip().lower(),
                    "news_aliases": aliases or [sym, str(company_var.get() or display_var.get() or sym).strip() or sym],
                },
            )
            register_instrument(sym, meta=meta, persist=True)
            open_workspace(sym, meta=meta, activate=True)
            self._refresh_instrument_choices()
            self._activate_workspace(sym, force_reload=True)
            status_var.set(f"{sym} workspace created. Starting the onboarding pipeline…")
            self._queue_onboarding_job(sym, meta=meta, train_models=start_training)
            try:
                win.destroy()
            except Exception:
                pass

        buttons = tk.Frame(win, bg="#081018")
        buttons.pack(fill="x", padx=18, pady=(8, 18))
        ttk.Button(buttons, text="Create Workspace", command=lambda: _create(False)).pack(side="left")
        ttk.Button(buttons, text="Create + Start Training", command=lambda: _create(True)).pack(side="left", padx=(8, 0))
        ttk.Button(buttons, text="Close", style="Secondary.TButton", command=win.destroy).pack(side="right")

    def _queue_onboarding_job(
        self,
        symbol: str,
        meta: dict | None = None,
        train_models: bool = True,
        *,
        started_note: str = "Onboarding pipeline started",
        completion_note: str | None = None,
        on_complete=None,
    ):
        sym = self._symbol_key(symbol)
        remember_workspace_view(sym, mode=self.mode_var.get(), profile_name=self.profile_var.get(), status="onboarding", note=started_note)
        self._refresh_workspace_tabs()

        def _progress(stage_key: str, stage_label: str, state: dict):
            def _apply():
                summary = summarize_onboarding_state(sym)
                note = str(summary.get("summary") or stage_label or "Onboarding update")
                remember_workspace_view(
                    sym,
                    mode=self.mode_var.get(),
                    profile_name=self.profile_var.get(),
                    status=f"onboarding_{stage_key}",
                    note=note,
                )
                self._refresh_workspace_tabs()
                if self._workspace_is_active(sym):
                    self.status(f"{sym} onboarding: {note}")

            try:
                self.root.after(0, _apply)
            except Exception:
                pass

        def _job():
            success = False
            try:
                result = run_onboarding_job(
                    sym,
                    meta=meta,
                    train_models=train_models,
                    progress_cb=_progress,
                    data_hub=getattr(self, "data_hub", None),
                )
                bundle = self._refresh_symbol_runtime(sym, force=True, reason="onboarding_complete", load_models=True)
                models = dict(bundle.get("models") or {})
                latest_df = self._trim_df_for_ui(bundle.get("df"))
                if self._workspace_is_active(sym):
                    with self._lock:
                        self.models = models
                        self.df = latest_df
                self._store_runtime_bundle(sym, models, latest_df)
                onboarding = summarize_onboarding_state(sym)
                status_note = str(completion_note or onboarding.get("summary") or result.get("summary") or "Onboarding complete")
                remember_workspace_view(
                    sym,
                    mode=self.mode_var.get(),
                    profile_name=self.profile_var.get(),
                    status="ready" if str(onboarding.get("status") or "") == "completed" else "onboarding",
                    note=status_note,
                )
                self.root.after(0, self._refresh_workspace_tabs)
                self.root.after(0, lambda: self.run_prediction_and_refresh_ui(reason=f"Refreshing {sym} after onboarding…"))
                self.root.after(0, self.refresh_overview_threaded)
                self.root.after(0, self.refresh_accuracy_summary_threaded)
                self.status(f"{sym} onboarding complete — refreshing workspace context…")
                success = True
            except Exception:
                logger.exception("Onboarding job failed")
                onboarding = summarize_onboarding_state(sym)
                note = str(onboarding.get("last_error") or onboarding.get("summary") or "Onboarding failed")
                remember_workspace_view(sym, mode=self.mode_var.get(), profile_name=self.profile_var.get(), status="onboarding_failed", note=note)
                self.root.after(0, self._refresh_workspace_tabs)
                self.status(f"{sym} onboarding failed; see logs")
            finally:
                if on_complete:
                    try:
                        self.root.after(0, lambda: on_complete(success))
                    except Exception:
                        try:
                            on_complete(success)
                        except Exception:
                            pass

        try:
            self.executor.submit(_job)
        except Exception:
            logger.exception("Failed to schedule onboarding job")
            if on_complete:
                try:
                    on_complete(False)
                except Exception:
                    pass
            raise

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

    def _prediction_payload_cache_key(self, mode: str, selected: str, df_copy, replay_snapshot: dict | None = None):
        last_bar_key = "-"
        try:
            if df_copy is not None and not getattr(df_copy, "empty", True):
                last_bar = pd.to_datetime(df_copy.index[-1], utc=True, errors="coerce")
                if pd.notna(last_bar):
                    last_bar_key = str(last_bar.isoformat())
        except Exception:
            pass
        replay_marker = "live"
        if isinstance(replay_snapshot, dict) and replay_snapshot.get("active"):
            replay_marker = "|".join(
                [
                    "replay",
                    str(replay_snapshot.get("status") or "-"),
                    str(replay_snapshot.get("cursor") or replay_snapshot.get("current_index") or "-"),
                    str(bool(replay_snapshot.get("completed"))),
                ]
            )
        return (
            str(selected or "").upper(),
            str(mode or ""),
            last_bar_key,
            replay_marker,
            str(self.profile_var.get() or ""),
        )

    def mode_apply(self):
        mode = self.mode_var.get()
        remember_workspace_view(self.instrument_var.get(), mode=mode, profile_name=self.profile_var.get(), note=f"View {mode}")
        self._refresh_workspace_tabs()
        self.status(f"Switching to {mode} view…")
        self._show_mode_ui()
        df_copy, _, selected, replay_snapshot = self._current_prediction_inputs()
        cache_key = self._prediction_payload_cache_key(mode, selected, df_copy, replay_snapshot)
        cached = self._prediction_payload_cache.get(cache_key)
        if isinstance(cached, dict):
            cached_payload = dict(cached)
            cached_payload["status_messages"] = list(cached_payload.get("status_messages") or []) + [f"Showing cached {mode} view while refreshing…"]
            self._apply_prediction_payload(cached_payload)
        self.run_prediction_and_refresh_ui(reason=f"Switching to {mode} view…")

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
        self.status(f"Opening {self.instrument_var.get()} workspace…")
        self._activate_workspace(self.instrument_var.get())

    def _on_profile_change(self, event=None):
        profile = get_trader_profile(self.profile_var.get())
        selected = self._symbol_key(self.instrument_var.get())
        remember_workspace_view(selected, mode=self.mode_var.get(), profile_name=self.profile_var.get(), note=f"Profile {profile.get('label', 'Balanced')}")
        self._invalidate_workspace_caches(selected)
        self.profile_summary_var.set(f"Profile: {profile.get('label', 'Balanced')}")
        self._refresh_workspace_tabs()
        self.status(f"Trader style set to {profile.get('label', 'Balanced')}; refreshing advisory context…")
        self.run_prediction_and_refresh_ui(reason="Refreshing advisory context…")

    def _current_replay_snapshot(self):
        try:
            return get_replay_dashboard_snapshot(str(self.instrument_var.get() or "").upper())
        except Exception:
            logger.debug("Replay snapshot lookup failed", exc_info=True)
            return {"active": False, "status": "inactive"}

    def _load_autonomy_ui_state(self):
        selected = str(self.instrument_var.get() or "").upper()
        snap = get_autonomy_snapshot(selected)
        self._latest_autonomy_meta = dict(snap or {})
        try:
            self.autonomy_mode_var.set(str((snap or {}).get("mode_label") or "OFF"))
            self.autonomy_decision_var.set(str((snap or {}).get("decision_style_label") or AUTONOMY_DECISION_STYLE_LABELS.get("full_autonomy", "Full Autonomy")))
            label = str((snap or {}).get("strategy_label") or "")
            if label and label in getattr(self, "autonomy_strategy_choices", {}):
                self.autonomy_strategy_var.set(label)
        except Exception:
            pass
        self._sync_action_buttons()

    @staticmethod
    def _decision_style_from_label(label: str | None) -> str:
        text = str(label or "").strip().lower()
        for key, value in AUTONOMY_DECISION_STYLE_LABELS.items():
            if text == str(value).strip().lower():
                return key
        if "strategy" in text:
            return "strategy_autonomy"
        return "full_autonomy"

    def _selected_strategy_key(self) -> str:
        var = getattr(self, "autonomy_strategy_var", None)
        try:
            label = str(var.get() if var is not None else "")
        except Exception:
            label = ""
        choices = getattr(self, "autonomy_strategy_choices", {}) or {}
        return str(choices.get(label) or "trend_following")

    def _apply_autonomy_mode(self):
        selected = str(self.instrument_var.get() or "").upper()
        snap = set_autonomy_mode(selected, self.autonomy_mode_var.get())
        snap = set_autonomy_decision_style(
            selected,
            self._decision_style_from_label(self.autonomy_decision_var.get()),
            strategy_key=self._selected_strategy_key(),
        )
        self._latest_autonomy_meta = dict(snap or {})
        self.status(
            f"Autonomy set to {snap.get('mode_label') or self.autonomy_mode_var.get()} / "
            f"{snap.get('decision_style_label') or self.autonomy_decision_var.get()} for {selected}"
        )
        self.run_prediction_and_refresh_ui(reason="Refreshing autonomy controller…")

    def _open_autonomy_settings_dialog(self):
        selected = str(self.instrument_var.get() or "").upper()
        snap = dict(get_autonomy_snapshot(selected) or {})
        win = tk.Toplevel(self.root)
        win.title("Autonomy Risk")
        win.geometry("560x410")
        win.configure(bg="#081018")
        win.transient(self.root)
        win.grab_set()
        self._apply_dialog_icon(win)

        tk.Label(win, text="Autonomy Risk", font=self.big_font, fg="#f8fafc", bg="#081018").pack(anchor="w", padx=18, pady=(16, 6))
        tk.Label(
            win,
            text="Set the controller-level stop loss. When realized plus tracked open PnL breaches this limit, autonomy halts and any tracked autonomous position is flattened.",
            font=self.small_font,
            fg="#cbd5e1",
            bg="#081018",
            wraplength=510,
            justify="left",
        ).pack(anchor="w", padx=18, pady=(0, 12))

        form = tk.Frame(win, bg="#081018")
        form.pack(fill="x", padx=18, pady=(4, 8))
        decision_var = tk.StringVar(value=str(snap.get("decision_style_label") or self.autonomy_decision_var.get()))
        strategy_var = tk.StringVar(value=str(snap.get("strategy_label") or self.autonomy_strategy_var.get()))
        dollars_var = tk.StringVar(value="" if snap.get("loss_limit_dollars") in (None, "") else str(snap.get("loss_limit_dollars")))
        pct_val = snap.get("loss_limit_pct")
        pct_var = tk.StringVar(value="" if pct_val in (None, "") else f"{100.0 * float(pct_val):.2f}")
        status_var = tk.StringVar(value=f"Current: {snap.get('loss_limit_detail') or '-'} | {snap.get('operator_label') or '-'}")

        rows = [
            ("Decision", ttk.Combobox(form, textvariable=decision_var, values=[AUTONOMY_DECISION_STYLE_LABELS[k] for k in ("full_autonomy", "strategy_autonomy")], state="readonly", width=22)),
            ("Strategy", ttk.Combobox(form, textvariable=strategy_var, values=list((getattr(self, "autonomy_strategy_choices", {}) or {}).keys()), state="readonly", width=24)),
            ("Stop $", ttk.Entry(form, textvariable=dollars_var, width=18)),
            ("Stop %", ttk.Entry(form, textvariable=pct_var, width=18)),
        ]
        for idx, (label, widget) in enumerate(rows):
            tk.Label(form, text=label, font=self.small_font, fg="#e2e8f0", bg="#081018").grid(row=idx, column=0, sticky="w", pady=(0, 8))
            widget.grid(row=idx, column=1, sticky="ew", pady=(0, 8))
        form.columnconfigure(1, weight=1)
        tk.Label(win, textvariable=status_var, font=self.small_font, fg="#93c5fd", bg="#081018", wraplength=510, justify="left").pack(anchor="w", padx=18, pady=(0, 12))

        def _parse_float(text: str) -> float | None:
            raw = str(text or "").strip().replace("$", "").replace(",", "").replace("%", "")
            if not raw:
                return None
            try:
                return float(raw)
            except Exception:
                return None

        def _save(clear_halt: bool = False):
            self.autonomy_decision_var.set(decision_var.get())
            if strategy_var.get():
                self.autonomy_strategy_var.set(strategy_var.get())
            set_autonomy_decision_style(
                selected,
                self._decision_style_from_label(decision_var.get()),
                strategy_key=(getattr(self, "autonomy_strategy_choices", {}) or {}).get(strategy_var.get()) or self._selected_strategy_key(),
            )
            dollars = _parse_float(dollars_var.get())
            pct = _parse_float(pct_var.get())
            snap2 = set_autonomy_loss_limit(
                selected,
                dollars=dollars,
                pct=(pct / 100.0) if pct is not None and pct > 0 else None,
                clear_halt=clear_halt,
            )
            self._latest_autonomy_meta = dict(snap2 or {})
            self.status(f"Autonomy risk saved for {selected}: {snap2.get('loss_limit_detail') or '-'}")
            try:
                win.destroy()
            except Exception:
                pass
            self.run_prediction_and_refresh_ui(reason="Refreshing autonomy controller…")

        buttons = tk.Frame(win, bg="#081018")
        buttons.pack(fill="x", padx=18, pady=(8, 18))
        ttk.Button(buttons, text="Save", command=lambda: _save(False)).pack(side="left")
        ttk.Button(buttons, text="Save + Clear Halt", command=lambda: _save(True)).pack(side="left", padx=(8, 0))
        ttk.Button(buttons, text="Close", style="Secondary.TButton", command=win.destroy).pack(side="right")

    def _toggle_autonomy_halt(self):
        selected = str(self.instrument_var.get() or "").upper()
        enabled = not bool((self._latest_autonomy_meta or {}).get("kill_switch"))
        snap = set_autonomy_kill_switch(
            selected,
            enabled,
            reason="Operator halt requested from the GUI" if enabled else "Operator halt cleared from the GUI",
        )
        self._latest_autonomy_meta = dict(snap or {})
        self.status("Autonomy halt enabled" if enabled else "Autonomy halt cleared")
        self.run_prediction_and_refresh_ui(reason="Refreshing autonomy controller…")

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
        self._invalidate_workspace_caches(selected)
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
        self._invalidate_workspace_caches(selected)
        self.run_prediction_and_refresh_ui()

    def _flatten_routed_position(self):
        selected = str(self.instrument_var.get() or "").upper()
        result = flatten_routed_position(
            selected,
            current_price=self._latest_price,
            profile_name=self.profile_var.get(),
        )
        route = str(result.get("routing_label") or result.get("routing_mode") or "route")
        status = str(result.get("status") or "-")
        if status in {"filled", "submitted"}:
            self.status(f"{route} FLAT sent for {selected}: {status}")
        else:
            self.status(f"{route} FLAT skipped: {result.get('reason') or status}")
        self._invalidate_workspace_caches(selected)
        self.run_prediction_and_refresh_ui()

    def _cancel_routed_order(self):
        selected = str(self.instrument_var.get() or "").upper()
        result = cancel_routed_order(symbol=selected)
        if str(result.get("status")) == "canceled":
            self.status(f"Canceled routed order for {selected}")
        else:
            self.status(f"Cancel skipped: {result.get('reason') or result.get('status') or 'blocked'}")
        self._invalidate_workspace_caches(selected)
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
                self._invalidate_workspace_caches(self.instrument_var.get())
                self._refresh_broker_status()
                self.run_prediction_and_refresh_ui()
                status_var.set(f"{success_text} Saved locally on this computer.")
                messagebox.showinfo("Alpaca Setup", f"{success_text}\n\nThe app will reuse these keys on future launches.")
                self._close_alpaca_setup()
            else:
                status_var.set(success_text)

        def _clear_saved():
            removed = clear_local_broker_credentials()
            self._invalidate_workspace_caches(self.instrument_var.get())
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
        self._invalidate_workspace_caches(selected)
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
        self._invalidate_workspace_caches(selected)
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
        self._invalidate_workspace_caches(selected)
        self.run_prediction_and_refresh_ui()

    def _start_replay(self):
        selected = str(self.instrument_var.get() or "").upper()
        with self._lock:
            df_copy = self.df.copy() if self.df is not None else None
        if df_copy is None or getattr(df_copy, "empty", True):
            self.status("Replay start skipped: no local data loaded")
            return
        snapshot = start_active_replay(df_copy, selected, profile_name=self.profile_var.get())
        self._invalidate_workspace_caches(selected)
        self.status(f"Replay simulation started: {snapshot.get('detail') or selected}")
        self.run_prediction_and_refresh_ui()

    def _step_replay(self):
        snapshot = step_active_replay()
        self._invalidate_workspace_caches(self.instrument_var.get())
        self.status(f"Replay step: {snapshot.get('detail') or snapshot.get('status') or 'updated'}")
        self.run_prediction_and_refresh_ui()

    def _toggle_replay(self):
        current = self._current_replay_snapshot()
        if not current.get("active"):
            self.status("Replay toggle skipped: no active replay session")
            return
        snapshot = set_replay_paused(not bool(current.get("paused", True)))
        self._invalidate_workspace_caches(self.instrument_var.get())
        state = "paused" if snapshot.get("paused", True) else "running"
        self.status(f"Replay {state}: {snapshot.get('detail') or '-'}")
        self.run_prediction_and_refresh_ui()

    def _reset_replay(self):
        snapshot = reset_active_replay()
        self._invalidate_workspace_caches(self.instrument_var.get())
        self.status(f"Replay reset: {snapshot.get('detail') or '-'}")
        self.run_prediction_and_refresh_ui()

    def _stop_replay(self):
        snapshot = stop_active_replay()
        self._invalidate_workspace_caches(self.instrument_var.get())
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
        self._invalidate_workspace_caches(selected)
        self.run_prediction_and_refresh_ui()

    def _flatten_replay_position(self):
        selected = str(self.instrument_var.get() or "").upper()
        result = flatten_replay_position(advisory=self._latest_advisory_meta)
        if str(result.get("status")) == "filled":
            order = result.get("order") or {}
            self.status(f"Replay flat: {selected} {int(float(order.get('qty') or 0))} @ ${float(order.get('fill_price') or 0.0):.2f}")
        else:
            self.status(f"Replay flat skipped: {result.get('reason') or result.get('status') or 'blocked'}")
        self._invalidate_workspace_caches(selected)
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
        selected = self._symbol_key(self.instrument_var.get())

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
                self.root.after(0, lambda: self.accuracy_var.set(text) if self._workspace_is_active(selected) else None)
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
                self._refresh_workspace_tabs()
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
        autonomy = dict(getattr(self, "_latest_autonomy_meta", {}) or {})
        autonomy_mode = str(autonomy.get("mode") or "off")
        kill_switch = bool(autonomy.get("kill_switch"))
        auto_locked = bool(autonomy.get("manual_trading_locked")) and not kill_switch
        self.route_buy_var.set(f"Buy ({route_label})")
        self.route_sell_var.set(f"Sell ({route_label})")
        if hasattr(self, "route_flat_var"):
            self.route_flat_var.set(f"Flat ({route_label})")
        self.route_cancel_var.set("Cancel Routed")
        self.route_setup_var.set(str((self._latest_execution_meta or {}).get("route_setup_label") or "Alpaca Setup"))
        self.route_refresh_var.set(str((self._latest_execution_meta or {}).get("route_refresh_label") or "Refresh Broker"))
        self.route_verify_var.set(str((self._latest_execution_meta or {}).get("route_verify_label") or "Verify Alpaca Paper"))
        can_place = bool(runtime.get("can_place_orders"))
        can_cancel = bool(runtime.get("can_cancel"))
        self.route_buy_btn.configure(state=("normal" if can_place and not auto_locked else "disabled"))
        self.route_sell_btn.configure(state=("normal" if can_place and not auto_locked else "disabled"))
        if hasattr(self, "route_flat_btn"):
            self.route_flat_btn.configure(state=("normal" if can_place else "disabled"))
        self.route_cancel_btn.configure(state=("normal" if can_cancel else "disabled"))
        self.route_setup_btn.configure(state="normal")
        self.route_refresh_btn.configure(state="normal")
        self.route_verify_btn.configure(state="normal")
        if hasattr(self, "paper_buy_btn"):
            self.paper_buy_btn.configure(state=("normal" if not auto_locked else "disabled"))
        if hasattr(self, "paper_sell_btn"):
            self.paper_sell_btn.configure(state=("normal" if not auto_locked else "disabled"))
        if hasattr(self, "paper_flatten_btn"):
            self.paper_flatten_btn.configure(state="normal")

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
        if hasattr(self, "autonomy_halt_var"):
            self.autonomy_halt_var.set("Clear Halt" if kill_switch else "Halt Auto")
        if hasattr(self, "autonomy_decision_var"):
            self.autonomy_decision_var.set(str(autonomy.get("decision_style_label") or self.autonomy_decision_var.get() or "Full Autonomy"))
        if hasattr(self, "autonomy_strategy_var") and autonomy.get("strategy_label"):
            self.autonomy_strategy_var.set(str(autonomy.get("strategy_label")))
        if hasattr(self, "autonomy_strategy_menu"):
            decision_style = str(autonomy.get("decision_style") or "full_autonomy")
            self.autonomy_strategy_menu.configure(state=("readonly" if decision_style == "strategy_autonomy" else "disabled"))
        if hasattr(self, "autonomy_badge_var"):
            self.autonomy_badge_var.set(str(autonomy.get("operator_label") or "Operator: Advisory Only"))
        if hasattr(self, "autonomy_badge_label"):
            badge_bg = "#94a3b8"
            badge_fg = "#081018"
            if kill_switch:
                badge_bg = "#ef4444"
                badge_fg = "#f8fafc"
            elif autonomy_mode == "shadow":
                badge_bg = "#fbbf24"
            elif autonomy_mode == "paper_auto":
                badge_bg = "#fb7185"
                badge_fg = "#f8fafc"
            elif autonomy_mode == "live_auto":
                badge_bg = "#dc2626"
                badge_fg = "#f8fafc"
            self.autonomy_badge_label.configure(bg=badge_bg, fg=badge_fg)
        if hasattr(self, "brand_subtitle_var"):
            subtitle = "Dual-mode workstation | advisory by default"
            if autonomy_mode == "shadow":
                subtitle = f"Dual-mode workstation | shadow autonomy on {route_label}"
            elif autonomy_mode == "paper_auto":
                subtitle = f"Dual-mode workstation | paper autonomy on {route_label}"
            elif autonomy_mode == "live_auto":
                subtitle = f"Dual-mode workstation | live autonomy on {route_label}"
            if kill_switch:
                subtitle = f"{subtitle} | HALTED"
            self.brand_subtitle_var.set(subtitle)

    def _compute_overview_rows(self):
        rows = []
        symbols = list_workspace_symbols(open_only=True, limit=8)
        if not symbols:
            try:
                ensure_runtime_instruments(force=False, auto_onboard=True)
                symbols = list_runtime_instruments()
            except Exception:
                symbols = list(INSTRUMENTS.keys())
        if not symbols:
            return rows
        bundle_map = {}
        hub = getattr(self, "data_hub", None)
        if hub is not None:
            try:
                bundle_map = dict(
                    hub.snapshot_symbols(
                        symbols,
                        allow_stale=False,
                        prefetch_missing=True,
                        reason="overview_refresh",
                        load_models=True,
                    )
                    or {}
                )
            except Exception:
                bundle_map = {}
        for sym in symbols:
            try:
                bundle = dict(bundle_map.get(sym) or {})
                if bundle.get("error"):
                    rows.append((sym, "-", "-", "-", "-", str(bundle.get("error") or "Error")))
                    continue
                if not bundle:
                    bundle = self._runtime_bundle(sym, allow_stale=False) or {}
                if bundle:
                    models = dict(bundle.get("models") or {})
                    df = bundle.get("df")
                else:
                    bundle = self._refresh_symbol_runtime(sym, force=False, reason="overview_refresh", load_models=True)
                    models = dict(bundle.get("models") or {})
                    df = bundle.get("df")
                    self._store_runtime_bundle(sym, models, df)
                lifecycle = dict(bundle.get("lifecycle_meta") or self._lifecycle_snapshot(sym, runtime=bundle))
                coverage = str(lifecycle.get("model_coverage_label") or lifecycle.get("data_state_label") or "-")

                four = "-"
                if df is not None and not getattr(df, "empty", True) and models:
                    p4 = dict(build_inference_payload(models, df, instrument_key=sym, mode="4min").get("prediction") or {})
                    if isinstance(p4, dict) and not p4.get("error"):
                        c = p4.get("consensus", {}) if isinstance(p4, dict) else {}
                        pred_price = c.get("pred_price")
                        direction = c.get("direction", "-")
                        four = (
                            f"{self._fmt_price(pred_price)} ({direction})"
                            if isinstance(pred_price, (int, float))
                            else str(lifecycle.get("model_state_label") or lifecycle.get("data_state_label") or "-")
                        )
                    else:
                        four = str(lifecycle.get("model_state_label") or "Waiting")
                else:
                    four = str(lifecycle.get("model_state_label") or lifecycle.get("data_state_label") or "-")

                daily = "-"
                if df is not None and not getattr(df, "empty", True) and models:
                    pdaily = dict(build_inference_payload(models, df, instrument_key=sym, mode="DailyHigh").get("prediction") or {})
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
                    elif "DailyHigh" not in list(lifecycle.get("available_modes") or []):
                        daily = str(lifecycle.get("feature_state_label") or "Daily pending")
                elif "DailyHigh" not in list(lifecycle.get("available_modes") or []):
                    daily = str(lifecycle.get("feature_state_label") or "Daily pending")

                week = "-"
                if df is not None and not getattr(df, "empty", True) and models:
                    pweek = dict(build_inference_payload(models, df, instrument_key=sym, mode="WeekEnd").get("prediction") or {})
                    if isinstance(pweek, dict) and not pweek.get("error"):
                        block = pweek.get("week", {})
                        vals = [v.get("pred_price") for v in block.values() if isinstance(v, dict) and isinstance(v.get("pred_price"), (int, float))]
                        if vals:
                            week = self._fmt_price(sum(vals) / len(vals))
                    elif "WeekEnd" not in list(lifecycle.get("available_modes") or []):
                        week = str(lifecycle.get("feature_state_label") or "WeekEnd pending")
                elif "WeekEnd" not in list(lifecycle.get("available_modes") or []):
                    week = str(lifecycle.get("feature_state_label") or "WeekEnd pending")

                status_bits = [
                    str(lifecycle.get("operator_state_label") or ""),
                    str(lifecycle.get("evidence_state_label") or ""),
                ]
                if str(lifecycle.get("primary_next_requirement") or "").strip():
                    status_bits.append(f"Next: {str(lifecycle.get('primary_next_requirement'))}")
                rows.append((sym, coverage, four, daily, week, " | ".join(bit for bit in status_bits if bit)))
            except Exception:
                rows.append((sym, "-", "-", "-", "-", "Error"))
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
        def _apply():
            try:
                self.status_label.config(text=f"Status: {text} — {ts} UTC")
                if getattr(self, "_loading_overlay_visible", False) and hasattr(self, "loading_hint_var"):
                    self.loading_hint_var.set(str(text or "Loading workspace"))
            except Exception:
                pass
        try:
            if threading.current_thread() is threading.main_thread():
                _apply()
            else:
                self.root.after(0, _apply)
        except Exception:
            pass
        logger.info(text)

    def _set_activity(self, text: str, active: bool = False, accent: str | None = None):
        label_text = str(text or "Ready").strip() or "Ready"

        def _apply():
            try:
                if hasattr(self, "activity_var"):
                    self.activity_var.set(f"Activity: {label_text}")
                if hasattr(self, "activity_label"):
                    self.activity_label.configure(fg=accent or ("#60a5fa" if active else "#86efac"))
                if hasattr(self, "activity_bar"):
                    if active and not getattr(self, "_activity_running", False):
                        self.activity_bar.start(10)
                        self._activity_running = True
                    elif not active and getattr(self, "_activity_running", False):
                        self.activity_bar.stop()
                        self._activity_running = False
            except Exception:
                pass

        try:
            if threading.current_thread() is threading.main_thread():
                _apply()
            else:
                self.root.after(0, _apply)
        except Exception:
            pass

    def _activity_state_from_payload(self, payload: dict) -> tuple[str, bool, str]:
        messages = [str(msg).strip() for msg in list(payload.get("status_messages") or []) if str(msg).strip()]
        last_msg = messages[-1] if messages else ""
        lower = last_msg.lower()
        if any(token in lower for token in ("error", "failed")):
            return last_msg, False, "#fda4af"
        if any(token in lower for token in ("loading", "waiting", "refresh", "training", "initializ")):
            return last_msg, True, "#60a5fa"
        autonomy = dict(payload.get("autonomy_meta") or {})
        if bool(autonomy.get("kill_switch")):
            return str(autonomy.get("operator_label") or "Autonomy halted"), False, "#f87171"
        if str(autonomy.get("mode") or "off") in {"shadow", "paper_auto", "live_auto"}:
            return str(autonomy.get("operator_label") or "Autonomy ready"), False, "#fbbf24"
        return "Ready for next cycle", False, "#86efac"

    def _build_loading_overlay(self):
        self.loading_title_var = tk.StringVar(value=APP_DISPLAY_NAME)
        self.loading_status_var = tk.StringVar(value="Booting workspace")
        self.loading_hint_var = tk.StringVar(value="Loading models, price data, and trading context")
        self.loading_overlay = tk.Frame(self.root, bg="#030712")
        self.loading_overlay.place(relx=0, rely=0, relwidth=1, relheight=1)
        shell = tk.Frame(self.loading_overlay, bg="#030712")
        shell.place(relx=0.5, rely=0.47, anchor="center")
        self.loading_canvas = tk.Canvas(shell, width=280, height=220, bg="#030712", highlightthickness=0, bd=0)
        self.loading_canvas.pack()
        self.loading_canvas.create_oval(70, 10, 210, 150, fill="#e2e8f0", outline="")
        self.loading_canvas.create_oval(56, -4, 224, 164, outline="#0f172a", width=2)
        self.loading_canvas.create_polygon(
            145, 18,
            112, 86,
            144, 86,
            123, 146,
            176, 74,
            146, 74,
            165, 18,
            fill="#020617",
            outline="#020617",
        )
        self._loading_charge_nodes = [
            (155, 28),
            (146, 48),
            (136, 68),
            (128, 88),
            (139, 104),
            (133, 124),
            (126, 144),
        ]
        charge_path = [coord for pt in self._loading_charge_nodes for coord in pt]
        self.loading_canvas.create_line(*charge_path, fill="#1e3a8a", width=3, smooth=True)
        self._loading_charge_segment = self.loading_canvas.create_line(0, 0, 0, 0, fill="#93c5fd", width=7, smooth=True, capstyle=tk.ROUND)
        self._loading_charge_outer = self.loading_canvas.create_oval(0, 0, 0, 0, fill="#60a5fa", outline="")
        self._loading_charge_inner = self.loading_canvas.create_oval(0, 0, 0, 0, fill="#eff6ff", outline="")
        tk.Label(shell, textvariable=self.loading_title_var, font=self.header_font, fg="#f8fafc", bg="#030712").pack(pady=(2, 4))
        tk.Label(shell, textvariable=self.loading_status_var, font=self.big_font, fg="#93c5fd", bg="#030712").pack()
        tk.Label(shell, textvariable=self.loading_hint_var, font=self.small_font, fg="#cbd5e1", bg="#030712", wraplength=540, justify="center").pack(pady=(6, 0))
        self._loading_overlay_visible = True
        self._animate_loading_overlay()

    def _set_loading_stage(self, title: str, detail: str | None = None):
        heading = str(title or "Booting workspace").strip() or "Booting workspace"
        hint = str(detail or title or "Loading workspace").strip() or "Loading workspace"

        def _apply():
            try:
                if hasattr(self, "loading_status_var"):
                    self.loading_status_var.set(heading)
                if hasattr(self, "loading_hint_var"):
                    self.loading_hint_var.set(hint)
                if hasattr(self, "loading_overlay"):
                    self.loading_overlay.lift()
                self._loading_overlay_visible = True
            except Exception:
                pass

        try:
            if threading.current_thread() is threading.main_thread():
                _apply()
            else:
                self.root.after(0, _apply)
        except Exception:
            pass
        self._set_activity(hint, active=True, accent="#60a5fa")

    def _animate_loading_overlay(self):
        if not getattr(self, "_loading_overlay_visible", False) or not hasattr(self, "loading_canvas"):
            return
        nodes = list(getattr(self, "_loading_charge_nodes", []) or [])
        if not nodes:
            return
        idx = int(self._loading_anim_step % len(nodes))
        core, glow = _loading_charge_palette(self._loading_anim_step)
        segment_nodes = nodes[max(0, idx - 1): min(len(nodes), idx + 2)]
        if len(segment_nodes) >= 2:
            self.loading_canvas.coords(self._loading_charge_segment, *[coord for pt in segment_nodes for coord in pt])
        x, y = nodes[idx]
        self.loading_canvas.coords(self._loading_charge_outer, x - 10, y - 10, x + 10, y + 10)
        self.loading_canvas.coords(self._loading_charge_inner, x - 4, y - 4, x + 4, y + 4)
        self.loading_canvas.itemconfigure(self._loading_charge_segment, fill=glow)
        self.loading_canvas.itemconfigure(self._loading_charge_outer, fill=core)
        self.loading_canvas.itemconfigure(self._loading_charge_inner, fill="#eff6ff")
        self._loading_anim_step += 1
        try:
            self._loading_after_id = self.root.after(115, self._animate_loading_overlay)
        except Exception:
            self._loading_after_id = None

    def _finish_loading_overlay(self, message: str | None = None, accent: str = "#86efac"):
        if message:
            self._set_activity(message, active=False, accent=accent)

        def _apply():
            try:
                if self._loading_after_id is not None:
                    self.root.after_cancel(self._loading_after_id)
                    self._loading_after_id = None
            except Exception:
                self._loading_after_id = None
            self._loading_overlay_visible = False
            try:
                if hasattr(self, "loading_overlay"):
                    self.loading_overlay.place_forget()
            except Exception:
                pass

        try:
            self.root.after(220, _apply)
        except Exception:
            _apply()

    def _refresh_card_wraplengths(self, width: int | None = None):
        try:
            base_width = int(width or self.right_canvas.winfo_width() or 0)
        except Exception:
            base_width = 0
        wrap = max(360, base_width - 70)
        for widget in list(getattr(self, "_card_labels", [])):
            try:
                widget.configure(wraplength=wrap)
            except Exception:
                pass

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
            app_menu.add_command(label="Download Latest Update", command=self._download_latest_update)
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
            help_menu.add_command(label="Download Latest Update", command=self._download_latest_update)
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
                            f"{meta.get('message')}\n\nDownload the latest GitHub release now?",
                        ):
                            self._download_latest_update()
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

    def _download_latest_update(self):
        if getattr(self, "_update_download_in_flight", False):
            self.status("Update download is already running…")
            return
        self._update_download_in_flight = True
        self.status("Downloading latest update from GitHub…")

        def _job():
            try:
                meta = download_latest_release(force=False)
            except Exception as exc:
                meta = {"status": "error", "message": str(exc)}
            finally:
                self._update_download_in_flight = False

            def _apply():
                if str(meta.get("status") or "") == "downloaded":
                    path = str(meta.get("path") or "")
                    self.status(f"Downloaded update: {path}")
                    messagebox.showinfo(
                        "Update Downloaded",
                        f"{meta.get('message') or 'Update downloaded.'}\n\nInstall it after closing the app.",
                    )
                else:
                    self.status(f"Update download failed: {meta.get('message') or meta.get('status') or '-'}")
                    messagebox.showinfo("Update Download", meta.get("message") or "Update download failed.")

            try:
                self.root.after(0, _apply)
            except Exception:
                pass

        try:
            self.executor.submit(_job)
        except Exception:
            self._update_download_in_flight = False
            logger.debug("Update download scheduling failed", exc_info=True)

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
        self._invalidate_workspace_caches(self.instrument_var.get())
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
        win.geometry("640x380")
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
        ttk.Button(actions, text="Download Update", command=self._download_latest_update).pack(side="left", padx=(8, 0))
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
        ttk.Button(buttons, text="Download Update", command=self._download_latest_update).pack(side="left", padx=(8, 0))
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
            "5. Add stocks from Add Stock; include company and news aliases when you know them.\n"
            "6. Choose Full Autonomy or Strategy Autonomy, then set Auto Risk before enabling paper or live auto.\n"
            "7. Use Check for Updates or Download Update before internal rollout or release testing."
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
        self._set_loading_stage("Booting workspace", "Checking models and cached market data")
        while not self._stop_event.is_set():
            try:
                selected = self.instrument_var.get()
                self._set_loading_stage("Refreshing market context", f"Loading {selected} prices, news, and support data")
                bundle = self._refresh_symbol_runtime(selected, force=True, reason="startup_boot", load_models=True)
                models = dict(bundle.get("models") or {})
                mode = self.mode_var.get()
                needs_retrain = self._needs_retrain(models, require_horizons=(mode != "4min"))
                latest_df = self._trim_df_for_ui(bundle.get("df"))
                if needs_retrain:
                    self.status("Models missing or legacy. Click 'Retrain Models' to build them.")
                    if self._workspace_is_active(selected):
                        with self._lock:
                            self.models = models
                            self.df = latest_df
                    self._store_runtime_bundle(selected, models, latest_df)
                    remember_workspace_view(selected, mode=self.mode_var.get(), profile_name=self.profile_var.get(), status="train_needed", note="Models missing or legacy")
                    self.root.after(0, self._refresh_workspace_tabs)
                    self.root.after(0, self.run_prediction_and_refresh_ui)
                    self.root.after(0, self.refresh_overview_threaded)
                    self.root.after(0, self.refresh_accuracy_summary_threaded)
                    self._finish_loading_overlay("Workspace ready - models need retrain", accent="#fbbf24")
                    return
                if self._workspace_is_active(selected):
                    with self._lock:
                        self.models = models
                        self.df = latest_df
                self._store_runtime_bundle(selected, models, latest_df)
                remember_workspace_view(selected, mode=self.mode_var.get(), profile_name=self.profile_var.get(), status="ready", note="Workspace warm")
                self.root.after(0, self._refresh_workspace_tabs)
                self.root.after(0, self.run_prediction_and_refresh_ui)
                self.root.after(0, self.refresh_overview_threaded)
                self.root.after(0, self.refresh_accuracy_summary_threaded)
                self.status("Models loaded — predictions available")
                self._finish_loading_overlay("Workspace ready", accent="#86efac")
                return
            except Exception as e:
                logger.exception("Model init loop error")
                self.status(f"Model init error: {e}")
                self._set_loading_stage("Startup retry", f"{e} | retrying in 5 seconds")
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
            self.status(f"Running staged training pipeline for {selected}…")
            run_onboarding_job(
                selected,
                meta=INSTRUMENTS.get(self._symbol_key(selected), {}),
                train_models=True,
                progress_cb=lambda stage_key, stage_label, state: self._set_training_stage(
                    str((state or {}).get("summary") or stage_label or stage_key)
                ),
                data_hub=getattr(self, "data_hub", None),
            )
            bundle = self._refresh_symbol_runtime(selected, force=True, reason="retrain_pipeline_complete", load_models=True)
            models = dict(bundle.get("models") or {})
            latest_df = self._trim_df_for_ui(bundle.get("df"))
            if self._workspace_is_active(selected):
                with self._lock:
                    self.models = models
                    self.df = latest_df
            self._store_runtime_bundle(selected, models, latest_df)
            remember_workspace_view(selected, mode=self.mode_var.get(), profile_name=self.profile_var.get(), status="ready", note="Retrained via staged pipeline")
            self._invalidate_workspace_caches(selected)
            self.root.after(0, self._refresh_workspace_tabs)
            self.root.after(0, lambda: self.run_prediction_and_refresh_ui(reason="Refreshing dashboard after training…"))
            self.root.after(0, self.refresh_overview_threaded)
            self.root.after(0, self.refresh_accuracy_summary_threaded)
            self.status("Training complete — refreshing dashboard…")
        except Exception:
            logger.exception("Background training failed")
            self.status("Background training failed; see logs")
        finally:
            self._retrain_in_progress = False

    def reload_once_threaded(self):
        def _job():
            try:
                self.status("Reloading prices, news, options, and models…")
                selected = self.instrument_var.get()
                bundle = self._refresh_symbol_runtime(selected, force=True, reason="manual_reload", load_models=True)
                models = dict(bundle.get("models") or {})
                df = bundle.get("df")
                mode = self.mode_var.get()
                if self._needs_retrain(models, require_horizons=(mode != "4min")):
                    self.status("Models missing or legacy. Click 'Retrain Models' to build them.")
                latest_df = self._trim_df_for_ui(df)
                if self._workspace_is_active(selected):
                    with self._lock:
                        self.models = models
                        self.df = latest_df
                self._store_runtime_bundle(selected, models, latest_df)
                remember_workspace_view(selected, mode=self.mode_var.get(), profile_name=self.profile_var.get(), status=("train_needed" if self._needs_retrain(models, require_horizons=(mode != "4min")) else "ready"), note="Reloaded")
                self._invalidate_workspace_caches(selected)
                self.root.after(0, self._refresh_workspace_tabs)
                self.root.after(0, lambda: self.run_prediction_and_refresh_ui(reason="Refreshing dashboard after reload…"))
                self.root.after(0, self.refresh_overview_threaded)
                self.root.after(0, self.refresh_accuracy_summary_threaded)
                self.status("Reload complete — dashboard refreshed")
            except Exception:
                logger.exception("Reload failed")
                self.status("Reload failed; see logs")
        self.executor.submit(_job)

    def retrain_once(self):
        if self._retrain_in_progress:
            self.status("Retrain already running…")
            return
        self._retrain_in_progress = True
        selected = self.instrument_var.get()
        self.status(f"Starting staged retrain pipeline for {selected}…")
        self._queue_onboarding_job(
            selected,
            meta=INSTRUMENTS.get(self._symbol_key(selected), {}),
            train_models=True,
            started_note="Staged retrain pipeline started",
            completion_note="Retrained via staged pipeline",
            on_complete=lambda _success: setattr(self, "_retrain_in_progress", False),
        )

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
            "decision_rows": [],
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
            "autonomy_card": "Autonomy: unavailable",
            "autonomy_detail": "-",
            "quality_card": "Quality: unavailable",
            "quality_detail": "-",
            "vol_card": "Volatility: unavailable",
            "vol_detail": "-",
            "options_card": "Options: unavailable",
            "options_detail": "-",
            "news_card": "News: unavailable",
            "news_detail": "-",
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
            "autonomy_meta": {},
            "section_errors": {},
            "status_messages": []
        }

    def _populate_panel_placeholders(self, payload: dict, reason: str, symbol: str | None = None):
        detail = str(reason or "Loading…").strip() or "Loading…"
        generic = ("Status", "-", detail)
        payload["market_rows"] = [generic]
        payload["breadth_rows"] = [generic]
        payload["macro_rows"] = [generic]
        payload["signal_rows"] = [generic]
        payload["stack_rows"] = [generic]
        payload["probability_rows"] = [generic]
        payload["quality_rows"] = [generic]
        payload["liquidity_rows"] = [generic]
        payload["catalyst_rows"] = [generic]
        payload["alerts_rows"] = [generic]
        payload["execution_rows"] = [generic]
        payload["replay_rows"] = [generic]
        payload["portfolio_rows"] = [generic]
        payload["performance_rows"] = [generic]
        payload["opportunity_rows"] = [generic]
        payload["setup_rows"] = [generic]
        payload["decision_rows"] = []
        payload["scenario_rows"] = [generic]
        payload["invalidation_rows"] = [generic]
        payload["reliability_rows"] = [generic]
        payload["levels_rows"] = [generic]
        payload["options_rows"] = [generic]
        payload["news_rows"] = [("-", "info", detail)]
        payload["setup_detail"] = detail
        payload["action_detail"] = detail
        payload["autonomy_detail"] = detail
        payload["quality_detail"] = detail
        payload["vol_detail"] = detail
        payload["options_detail"] = detail
        payload["news_detail"] = detail
        current_symbol = symbol
        if current_symbol is None:
            try:
                current_symbol = self.instrument_var.get()
            except Exception:
                current_symbol = None
        sym = self._symbol_key(current_symbol)
        if sym:
            try:
                self._apply_lifecycle_placeholders(payload, sym, detail)
            except Exception:
                logger.debug("Lifecycle placeholder build failed", exc_info=True)

    def _build_waiting_payload(self, mode: str, msg: str, status: str | None = None, symbol: str | None = None) -> dict:
        payload = self._base_payload(mode)
        self._populate_panel_placeholders(payload, msg, symbol=symbol)
        payload["quality_card"] = "Quality: loading…"
        payload["vol_card"] = "Volatility: loading…"
        payload["options_card"] = "Options: loading…"
        payload["autonomy_card"] = "Autonomy: loading…"
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
        payload = self._build_waiting_payload(mode, "Prediction error", symbol=self.instrument_var.get())
        payload["quality_card"] = "Quality: error"
        payload["vol_card"] = "Volatility: error"
        payload["options_card"] = "Options: error"
        payload["news_card"] = "News: error"
        payload["autonomy_card"] = "Autonomy: error"
        payload["quality_detail"] = err
        payload["vol_detail"] = err
        payload["options_detail"] = err
        payload["news_detail"] = err
        payload["autonomy_detail"] = err
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
        self.autonomy_card_var.set(payload.get("autonomy_card", "Autonomy: unavailable"))
        self.autonomy_detail_var.set(payload.get("autonomy_detail", "-"))
        self.quality_card_var.set(payload.get("quality_card", "Quality: unavailable"))
        self.quality_detail_var.set(payload.get("quality_detail", "-"))
        self.vol_card_var.set(payload.get("vol_card", "Volatility: unavailable"))
        self.vol_detail_var.set(payload.get("vol_detail", "-"))
        self.options_card_var.set(payload.get("options_card", "Options: unavailable"))
        self.options_detail_var.set(payload.get("options_detail", "-"))
        self.news_card_var.set(payload.get("news_card", "News: unavailable"))
        self.news_detail_var.set(payload.get("news_detail", "-"))
        self.profile_card_var.set(payload.get("profile_card", "Profile: unavailable"))
        self.profile_detail_var.set(payload.get("profile_detail", "-"))
        self.broker_card_var.set(payload.get("broker_card", "Broker: unavailable"))
        self.broker_detail_var.set(payload.get("broker_detail", "-"))
        self.replay_card_var.set(payload.get("replay_card", "Replay: unavailable"))
        self.replay_detail_var.set(payload.get("replay_detail", "-"))
        self._latest_advisory_meta = dict(payload.get("advisory_meta") or {})
        self._latest_execution_meta = dict(payload.get("execution_meta") or {})
        self._latest_replay_meta = dict(payload.get("replay_meta") or {})
        self._latest_autonomy_meta = dict(payload.get("autonomy_meta") or {})
        try:
            current_label = str(payload.get("current_price", ""))
            self._latest_price = float(current_label.split("$", 1)[1].replace(",", "")) if "$" in current_label else None
        except Exception:
            self._latest_price = None
        self._sync_action_buttons()
        self._refresh_support_dialogs()
        activity_text, activity_busy, activity_accent = self._activity_state_from_payload(payload)
        selected = self._symbol_key(self.instrument_var.get())
        remember_workspace_view(
            selected,
            mode=self.mode_var.get(),
            profile_name=self.profile_var.get(),
            status=("busy" if activity_busy else "ready"),
            note=str(payload.get("action_card") or activity_text or "").strip(),
        )
        self._refresh_workspace_tabs()
        self._set_activity(activity_text, active=activity_busy, accent=activity_accent)
        if getattr(self, "_loading_overlay_visible", False):
            self._finish_loading_overlay(activity_text, accent=activity_accent)
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
            return self._build_waiting_payload(mode, "Models missing. Click 'Retrain Models'.", status="Models missing. Click 'Retrain Models'.", symbol=selected)

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
                return self._build_waiting_payload(mode, results.get("error"), status=results.get("error"), symbol=selected)
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
                return self._build_waiting_payload(mode, msg, status=msg, symbol=selected)
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
        dashboard_error = None
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
            dashboard_error = "Dashboard analytics failed; showing available data only"
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
                payload["decision_rows"] = dashboard.get("decision_rows", [])
                payload["scenario_rows"] = dashboard.get("scenario_rows", [])
                payload["invalidation_rows"] = dashboard.get("invalidation_rows", [])
                payload["reliability_rows"] = dashboard.get("reliability_rows", [])
                payload["levels_rows"] = dashboard.get("levels_rows", [])
                payload["options_rows"] = dashboard.get("options_rows", [])
                payload["news_rows"] = dashboard.get("news_rows", [])
                payload["advisory_meta"] = dashboard.get("advisory_meta", {})
                payload["execution_meta"] = dashboard.get("execution_meta", {})
                payload["replay_meta"] = dashboard.get("replay_meta", {})
                payload["section_errors"] = dashboard.get("section_errors", {})
                payload["options_detail"] = dashboard.get("options_detail", payload.get("options_detail", "-"))
                payload["news_detail"] = dashboard.get("news_detail", payload.get("news_detail", "-"))
                try:
                    autonomy_snapshot = evaluate_autonomy_cycle(
                        selected,
                        df_copy,
                        advisory=payload.get("advisory_meta"),
                        execution_snapshot=payload.get("execution_meta"),
                        profile_name=self.profile_var.get(),
                        replay_snapshot=replay_snapshot,
                        options_snapshot=dashboard.get("options_snapshot"),
                    )
                except Exception:
                    logger.exception("Autonomy controller evaluation failed")
                    autonomy_snapshot = get_autonomy_snapshot(selected, advisory=payload.get("advisory_meta"), execution_snapshot=payload.get("execution_meta"))
                    autonomy_snapshot["detail"] = "Autonomy controller error; controller left unchanged"
                payload["autonomy_card"] = autonomy_snapshot.get("card", payload["autonomy_card"])
                payload["autonomy_detail"] = autonomy_snapshot.get("detail", payload["autonomy_detail"])
                payload["autonomy_meta"] = autonomy_snapshot
                auto_rows = list(autonomy_snapshot.get("rows") or [])
                if auto_rows:
                    payload["execution_rows"] = auto_rows + list(payload.get("execution_rows") or [])
                exec_override = autonomy_snapshot.get("execution_override")
                if isinstance(exec_override, dict):
                    payload["execution_meta"] = exec_override
                    payload["execution_rows"] = auto_rows + list(exec_override.get("rows") or [])
                    payload["portfolio_rows"] = list(exec_override.get("portfolio_rows") or payload.get("portfolio_rows") or [])
            else:
                self._populate_panel_placeholders(payload, dashboard_error or "Dashboard context is still loading")
                payload["quality_card"] = "Quality: unavailable"
                payload["vol_card"] = "Volatility: unavailable"
                payload["options_card"] = "Options: unavailable"
                payload["news_card"] = "News: unavailable"
                payload["autonomy_card"] = "Autonomy: unavailable"
                if dashboard_error:
                    payload["status_messages"].append(dashboard_error)
        except Exception:
            logger.exception("Dashboard analytics build failed")
            self._populate_panel_placeholders(payload, "Dashboard context is still loading")
            payload["status_messages"].append("Dashboard analytics failed; fallback displayed")
        if not payload.get("autonomy_meta"):
            payload["autonomy_meta"] = get_autonomy_snapshot(selected)
            payload["autonomy_card"] = payload["autonomy_meta"].get("card", payload["autonomy_card"])
            payload["autonomy_detail"] = payload["autonomy_meta"].get("detail", payload["autonomy_detail"])
            auto_rows = list((payload["autonomy_meta"] or {}).get("rows") or [])
            if auto_rows:
                payload["execution_rows"] = auto_rows + list(payload.get("execution_rows") or [])
        payload["status_messages"].append("Predicted")
        return payload

    def run_prediction_and_refresh_ui(self, reason: str | None = None):
        mode = self.mode_var.get()
        df_copy, models_copy, selected, replay_snapshot = self._current_prediction_inputs()
        if reason:
            self.status(reason)
            self._set_activity(reason, active=True, accent="#60a5fa")

        if df_copy is None or getattr(df_copy, "empty", True) or not models_copy:
            payload = self._build_waiting_payload(mode, "Waiting…", symbol=selected)
            self._set_activity("Waiting for models and live data", active=True, accent="#60a5fa")
            self.root.after(0, lambda: self._apply_prediction_payload(payload))
            return

        cache_key = self._prediction_payload_cache_key(mode, selected, df_copy, replay_snapshot)

        with self._pred_lock:
            if self._pred_in_flight:
                self._pending_prediction_request = True
                if reason:
                    self._pending_prediction_reason = reason
                self._set_activity("Refresh queued while the current cycle finishes", active=True, accent="#93c5fd")
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
        self._set_activity(reason or f"Refreshing {selected} {mode} stack", active=True, accent="#60a5fa")

        def _job():
            pending = False
            pending_reason = None
            try:
                payload = self._compute_prediction_payload(df_copy, models_copy, mode, selected, replay_snapshot=replay_snapshot)
            except Exception as e:
                logger.exception("Prediction UI update failed")
                payload = self._build_error_payload(mode, str(e))
            finally:
                with self._pred_lock:
                    self._pred_in_flight = False
                    self._pred_started_at = None
                    pending = bool(self._pending_prediction_request)
                    pending_reason = self._pending_prediction_reason
                    self._pending_prediction_request = False
                    self._pending_prediction_reason = None
            self._prediction_payload_cache[cache_key] = dict(payload)
            try:
                def _apply_if_latest():
                    with self._pred_lock:
                        if seq != self._pred_seq:
                            return
                    self._apply_prediction_payload(payload)
                    if pending:
                        self.run_prediction_and_refresh_ui(reason=pending_reason or "Refreshing latest view…")
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
                    bundle = self._refresh_symbol_runtime(selected, force=False, reason="background_auto", load_models=True)
                    df = bundle.get("df")
                    models = dict(bundle.get("models") or {})
                    latest_df = self._trim_df_for_ui(df)
                    if self._workspace_is_active(selected):
                        with self._lock:
                            self.df = latest_df
                            self.models = models
                    self._store_runtime_bundle(selected, models, latest_df)
                    remember_workspace_view(selected, mode=self.mode_var.get(), profile_name=self.profile_var.get(), status=("train_needed" if self._needs_retrain(models, require_horizons=(self.mode_var.get() != "4min")) else "ready"), note="Auto-refreshed")
                    self.root.after(0, self._refresh_workspace_tabs)
                    self.root.after(0, self.run_prediction_and_refresh_ui)
                    if (time.time() - self._overview_last_refresh) >= float(self._overview_refresh_sec):
                        self.root.after(0, self.refresh_overview_threaded)
                except Exception:
                    logger.exception("Background update error")
                    self.status("Background update error; see logs")
            time.sleep(AUTO_PREDICT_SECONDS)

    def shutdown(self):
        self._stop_event.set()
        self._unbind_runtime_subscription()
        try:
            self.executor.shutdown(wait=False)
        except Exception:
            pass

def launch_gui():
    root = tk.Tk()
    app = LightningGUI(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.shutdown(), root.destroy()))
    root.mainloop()
