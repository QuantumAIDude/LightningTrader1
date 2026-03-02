#!/usr/bin/env python3
"""
deployment_gate.py
Hard deployment gate for LightningTrader4000.

Exits 0 only when all selected instruments pass operational checks.
"""

import argparse
import json
import os
from datetime import datetime, timezone

from config import INSTRUMENTS, DEFAULT_INSTRUMENT, MODEL_DIR
from system_checkup import _check_instrument


def _ts_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _model_dir_size_bytes() -> int:
    total = 0
    try:
        for name in os.listdir(MODEL_DIR):
            if not str(name).endswith(".pkl"):
                continue
            p = os.path.join(MODEL_DIR, name)
            if os.path.isfile(p):
                total += int(os.path.getsize(p))
    except Exception:
        return 0
    return total


def _parse_instruments(raw: str) -> list[str]:
    val = str(raw or "").strip()
    if not val or val.lower() == "all":
        return list(INSTRUMENTS.keys())
    return [s.strip() for s in val.split(",") if s.strip()]


def main():
    parser = argparse.ArgumentParser(description="Deployment gate for LightningTrader4000")
    parser.add_argument(
        "--instrument",
        default="all",
        help="Comma-separated instruments or 'all' (default).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip daily/weekend checks (faster).",
    )
    parser.add_argument(
        "--online",
        action="store_true",
        help="Use live fetch during checks.",
    )
    parser.add_argument(
        "--allow-warn",
        action="store_true",
        help="Allow warning status to pass the gate.",
    )
    parser.add_argument(
        "--max-model-dir-gb",
        type=float,
        default=6.0,
        help="Fail gate if models directory exceeds this size (GB).",
    )
    parser.add_argument(
        "--max-4min-latency-s",
        type=float,
        default=20.0,
        help="Fail gate if 4min prediction latency exceeds this threshold.",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Optional JSON report path.",
    )
    args = parser.parse_args()

    instruments = _parse_instruments(args.instrument)
    out = {
        "ts": _ts_now(),
        "status": "ok",
        "limits": {
            "max_model_dir_gb": float(args.max_model_dir_gb),
            "max_4min_latency_s": float(args.max_4min_latency_s),
            "allow_warn": bool(args.allow_warn),
        },
        "instruments": {},
        "global": {},
    }

    # Global size gate
    model_dir_bytes = _model_dir_size_bytes()
    model_dir_gb = model_dir_bytes / (1024.0 ** 3)
    out["global"]["model_dir_gb"] = round(float(model_dir_gb), 3)
    if model_dir_gb > float(args.max_model_dir_gb):
        out["status"] = "fail"
        out["global"]["model_dir_gate"] = (
            f"FAIL: model dir {model_dir_gb:.2f}GB > limit {args.max_model_dir_gb:.2f}GB"
        )
    else:
        out["global"]["model_dir_gate"] = "pass"

    for instr in instruments:
        if instr not in INSTRUMENTS:
            out["instruments"][instr] = {"status": "fail", "gate_failures": ["Instrument not defined"]}
            out["status"] = "fail"
            continue

        rep = _check_instrument(instr, quick=args.quick, online=args.online)
        gate_failures = []

        status = str(rep.get("status", "fail")).lower()
        if status == "fail":
            gate_failures.append("System check status is fail")
        elif status == "warn" and not args.allow_warn:
            gate_failures.append("System check status is warn (not allowed)")

        pred = rep.get("predictions", {}).get("4min", {})
        if isinstance(pred, dict) and pred.get("error"):
            gate_failures.append(f"4min prediction error: {pred.get('error')}")

        latency = rep.get("predictions", {}).get("4min_latency_s")
        try:
            if latency is not None and float(latency) > float(args.max_4min_latency_s):
                gate_failures.append(
                    f"4min latency {float(latency):.3f}s exceeds {float(args.max_4min_latency_s):.3f}s"
                )
        except Exception:
            pass

        missing = rep.get("models", {}).get("missing", [])
        if missing:
            gate_failures.append(f"Missing models: {missing}")
        try:
            diag = rep.get("predictions", {}).get("4min", {}).get("diagnostics", {})
            critical_stale = bool(diag.get("critical_stale", False))
            if critical_stale:
                gate_failures.append("Critical stale-data no-trade guard triggered")
            div = diag.get("diversity")
            if isinstance(div, dict) and bool(div.get("degenerate", False)) and not critical_stale:
                gate_failures.append("Degenerate 4min model spread (near-identical outputs)")
        except Exception:
            pass

        rep["gate_failures"] = gate_failures
        out["instruments"][instr] = rep
        if gate_failures:
            out["status"] = "fail"

    if args.report:
        with open(args.report, "w") as f:
            json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    raise SystemExit(0 if out["status"] == "ok" else 2)


if __name__ == "__main__":
    main()
