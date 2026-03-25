from __future__ import annotations

import argparse

from execution_engine import get_broker_verification_snapshot, refresh_broker_connection, run_broker_verification


def _print_rows(title: str, rows: list[tuple[str, str, str]] | None):
    print(f"\n{title}")
    print("-" * len(title))
    for item, value, detail in list(rows or []):
        print(f"{item}: {value}")
        if detail and detail != "-":
            print(f"  {detail}")


def main():
    parser = argparse.ArgumentParser(description="Run a local broker acceptance check for the trading workstation.")
    parser.add_argument("--symbol", default="QQQ", help="Symbol to use for broker snapshot and optional verification.")
    parser.add_argument("--verify", action="store_true", help="Attempt the Alpaca paper verification flow if connected.")
    parser.add_argument("--price", type=float, default=None, help="Optional reference price for the verification order.")
    parser.add_argument("--profile", default="balanced", help="Trader profile name for sizing if verification runs.")
    args = parser.parse_args()

    symbol = str(args.symbol or "QQQ").upper()
    diagnostics = refresh_broker_connection(symbol)
    print(f"Symbol: {symbol}")
    print(f"Requested mode: {diagnostics.get('requested_label')}")
    print(f"Effective route: {diagnostics.get('route_label')}")
    print(f"Status: {diagnostics.get('status')}")
    print(f"Connected: {diagnostics.get('connected')}")
    print(f"Checked at: {diagnostics.get('checked_at')}")
    if diagnostics.get("operator_hint"):
        print(f"Hint: {diagnostics.get('operator_hint')}")
    _print_rows("Broker Diagnostics", diagnostics.get("rows"))

    if args.verify:
        result = run_broker_verification(
            symbol,
            current_price=args.price,
            advisory=None,
            profile_name=args.profile,
        )
        print(f"\nVerification status: {result.get('status')}")
        if result.get("detail"):
            print(f"Verification detail: {result.get('detail')}")
        _print_rows("Verification Rows", result.get("rows"))
    else:
        latest = get_broker_verification_snapshot(symbol)
        if latest:
            print(f"\nLatest verification: {latest.get('status')}")
            if latest.get("detail"):
                print(f"Latest verification detail: {latest.get('detail')}")


if __name__ == "__main__":
    main()
