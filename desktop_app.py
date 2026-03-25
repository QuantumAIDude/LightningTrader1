from __future__ import annotations

from app_paths import bootstrap_runtime_environment


def main() -> int:
    bootstrap_runtime_environment()
    from stock_gui import launch_gui

    launch_gui()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
