#run_backtest.py
from data_fetch import update_price_data
from model_trainers import walk_forward_validate
try:
    from config import DEFAULT_INSTRUMENT
except Exception:
    DEFAULT_INSTRUMENT = None

if __name__ == "__main__":
    df = update_price_data(DEFAULT_INSTRUMENT)
    metrics = walk_forward_validate(df, instrument_key=DEFAULT_INSTRUMENT)
    print("Walk-forward folds:", metrics.get("folds", 0))
    if metrics.get("folds", 0) > 0:
        import statistics
        print("Base MAE:", statistics.mean(metrics.get("mae_base", [0])))
        print("Forest MAE:", statistics.mean(metrics.get("mae_forest", [0])))
        print("Boost MAE:", statistics.mean(metrics.get("mae_boost", [0])))
        try:
            print("Extra Trees MAE:", statistics.mean(metrics.get("mae_extra_trees", [0])))
        except Exception:
            pass
        try:
            print("Directional Acc:", statistics.mean([x for x in metrics.get("directional_acc", []) if x == x]))
        except Exception:
            pass
        try:
            print("Sharpe-like:", statistics.mean([x for x in metrics.get("sharpe_like", []) if x == x]))
        except Exception:
            pass
        try:
            print("Net Return:", statistics.mean(metrics.get("net_returns", [0])))
        except Exception:
            pass
        try:
            print("Active Fraction:", statistics.mean([x for x in metrics.get("active_fraction", []) if x == x]))
        except Exception:
            pass
