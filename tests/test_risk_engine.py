import unittest

import pandas as pd

import risk_engine as re


def _df_from_close(values):
    idx = pd.date_range("2026-04-01 14:30:00+00:00", periods=len(values), freq="2min")
    closes = [float(v) for v in values]
    return pd.DataFrame(
        {
            "open": closes,
            "high": [v + 0.1 for v in closes],
            "low": [v - 0.1 for v in closes],
            "close": closes,
            "volume": [1000.0] * len(closes),
        },
        index=idx,
    )


def _advisory(direction: str = "UP") -> dict:
    return {
        "quality_meta": {"score": 0.72},
        "plan_meta": {"entry": 100.0, "target1": 101.5},
        "pred_state": {"intraday": {"direction": direction, "position_size": 0.55}},
        "news_meta": {"headline_state": "ok", "alignment": "neutral"},
    }


class RiskEngineTests(unittest.TestCase):
    def test_regime_detector_returns_unknown_when_evidence_is_insufficient(self):
        df = _df_from_close([100.0 + 0.02 * i for i in range(8)])
        out = re.classify_market_regime(df)
        self.assertEqual(out["label"], "unknown_or_insufficient_evidence")
        self.assertTrue(bool(out["unknown"]))

    def test_regime_detector_distinguishes_high_vs_low_volatility_on_synthetic_data(self):
        low_vol = _df_from_close([100.0 + 0.01 * i for i in range(40)])
        high_vol = _df_from_close(([100.0, 102.5, 100.0, 97.5, 100.0] * 8))

        low_out = re.classify_market_regime(low_vol)
        high_out = re.classify_market_regime(high_vol)

        self.assertEqual(low_out["label"], "low_volatility")
        self.assertEqual(high_out["label"], "high_volatility")

    def test_risk_engine_blocks_trade_when_required_inputs_missing(self):
        df = _df_from_close([100.0 + 0.01 * i for i in range(40)])
        out = re.risk_summary_for_symbol("QQQ", df=df, advisory=_advisory(), require_portfolio=True)
        self.assertTrue(out["blocked"])
        self.assertEqual(out["primary_blocker"], "portfolio snapshot is unavailable")

    def test_risk_engine_computes_smaller_size_for_higher_volatility(self):
        low_vol = _df_from_close([100.0 + 0.01 * i for i in range(40)])
        high_vol = _df_from_close(([100.0, 102.5, 100.0, 97.5, 100.0] * 8))

        low_out = re.risk_summary_for_symbol("QQQ", df=low_vol, advisory=_advisory(), require_portfolio=False)
        high_out = re.risk_summary_for_symbol("QQQ", df=high_vol, advisory=_advisory(), require_portfolio=False)

        self.assertFalse(low_out["blocked"])
        self.assertFalse(high_out["blocked"])
        self.assertGreater(low_out["suggested_qty"], high_out["suggested_qty"])

    def test_risk_engine_respects_per_trade_risk_budget(self):
        out = re.compute_risk_budget(
            equity=100000.0,
            risk_fraction=0.002,
            quality_score=0.60,
            position_size_signal=1.0,
        )
        self.assertTrue(out["allowed"])
        self.assertEqual(out["risk_budget"], 200.0)

    def test_risk_engine_flags_symbol_or_portfolio_exposure_breach(self):
        out = re.portfolio_exposure_check(
            portfolio={
                "equity": 100000.0,
                "gross_exposure": 20000.0,
                "net_exposure": 5000.0,
                "positions": [{"symbol": "QQQ", "notional": 10000.0}],
            },
            symbol="QQQ",
            proposed_notional=5000.0,
            side="buy",
        )
        self.assertFalse(out["allowed"])
        self.assertIn("symbol concentration cap breached", out["reasons"])

    def test_risk_gate_guidance_interprets_budget_and_regime_blockers(self):
        budget = re.risk_gate_guidance({"blocked": True, "primary_blocker": "risk budget is too small"})
        regime = re.risk_gate_guidance({"blocked": True, "primary_blocker": "regime is unknown or evidence is insufficient"})
        portfolio = re.risk_gate_guidance({"blocked": True, "primary_blocker": "portfolio snapshot is unavailable"})
        self.assertEqual(budget["key"], "risk_budget_blocked")
        self.assertIn("safer stop distance", budget["next_requirement"])
        self.assertEqual(regime["key"], "unknown_regime")
        self.assertIn("classify the current regime safely", regime["next_requirement"])
        self.assertEqual(portfolio["key"], "portfolio_unavailable")
        self.assertIn("portfolio-aware autonomy cannot be proven", portfolio["detail"].lower())

    def test_risk_engine_uses_execution_recommendation_qty_for_exposure_check(self):
        df = _df_from_close([100.0 + 0.01 * i for i in range(40)])
        out = re.risk_summary_for_symbol(
            "QQQ",
            df=df,
            advisory=_advisory(),
            execution_snapshot={
                "recommendation": {"allowed": True, "qty": 1},
                "portfolio": {
                    "equity": 100000.0,
                    "gross_exposure": 49800.0,
                    "net_exposure": 0.0,
                    "positions": [],
                },
            },
            require_portfolio=True,
        )
        self.assertFalse(out["blocked"])
        self.assertEqual(out["suggested_qty"], 1)
        self.assertEqual(out["requested_qty"], 1)
        self.assertEqual(out["quantity_source"], "execution_recommendation")


if __name__ == "__main__":
    unittest.main()
