import sys
import unittest
from pathlib import Path

AGENT_FRAMEWORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_FRAMEWORK))

from AlphaCrafter.portfolio_contract import evaluate_trade


def _decision(edge_bps: float):
    return evaluate_trade(
        current_weights={"A": 0.5, "B": 0.5},
        proposed_target_weights={"A": 0.4, "B": 0.6},
        forecast_returns={"A": 0.0, "B": edge_bps / 1000.0},
        pre_trade_nav=1_000_000.0,
    )


class PortfolioContractTests(unittest.TestCase):
    def test_edge_at_or_below_three_bps_is_skipped(self):
        self.assertFalse(_decision(0.29).executed)
        self.assertFalse(_decision(0.30).executed)
        self.assertEqual(_decision(0.29).skip_reason, "gross_edge_not_above_migration_cost")
        self.assertEqual(_decision(0.30).skip_reason, "gross_edge_not_above_migration_cost")


    def test_edge_above_three_bps_executes_and_charges_one_way_cost(self):
        decision = _decision(0.31)
        self.assertTrue(decision.executed)
        self.assertAlmostEqual(decision.one_way_turnover, 0.1)
        self.assertAlmostEqual(decision.actual_cost, 30.0)


    def test_initial_allocation_is_unconditional(self):
        decision = evaluate_trade(
            current_weights={},
            proposed_target_weights={"A": 1.0},
            forecast_returns={},
            pre_trade_nav=1_000_000.0,
            initial_allocation=True,
        )
        self.assertTrue(decision.executed)
        self.assertEqual(decision.actual_cost, 0.0)
