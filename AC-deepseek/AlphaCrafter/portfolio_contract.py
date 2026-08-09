"""Shared deterministic portfolio proposal and transaction-decision contract.

This module is deliberately dependency-free so FM and both AC copies can use
the same calculations in live code and in offline fixtures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


DEFAULT_DECISION_EDGE_BPS = 3.0
DEFAULT_COST_BPS = 3.0
# A new online run must not resume an account produced under the abandoned
# pre-proposal rebalance semantics.  Warmup accounts are uninitialized and do
# not carry this marker, so the guard does not touch either shared warmup.
PORTFOLIO_CONTRACT_VERSION = "ac-worldline-v2-migration-gate"


def assert_current_portfolio_contract(account: Mapping[str, object]) -> None:
    """Fail closed for initialized accounts from an abandoned online run."""
    if not bool(account.get("portfolio_initialized", False)):
        return
    actual = account.get("portfolio_contract_version")
    if actual != PORTFOLIO_CONTRACT_VERSION:
        raise RuntimeError(
            "stale AC online account contract: "
            f"expected {PORTFOLIO_CONTRACT_VERSION}, got {actual!r}; "
            "do not resume the abandoned online stage"
        )


def normalize_weights(weights: Mapping[str, float], assets: list[str]) -> dict[str, float]:
    """Return a complete, non-negative 15-asset weight vector."""
    expected = {str(asset) for asset in assets}
    actual = {str(asset): float(value) for asset, value in weights.items()}
    unknown = set(actual) - expected
    if unknown:
        raise ValueError(f"target contains unknown assets: {sorted(unknown)}")
    complete = {asset: max(0.0, actual.get(asset, 0.0)) for asset in expected}
    total = sum(complete.values())
    if total <= 0.0:
        raise ValueError("target weights must have positive total")
    return {asset: value / total for asset, value in sorted(complete.items())}


def one_way_turnover(current_weights: Mapping[str, float], target_weights: Mapping[str, float]) -> float:
    """Return migrated notional as a fraction of NAV, not bilateral L1."""
    assets = set(current_weights) | set(target_weights)
    return 0.5 * sum(
        abs(float(target_weights.get(asset, 0.0)) - float(current_weights.get(asset, 0.0)))
        for asset in assets
    )


def gross_edge_bps(
    current_weights: Mapping[str, float],
    target_weights: Mapping[str, float],
    forecast_returns: Mapping[str, float],
) -> float:
    """Return forecast incremental return in basis points."""
    assets = set(current_weights) | set(target_weights) | set(forecast_returns)
    return 10_000.0 * sum(
        (float(target_weights.get(asset, 0.0)) - float(current_weights.get(asset, 0.0)))
        * float(forecast_returns.get(asset, 0.0))
        for asset in assets
    )


@dataclass(frozen=True)
class TradeDecision:
    current_weights: dict[str, float]
    proposed_target_weights: dict[str, float]
    executed_target_weights: dict[str, float]
    forecast_returns: dict[str, float]
    factor_ids: list[str]
    horizon_days: int
    one_way_turnover: float
    gross_edge_bps: float
    decision_edge_threshold_bps: float
    actual_cost: float
    executed: bool
    skip_reason: str

    def as_dict(self) -> dict:
        return {
            "current_weights": self.current_weights,
            "proposed_target_weights": self.proposed_target_weights,
            # Keep the old name as a compatibility alias for consumers that
            # have not migrated, while making the executed target explicit.
            "target_weights": self.proposed_target_weights,
            "executed_target_weights": self.executed_target_weights,
            "forecast_returns": self.forecast_returns,
            "factor_ids": self.factor_ids,
            "horizon_days": self.horizon_days,
            "one_way_turnover": self.one_way_turnover,
            "turnover": self.one_way_turnover,
            "gross_edge_bps": self.gross_edge_bps,
            "predicted_incremental_edge_bps": self.gross_edge_bps,
            "decision_edge_threshold_bps": self.decision_edge_threshold_bps,
            "required_edge_bps": self.decision_edge_threshold_bps,
            "actual_cost": self.actual_cost,
            "cost": self.actual_cost,
            "portfolio_contract_version": PORTFOLIO_CONTRACT_VERSION,
            "executed": self.executed,
            "skip_reason": self.skip_reason,
        }


def evaluate_trade(
    *,
    current_weights: Mapping[str, float],
    proposed_target_weights: Mapping[str, float],
    forecast_returns: Mapping[str, float],
    pre_trade_nav: float,
    factor_ids: list[str] | None = None,
    horizon_days: int = 10,
    initial_allocation: bool = False,
    force_execute: bool = False,
    cost_bps: float = DEFAULT_COST_BPS,
) -> TradeDecision:
    """Evaluate one proposal without mutating account state.

    Initial allocation is the only unconditional execution.  All later
    proposals require gross edge strictly above the 3bp cost of the migrated
    notional.  The threshold is therefore ``one_way_turnover * cost_bps``;
    it is not a fixed 3bp of total NAV.
    """
    proposed = dict(proposed_target_weights)
    current = dict(current_weights)
    forecasts = {str(k): float(v) for k, v in forecast_returns.items()}
    turnover = one_way_turnover(current, proposed)
    edge = gross_edge_bps(current, proposed, forecasts)
    required_edge_bps = float(cost_bps) * turnover
    execute = bool(force_execute or initial_allocation or (turnover > 1e-12 and edge > required_edge_bps))
    if turnover <= 1e-12:
        reason = "target_unchanged"
    elif force_execute:
        reason = "account_repair"
    elif initial_allocation:
        reason = "initial_allocation"
    elif edge <= required_edge_bps:
        reason = "gross_edge_not_above_migration_cost"
    else:
        reason = ""
    actual_cost = float(pre_trade_nav) * turnover * float(cost_bps) / 10_000.0 if execute and not initial_allocation else 0.0
    executed_target = proposed if execute else current
    return TradeDecision(
        current_weights={str(k): float(v) for k, v in sorted(current.items())},
        proposed_target_weights={str(k): float(v) for k, v in sorted(proposed.items())},
        executed_target_weights={str(k): float(v) for k, v in sorted(executed_target.items())},
        forecast_returns={str(k): float(v) for k, v in sorted(forecasts.items())},
        factor_ids=[str(value) for value in (factor_ids or [])],
        horizon_days=int(horizon_days),
        one_way_turnover=float(turnover),
        gross_edge_bps=float(edge),
        decision_edge_threshold_bps=float(required_edge_bps),
        actual_cost=float(actual_cost),
        executed=execute,
        skip_reason=reason,
    )
