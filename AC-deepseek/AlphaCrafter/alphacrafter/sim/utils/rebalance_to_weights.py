"""Atomic fractional portfolio rebalance for the 15-asset benchmark."""

from __future__ import annotations

import math
import os
from pathlib import Path

import pandas as pd

from alphacrafter.utils.atomic_io import atomic_write_json, load_json


def _execution_price(dataset_dir: Path, symbol: str, current_date: str) -> float:
    path = dataset_dir / f"{symbol}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing execution data for {symbol}: {path}")
    frame = pd.read_csv(path)
    required = {"date", "open", "close"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Execution data for {symbol} lacks date/open/close")
    frame["date"] = pd.to_datetime(frame["date"])
    day = pd.Timestamp(current_date)
    exact = frame.loc[frame["date"] == day]
    if not exact.empty:
        value = float(exact.iloc[-1]["open"])
        if math.isfinite(value) and value > 0:
            return value
    previous = frame.loc[frame["date"] < day].sort_values("date")
    if previous.empty:
        raise ValueError(f"No execution price is available for {symbol} on {current_date}")
    value = float(previous.iloc[-1]["close"])
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"Invalid execution price for {symbol} on {current_date}")
    return value


def _solve_transfer(
    current_values: dict[str, float],
    weights: dict[str, float],
    pre_trade_nav: float,
    cost_bps: float,
) -> tuple[dict[str, float], float, float]:
    post_trade_nav = pre_trade_nav
    rate = float(cost_bps) / 10_000.0
    for _ in range(100):
        targets = {asset: post_trade_nav * weight for asset, weight in weights.items()}
        transferred = sum(
            max(current_values.get(asset, 0.0) - target, 0.0)
            for asset, target in targets.items()
        )
        cost = transferred * rate
        updated_nav = pre_trade_nav - cost
        if abs(updated_nav - post_trade_nav) <= max(1e-9, pre_trade_nav * 1e-12):
            return (
                {asset: updated_nav * weight for asset, weight in weights.items()},
                transferred,
                cost,
            )
        post_trade_nav = updated_nav
    raise RuntimeError("asset-transfer cost fixed point did not converge")


def rebalance_to_weights(
    target_weights: dict[str, float],
    account_file_path: str = "../persistent/account.json",
    date_file_path: str = "../persistent/date.json",
    dataset_dir_path: str = "../persistent/stock_data",
    cost_bps: float | None = None,
) -> dict:
    """Rebuild the portfolio at target weights using fractional asset units.

    The first allocation is free. Later rebalances charge 3 bps once on the
    notional transferred out of overweight assets. Cash is zero afterwards.
    """
    account_path = Path(account_file_path)
    account = load_json(account_path)
    date_state = load_json(date_file_path)
    current_date = str(date_state["current_date"])
    assets = [str(asset) for asset in account.get("watch_list", [])]
    if len(assets) != 15 or len(set(assets)) != 15:
        raise ValueError("benchmark rebalance requires exactly 15 unique tradable assets")
    if set(target_weights) != set(assets):
        missing = sorted(set(assets) - set(target_weights))
        extra = sorted(set(target_weights) - set(assets))
        raise ValueError(
            f"target_weights must cover all 15 assets; missing={missing}, extra={extra}"
        )
    weights = {asset: float(target_weights[asset]) for asset in assets}
    if any(not math.isfinite(weight) or weight < 0.0 for weight in weights.values()):
        raise ValueError("target weights must be finite and non-negative")
    total_weight = sum(weights.values())
    if abs(total_weight - 1.0) > 1e-6:
        raise ValueError(f"target weights must sum to 1, got {total_weight}")
    weights[assets[-1]] += 1.0 - total_weight

    dataset_dir = Path(dataset_dir_path)
    prices = {asset: _execution_price(dataset_dir, asset, current_date) for asset in assets}
    current_values = {asset: 0.0 for asset in assets}
    for position in account.get("positions", []):
        symbol = str(position["symbol"])
        if symbol not in current_values:
            raise ValueError(f"non-benchmark position cannot be rebalanced: {symbol}")
        current_values[symbol] += float(position["quantity"]) * prices[symbol]
    pre_trade_nav = float(account.get("available_cash", 0.0)) + sum(current_values.values())
    if pre_trade_nav <= 0:
        raise ValueError("portfolio NAV must be positive")

    initial_allocation = not bool(account.get("portfolio_initialized", False))
    applied_cost_bps = 0.0 if initial_allocation else float(
        cost_bps if cost_bps is not None else os.environ.get("AC_REBALANCE_COST_BPS", "3")
    )
    if initial_allocation:
        target_values = {asset: pre_trade_nav * weight for asset, weight in weights.items()}
        transferred = pre_trade_nav
        cost = 0.0
    else:
        target_values, transferred, cost = _solve_transfer(
            current_values, weights, pre_trade_nav, applied_cost_bps
        )
    post_trade_nav = pre_trade_nav - cost
    positions = []
    for asset in assets:
        target_value = target_values[asset]
        if target_value <= 1e-12:
            continue
        quantity = target_value / prices[asset]
        positions.append({
            "symbol": asset,
            "direction": "LONG",
            "quantity": quantity,
            "available_quantity": quantity,
            "cost_price": prices[asset],
            "current_price": prices[asset],
            "market_value": target_value,
            "profit_loss": 0.0,
            "profit_loss_rate": 0.0,
        })

    initial_capital = float(account.get("initial_capital", pre_trade_nav))
    account.update({
        "total_assets": post_trade_nav,
        "net_assets": post_trade_nav,
        "available_cash": 0.0,
        "market_value": post_trade_nav,
        "total_profit_loss": post_trade_nav - initial_capital,
        "total_profit_loss_rate": post_trade_nav / initial_capital - 1.0,
        "gross_position_rate": 1.0,
        "net_position_rate": 1.0,
        "positions": positions,
        "orders": [],
        "portfolio_initialized": True,
        "last_rebalance_date": current_date,
        "last_target_weights": weights,
        "cumulative_transaction_cost": float(
            account.get("cumulative_transaction_cost", 0.0)
        ) + cost,
    })
    record = {
        "date": current_date,
        "initial_allocation": initial_allocation,
        "pre_trade_nav": pre_trade_nav,
        "post_trade_nav": post_trade_nav,
        "transferred_notional": transferred,
        "cost_bps": applied_cost_bps,
        "cost": cost,
        "target_weights": weights,
    }
    account.setdefault("rebalance_history", []).append(record)
    atomic_write_json(account_path, account)
    return record


def ensure_fully_invested(
    account_file_path: str = "../persistent/account.json",
    date_file_path: str = "../persistent/date.json",
    dataset_dir_path: str = "../persistent/stock_data",
) -> dict | None:
    """Self-heal the benchmark invariant after a strategy hook."""
    account_path = Path(account_file_path)
    account = load_json(account_path)
    assets = [str(asset) for asset in account.get("watch_list", [])]
    if len(assets) != 15:
        raise ValueError("benchmark rebalance requires exactly 15 tradable assets")

    orders = account.get("orders", [])
    pending = [order for order in orders if str(order.get("status")) == "PENDING"]
    if pending:
        account["orders"] = [
            order for order in orders if str(order.get("status")) != "PENDING"
        ]
        atomic_write_json(account_path, account)

    def _rebalance(target_weights: dict[str, float]) -> dict:
        return rebalance_to_weights(
            target_weights,
            account_file_path=account_file_path,
            date_file_path=date_file_path,
            dataset_dir_path=dataset_dir_path,
        )

    if not account.get("portfolio_initialized", False):
        equal = 1.0 / len(assets)
        return _rebalance({asset: equal for asset in assets})

    last = account.get("last_target_weights") or {}
    fallback = (
        {asset: float(last[asset]) for asset in assets}
        if all(asset in last for asset in assets)
        else {asset: 1.0 / len(assets) for asset in assets}
    )
    nav = float(account.get("net_assets", 0.0))
    cash = float(account.get("available_cash", 0.0))
    if abs(cash) > max(1e-6, abs(nav) * 1e-9) or not account.get("positions"):
        return _rebalance(fallback)
    return None
