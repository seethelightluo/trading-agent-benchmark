"""Leak-free FactorMiner forward portfolio runner.

FactorMiner remains responsible for its native job: mine a factor library on
the frozen historical window.  This module consumes that frozen library in a
strict expanding-window walk-forward.  At each decision date it recomputes the
factor signals using only data visible through that date and atomically
rebalances a fully-invested long-only portfolio.  The initial allocation is
free; later asset-to-asset transfers lose 3 bps of the amount moved before the
net amount becomes destination-asset units.  Cash is not a portfolio asset and
fractional units are supported.

No LLM call is made inside the daily mark-to-market loop.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def slice_panel(
    panel: Path,
    *,
    cutoff: str,
    tradable_ids: list[str],
    out: Path,
) -> Path:
    """Persist the exact no-future, tradable-only panel supplied to FM."""
    df = pd.read_parquet(panel) if panel.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(panel)
    date_col = "datetime" if "datetime" in df.columns else "date"
    dates = pd.to_datetime(df[date_col])
    keep = (dates <= pd.Timestamp(cutoff)) & df["asset_id"].astype(str).isin(tradable_ids)
    sliced = df.loc[keep].copy()
    if sliced.empty:
        raise ValueError(f"FM panel has no tradable rows through {cutoff}")
    sliced[date_col] = pd.to_datetime(sliced[date_col])
    if date_col != "datetime":
        sliced = sliced.rename(columns={date_col: "datetime"})
    sliced = sliced.sort_values(["datetime", "asset_id"]).reset_index(drop=True)
    if sliced["datetime"].max() > pd.Timestamp(cutoff):
        raise AssertionError("future row escaped FactorMiner cutoff")
    out.parent.mkdir(parents=True, exist_ok=True)
    sliced.to_parquet(out, index=False)
    return out


def _target_weights(
    visible: pd.DataFrame,
    library_path: Path,
    config_path: Path,
    max_factors: int,
    forecast_horizon: int,
) -> tuple[dict[str, float], list[int], dict[str, float]]:
    """Return target weights and a conservative, visible-data-only forecast.

    IC is a correlation rather than a return forecast, so it is converted into
    return units using the trailing cross-sectional volatility of observed
    ``forecast_horizon``-day returns.  This is deliberately simple and fully
    auditable; it exists to decide whether a proposed rebalance can plausibly
    clear transaction costs, not to manufacture an exact expected PnL.
    """
    from factorminer.core.library_io import load_library
    from factorminer.evaluation.runtime import evaluate_factors, load_runtime_dataset
    from factorminer.utils.config import load_config

    library = load_library(library_path)
    # F6: rank by predictive strength × consistency (IC × ICIR); a factor must
    # be both strong and stable to enter the active ensemble.
    factors = sorted(
        library.list_factors(),
        key=lambda factor: abs(float(factor.ic_mean))
        * abs(float(getattr(factor, "icir", 0.0) or 0.0)),
        reverse=True,
    )[:max_factors]
    if not factors:
        return {}, [], {}

    cfg = load_config(config_path=config_path)
    dataset = load_runtime_dataset(visible, cfg)
    artifacts = evaluate_factors(
        factors,
        dataset,
        signal_failure_policy="reject",
    )

    combined = np.zeros(len(dataset.asset_ids), dtype=np.float64)
    support_weight = np.zeros(len(dataset.asset_ids), dtype=np.float64)
    used: list[int] = []
    for factor, artifact in zip(factors, artifacts):
        if not artifact.succeeded or artifact.signals_full is None:
            continue
        values = np.asarray(artifact.signals_full[:, -1], dtype=np.float64)
        finite = np.isfinite(values)
        if finite.sum() < 2:
            continue
        ranked = pd.Series(values[finite]).rank(pct=True).to_numpy() - 0.5
        direction = 1.0 if float(factor.ic_mean) >= 0 else -1.0
        contribution = np.zeros_like(combined)
        contribution[finite] = direction * ranked
        ic_weight = max(abs(float(factor.ic_mean)), 1e-6)
        icir = abs(float(getattr(factor, "icir", 0.0) or 0.0))
        quality_weight = ic_weight * max(icir, 1e-6)
        combined += contribution * quality_weight
        support_weight[finite] += quality_weight
        used.append(int(factor.id))

    if not used:
        return {}, [], {}
    # A missing signal must not look like a neutral zero-valued signal. Average
    # each asset only over factors that produced a valid last-date observation.
    eligible = support_weight > 0
    combined[eligible] /= support_weight[eligible]
    combined[~eligible] = np.nan
    finite_idx = np.flatnonzero(np.isfinite(combined))
    if len(finite_idx) < 2:
        return {}, used, {}

    centered = combined[finite_idx] - np.nanmean(combined[finite_idx])
    score_std = float(np.nanstd(centered))
    if not np.isfinite(score_std) or score_std <= 1e-12:
        return {}, used, {str(dataset.asset_ids[idx]): 0.0 for idx in finite_idx}

    close_panel = visible.pivot(index="datetime", columns="asset_id", values="close")
    horizon_returns = close_panel.pct_change(periods=max(1, forecast_horizon), fill_method=None)
    cross_sectional_vol = horizon_returns.std(axis=1, ddof=0).replace([np.inf, -np.inf], np.nan)
    recent_vol = cross_sectional_vol.dropna().tail(252)
    return_scale = float(recent_vol.median()) if not recent_vol.empty else 0.01
    if not np.isfinite(return_scale) or return_scale <= 0:
        return_scale = 0.01
    mean_abs_ic = float(np.mean([abs(float(f.ic_mean)) for f in factors if int(f.id) in used]))
    forecasts = {
        str(dataset.asset_ids[idx]): float(
            ((combined[idx] - np.nanmean(combined[finite_idx])) / score_std)
            * mean_abs_ic
            * return_scale
        )
        for idx in finite_idx
    }

    # F6: full-universe long-only tilt around equal weight, scaled by the
    # standardized composite score z; clip negatives and renormalize so the
    # signal-covered assets stay fully invested with zero cash. κ tunes tilt
    # aggressiveness (κ=0 ⇒ equal weight, large κ ⇒ concentration).
    n = len(finite_idx)
    finite_list = [int(idx) for idx in finite_idx]
    z = (combined[finite_list] - np.nanmean(combined[finite_list])) / score_std
    kappa = 1.0
    raw = {idx: (1.0 + kappa * float(z[i])) / n for i, idx in enumerate(finite_list)}
    clipped = {idx: max(0.0, weight) for idx, weight in raw.items()}
    total = sum(clipped.values())
    if total <= 1e-12:
        equal = 1.0 / n
        return (
            {str(dataset.asset_ids[idx]): equal for idx in finite_list},
            used,
            forecasts,
        )
    return (
        {str(dataset.asset_ids[idx]): float(clipped[idx] / total) for idx in finite_list},
        used,
        forecasts,
    )


def _full_investment_weights(
    target: dict[str, float], tradable_ids: list[str]
) -> dict[str, float]:
    """Return a non-negative 15-asset vector summing exactly to one.

    An empty/invalid research signal falls back to equal weight instead of cash.
    """
    assets = [str(asset) for asset in tradable_ids]
    if not assets:
        raise ValueError("tradable universe cannot be empty")
    cleaned: dict[str, float] = {}
    for asset in assets:
        value = float(target.get(asset, 0.0))
        cleaned[asset] = value if np.isfinite(value) and value >= 0.0 else 0.0
    total = sum(cleaned.values())
    if total <= 1e-12:
        weight = 1.0 / len(assets)
        return {asset: weight for asset in assets}
    normalized = {asset: cleaned[asset] / total for asset in assets}
    normalized[assets[-1]] += 1.0 - sum(normalized.values())
    return normalized


def _asset_transfer_rebalance(
    current_values: dict[str, float],
    target_weights: dict[str, float],
    pre_trade_nav: float,
    *,
    cost_bps: float,
    initial_allocation: bool,
) -> tuple[dict[str, float], float, float, float]:
    """Solve a no-cash asset transfer with friction charged once on outflow."""
    if pre_trade_nav <= 0:
        raise ValueError("portfolio NAV must be positive before rebalance")
    if initial_allocation:
        post_trade_nav = pre_trade_nav
        transferred = pre_trade_nav
        cost = 0.0
    else:
        rate = float(cost_bps) / 10_000.0
        post_trade_nav = pre_trade_nav
        for _ in range(100):
            target_values = {
                asset: post_trade_nav * weight
                for asset, weight in target_weights.items()
            }
            transferred = sum(
                max(float(current_values.get(asset, 0.0)) - target_value, 0.0)
                for asset, target_value in target_values.items()
            )
            cost = transferred * rate
            updated_nav = pre_trade_nav - cost
            if abs(updated_nav - post_trade_nav) <= max(1e-9, pre_trade_nav * 1e-12):
                post_trade_nav = updated_nav
                break
            post_trade_nav = updated_nav
        else:
            raise RuntimeError("asset-transfer cost fixed point did not converge")
    target_values = {
        asset: post_trade_nav * weight for asset, weight in target_weights.items()
    }
    turnover = 0.0 if initial_allocation else transferred / pre_trade_nav
    return target_values, transferred, cost, turnover


def run_forward(
    panel: Path,
    *,
    library_path: Path,
    config_path: Path,
    output_dir: Path,
    tradable_ids: list[str],
    history_end: str,
    baseline_date: str,
    online_end: str,
    cadence: int = 10,
    initial_capital: float = 1_000_000.0,
    cost_bps: float = 3.0,
    initial_allocation_cost_bps: float = 0.0,
    max_factors: int = 10,
) -> dict:
    """Run or resume the deterministic FM portfolio leg."""
    if cadence <= 0:
        raise ValueError("cadence must be positive")
    raw = pd.read_parquet(panel) if panel.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(panel)
    date_col = "datetime" if "datetime" in raw.columns else "date"
    raw = raw[raw["asset_id"].astype(str).isin(tradable_ids)].copy()
    raw["datetime"] = pd.to_datetime(raw[date_col])
    required = ["datetime", "asset_id", "open", "high", "low", "close", "volume", "amount"]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise ValueError(f"FM panel is missing required columns: {missing}")
    raw = raw[required].sort_values(["datetime", "asset_id"]).reset_index(drop=True)
    if raw.duplicated(["datetime", "asset_id"]).any():
        raise ValueError("FM panel contains duplicate (datetime, asset_id) rows")
    closes = raw.pivot(index="datetime", columns="asset_id", values="close").sort_index()
    opens = raw.pivot(index="datetime", columns="asset_id", values="open").sort_index()
    closes = closes.ffill()
    # A missing execution-day open must never be filled with that day's close:
    # the close is not known when the order is priced.  Previous close is a
    # conservative, already-visible fallback; any remaining leading gaps stay
    # unavailable and therefore cannot be traded.
    opens = opens.reindex(closes.index).fillna(closes.shift(1)).ffill()
    forward_days = [
        day for day in closes.index
        if pd.Timestamp(baseline_date) <= day <= pd.Timestamp(online_end)
    ]
    if not forward_days:
        raise ValueError("FM panel has no forward trading days")

    state_path = output_dir / "forward_state.json"
    equity_path = output_dir / "equity.csv"
    contract = {
        "history_end": history_end,
        "baseline_date": baseline_date,
        "cadence": int(cadence),
        "initial_capital": float(initial_capital),
        "cost_bps": float(cost_bps),
        "initial_allocation_cost_bps": float(initial_allocation_cost_bps),
        "cash_weight": 0.0,
        "fractional_positions": True,
        "rebalance_model": "atomic_asset_transfer_v1",
        "max_factors": int(max_factors),
        "tradable_ids": sorted(str(asset) for asset in tradable_ids),
    }
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("state_version") != 4:
            raise ValueError(
                f"incompatible FM forward state at {state_path}; "
                "use a new output directory so the old result remains preserved"
            )
        saved_contract = state.get("contract")
        if saved_contract is not None and saved_contract != contract:
            raise ValueError(
                f"FM forward contract changed at {state_path}; use a new output directory"
            )
        state["state_version"] = 4
        state["schema_version"] = 1
        state["contract"] = contract
        state["initial_capital"] = float(initial_capital)
        _atomic_json(state_path, state)
    else:
        state = {
            "state_version": 4,
            "schema_version": 1,
            "contract": contract,
            "initial_capital": float(initial_capital),
            "nav": float(initial_capital),
            "cash": float(initial_capital),
            "shares": {},
            "portfolio_initialized": False,
            "cumulative_transaction_cost": 0.0,
            "last_processed_date": None,
            "last_visible_date": history_end,
            "decisions": [],
        }

    existing_rows = pd.read_csv(equity_path).to_dict("records") if equity_path.exists() else []
    completed = state.get("last_processed_date")
    start_idx = 0
    if completed:
        completed_ts = pd.Timestamp(completed)
        matching_indices = [
            i for i, day in enumerate(forward_days) if day == completed_ts
        ]
        if not matching_indices:
            if completed_ts > forward_days[-1]:
                return state
            raise ValueError(
                f"persisted FM date {completed} is absent from the current forward panel"
            )
        start_idx = matching_indices[0] + 1
    if start_idx >= len(forward_days):
        return state

    shares = {str(k): float(v) for k, v in state.get("shares", {}).items()}
    cash = float(state.get("cash", initial_capital))
    nav = float(state.get("nav", initial_capital))
    active_decision_date = str(state.get("active_decision_date") or history_end)

    # Cadence is anchored to the first forward trading day.  Using the absolute
    # index (rather than restarting blocks at ``start_idx``) makes a power-loss
    # resume continue the existing holding block without an early rebalance.
    for idx in range(start_idx, len(forward_days)):
        day = forward_days[idx]
        turnover = 0.0
        cost = 0.0
        executed = False
        skip_reason = "not_a_decision_day"

        open_values = {
            asset: quantity * float(opens.at[day, asset])
            for asset, quantity in shares.items()
            if asset in opens.columns and np.isfinite(opens.at[day, asset])
        }
        pre_trade_nav = cash + sum(open_values.values())
        if idx % cadence == 0:
            active_decision_date = (
                history_end if idx == 0 else forward_days[idx - 1].strftime("%Y-%m-%d")
            )
            visible = raw[raw["datetime"] <= pd.Timestamp(active_decision_date)].copy()
            if visible["datetime"].max() > pd.Timestamp(active_decision_date):
                raise AssertionError("FM decision received future data")
            raw_target, factor_ids, forecasts = _target_weights(
                visible, library_path, config_path, max_factors, cadence
            )
            target = _full_investment_weights(raw_target, tradable_ids)

            current_weights = {
                asset: open_values.get(asset, 0.0) / pre_trade_nav
                for asset in tradable_ids
            }
            initial_allocation = not bool(state.get("portfolio_initialized"))
            target_delta = sum(
                abs(target[asset] - current_weights.get(asset, 0.0))
                for asset in tradable_ids
            )

            do_execute = True
            if not initial_allocation:
                if target_delta <= 1e-10:
                    do_execute = False
                    skip_reason = "target_unchanged"
                elif forecasts and any(abs(float(v)) > 0 for v in forecasts.values()):
                    # Cost dead-band (user-confirmed): rebalance only when the
                    # expected one-period gain clears the 3 bps migration cost.
                    # Requires per-asset forecasts; without them we cannot
                    # evaluate the dead-band and fall through to execute.
                    # Both sides are fractions of NAV, so directly comparable.
                    migration = sum(
                        max(current_weights[asset] - target[asset], 0.0)
                        for asset in tradable_ids
                    )
                    cost_frac = (cost_bps / 10_000.0) * migration
                    expected_gain = sum(
                        (target[asset] - current_weights.get(asset, 0.0))
                        * forecasts.get(asset, 0.0)
                        for asset in tradable_ids
                    )
                    if expected_gain <= cost_frac:
                        do_execute = False
                        skip_reason = (
                            f"below_cost_deadband(gain={expected_gain:.3e},"
                            f"cost={cost_frac:.3e})"
                        )
            if do_execute:
                executed = True
                skip_reason = ""
                target_values, transferred, cost, turnover = _asset_transfer_rebalance(
                    open_values,
                    target,
                    pre_trade_nav,
                    cost_bps=(initial_allocation_cost_bps if initial_allocation else cost_bps),
                    initial_allocation=initial_allocation,
                )
                shares = {
                    asset: target_values[asset] / float(opens.at[day, asset])
                    for asset in tradable_ids
                    if target[asset] > 0.0
                    and asset in opens.columns
                    and np.isfinite(opens.at[day, asset])
                    and float(opens.at[day, asset]) > 0.0
                }
                expected_assets = {
                    asset for asset, weight in target.items() if weight > 0.0
                }
                if set(shares) != expected_assets:
                    raise ValueError(
                        "cannot fully invest because an execution price is unavailable"
                    )
                cash = 0.0
                state["portfolio_initialized"] = True
                state["cumulative_transaction_cost"] = float(
                    state.get("cumulative_transaction_cost", 0.0)
                ) + cost

            state.setdefault("decisions", []).append({
                "decision_date": active_decision_date,
                "execution_date": day.strftime("%Y-%m-%d"),
                "factor_ids": factor_ids,
                "target_weights": target,
                "current_weights": current_weights,
                "forecast_returns": forecasts,
                "turnover": turnover,
                "transferred_notional": (
                    pre_trade_nav if initial_allocation else turnover * pre_trade_nav
                ),
                "cost": cost,
                "initial_allocation": initial_allocation,
                "executed": executed,
                "skip_reason": skip_reason,
                "pre_trade_nav": pre_trade_nav,
            })

        close_values = {
            asset: quantity * float(closes.at[day, asset])
            for asset, quantity in shares.items()
            if asset in closes.columns and np.isfinite(closes.at[day, asset])
        }
        nav = cash + sum(close_values.values())
        row = {
            "date": day.strftime("%Y-%m-%d"),
            "visible_through_at_decision": active_decision_date,
            "nav": nav,
            "cash": cash,
            "market_value": sum(close_values.values()),
            "n_positions": len(shares),
            "turnover_at_rebalance": turnover,
            "cost_at_rebalance": cost,
            "rebalance_executed": executed,
            "skip_reason": skip_reason,
        }
        existing_rows.append(row)
        state.update({
            "nav": nav,
            "cash": cash,
            "shares": shares,
            "last_processed_date": row["date"],
            "last_visible_date": row["date"],
            "active_decision_date": active_decision_date,
        })
        output_dir.mkdir(parents=True, exist_ok=True)
        equity = pd.DataFrame(existing_rows).drop_duplicates("date", keep="last")
        _atomic_csv(equity_path, equity)
        _atomic_json(state_path, state)

    return state
