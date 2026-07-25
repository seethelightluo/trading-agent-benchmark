"""Leak-free FactorMiner forward portfolio runner.

FactorMiner remains responsible for its native job: mine a factor library on
the frozen historical window.  This module consumes that frozen library in a
strict expanding-window walk-forward.  At each decision date it recomputes the
factor signals using only data visible through that date, rebalances a long-only
top-quintile portfolio, charges one-way transaction cost, and then locally marks
the holdings for the next ``cadence`` trading days.

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
    factors = sorted(
        library.list_factors(),
        key=lambda factor: abs(float(factor.ic_mean)),
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
        combined += contribution * ic_weight
        support_weight[finite] += ic_weight
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

    count = max(1, math.ceil(len(finite_idx) * 0.20))
    ranked = finite_idx[np.argsort(combined[finite_idx])[-count:]]
    selected = [idx for idx in ranked if forecasts[str(dataset.asset_ids[idx])] > 0]
    if not selected:
        return {}, used, forecasts
    weight = 1.0 / len(selected)
    return (
        {str(dataset.asset_ids[idx]): weight for idx in selected},
        used,
        forecasts,
    )


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
    initial_capital: float = 100_000_000.0,
    cost_bps: float = 3.0,
    min_round_trip_edge_bps: float = 6.0,
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
        "min_round_trip_edge_bps": float(min_round_trip_edge_bps),
        "max_factors": int(max_factors),
        "tradable_ids": sorted(str(asset) for asset in tradable_ids),
    }
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("state_version") not in (2, 3):
            raise ValueError(
                f"incompatible FM forward state at {state_path}; "
                "use a new output directory so the old result remains preserved"
            )
        saved_contract = state.get("contract")
        if saved_contract is not None and saved_contract != contract:
            raise ValueError(
                f"FM forward contract changed at {state_path}; use a new output directory"
            )
        state["state_version"] = 3
        state["schema_version"] = 1
        state["contract"] = contract
        state["initial_capital"] = float(initial_capital)
        _atomic_json(state_path, state)
    else:
        state = {
            "state_version": 3,
            "schema_version": 1,
            "contract": contract,
            "initial_capital": float(initial_capital),
            "nav": float(initial_capital),
            "cash": float(initial_capital),
            "shares": {},
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
            target, factor_ids, forecasts = _target_weights(
                visible, library_path, config_path, max_factors, cadence
            )

            current_weights = {
                asset: value / pre_trade_nav
                for asset, value in open_values.items()
                if pre_trade_nav > 0
            }
            all_assets = set(current_weights) | set(target)
            turnover = sum(
                abs(target.get(asset, 0.0) - current_weights.get(asset, 0.0))
                for asset in all_assets
            )
            predicted_edge = sum(
                (target.get(asset, 0.0) - current_weights.get(asset, 0.0))
                * forecasts.get(asset, 0.0)
                for asset in all_assets
            )
            predicted_edge_bps = predicted_edge * 10_000.0
            estimated_cost_bps = cost_bps * turnover
            required_edge_bps = max(min_round_trip_edge_bps, estimated_cost_bps)

            if turnover <= 1e-10:
                skip_reason = "target_unchanged"
            elif predicted_edge_bps <= required_edge_bps:
                skip_reason = "predicted_edge_not_above_cost_threshold"
            else:
                executed = True
                skip_reason = ""
                cost = pre_trade_nav * (cost_bps / 10_000.0) * turnover
                investable = max(pre_trade_nav - cost, 0.0)
                shares = {
                    asset: (investable * weight) / float(opens.at[day, asset])
                    for asset, weight in target.items()
                    if asset in opens.columns
                    and np.isfinite(opens.at[day, asset])
                    and opens.at[day, asset] > 0
                }
                invested_weight = sum(
                    target[asset] for asset in shares if asset in target
                )
                cash = investable * max(0.0, 1.0 - invested_weight)

            state.setdefault("decisions", []).append({
                "decision_date": active_decision_date,
                "execution_date": day.strftime("%Y-%m-%d"),
                "factor_ids": factor_ids,
                "target_weights": target,
                "current_weights": current_weights,
                "forecast_returns": forecasts,
                "turnover": turnover,
                "cost": cost,
                "predicted_incremental_edge_bps": predicted_edge_bps,
                "estimated_one_way_cost_bps": estimated_cost_bps,
                "required_edge_bps": required_edge_bps,
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
