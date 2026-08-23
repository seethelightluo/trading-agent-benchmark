import json
from pathlib import Path
import numpy as np
from alphacrafter.sim.utils import (
    register_hook, get_stock_daily_data, get_index_daily_data,
    rebalance_to_weights,
)

UNIVERSE = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
ROOT = Path(__file__).parent


def load_ensemble():
    try:
        selected = json.loads((ROOT / "factors" / "factor_ensemble.json").read_text()).get("selected_factors", [])
        return selected if 0 < len(selected) <= 10 else []
    except Exception:
        return []


def factor_value(fid, prices):
    # Signals use only completed observations supplied by the simulator.
    p = prices[:-1]
    r = p[1:] / p[:-1] - 1.0
    def ret(n): return float(p[-1] / p[-n - 1] - 1.0)
    def vol(n): return max(float(np.std(r[-n:])), 0.008)
    v20, v40 = vol(20), vol(40)
    r20, r30, r40 = ret(20), ret(30), ret(40)
    neg = r[-40:][r[-40:] < 0]
    downside = max(float(np.std(neg)) if len(neg) >= 5 else v40, 0.008)
    formulas = {
        "volatility_premium_20d": v20,
        "miner_3_20310515_risk_scaled_relative_deceleration_20d": (r40 - 0.5 * r20) / v40,
        "breadth_persistence_quality_40d": r40 / v40,
        "breadth_conditioned_calm_reversal_10d": -r20 / v20,
        "miner_1_20350426_volatility_shock_reversal_40d": -r40 / v40,
        "miner_2_downside_adjusted_momentum_30d": r30 / downside,
    }
    return formulas.get(fid, 0.0)


def rank_normalize(values):
    result = {s: 0.5 for s in UNIVERSE}
    valid = sorted((float(v), s) for s, v in values.items() if np.isfinite(v))
    if not valid:
        return result
    for i, (_, symbol) in enumerate(valid):
        result[symbol] = (i + 1.0) / len(valid)
    return result


def bounded_full_investment(scores):
    raw = np.array([max(float(scores[s]), 1e-8) for s in UNIVERSE])
    raw /= raw.sum()
    # 3.5%-12.5% bounds are suitable for this 15-asset cross-section.
    weights = raw
    for _ in range(20):
        clipped = np.clip(weights, 0.035, 0.125)
        clipped /= clipped.sum()
        if np.max(np.abs(clipped - weights)) < 1e-12:
            break
        weights = clipped
    weights = np.clip(weights, 0.035, 0.125)
    weights /= weights.sum()
    return {s: float(weights[i]) for i, s in enumerate(UNIVERSE)}


@register_hook
def strategy():
    factors = load_ensemble()
    if not factors:
        return
    prices = {}
    for symbol in UNIVERSE:
        try:
            data = get_stock_daily_data(symbol=symbol, days=260)
            if data is None:
                continue
            p = np.asarray(data.sort_values("date")["close"], dtype=float)
            if len(p) >= 130 and np.all(np.isfinite(p)) and np.all(p > 0):
                prices[symbol] = p
        except Exception:
            continue
    if len(prices) < 12:
        return

    scores = {s: 0.5 for s in UNIVERSE}
    for factor in factors:
        fid = str(factor["factor_id"])
        direction = float(factor.get("direction", 1))
        values = {s: direction * factor_value(fid, p) for s, p in prices.items()}
        ranks = rank_normalize(values)
        for s in UNIVERSE:
            scores[s] += float(factor["weight"]) * (ranks[s] - 0.5)

    # Observable VIX and cross-sectional breadth adjust posture, while all
    # 15 tradable assets remain invested (macro signals are never ordered).
    breadth = float(np.mean([factor_value("breadth_persistence_quality_40d", p) > 0 for p in prices.values()]))
    stressed = False
    try:
        vix = get_index_daily_data(symbol="VIX", days=160)
        v = np.asarray(vix.sort_values("date")["close"], dtype=float)
        stressed = len(v) >= 120 and v[-1] > 1.05 * np.median(v[-120:])
    except Exception:
        pass
    if stressed or breadth < 0.40:
        for s in ("XAU", "US10Y", "CN10Y"):
            scores[s] *= 1.30
        for s in ("BTC", "ETH", "SOX", "NDX", "WTI", "COPPER"):
            scores[s] *= 0.75
    elif breadth > 0.65:
        for s in ("SPX", "N225", "SX5E", "000300.SH"):
            scores[s] *= 1.08

    target = bounded_full_investment(scores)
    z = (np.array([scores[s] for s in UNIVERSE]) - np.mean(list(scores.values()))) / max(np.std(list(scores.values())), 1e-9)
    forecast = {s: float(0.01 * z[i]) for i, s in enumerate(UNIVERSE)}
    rebalance_to_weights(
        target, forecast_returns=forecast,
        factor_ids=[str(f["factor_id"]) for f in factors], horizon_days=10,
    )
