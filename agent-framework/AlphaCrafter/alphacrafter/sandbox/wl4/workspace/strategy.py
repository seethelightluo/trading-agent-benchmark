import json
import numpy as np
from alphacrafter.sim.utils import (
    register_hook, get_stock_daily_data, get_index_daily_data,
    rebalance_to_weights,
)

UNIVERSE = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
with open("factors/factor_ensemble.json", encoding="utf-8") as f:
    _selected = json.load(f).get("selected_factors", [])[:10]
ENSEMBLE = [(x["factor_id"], float(x["weight"]), int(x["direction"])) for x in _selected]
_day = 0


def _vol(r, n):
    return max(float(np.std(r[-n:])), 1e-8)


def _rank(values):
    a = np.array([values.get(s, np.nan) for s in UNIVERSE], dtype=float)
    good = a[np.isfinite(a)]
    a[~np.isfinite(a)] = float(np.median(good)) if len(good) else 0.0
    ranks = np.argsort(np.argsort(a, kind="mergesort"), kind="mergesort")
    return {s: float(ranks[i] / (len(UNIVERSE) - 1) - 0.5) for i, s in enumerate(UNIVERSE)}


def _signal(fid, r, market, typical):
    v10, v20 = _vol(r, 10), _vol(r, 20)
    if "defensive_volatility_quality" in fid:
        downside = np.sqrt(np.mean(np.minimum(r[-40:], 0.0) ** 2))
        return -(v20 + 0.25 * downside) / typical
    if "relative_lowvolatility" in fid:
        return -v20 / typical
    if "volatility_compression" in fid:
        return -v10 / _vol(r[:-10], 10) - 0.15 * v20 / typical
    if "market_residual_reversal" in fid:
        beta = np.cov(r[-60:], market[-60:])[0, 1] / max(np.var(market[-60:]), 1e-8)
        return -(np.sum(r[-20:]) - beta * np.sum(market[-20:])) / v20
    if "drawdown_convex_reversal" in fid:
        path = np.exp(np.cumsum(r[-60:]))
        dd = path / np.maximum.accumulate(path) - 1.0
        return -float(dd[-1]) / v20
    if "consistency_volscaled_momentum" in fid:
        q = r[-20:]
        return np.sum(q) / v20 * (0.5 + np.mean(q > 0))
    return 0.0


@register_hook
def strategy():
    global _day
    _day += 1
    if _day < 10 or not ENSEMBLE:
        return

    closes = {}
    for symbol in UNIVERSE:
        df = get_stock_daily_data(symbol=symbol, days=360)
        if df is None or len(df) < 130:
            continue
        x = np.asarray(df.sort_values("date")["close"], dtype=float)
        if np.all(np.isfinite(x)) and np.all(x > 0):
            closes[symbol] = x
    if len(closes) < 10:
        _day = 0
        return

    n = min(len(x) for x in closes.values())
    returns = {s: np.diff(np.log(x[-n:])) for s, x in closes.items()}
    market = np.mean(np.asarray(list(returns.values())), axis=0)
    typical = max(float(np.median([_vol(r, 20) for r in returns.values()])), 1e-8)
    score = {s: 0.0 for s in UNIVERSE}
    for fid, weight, direction in ENSEMBLE:
        raw = {s: float(np.clip(_signal(fid, r, market, typical), -10, 10)) for s, r in returns.items()}
        ranked = _rank(raw)
        for s in UNIVERSE:
            score[s] += weight * direction * ranked[s]

    # Bullish regime stays diversified; only add defensive tradables when
    # breadth or the observation-only volatility signal confirms stress.
    breadth = float(np.mean([np.mean(r[-20:] > 0) for r in returns.values()]))
    stressed = breadth < 0.45
    vix = get_index_daily_data(symbol="VIX", days=30)
    if vix is not None and len(vix) >= 21:
        vx = np.asarray(vix.sort_values("date")["close"], dtype=float)
        stressed = stressed or (np.isfinite(vx[-1]) and vx[-1] > 1.10 * np.median(vx[-21:]))
    if stressed:
        for symbol, bonus in (("XAU", 0.12), ("US10Y", 0.07), ("CN10Y", 0.05)):
            score[symbol] += bonus
        for symbol in ("WTI", "BTC", "ETH"):
            score[symbol] -= 0.035

    z = np.array([score[s] for s in UNIVERSE], dtype=float)
    z = np.clip((z - z.mean()) / max(z.std(), 1e-8), -1.4, 1.4)
    p = np.exp(0.30 * z)
    p /= p.sum()
    target = {s: float(0.55 * p[i] + 0.45 / len(UNIVERSE)) for i, s in enumerate(UNIVERSE)}
    # Complete deterministic 10-day forecast used by the migration-cost gate.
    forecast = {s: float(0.0025 * z[i]) for i, s in enumerate(UNIVERSE)}
    rebalance_to_weights(
        target, forecast_returns=forecast,
        factor_ids=[x[0] for x in ENSEMBLE], horizon_days=10,
    )
    _day = 0
