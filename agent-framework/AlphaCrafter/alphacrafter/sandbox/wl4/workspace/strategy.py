import json
import numpy as np
from alphacrafter.sim.utils import register_hook, get_stock_daily_data, rebalance_to_weights

UNIVERSE = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
with open("factors/factor_ensemble.json", encoding="utf-8") as f:
    _raw = json.load(f).get("selected_factors", [])[:10]
ENSEMBLE = [(str(x["factor_id"]), float(x["weight"]), int(x.get("direction", 1))) for x in _raw]
_day = 0

def cs_rank(values):
    a = np.array([values.get(s, np.nan) for s in UNIVERSE], float)
    good = a[np.isfinite(a)]
    a[~np.isfinite(a)] = np.median(good) if len(good) else 0.0
    order = np.argsort(np.argsort(a, kind="mergesort"), kind="mergesort")
    return {s: float(order[i] / 14.0 - 0.5) for i, s in enumerate(UNIVERSE)}

def sd(x, n):
    return max(float(np.std(x[-n:])), 1e-8)

def factor_signal(fid, r, panel):
    if len(r) < 130:
        return 0.0
    v10, v20, v40, v60 = sd(r, 10), sd(r, 20), sd(r, 40), sd(r, 60)
    if "defensive_volatility_quality" in fid:
        # Own-volatility level and improvement, matching the persisted expression.
        history = [sd(r[:i], 20) for i in range(20, len(r) + 1, 10)]
        baseline = max(float(np.median(history[-12:])), 1e-8)
        improvement = v20 / max(sd(r[:-10], 20), 1e-8) - 1.0
        return -(v20 / baseline) - 0.5 * improvement
    if "volatility_compression" in fid:
        return -(v10 / v40 - 1.0)
    if "relative_lowvolatility" in fid:
        peer = [sd(x, 20) for x in panel.values() if len(x) >= 20]
        return -np.log(max(v20 / max(float(np.median(peer)), 1e-8), 1e-8))
    if "consistency_volscaled_momentum20" in fid:
        return float(np.sum(r[-20:]) / (v20 * np.sqrt(20))) * (0.5 + float(np.mean(r[-20:] > 0)))
    if "trend_persistence" in fid:
        persistence = 2.0 * float(np.mean(r[-30:] > 0)) - 1.0
        return float(np.sum(r[-60:]) / (v60 * np.sqrt(60))) * persistence
    if "dispersion_conditioned_reversal" in fid:
        cross = np.array([np.sum(x[-20:]) for x in panel.values() if len(x) >= 20])
        dispersion = float(np.std(cross)) if len(cross) else 0.0
        return -float(np.sum(r[-20:])) / (v40 * np.sqrt(40)) * np.clip(1.0 + dispersion / max(float(np.std(cross[-10:] if len(cross) >= 10 else cross)), 1e-8), .25, 2.5)
    return 0.0

@register_hook
def strategy():
    global _day
    _day += 1
    if _day < 10 or not ENSEMBLE:
        return
    closes = {}
    for symbol in UNIVERSE:
        df = get_stock_daily_data(symbol=symbol, days=420)
        if df is not None and len(df) >= 180:
            x = np.asarray(df.sort_values("date")["close"], float)
            if len(x) and np.all(np.isfinite(x)) and np.all(x > 0):
                closes[symbol] = x
    if len(closes) < 10:
        _day = 0
        return
    n = min(len(x) for x in closes.values())
    returns = {s: np.diff(np.log(x[-n:])) for s, x in closes.items()}
    scores = {s: 0.0 for s in UNIVERSE}
    for fid, weight, direction in ENSEMBLE:
        raw = {s: factor_signal(fid, r[:-1], returns) for s, r in returns.items()}
        ranked = cs_rank(raw)
        for s in UNIVERSE:
            scores[s] += weight * direction * ranked[s]
    # High-risk sideways/bearish regime: full investment, but explicitly favor tradable defensives.
    breadth = float(np.mean([np.mean(r[-30:] > 0) for r in returns.values()]))
    if breadth < 0.46:
        for s, bonus in (("XAU", .12), ("US10Y", .08), ("CN10Y", .06)):
            scores[s] += bonus
        for s in ("WTI", "BTC", "ETH"):
            scores[s] -= .035
    z = np.array([scores[s] for s in UNIVERSE], float)
    z = np.clip((z - z.mean()) / max(float(z.std()), 1e-8), -1.4, 1.4)
    p = np.exp(0.30 * z); p /= p.sum()
    target = {s: float(.55 * p[i] + .45 / len(UNIVERSE)) for i, s in enumerate(UNIVERSE)}
    forecast = {s: float(.0025 * z[i]) for i, s in enumerate(UNIVERSE)}
    rebalance_to_weights(target, forecast_returns=forecast,
                         factor_ids=[x[0] for x in ENSEMBLE], horizon_days=10)
    _day = 0
