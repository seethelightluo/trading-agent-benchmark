import numpy as np
from alphacrafter.sim.utils import register_hook, get_stock_daily_data, rebalance_to_weights

UNIVERSE = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
# Screener ensemble: CLV, peer lead-lag, 5d reversal, risk-adjusted 20d momentum.
FACTORS = {"clv": 0.3395, "peer": 0.2612, "reversal": 0.2012, "momentum": 0.1981}
REBALANCE_EVERY = 10
MIN_WEIGHT, MAX_WEIGHT = 0.02, 0.14
_calls = 0

def ranks(x):
    valid = sorted((s, v) for s, v in x.items() if np.isfinite(v))
    out = {s: 0.5 for s in UNIVERSE}
    n = len(valid)
    for i, (s, _) in enumerate(valid):
        out[s] = (i + 1.0) / n if n > 1 else 0.5
    return out

def project_box(raw):
    # Iterative box-constrained simplex projection; fractional weights are intended.
    w = {s: max(float(raw.get(s, 1.0)), 1e-12) for s in UNIVERSE}
    fixed = {}
    free = set(UNIVERSE)
    for _ in range(30):
        remaining = 1.0 - sum(fixed.values())
        scale = remaining / max(sum(w[s] for s in free), 1e-12)
        clipped = []
        for s in list(free):
            q = w[s] * scale
            if q < MIN_WEIGHT:
                fixed[s] = MIN_WEIGHT; clipped.append(s)
            elif q > MAX_WEIGHT:
                fixed[s] = MAX_WEIGHT; clipped.append(s)
            else:
                w[s] = q
        if not clipped:
            break
        free -= set(clipped)
    if free:
        left = 1.0 - sum(fixed.values())
        z = sum(w[s] for s in free)
        for s in free:
            w[s] = left * w[s] / max(z, 1e-12)
    w.update(fixed)
    z = sum(w.values())
    return {s: w[s] / z for s in UNIVERSE}

@register_hook
def cross_asset_strategy():
    global _calls
    _calls += 1
    if _calls > 1 and (_calls - 1) % REBALANCE_EVERY != 0:
        return
    returns, clv, reversal, momentum, volatility = {}, {}, {}, {}, {}
    for symbol in UNIVERSE:
        df = get_stock_daily_data(symbol=symbol, days=90)
        if df is None or len(df) < 26:
            continue
        df = df.sort_values("date").reset_index(drop=True)
        close = np.asarray(df["close"], dtype=float)
        high = np.asarray(df["high"], dtype=float)
        low = np.asarray(df["low"], dtype=float)
        if not np.all(np.isfinite(close[-26:])) or close[-1] <= 0:
            continue
        daily = close[1:] / np.maximum(close[:-1], 1e-12) - 1.0
        sigma = max(float(np.std(daily[-20:])), 0.008)
        returns[symbol] = close[-1] / max(close[-21], 1e-12) - 1.0
        clv[symbol] = (close[-1] - low[-1]) / max(high[-1] - low[-1], close[-1] * 1e-8)
        reversal[symbol] = -float(np.mean(daily[-5:]))
        volatility[symbol] = sigma
        momentum[symbol] = returns[symbol] / (sigma + 0.01)
    if len(returns) < 8:
        return
    peer = {s: returns[s] - np.median([returns[t] for t in returns if t != s]) for s in returns}
    rr = {"clv": ranks(clv), "peer": ranks(peer), "reversal": ranks(reversal), "momentum": ranks(momentum)}
    score = {s: sum(FACTORS[k] * rr[k][s] for k in FACTORS) for s in returns}
    breadth = float(np.mean([v > 0 for v in returns.values()]))
    cyclical = {"000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "COPPER", "WTI", "BTC", "ETH"}
    defensive = {"XAU", "US10Y", "CN10Y"}
    tilt = {s: 1.0 for s in UNIVERSE}
    if breadth < 0.40:
        for s in cyclical: tilt[s] = 0.82
        for s in defensive: tilt[s] = 1.55
    elif breadth < 0.55:
        for s in cyclical: tilt[s] = 0.93
        for s in defensive: tilt[s] = 1.20
    inv_mean = np.mean([1.0 / v for v in volatility.values()])
    raw = {}
    for s in UNIVERSE:
        base = max(score.get(s, 0.5), 0.05)
        iv = (1.0 / volatility[s]) / max(inv_mean, 1e-12) if s in volatility else 1.0
        raw[s] = base * tilt[s] * (0.85 + 0.15 * iv)
    rebalance_to_weights(project_box(raw))
