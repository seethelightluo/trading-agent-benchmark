import numpy as np
from alphacrafter.sim.utils import register_hook, get_stock_daily_data, rebalance_to_weights

UNIVERSE = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
FACTORS = {"clv": 0.3395, "peer": 0.2612, "reversal": 0.2012, "momentum": 0.1981}
REBALANCE_EVERY = 10
MIN_WEIGHT, MAX_WEIGHT = 0.02, 0.14
_call = 0


def percentile(values):
    good = [(s, float(v)) for s, v in values.items() if np.isfinite(v)]
    out = {s: 0.5 for s in UNIVERSE}
    if len(good) > 1:
        ordered = sorted(good, key=lambda x: x[1])
        n = len(ordered)
        for i, (s, _) in enumerate(ordered):
            out[s] = (i + 1.0) / n
    return out


def bounded(raw):
    # Iterative box-constrained simplex projection; always returns 15 weights.
    w = {s: max(float(raw.get(s, 0.0)), 1e-10) for s in UNIVERSE}
    fixed = {}
    free = set(UNIVERSE)
    for _ in range(30):
        left = 1.0 - sum(fixed.values())
        scale = left / max(sum(w[s] for s in free), 1e-12)
        clipped = []
        for s in tuple(free):
            x = w[s] * scale
            if x < MIN_WEIGHT:
                fixed[s] = MIN_WEIGHT
                clipped.append(s)
            elif x > MAX_WEIGHT:
                fixed[s] = MAX_WEIGHT
                clipped.append(s)
            else:
                w[s] = x
        if not clipped:
            break
        free -= set(clipped)
    if free:
        left = 1.0 - sum(fixed.values())
        z = sum(w[s] for s in free)
        for s in free:
            w[s] = left * w[s] / max(z, 1e-12)
    w.update(fixed)
    total = sum(w.values())
    return {s: float(w[s] / total) for s in UNIVERSE}


@register_hook
def cross_asset_strategy():
    global _call
    _call += 1
    if _call > 1 and (_call - 1) % REBALANCE_EVERY:
        return

    prices, clv, rev, mom, ret20, vol = {}, {}, {}, {}, {}, {}
    for s in UNIVERSE:
        df = get_stock_daily_data(symbol=s, days=90)
        if df is None or len(df) < 26:
            continue
        df = df.sort_values("date").reset_index(drop=True)
        c = np.asarray(df["close"], dtype=float)
        h = np.asarray(df["high"], dtype=float)
        l = np.asarray(df["low"], dtype=float)
        if not np.all(np.isfinite(c[-26:])):
            continue
        r = c[1:] / np.maximum(c[:-1], 1e-12) - 1.0
        sigma = max(float(np.std(r[-20:])), 0.008)
        prices[s] = c
        clv[s] = (c[-1] - l[-1]) / max(h[-1] - l[-1], abs(c[-1]) * 1e-8)
        rev[s] = -float(np.mean(r[-5:]))
        ret20[s] = c[-1] / max(c[-21], 1e-12) - 1.0
        mom[s] = ret20[s] / (sigma + 0.01)
        vol[s] = sigma
    if len(prices) < 12:
        return

    # Relative peer lead-lag: each asset's recent move versus the leave-one-out peer median.
    peer = {}
    for s in prices:
        others = [ret20[t] for t in prices if t != s]
        peer[s] = ret20[s] - float(np.median(others)) if others else 0.0
    ranks = {"clv": percentile(clv), "peer": percentile(peer),
             "reversal": percentile(rev), "momentum": percentile(mom)}
    score = {s: sum(FACTORS[k] * ranks[k][s] for k in FACTORS) for s in UNIVERSE}

    # Regime overlay: bullish breadth retains risk assets; weak breadth shifts toward
    # tradable defensives while remaining fully invested and long-only.
    breadth = np.mean([ret20[s] > 0 for s in ret20])
    risk_assets = {"000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "COPPER", "WTI", "BTC", "ETH"}
    defensives = {"XAU", "US10Y", "CN10Y"}
    overlay = {s: 1.0 for s in UNIVERSE}
    if breadth < 0.40:
        for s in risk_assets: overlay[s] = 0.82
        for s in defensives: overlay[s] = 1.55
    elif breadth < 0.55:
        for s in risk_assets: overlay[s] = 0.93
        for s in defensives: overlay[s] = 1.20

    inv_average = float(np.mean([1.0 / vol[s] for s in vol]))
    raw = {s: max(score[s], 0.05) * overlay[s] *
           (0.85 + 0.15 * (1.0 / vol[s]) / max(inv_average, 1e-12)) for s in UNIVERSE}
    rebalance_to_weights(bounded(raw))
