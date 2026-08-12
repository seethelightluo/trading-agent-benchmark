import numpy as np
from alphacrafter.sim.utils import register_hook, get_stock_daily_data, rebalance_to_weights

UNIVERSE = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
DEFENSIVE = {"XAU", "US10Y", "CN10Y"}
RISKY = {"BTC", "ETH", "WTI", "COPPER"}
# Current screener ensemble (7 active factors; all positive direction).
FACTORS = ("consistency", "breakout", "residual", "failure", "dispersion", "volstate", "relative")
FACTOR_W = {"consistency": .24, "breakout": .18, "residual": .17,
            "failure": .15, "dispersion": .10, "volstate": .09, "relative": .07}
CADENCE = 10
MIN_W, MAX_W = .04, .18
_previous = None
_day = 0

def cs_rank(values):
    good = sorted((s, v) for s, v in values.items() if np.isfinite(v))
    out = {s: .5 for s in UNIVERSE}
    n = len(good)
    for i, (s, _) in enumerate(good):
        out[s] = (i + 1.0) / max(n, 1)
    return out

def bounded_simplex(raw):
    # Iterative bounded allocation; preserves a complete, cash-free target.
    active = set(UNIVERSE)
    fixed = {}
    base = {s: max(float(raw.get(s, 1.0)), 1e-9) for s in UNIVERSE}
    for _ in range(20):
        remaining = 1.0 - sum(fixed.values())
        z = sum(base[s] for s in active)
        proposed = {s: remaining * base[s] / max(z, 1e-12) for s in active}
        low = [s for s in active if proposed[s] < MIN_W]
        high = [s for s in active if proposed[s] > MAX_W]
        if not low and not high:
            fixed.update(proposed)
            break
        for s in low:
            fixed[s] = MIN_W
            active.remove(s)
        for s in high:
            fixed[s] = MAX_W
            active.remove(s)
        if not active:
            break
    if active:
        rem = 1.0 - sum(fixed.values())
        z = sum(base[s] for s in active)
        fixed.update({s: rem * base[s] / max(z, 1e-12) for s in active})
    total = sum(fixed.values())
    return {s: max(0.0, fixed.get(s, 0.0)) / max(total, 1e-12) for s in UNIVERSE}

@register_hook
def cross_asset_strategy():
    global _day, _previous
    _day += 1
    if _day != 1 and (_day - 1) % CADENCE != 0:
        return
    data = {}
    for symbol in UNIVERSE:
        df = get_stock_daily_data(symbol=symbol, days=240)
        if df is None or len(df) < 130:
            continue
        close = np.asarray(df.sort_values("date")["close"], dtype=float)[:-1]
        if len(close) < 121 or np.any(~np.isfinite(close)) or np.any(close <= 0):
            continue
        ret = close[1:] / close[:-1] - 1.0
        v20 = max(float(np.std(ret[-20:])), .006)
        v60 = max(float(np.std(ret[-60:])), .006)
        data[symbol] = {"r": ret, "v20": v20, "v60": v60}
        data[symbol]["volstate"] = -float(np.prod(1 + ret[-3:]) - 1) / v20
        data[symbol]["failure"] = -float(np.prod(1 + ret[-5:]) - 1) / v20
        data[symbol]["dispersion"] = -float(np.prod(1 + ret[-20:]) - 1) / v60
        data[symbol]["consistency"] = float(np.mean(ret[-60:] > 0) - .5) / v60
        data[symbol]["breakout"] = float(close[-1] / max(np.max(close[-121:-1]), 1e-12) - 1) / v60
        data[symbol]["relative"] = float(close[-1] / max(close[-21], 1e-12) - 1) / v60
    if len(data) < 10:
        return
    market = data.get("000300.SH", data.get("SPX"))
    mr = market["r"][-20:] if market else np.zeros(20)
    vm = max(float(np.var(mr)), 1e-8)
    for x in data.values():
        ar = x["r"][-20:]
        beta = float(np.cov(ar, mr, ddof=0)[0, 1] / vm)
        x["residual"] = float(np.mean(ar - beta * mr)) / x["v60"]
    ranks = {f: cs_rank({s: x[f] for s, x in data.items()}) for f in FACTORS}
    score = {s: sum(FACTOR_W[f] * ranks[f][s] for f in FACTORS) for s in UNIVERSE}
    if _previous is not None:
        score = {s: .70 * score[s] + .30 * _previous[s] for s in UNIVERSE}
    _previous = score.copy()
    median_v = float(np.median([x["v20"] for x in data.values()]))
    breadth = float(np.mean([x["consistency"] > 0 for x in data.values()]))
    m30 = float(np.prod(1 + market["r"][-30:]) - 1) if market else 0.0
    stressed = median_v > .015 or m30 < -.06 or breadth < .40
    invmean = float(np.mean([1.0 / x["v20"] for x in data.values()]))
    raw = {}
    for symbol in UNIVERSE:
        x = data.get(symbol)
        vol = x["v20"] if x else median_v
        damp = np.clip((1.0 / max(vol, .006)) / max(invmean, 1e-9), .78, 1.10)
        raw[symbol] = max(score[symbol], .02) * (.88 + .12 * damp)
        if stressed:
            raw[symbol] *= 1.70 if symbol in DEFENSIVE else (.48 if symbol in RISKY else .86)
        elif symbol in RISKY:
            raw[symbol] *= .90
    target = bounded_simplex(raw)
    if set(target) == set(UNIVERSE) and all(np.isfinite(v) and v >= 0 for v in target.values()):
        rebalance_to_weights(target)
