import numpy as np
from alphacrafter.sim.utils import register_hook, get_stock_daily_data, rebalance_to_weights

UNIVERSE = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
DEFENSIVE = {"XAU", "US10Y", "CN10Y"}
RISKY = {"BTC", "ETH", "WTI", "COPPER"}
# Current screener ensemble: six validated factors, capped below ten.
FACTORS = ("failure", "volstate", "dispersion", "consistency", "lowvol", "residual")
FACTOR_W = {"failure": .28, "volstate": .25, "dispersion": .18,
            "consistency": .15, "lowvol": .08, "residual": .06}
CADENCE = 10
MIN_W, MAX_W = .04, .18
_day = 0
_previous = None

def ranks(values):
    good = sorted((s, v) for s, v in values.items() if np.isfinite(v))
    out = {s: .5 for s in UNIVERSE}
    for i, (s, _) in enumerate(good):
        out[s] = (i + 1.0) / max(len(good), 1)
    return out

def bounded(raw):
    # Iterative projection gives every tradable asset a legal fractional weight.
    w = {s: max(float(raw.get(s, 1.0)), 1e-12) for s in UNIVERSE}
    fixed = {}
    for _ in range(100):
        free = [s for s in UNIVERSE if s not in fixed]
        rem = 1.0 - sum(fixed.values())
        z = sum(w[s] for s in free)
        for s in free:
            w[s] = rem * w[s] / max(z, 1e-12)
        hit = [s for s in free if w[s] < MIN_W or w[s] > MAX_W]
        if not hit:
            break
        for s in hit:
            fixed[s] = MIN_W if w[s] < MIN_W else MAX_W
            w[s] = fixed[s]
    z = sum(w.values())
    return {s: w[s] / z for s in UNIVERSE}

@register_hook
def cross_asset_strategy():
    global _day, _previous
    _day += 1
    # Decisions are made only on the first day of each ten-trading-day block.
    if _day != 1 and (_day - 1) % CADENCE != 0:
        return
    data = {}
    for s in UNIVERSE:
        df = get_stock_daily_data(symbol=s, days=220)
        if df is None or len(df) < 100:
            continue
        close = np.asarray(df.sort_values("date")["close"], dtype=float)[:-1]
        if len(close) < 80 or np.any(~np.isfinite(close)) or np.any(close <= 0):
            continue
        r = close[1:] / close[:-1] - 1.0
        v20 = max(float(np.std(r[-20:])), .006)
        v60 = max(float(np.std(r[-60:])), .006)
        data[s] = {"r": r, "v20": v20, "v60": v60}
        data[s].update({
            "failure": -(np.prod(1+r[-5:])-1) / v20,
            "volstate": -(np.prod(1+r[-3:])-1) / v20,
            "dispersion": -((np.prod(1+r[-20:])-1) / v60),
            "consistency": (np.mean(r[-60:] > 0)-.5) / v60,
            "lowvol": -v20,
        })
    if len(data) < 10:
        return
    market = data.get("000300.SH", data.get("SPX"))
    if market is not None:
        mr = market["r"][-20:]
        vm = max(float(np.var(mr)), 1e-8)
        for s, x in data.items():
            ar = x["r"][-20:]
            beta = float(np.cov(ar, mr, ddof=0)[0, 1] / vm)
            x["residual"] = float(np.mean(ar - beta * mr)) / x["v60"]
    else:
        for x in data.values():
            x["residual"] = 0.0
    rr = {f: ranks({s: x[f] for s, x in data.items()}) for f in FACTORS}
    score = {s: sum(FACTOR_W[f] * rr[f][s] for f in FACTORS) for s in UNIVERSE}
    if _previous is not None:
        score = {s: .70 * score[s] + .30 * _previous[s] for s in UNIVERSE}
    _previous = score.copy()
    vols = [x["v20"] for x in data.values()]
    med = float(np.median(vols))
    breadth = float(np.mean([x["consistency"] > 0 for x in data.values()]))
    m30 = np.prod(1 + market["r"][-30:]) - 1 if market is not None else 0.0
    stressed = med > .015 or m30 < -.06 or breadth < .40
    invmean = float(np.mean([1.0 / v for v in vols]))
    raw = {}
    for s in UNIVERSE:
        x = data.get(s)
        v = x["v20"] if x is not None else med
        damp = np.clip((1.0 / max(v, .006)) / invmean, .78, 1.10)
        raw[s] = max(score[s], .03) * (.88 + .12*damp)
        if stressed:
            raw[s] *= 1.70 if s in DEFENSIVE else (.48 if s in RISKY else .86)
        elif s in RISKY:
            raw[s] *= .90
    target = bounded(raw)
    if set(target) == set(UNIVERSE) and all(np.isfinite(v) and v >= 0 for v in target.values()):
        rebalance_to_weights(target)
