import numpy as np
from alphacrafter.sim.utils import register_hook, get_stock_daily_data, rebalance_to_weights

ASSETS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
# Current Screener ensemble (8 active factors; all computed from completed bars).
FACTORS = {"downbreadth": .18, "breadth_mom": .17, "consistency_mom": .16,
           "recovery": .13, "switch": .12, "recovery_accel": .10,
           "persistence": .08, "downvol": .06}
REBALANCE_DAYS = 10
MIN_W, MAX_W = .04, .14
_day = 0


def rank(values):
    good = sorted((s, float(v)) for s, v in values.items() if np.isfinite(v))
    out = {s: .5 for s in ASSETS}
    n = len(good)
    if n:
        for i, (s, _) in enumerate(good): out[s] = (i + 1.) / n
    return out


def weights(raw):
    raw = {s: max(float(raw.get(s, .01)), 1e-8) for s in ASSETS}
    free = 1. - len(ASSETS) * MIN_W
    w = {s: MIN_W + free * raw[s] / sum(raw.values()) for s in ASSETS}
    for _ in range(30):
        over = [s for s in ASSETS if w[s] > MAX_W + 1e-12]
        if not over: break
        excess = sum(w[s] - MAX_W for s in over)
        for s in over: w[s] = MAX_W
        rest = [s for s in ASSETS if s not in over]
        den = sum(raw[s] for s in rest)
        for s in rest: w[s] += excess * raw[s] / den
    z = sum(w.values())
    return {s: w[s] / z for s in ASSETS}


@register_hook
def cross_asset_strategy():
    global _day
    _day += 1
    if (_day - 1) % REBALANCE_DAYS != 0:
        return
    d = {}
    for s in ASSETS:
        df = get_stock_daily_data(symbol=s, days=230)
        if df is None or len(df) < 100: continue
        c = np.asarray(df.sort_values("date").iloc[:-1]["close"], dtype=float)
        if len(c) < 65 or np.any(~np.isfinite(c[-65:])) or np.any(c[-65:] <= 0): continue
        r = c[1:] / c[:-1] - 1.
        v20 = max(float(np.std(r[-20:])), .006)
        neg = r[-20:][r[-20:] < 0]
        t1 = float(r[-1]); t5 = c[-1] / c[-6] - 1.; t10 = c[-1] / c[-11] - 1.
        t20 = c[-1] / c[-21] - 1.; t30 = c[-1] / c[-31] - 1.; t60 = c[-1] / c[-61] - 1.
        cons = float(np.mean(r[-20:] > 0))
        d[s] = {
            # Higher is safer: fewer negative days and less downside volatility.
            "downbreadth": -(0.6 * np.mean(r[-20:] < 0) + 0.4 * np.mean(r[-60:] < 0)),
            "breadth_mom": t20 / v20 * (0.35 + 0.65 * cons),
            "consistency_mom": t20 / v20 * cons,
            "recovery": t5 - .30 * min(t30, 0.),
            "switch": t10 / (v20 + .01) if t30 >= 0 else -t5 / (v20 + .01),
            "recovery_accel": (t10 - t30) / (v20 + .01),
            "persistence": t30 / (v20 + .01) * (.4 + .6 * cons),
            "downvol": -(float(np.std(neg)) if len(neg) > 1 else v20),
            "vol": v20, "t30": t30, "t60": t60
        }
    if len(d) < 10: return
    rr = {k: rank({s: x[k] for s, x in d.items()}) for k in FACTORS}
    score = {s: sum(FACTORS[k] * rr[k][s] for k in FACTORS) for s in ASSETS}
    medvol = float(np.median([x["vol"] for x in d.values()]))
    breadth = float(np.mean([x["t30"] > 0 for x in d.values()]))
    # High-risk sideways/bear overlay: defensive tradables, not cash.
    if medvol > .018 or breadth < .60:
        for s in ("XAU", "US10Y", "CN10Y"): score[s] += .24
        for s in ("BTC", "ETH", "WTI"): score[s] *= .60
    raw = {s: (max(score[s], .03) * np.clip(medvol / d[s]["vol"], .70, 1.25)
               if s in d else .03) for s in ASSETS}
    target = weights(raw)
    if all(np.isfinite(target[s]) and target[s] >= 0 for s in ASSETS) and abs(sum(target.values()) - 1.) < 1e-8:
        rebalance_to_weights(target)


def strategy():
    return cross_asset_strategy()
