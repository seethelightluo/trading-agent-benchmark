import numpy as np
from alphacrafter.sim.utils import (register_hook, get_stock_daily_data,
                                     get_index_daily_data, rebalance_to_weights)

ASSETS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
FACTORS = {"lead5": .20, "rev5": .16, "mom30": .16, "mom20": .13,
           "clv": .10, "trend20": .09, "dxyres": .07, "mktres": .05,
           "shock": .04}
REBALANCE_DAYS = 10
POSITION_SCALING = .10
MIN_W, MAX_W = .035, .16
_day = 0


def _rank(vals):
    good = [(s, float(v)) for s, v in vals.items() if np.isfinite(v)]
    out = {s: .5 for s in ASSETS}
    if not good:
        return out
    good.sort(key=lambda z: z[1])
    for i, (s, _) in enumerate(good):
        out[s] = (i + 1.) / len(good)
    return out


def _weights(score, invvol):
    raw = {s: max(.20, .5 + POSITION_SCALING * (score.get(s, .5) - .5)) *
           np.clip(invvol.get(s, 1.), .70, 1.30) for s in ASSETS}
    free = 1.0 - len(ASSETS) * MIN_W
    w = {s: MIN_W + free * raw[s] / sum(raw.values()) for s in ASSETS}
    # Cap concentration and redistribute excess until stable.
    for _ in range(30):
        over = [s for s in ASSETS if w[s] > MAX_W]
        if not over:
            break
        excess = sum(w[s] - MAX_W for s in over)
        for s in over:
            w[s] = MAX_W
        rest = [s for s in ASSETS if s not in over]
        den = sum(raw[s] for s in rest)
        if not rest or den <= 0:
            break
        for s in rest:
            w[s] += excess * raw[s] / den
    z = sum(w.values())
    return {s: max(0., w[s] / z) for s in ASSETS}


@register_hook
def cross_asset_strategy():
    global _day
    _day += 1
    if (_day - 1) % REBALANCE_DAYS:
        return
    data, returns = {}, {}
    for s in ASSETS:
        df = get_stock_daily_data(symbol=s, days=150)
        if df is None or len(df) < 75:
            continue
        d = df.sort_values("date").iloc[:-1]  # completed bars only
        c = np.asarray(d["close"], dtype=float)
        if len(c) < 65 or np.any(~np.isfinite(c[-65:])) or np.any(c[-65:] <= 0):
            continue
        r = c[1:] / c[:-1] - 1.
        r5, r20, r30 = r[-5:], r[-20:], r[-30:]
        vol = max(float(np.std(r20)), .006)
        p20, p30 = np.mean(r20 > 0), np.mean(r30 > 0)
        t20, t30 = c[-1] / c[-21] - 1., c[-1] / c[-31] - 1.
        clv = 0.
        if all(k in d.columns for k in ("high", "low", "close")):
            hi = np.asarray(d["high"].iloc[-10:], float)
            lo = np.asarray(d["low"].iloc[-10:], float)
            cc = np.asarray(d["close"].iloc[-10:], float)
            clv = float(np.mean(np.where(hi > lo, (2*cc-hi-lo)/(hi-lo), 0.)))
        data[s] = {"lead5": float(np.sum(r5)), "rev5": -float(np.sum(r5)),
                   "mom30": t30/(vol+.01) * (.5+.5*max(0., 2*p30-1)),
                   "mom20": t20/(vol+.01) * (.5+.5*max(0., 2*p20-1)),
                   "clv": clv, "trend20": t20 * (.5+.5*max(0., 2*p20-1)),
                   "shock": -float(r[-1])/(vol+.01), "t30": t30, "vol": vol}
        returns[s] = r[-30:]
    if len(data) < 10:
        return

    median5 = np.median([x["lead5"] for x in data.values()])
    for s, x in data.items():
        x["lead5"] -= median5
        x["mktres"] = x["t30"] - np.median([z["t30"] for z in data.values()])
        x["dxyres"] = x["t30"]
    dxy = get_index_daily_data(symbol="DXY", days=150)
    if dxy is not None and len(dxy) > 35:
        dc = np.asarray(dxy.sort_values("date")["close"], float)[:-1]
        if len(dc) >= 31 and np.all(np.isfinite(dc[-31:])) and np.all(dc[-31:] > 0):
            dr = dc[1:] / dc[:-1] - 1.
            for s, x in data.items():
                ar = returns[s]
                q = dr[-len(ar):]
                var = np.var(q)
                beta = np.cov(ar, q, ddof=0)[0, 1] / var if var > 1e-10 else 0.
                x["dxyres"] = x["t30"] - beta * np.sum(q)

    ranks = {k: _rank({s: x[k] for s, x in data.items()}) for k in FACTORS}
    score = {s: sum(v * ranks[k][s] for k, v in FACTORS.items()) for s in ASSETS}
    breadth = np.mean([x["t30"] > 0 for x in data.values()])
    stress = breadth < .50 or np.median([x["vol"] for x in data.values()]) > .018
    if stress:
        for s in ("XAU", "US10Y", "CN10Y"):
            score[s] += .28
        for s in ("BTC", "ETH", "WTI"):
            score[s] -= .20
    medvol = np.median([x["vol"] for x in data.values()])
    invvol = {s: np.clip(medvol / x["vol"], .70, 1.30) for s, x in data.items()}
    rebalance_to_weights(_weights(score, invvol))


def strategy():
    return cross_asset_strategy()
