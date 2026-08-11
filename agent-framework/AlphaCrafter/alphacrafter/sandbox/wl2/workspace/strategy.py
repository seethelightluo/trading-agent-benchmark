import numpy as np
from alphacrafter.sim.utils import register_hook, get_stock_daily_data, rebalance_to_weights

ASSETS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
# Screener ensemble (8 active factors; all positive directions).
FACTORS = {"eff": .20, "peer": .20, "rev": .17, "down": .16,
           "mom20": .12, "mom30": .07, "resid": .05, "clv": .03}
REBALANCE_DAYS = 10
POSITION_SCALING = .34
MIN_W, MAX_W = .04, .14
_day = 0


def cs_rank(vals):
    good = sorted((s, float(v)) for s, v in vals.items() if np.isfinite(v))
    out = {s: .5 for s in ASSETS}
    n = len(good)
    if n:
        for i, (s, _) in enumerate(good):
            out[s] = (i + 1.0) / n
    return out


def make_weights(score, invvol):
    raw = {s: max(.20, .5 + POSITION_SCALING * (score[s] - .5)) *
           np.clip(invvol.get(s, 1.0), .72, 1.28) for s in ASSETS}
    w = {s: MIN_W + (1 - len(ASSETS) * MIN_W) * raw[s] / sum(raw.values())
         for s in ASSETS}
    # Enforce the stated asset-level concentration bound while retaining full investment.
    for _ in range(30):
        over = [s for s in ASSETS if w[s] > MAX_W + 1e-12]
        if not over:
            break
        excess = sum(w[s] - MAX_W for s in over)
        for s in over:
            w[s] = MAX_W
        free = [s for s in ASSETS if s not in over]
        den = sum(raw[s] for s in free)
        if not free or den <= 0:
            break
        for s in free:
            w[s] += excess * raw[s] / den
    total = sum(w.values())
    return {s: max(0.0, w[s] / total) for s in ASSETS}


@register_hook
def cross_asset_strategy():
    global _day
    _day += 1
    if (_day - 1) % REBALANCE_DAYS:
        return
    data = {}
    for s in ASSETS:
        df = get_stock_daily_data(symbol=s, days=100)
        if df is None or len(df) < 70:
            continue
        df = df.sort_values("date")
        # Exclude the current incomplete bar: decisions use completed prior day only.
        c = np.asarray(df["close"], dtype=float)[:-1]
        hi = np.asarray(df["high"], dtype=float)[:-1]
        lo = np.asarray(df["low"], dtype=float)[:-1]
        if len(c) < 65 or np.any(c[-65:] <= 0):
            continue
        r = c[1:] / c[:-1] - 1.0
        v20 = max(float(np.std(r[-20:])), .006)
        dn20 = max(float(np.std(np.minimum(r[-20:], 0.0))), .003)
        t5, t20, t30 = c[-1]/c[-6]-1, c[-1]/c[-21]-1, c[-1]/c[-31]-1
        p20, p30 = np.mean(r[-20:] > 0), np.mean(r[-30:] > 0)
        # Smoothed short reversal and a small candle-pressure diversifier.
        rev = np.average(-r[-5:], weights=np.array([1, 2, 3, 4, 5])) / (v20 + .01)
        clv = np.mean((2*c[-3:] - hi[-3:] - lo[-3:]) /
                      np.maximum(hi[-3:] - lo[-3:], c[-3:] * .001))
        data[s] = {"peer": t5, "eff": t20/(dn20+.015),
                   "mom20": t20/(v20+.01)*(.5+p20), "down": t20/(dn20+.01),
                   "rev": rev, "mom30": t30/(v20+.01)*(.5+p30),
                   "clv": float(clv), "v": v20, "t30": t30}
    if len(data) < 10:
        return
    med5 = np.median([x["peer"] for x in data.values()])
    eq = [s for s in ASSETS[:8] if s in data]
    med30 = np.median([data[s]["t30"] for s in eq]) if eq else 0.0
    for x in data.values():
        x["peer"] -= med5
        x["resid"] = x["t30"] - med30
    ranks = {k: cs_rank({s: x.get(k, np.nan) for s, x in data.items()}) for k in FACTORS}
    score = {s: sum(FACTORS[k] * ranks[k][s] for k in FACTORS) for s in ASSETS}
    breadth = np.mean([data[s]["t30"] > 0 for s in eq]) if eq else .5
    market_vol = float(np.median([x["v"] for x in data.values()]))
    # Bearish correction: remain fully invested but favor defensive tradable assets.
    if breadth < .50 or market_vol > .018:
        for s in ("XAU", "US10Y", "CN10Y"):
            score[s] += .12
        for s in ("BTC", "ETH", "WTI"):
            score[s] -= .06
    invvol = {s: np.clip(market_vol / max(x["v"], .006), .72, 1.28)
              for s, x in data.items()}
    rebalance_to_weights(make_weights(score, invvol))


def strategy():
    return cross_asset_strategy()
