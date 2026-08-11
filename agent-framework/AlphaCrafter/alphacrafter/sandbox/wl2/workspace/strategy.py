import numpy as np
from alphacrafter.sim.utils import register_hook, get_stock_daily_data, rebalance_to_weights

ASSETS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
FACTORS = {"clv": .275, "leadlag": .220, "mom20": .170, "rev5": .165, "rev3": .085, "down20": .085}
MIN_W, MAX_W = .02, .15
_day = 0

def rank_cs(x):
    r = {s: .5 for s in ASSETS}
    good = sorted((s, v) for s, v in x.items() if np.isfinite(v))
    n = len(good)
    if n > 1:
        for i, (s, _) in enumerate(good): r[s] = (i + 1.) / n
    return r

def make_weights(score):
    # Allocate the residual above the 2% floor by score, capped at 15%.
    w = {s: MIN_W for s in ASSETS}; active = set(ASSETS)
    residual = 1. - MIN_W * len(ASSETS)
    while active and residual > 1e-12:
        q = {s: max(float(score[s]), .01) for s in active}
        z = sum(q.values())
        capped = [s for s in active if residual * q[s] / z > MAX_W - MIN_W]
        if not capped:
            for s in active: w[s] += residual * q[s] / z
            break
        for s in capped:
            w[s] = MAX_W; residual -= MAX_W - MIN_W; active.remove(s)
    z = sum(w.values())
    return {s: w[s] / z for s in ASSETS}

@register_hook
def cross_asset_strategy():
    global _day
    _day += 1
    if (_day - 1) % 10: return
    d = {}
    for s in ASSETS:
        df = get_stock_daily_data(symbol=s, days=95)
        if df is None or len(df) < 35: continue
        c = np.asarray(df.sort_values("date").close, dtype=float)
        h = np.asarray(df.sort_values("date").high, dtype=float)
        l = np.asarray(df.sort_values("date").low, dtype=float)
        if len(c) < 32 or not np.all(np.isfinite(c[-32:])) or c[-1] <= 0: continue
        rr = c[1:] / np.maximum(c[:-1], 1e-12) - 1.
        vol = max(float(np.std(rr[-20:])), .008)
        dvol = max(float(np.std(np.minimum(rr[-20:], 0.))), .004)
        d[s] = {"clv": (2*c[-1]-h[-1]-l[-1]) / max(h[-1]-l[-1], 1e-12),
                "r5": c[-1]/c[-6]-1., "rev5": -float(np.mean(rr[-5:])),
                "rev3": -float(np.mean(rr[-3:])), "mom20": (c[-1]/c[-21]-1.)/(vol+.01),
                "down20": (c[-1]/c[-21]-1.)/(dvol+.01), "trend": c[-1]/c[-31]-1.,
                "invvol": 1./vol}
    if len(d) < 10: return
    med = float(np.median([x["r5"] for x in d.values()]))
    raw = {f: {} for f in FACTORS}
    for s, x in d.items():
        raw["clv"][s] = x["clv"]; raw["leadlag"][s] = x["r5"] - med
        for f in ("mom20", "rev5", "rev3", "down20"): raw[f][s] = x[f]
    ranks = {f: rank_cs(v) for f, v in raw.items()}
    score = {s: sum(FACTORS[f] * ranks[f][s] for f in FACTORS) for s in ASSETS}
    eq = [d[s] for s in ("000300.SH", "SPX", "HSI", "N225", "SX5E", "NDX") if s in d]
    weak = eq and sum(x["trend"] < 0 for x in eq) / len(eq) >= .5
    weak = weak or (d.get("SPX", {}).get("trend", 0) < 0 and d.get("SPX", {}).get("r5", 0) < 0)
    if weak:
        for s in ("XAU", "US10Y", "CN10Y"): score[s] += .10
        for s in ("BTC", "ETH", "WTI"): score[s] = max(.02, score[s]-.05)
    med_iv = float(np.median([x["invvol"] for x in d.values()]))
    for s in ASSETS:
        score[s] = max(.02, score[s]) * (.92 + .08*d.get(s, {}).get("invvol", med_iv)/max(med_iv, 1e-12))
    target = make_weights(score)
    if set(target) == set(ASSETS) and all(np.isfinite(v) and v >= 0 for v in target.values()) and abs(sum(target.values())-1.) < 1e-8:
        rebalance_to_weights(target)
