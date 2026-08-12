import numpy as np
from alphacrafter.sim.utils import register_hook, get_stock_daily_data, rebalance_to_weights

UNIVERSE = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
DEFENSIVE = {"XAU", "US10Y", "CN10Y"}
RISKY = {"BTC", "ETH", "WTI", "COPPER"}
FACTOR_IDS = [
    "miner_2_20280615_volmanaged_consistency30",
    "miner_3_20280601_beta_residual_momentum20",
    "miner_2_20280907_breakout_failure_reversal",
    "miner_1_20280601_relative_strength20",
    "miner_2_20270603_medium_dispersion_reversal",
    "miner_1_20260730_lowvol_scaled_reversal_1d",
]
FACTOR_WEIGHTS = [0.25, 0.22, 0.20, 0.15, 0.12, 0.06]
CADENCE = 10
_day = 0
_previous = None

def rank(values):
    good = sorted((s, float(v)) for s, v in values.items() if np.isfinite(v))
    out = {s: 0.5 for s in UNIVERSE}
    for i, (s, _) in enumerate(good):
        out[s] = (i + 1.0) / max(1, len(good))
    return out

def bounded(raw):
    lo, hi = 0.035, 0.17
    w = {s: max(float(raw.get(s, .01)), 1e-12) for s in UNIVERSE}
    fixed, free = {}, set(UNIVERSE)
    for _ in range(40):
        if not free: break
        rem = 1.0 - sum(fixed.values())
        z = sum(w[s] for s in free)
        p = {s: rem * w[s] / z for s in free}
        low = [s for s in free if p[s] < lo]
        high = [s for s in free if p[s] > hi]
        if not low and not high:
            fixed.update(p); free.clear(); break
        for s in low: fixed[s] = lo; free.remove(s)
        for s in high: fixed[s] = hi; free.remove(s)
    if free:
        rem = 1.0 - sum(fixed.values()); z = sum(w[s] for s in free)
        fixed.update({s: rem*w[s]/z for s in free})
    z = sum(fixed.values())
    return {s: fixed.get(s, 0.0) / z for s in UNIVERSE}

@register_hook
def cross_asset_strategy():
    global _day, _previous
    _day += 1
    if _day != 1 and (_day - 1) % CADENCE != 0:
        return
    data = {}
    for s in UNIVERSE:
        df = get_stock_daily_data(symbol=s, days=270)
        if df is None or len(df) < 140: continue
        c = np.asarray(df.sort_values("date")["close"], dtype=float)[:-1]
        if len(c) < 125 or np.any(~np.isfinite(c)) or np.any(c <= 0): continue
        r = c[1:] / c[:-1] - 1.0
        v20 = max(float(np.std(r[-20:])), .006)
        v30 = max(float(np.std(r[-30:])), .006)
        v60 = max(float(np.std(r[-60:])), .006)
        r1, r20 = float(r[-1]), float(np.prod(1+r[-20:])-1)
        data[s] = {"r": r, "v20": v20,
                   "consistency": (float(np.mean(r[-30:] > 0))-.5)/v30,
                   "beta": 0.0, "failure": -r20/v60, "relative": r20/v20,
                   "dispersion": -r20/v30, "lowrev": -r1/v20}
    if len(data) < 10: return
    market = data.get("000300.SH", data.get("SPX"))
    if market is None: return
    mr = market["r"][-20:]
    var = max(float(np.var(mr)), 1e-8)
    for x in data.values():
        ar = x["r"][-20:]
        beta = float(np.cov(ar, mr, ddof=0)[0, 1]) / var
        x["beta"] = float(np.mean(ar - beta * mr) / x["v20"])
    names = ["consistency", "beta", "failure", "relative", "dispersion", "lowrev"]
    ranks = {n: rank({s: x[n] for s, x in data.items()}) for n in names}
    score = {s: sum(w * ranks[n].get(s, .5) for n, w in zip(names, FACTOR_WEIGHTS)) for s in UNIVERSE}
    if _previous is not None:
        score = {s: .70 * score[s] + .30 * _previous[s] for s in UNIVERSE}
    _previous = score.copy()
    medvol = float(np.median([x["v20"] for x in data.values()]))
    breadth = float(np.mean([x["consistency"] > 0 for x in data.values()]))
    stressed = medvol > .015 or float(np.prod(1+mr)-1) < -.06 or breadth < .40
    invmean = float(np.mean([1/max(x["v20"], .006) for x in data.values()]))
    raw = {}
    for s in UNIVERSE:
        x = data.get(s); vol = x["v20"] if x else medvol
        damp = np.clip((1/max(vol, .006))/max(invmean, 1e-12), .78, 1.10)
        raw[s] = max(score[s], .02) * (.88 + .12*damp)
        raw[s] *= (2.15 if s in DEFENSIVE else (.30 if s in RISKY else .75)) if stressed else (1.10 if s in DEFENSIVE else (.86 if s in RISKY else 1.0))
    target = bounded(raw)
    forecast_returns = {s: float(np.clip((score[s]-.5)*.08, -.04, .04)) for s in UNIVERSE}
    if all(np.isfinite(target[s]) and target[s] >= 0 for s in UNIVERSE) and abs(sum(target.values())-1.0) < 1e-8:
        rebalance_to_weights(target, forecast_returns=forecast_returns, factor_ids=FACTOR_IDS)
