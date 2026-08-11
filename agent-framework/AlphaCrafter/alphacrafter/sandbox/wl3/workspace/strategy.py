import numpy as np
from alphacrafter.sim.utils import register_hook, get_stock_daily_data, rebalance_to_weights

UNIVERSE = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
# Current Screener ensemble; eight active factors, all long-predictive.
FACTORS = {"clv": .18, "peer": .15, "mom": .17, "res20": .12,
           "rev5": .10, "volrev": .10, "lowvolrev": .10, "macrorev": .08}
MIN_W, MAX_W, CADENCE = .02, .14, 10
_day = 0

def cross_rank(values):
    valid = sorted((s, float(v)) for s, v in values.items() if np.isfinite(v))
    out = {s: .5 for s in UNIVERSE}
    n = len(valid)
    for i, (s, _) in enumerate(valid):
        out[s] = (i + 1.0) / n
    return out

def project(raw):
    # Projection onto the full-investment simplex with sensible concentration bounds.
    w = {s: max(float(raw.get(s, 1.0 / len(UNIVERSE))), 1e-12) for s in UNIVERSE}
    fixed, active = {}, set(UNIVERSE)
    for _ in range(30):
        left = 1.0 - sum(fixed.values())
        z = sum(w[s] for s in active)
        trial = {s: left * w[s] / z for s in active}
        clipped = False
        for s, x in list(trial.items()):
            if x < MIN_W:
                fixed[s] = MIN_W; active.remove(s); clipped = True
            elif x > MAX_W:
                fixed[s] = MAX_W; active.remove(s); clipped = True
        if not clipped:
            fixed.update(trial); break
    total = sum(fixed.values())
    return {s: fixed.get(s, 0.0) / total for s in UNIVERSE}

def cum(r, n):
    return float(np.prod(1.0 + r[-n:]) - 1.0)

@register_hook
def cross_asset_strategy():
    global _day
    _day += 1
    if _day != 1 and (_day - 1) % CADENCE != 0:
        return
    returns, vols, sig = {}, {}, {k: {} for k in FACTORS}
    for s in UNIVERSE:
        df = get_stock_daily_data(symbol=s, days=160)
        if df is None or len(df) < 45:
            continue
        df = df.sort_values("date").reset_index(drop=True)
        c = np.asarray(df["close"], float)
        o = np.asarray(df["open"], float)
        h = np.asarray(df["high"], float)
        l = np.asarray(df["low"], float)
        q = np.asarray(df["volume"], float)
        if np.any(~np.isfinite(c[-45:])) or np.any(c[-45:] <= 0):
            continue
        r = c[1:] / c[:-1] - 1.0
        v = max(float(np.nanstd(r[-20:])), .008)
        returns[s], vols[s] = r, v
        span = h[-1] - l[-1]
        sig["clv"][s] = (2*c[-1] - h[-1] - l[-1]) / span if span > 0 else 0.0
        sig["mom"][s] = cum(r, 20) / (v*np.sqrt(20) + .01)
        sig["rev5"][s] = -cum(r, 5) / (v*np.sqrt(5) + .005)
        sig["res20"][s] = 0.0
        downside = max(float(np.std(np.minimum(r[-10:], 0.0))), .004)
        sig["lowvolrev"][s] = -cum(r, 3) / (v*np.sqrt(3) + .005)
        sig["macrorev"][s] = -cum(r, 3) / (downside*np.sqrt(3) + .005)
        medq = np.nanmedian(q[-21:-1]) if len(q) >= 21 else np.nan
        boost = np.clip(q[-1]/medq, .75, 1.5) if np.isfinite(medq) and medq > 0 else 1.0
        sig["volrev"][s] = (-r[-1] / v) * (1.0 + .20*max(boost-1.0, 0.0))
    if len(returns) < 8:
        return
    r5 = {s: cum(r, 5) for s, r in returns.items()}
    r20 = {s: cum(r, 20) for s, r in returns.items()}
    for s in returns:
        others5 = [r5[t] for t in returns if t != s]
        others20 = [r20[t] for t in returns if t != s]
        sig["peer"][s] = r5[s] - float(np.median(others5))
        # Residual reversal: mean-revert relative cross-asset performance.
        sig["res20"][s] = -(r20[s] - float(np.median(others20)))
    ranks = {k: cross_rank(v) for k, v in sig.items()}
    score = {s: sum(FACTORS[k] * ranks[k][s] for k in FACTORS) for s in UNIVERSE}
    median_vol = float(np.median(list(vols.values())))
    breadth = float(np.mean([r20[s] > 0 for s in returns]))
    stress = median_vol > .018 or breadth < .45
    inv_mean = float(np.mean([1.0 / v for v in vols.values()]))
    raw = {}
    for s in UNIVERSE:
        v = vols.get(s, median_vol)
        risk_scale = min((1.0 / v) / inv_mean, 1.5)
        raw[s] = max(score[s], .05) * (.85 + .15 * risk_scale)
        if stress and s in {"XAU", "US10Y", "CN10Y"}:
            raw[s] *= 1.15
    # Always submit a complete 15-asset, non-negative, sum-to-one target; no cash sleeve.
    rebalance_to_weights(project(raw))
