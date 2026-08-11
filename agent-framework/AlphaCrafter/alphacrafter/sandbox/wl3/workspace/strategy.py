import numpy as np
from alphacrafter.sim.utils import register_hook, get_stock_daily_data, rebalance_to_weights

UNIVERSE = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
# Current screener ensemble: eight active factors, capped at the full 15-asset universe.
FACTORS = {"volstate": .27, "medium_rev": .15, "leadlag": .17, "lowvol": .15,
           "residual": .08, "idio": .07, "eqresid": .05, "clv": .06}
CADENCE = 10
MIN_W, MAX_W = .03, .15
_previous = None
_day = 0


def ranks(x):
    good = [(s, float(v)) for s, v in x.items() if np.isfinite(v)]
    good.sort(key=lambda z: z[1])
    ans = {s: .5 for s in UNIVERSE}
    n = len(good)
    if n:
        for i, (s, _) in enumerate(good):
            ans[s] = (i + 1.) / n
    return ans


def capped(raw):
    # Iterative water filling gives an exact, non-negative, full-investment vector.
    fixed, active = {}, set(UNIVERSE)
    for _ in range(40):
        left = 1. - sum(fixed.values())
        den = sum(max(float(raw.get(s, .01)), 1e-9) for s in active)
        trial = {s: left * max(float(raw.get(s, .01)), 1e-9) / den for s in active}
        changed = False
        for s, w in list(trial.items()):
            if w < MIN_W:
                fixed[s] = MIN_W; active.remove(s); changed = True
            elif w > MAX_W:
                fixed[s] = MAX_W; active.remove(s); changed = True
        if not changed:
            fixed.update(trial); break
    z = sum(fixed.values())
    return {s: fixed.get(s, 0.) / z for s in UNIVERSE}


@register_hook
def cross_asset_strategy():
    global _day, _previous
    _day += 1
    if _day != 1 and (_day - 1) % CADENCE:
        return

    returns, vols, bars = {}, {}, {}
    for s in UNIVERSE:
        df = get_stock_daily_data(symbol=s, days=230)
        if df is None or len(df) < 70:
            continue
        df = df.sort_values("date")
        c = np.asarray(df["close"], dtype=float)
        if np.any(~np.isfinite(c[-70:])) or np.any(c[-70:] <= 0):
            continue
        r = c[1:] / c[:-1] - 1.
        returns[s] = r
        vols[s] = max(float(np.std(r[-20:])), .006)
        bars[s] = df.iloc[-1]
    if len(returns) < 10:
        return

    def cum(r, n): return float(np.prod(1. + r[-n:]) - 1.)
    r1 = {s: r[-1] for s, r in returns.items()}
    r5 = {s: cum(r, 5) for s, r in returns.items()}
    r10 = {s: cum(r, 10) for s, r in returns.items()}
    r20 = {s: cum(r, 20) for s, r in returns.items()}
    r40 = {s: cum(r, 40) for s, r in returns.items()}
    eq = [s for s in UNIVERSE[:8] if s in returns]
    eq_med = float(np.median([r40[s] for s in eq])) if eq else 0.
    all_syms = list(returns)
    factors = {k: {} for k in FACTORS}
    for s, r in returns.items():
        v = vols[s]
        peer = np.median([r5[t] for t in all_syms if t != s])
        rev = -r1[s] / (v + .006)
        factors["volstate"][s] = rev * (1. + min(v / .02, 1.))
        factors["medium_rev"][s] = -r10[s] / (v * np.sqrt(10.) + .01)
        # Positive peer lead-lag rewards assets following the cross-asset median.
        factors["leadlag"][s] = peer / (v * np.sqrt(5.) + .01)
        factors["lowvol"][s] = rev / (1. + 2. * v)
        factors["residual"][s] = -(r20[s] - peer) / (v * np.sqrt(20.) + .01)
        factors["idio"][s] = rev - (r5[s] - peer) / (v + .01)
        factors["eqresid"][s] = -(r40[s] - eq_med) / (v * np.sqrt(40.) + .02)
        b = bars[s]
        hi, lo, close = float(b.get("high", 0)), float(b.get("low", 0)), float(b["close"])
        factors["clv"][s] = (2.*close-hi-lo)/(hi-lo) if hi > lo else 0.

    score = {s: sum(w * ranks(factors[k])[s] for k, w in FACTORS.items()) for s in UNIVERSE}
    if _previous is not None:
        score = {s: .50 * score[s] + .50 * _previous.get(s, .5) for s in UNIVERSE}
    _previous = score.copy()

    # High-volatility, weak-breadth regime: defensive tradable assets replace cash.
    breadth = np.mean([r20[s] > 0 for s in returns])
    stress = breadth < .50 or np.median(list(r10.values())) < 0. or np.median(list(vols.values())) > .018
    defensive = {"XAU", "US10Y", "CN10Y"}
    invmean = np.mean([1. / v for v in vols.values()])
    raw = {}
    for s in UNIVERSE:
        damp = .90 + .10 * min((1. / vols.get(s, .02)) / invmean, 1.20)
        raw[s] = max(score[s], .05) * damp
        if stress and s in defensive: raw[s] *= 1.85
        if stress and s not in defensive and r20.get(s, 0.) < -.08: raw[s] *= .75
    target = capped(raw)
    if abs(sum(target.values()) - 1.) < 1e-8:
        rebalance_to_weights(target)
