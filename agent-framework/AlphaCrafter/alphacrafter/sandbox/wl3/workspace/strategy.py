import numpy as np
from alphacrafter.sim.utils import register_hook, get_stock_daily_data, rebalance_to_weights

UNIVERSE = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
# Current Screener quality/IC ensemble; seven active factors, all long-positive.
FACTORS = {
    "lowvol_rev": 0.22,
    "volstate_rev": 0.18,
    "peer": 0.18,
    "resid20": 0.16,
    "eq_resid40": 0.12,
    "idio_rev": 0.08,
    "clv": 0.06,
}
REBALANCE_DAYS = 10
MIN_W, MAX_W = 0.03, 0.15
_previous = None
_day = 0


def rank(values):
    good = sorted((s, x) for s, x in values.items() if np.isfinite(x))
    out = {s: 0.5 for s in UNIVERSE}
    if good:
        n = len(good)
        for i, (s, _) in enumerate(good):
            out[s] = (i + 1.0) / n
    return out


def bounded(raw):
    # Water-fill into the required 3%-15% long-only range.
    fixed, active = {}, set(UNIVERSE)
    for _ in range(30):
        left = 1.0 - sum(fixed.values())
        den = sum(max(raw[s], 1e-9) for s in active)
        trial = {s: left * max(raw[s], 1e-9) / den for s in active}
        clipped = False
        for s, w in list(trial.items()):
            if w < MIN_W:
                fixed[s] = MIN_W; active.remove(s); clipped = True
            elif w > MAX_W:
                fixed[s] = MAX_W; active.remove(s); clipped = True
        if not clipped:
            fixed.update(trial)
            break
    z = sum(fixed.values())
    return {s: fixed.get(s, 0.0) / z for s in UNIVERSE}


@register_hook
def cross_asset_strategy():
    global _day, _previous
    _day += 1
    if _day != 1 and (_day - 1) % REBALANCE_DAYS != 0:
        return

    rets, vols, rows = {}, {}, {}
    for s in UNIVERSE:
        df = get_stock_daily_data(symbol=s, days=230)
        if df is None or len(df) < 70:
            continue
        df = df.sort_values("date")
        c = np.asarray(df["close"], dtype=float)
        if np.any(~np.isfinite(c[-70:])) or np.any(c[-70:] <= 0):
            continue
        r = c[1:] / c[:-1] - 1.0
        rets[s] = r
        vols[s] = max(float(np.std(r[-20:])), 0.006)
        rows[s] = df.iloc[-1]
    if len(rets) < 10:
        return

    r1 = {s: r[-1] for s, r in rets.items()}
    r5 = {s: float(np.prod(1 + r[-5:]) - 1) for s, r in rets.items()}
    r10 = {s: float(np.prod(1 + r[-10:]) - 1) for s, r in rets.items()}
    r20 = {s: float(np.prod(1 + r[-20:]) - 1) for s, r in rets.items()}
    r40 = {s: float(np.prod(1 + r[-40:]) - 1) for s, r in rets.items()}
    peers = list(rets)
    equities = [s for s in UNIVERSE[:8] if s in rets]
    eq40 = float(np.median([r40[s] for s in equities])) if equities else 0.0
    breadth = float(np.mean([x > 0 for x in r20.values()]))
    stress = breadth < 0.50 or float(np.median(list(r10.values()))) < 0 or float(np.median(list(vols.values()))) > 0.018

    f = {k: {} for k in FACTORS}
    for s, r in rets.items():
        v = vols[s]
        peer5 = np.median([r5[t] for t in peers if t != s])
        scale20 = v * np.sqrt(20) + 0.01
        scale40 = v * np.sqrt(40) + 0.02
        # One-day signals are volatility scaled and smoothed by the 10-day schedule.
        rev = -r1[s] / (v + 0.006)
        f["lowvol_rev"][s] = rev
        f["volstate_rev"][s] = rev * (1.0 + min(v / 0.02, 1.0))
        f["peer"][s] = r5[s] - peer5
        f["resid20"][s] = (r20[s] - peer5) / scale20
        f["eq_resid40"][s] = (r40[s] - eq40) / scale40
        f["idio_rev"][s] = rev - (r5[s] - peer5) / (v + 0.01)
        hi, lo, cl = float(rows[s].get("high", 0)), float(rows[s].get("low", 0)), float(rows[s]["close"])
        f["clv"][s] = (2 * cl - hi - lo) / (hi - lo) if hi > lo else 0.0

    ranked = {k: rank(v) for k, v in f.items()}
    score = {s: sum(FACTORS[k] * ranked[k].get(s, 0.5) for k in FACTORS) for s in UNIVERSE}
    if _previous is not None:
        score = {s: 0.50 * score[s] + 0.50 * _previous.get(s, 0.5) for s in UNIVERSE}
    _previous = score.copy()

    invmean = float(np.mean([1.0 / v for v in vols.values()]))
    defensive = {"XAU", "US10Y", "CN10Y"}
    raw = {}
    for s in UNIVERSE:
        invvol = min((1.0 / vols.get(s, 0.02)) / invmean, 1.20)
        raw[s] = max(score[s], 0.05) * (0.90 + 0.10 * invvol)
        if stress and s in defensive:
            raw[s] *= 1.85
        if stress and s not in defensive and r20.get(s, 0.0) < -0.08:
            raw[s] *= 0.75
    target = bounded(raw)
    if all(np.isfinite(target[s]) and target[s] >= 0 for s in UNIVERSE) and abs(sum(target.values()) - 1.0) < 1e-8:
        rebalance_to_weights(target)
