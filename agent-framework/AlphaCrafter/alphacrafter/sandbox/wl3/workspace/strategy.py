import numpy as np
from alphacrafter.sim.utils import register_hook, get_stock_daily_data, rebalance_to_weights

UNIVERSE = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
# Screener 2027-06-03: six-factor ensemble, capped reversal cluster.
FACTORS = {
    "disp_rev5": 0.38,
    "vol_rev5": 0.15,
    "peer5": 0.15,
    "skip120": 0.12,
    "scaled_rev5": 0.10,
    "medium_rev10": 0.10,
}
CADENCE = 10
MIN_W, MAX_W = 0.03, 0.15
DEFENSIVE = {"XAU", "US10Y", "CN10Y"}
_previous = None
_day = 0


def ranks(vals):
    good = sorted((s, v) for s, v in vals.items() if np.isfinite(v))
    out = {s: 0.5 for s in UNIVERSE}
    n = len(good)
    for i, (s, _) in enumerate(good):
        out[s] = (i + 1.0) / n
    return out


def bounded(raw):
    # Water filling enforces the online contract without board-lot rounding.
    fixed, active = {}, set(UNIVERSE)
    for _ in range(40):
        left = 1.0 - sum(fixed.values())
        den = sum(max(float(raw.get(s, 0.01)), 1e-6) for s in active)
        trial = {s: left * max(float(raw.get(s, 0.01)), 1e-6) / den for s in active}
        changed = False
        for s, w in list(trial.items()):
            if w < MIN_W:
                fixed[s] = MIN_W; active.remove(s); changed = True
            elif w > MAX_W:
                fixed[s] = MAX_W; active.remove(s); changed = True
        if not changed:
            fixed.update(trial); break
    ans = {s: fixed.get(s, 0.0) for s in UNIVERSE}
    z = sum(ans.values())
    return {s: ans[s] / z for s in UNIVERSE}


@register_hook
def cross_asset_strategy():
    global _day, _previous
    _day += 1
    if _day != 1 and (_day - 1) % CADENCE != 0:
        return

    returns, vols = {}, {}
    for s in UNIVERSE:
        df = get_stock_daily_data(symbol=s, days=240)
        if df is None or len(df) < 140:
            continue
        close = np.asarray(df.sort_values("date")["close"], dtype=float)
        if len(close) < 140 or np.any(~np.isfinite(close)) or np.any(close <= 0):
            continue
        r = close[1:] / close[:-1] - 1.0
        returns[s] = r
        vols[s] = max(float(np.std(r[-30:])), 0.006)
    if len(returns) < 10:
        return

    def ret(r, n):
        return float(np.prod(1.0 + r[-n:]) - 1.0)

    h = {n: {s: ret(r, n) for s, r in returns.items()} for n in (3, 5, 10, 20)}
    dispersion = float(np.std([h[3][s] for s in returns]))
    f = {k: {} for k in FACTORS}
    syms = list(returns)
    for s in syms:
        v = vols[s]
        peer3 = np.median([h[3][t] for t in syms if t != s])
        peer5 = np.median([h[5][t] for t in syms if t != s])
        scale5 = v * np.sqrt(5.0) + 0.01
        f["disp_rev5"][s] = (peer3 - h[3][s]) / (v * np.sqrt(3.0) + 0.01) * min(dispersion / 0.025, 1.0)
        f["vol_rev5"][s] = -h[5][s] / scale5
        f["scaled_rev5"][s] = np.clip(-h[5][s] / (v * np.sqrt(30.0) + 1e-6), -6, 6)
        f["medium_rev10"][s] = -h[10][s] / (v * np.sqrt(10.0) + 0.01)
        f["peer5"][s] = peer5 / scale5
        # Skip recent noise: medium-term relative strength.
        f["skip120"][s] = np.prod(1.0 + returns[s][-135:-15]) - 1.0

    score = {s: sum(w * ranks(f[k])[s] for k, w in FACTORS.items()) for s in UNIVERSE}
    if _previous is not None:
        score = {s: 0.50 * score[s] + 0.50 * _previous.get(s, 0.5) for s in UNIVERSE}
    _previous = score.copy()

    breadth = np.mean([h[20][s] > 0 for s in syms])
    high_risk = breadth < 0.50 or np.median(list(h[10].values())) < 0 or np.median(list(vols.values())) > 0.018
    inv_mean = np.mean([1.0 / v for v in vols.values()])
    raw = {}
    for s in UNIVERSE:
        invvol = (1.0 / vols.get(s, 0.02)) / inv_mean
        raw[s] = max(score[s], 0.05) * (0.90 + 0.10 * min(invvol, 1.20))
        if high_risk and s in DEFENSIVE:
            raw[s] *= 2.10
        elif high_risk and h[20].get(s, 0.0) < -0.08:
            raw[s] *= 0.70
    target = bounded(raw)
    if all(np.isfinite(target[s]) and target[s] >= 0 for s in UNIVERSE) and abs(sum(target.values()) - 1.0) < 1e-8:
        rebalance_to_weights(target)
