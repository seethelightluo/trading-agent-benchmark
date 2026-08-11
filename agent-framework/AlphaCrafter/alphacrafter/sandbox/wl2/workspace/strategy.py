import numpy as np
from alphacrafter.sim.utils import register_hook, get_stock_daily_data, rebalance_to_weights

ASSETS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
# Screener ensemble: downside-aware, consistency-gated momentum, and residual momentum.
FACTORS = {"downmom": 0.24, "eff": 0.22, "cons20": 0.24, "cons30": 0.20, "resid": 0.10}
MIN_W, MAX_W = 0.03, 0.15
_day = 0


def cs_rank(values):
    out = {s: 0.5 for s in ASSETS}
    good = sorted((s, float(v)) for s, v in values.items() if np.isfinite(v))
    n = len(good)
    for i, (s, _) in enumerate(good):
        out[s] = (i + 1) / n if n > 1 else 0.5
    return out


def bounded(score):
    raw = {s: max(0.05, float(score.get(s, 0.5))) for s in ASSETS}
    w = {s: MIN_W + (1 - MIN_W * len(ASSETS)) * raw[s] / sum(raw.values()) for s in ASSETS}
    for _ in range(20):
        over = [s for s in ASSETS if w[s] > MAX_W + 1e-10]
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
    z = sum(w.values())
    return {s: max(0.0, w[s] / z) for s in ASSETS}


@register_hook
def cross_asset_strategy():
    global _day
    _day += 1
    if (_day - 1) % 10 != 0:
        return
    data = {}
    for s in ASSETS:
        df = get_stock_daily_data(symbol=s, days=190)
        if df is None or len(df) < 65:
            continue
        c = np.asarray(df.sort_values("date")["close"], dtype=float)
        if np.any(~np.isfinite(c[-65:])) or np.any(c[-65:] <= 0):
            continue
        r = c[1:] / c[:-1] - 1.0
        v20 = max(float(np.std(r[-20:])), 0.006)
        dn20 = max(float(np.std(np.minimum(r[-20:], 0.0))), 0.003)
        t20 = c[-1] / c[-21] - 1.0
        t30 = c[-1] / c[-31] - 1.0
        p20 = float(np.mean(r[-20:] > 0))
        p30 = float(np.mean(r[-30:] > 0))
        data[s] = {
            "downmom": t20 / (dn20 + 0.01),
            "eff": t20 / (dn20 + 0.015),
            "cons20": t20 / (v20 + 0.01) * (0.5 + p20),
            "cons30": t30 / (v20 + 0.01) * (0.5 + p30),
            "resid_raw": t30, "vol": v20, "t30": t30,
        }
    if len(data) < 10:
        return
    median_t30 = float(np.median([x["t30"] for x in data.values()]))
    for x in data.values():
        x["resid"] = x["resid_raw"] - median_t30
    ranks = {f: cs_rank({s: x[f] for s, x in data.items()}) for f in FACTORS}
    score = {s: sum(FACTORS[f] * ranks[f][s] for f in FACTORS) for s in ASSETS}

    equity = [data[s] for s in ("000300.SH", "SPX", "HSI", "N225", "SX5E", "NDX") if s in data]
    breadth = float(np.mean([x["t30"] > 0 for x in equity])) if equity else 0.5
    medvol = float(np.median([x["vol"] for x in data.values()]))
    spx = data.get("SPX", {})
    bear = breadth <= 0.50 or (spx.get("t30", 0.0) < 0 and spx.get("resid", 0.0) < 0)
    high_risk = medvol > 0.018 or breadth < 0.67
    if bear or high_risk:
        boost = 0.12 if bear else 0.07
        for s in ("XAU", "US10Y", "CN10Y"):
            score[s] += boost
        for s in ("BTC", "ETH", "WTI"):
            score[s] = max(0.05, score[s] - (0.06 if bear else 0.03))
    # Small inverse-vol adjustment; still produces a complete, fully invested target.
    for s in ASSETS:
        if s in data:
            score[s] *= 0.99 + 0.01 * medvol / max(data[s]["vol"], 0.004)
    rebalance_to_weights(bounded(score))


def strategy():
    return cross_asset_strategy()
