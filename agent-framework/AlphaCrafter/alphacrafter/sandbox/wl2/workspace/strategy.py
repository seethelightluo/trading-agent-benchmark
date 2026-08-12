import numpy as np
from alphacrafter.sim.utils import register_hook, get_stock_daily_data, rebalance_to_weights

ASSETS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
# Current screener ensemble (9 active factors; weights sum to 1).
FACTOR_W = {
    "quiet": 0.18, "relative": 0.16, "downtrend": 0.14,
    "accel": 0.10, "compression": 0.12, "breadth": 0.09,
    "stress": 0.08, "agreement_rev": 0.07, "volume_rev": 0.06,
}
CADENCE = 10
_day = 0

def cs_rank(values):
    valid = sorted((s, float(v)) for s, v in values.items() if np.isfinite(v))
    out = {s: 0.5 for s in ASSETS}
    if len(valid) > 1:
        for i, (s, _) in enumerate(valid):
            out[s] = i / (len(valid) - 1)
    return out

def bounded_weights(scores):
    # Full investment, diversified 2% floor and 20% ceiling.
    floor, cap = 0.02, 0.20
    x = {s: max(float(scores.get(s, 0.5)), 1e-6) for s in ASSETS}
    remaining = 1.0 - floor * len(ASSETS)
    w = {s: floor + remaining * x[s] / sum(x.values()) for s in ASSETS}
    for _ in range(30):
        high = [s for s in ASSETS if w[s] > cap]
        if not high:
            break
        excess = sum(w[s] - cap for s in high)
        for s in high:
            w[s] = cap
        rest = [s for s in ASSETS if s not in high]
        z = sum(x[s] for s in rest)
        for s in rest:
            w[s] += excess * x[s] / max(z, 1e-12)
    z = sum(w.values())
    return {s: max(0.0, w[s] / z) for s in ASSETS}

@register_hook
def cross_asset_strategy():
    global _day
    _day += 1
    if (_day - 1) % CADENCE != 0:
        return
    data = {}
    for s in ASSETS:
        df = get_stock_daily_data(symbol=s, days=280)
        if df is None or len(df) < 100:
            continue
        c = np.asarray(df.sort_values("date").iloc[:-1]["close"], dtype=float)
        c = c[np.isfinite(c) & (c > 0)]
        if len(c) < 80:
            continue
        r = c[1:] / c[:-1] - 1.0
        neg = r[-30:][r[-30:] < 0]
        data[s] = {
            "t3": c[-1] / c[-4] - 1, "t10": c[-1] / c[-11] - 1,
            "t20": c[-1] / c[-21] - 1, "t30": c[-1] / c[-31] - 1,
            "t60": c[-1] / c[-61] - 1, "v10": max(np.std(r[-10:]), .003),
            "v20": max(np.std(r[-20:]), .003), "v60": max(np.std(r[-60:]), .003),
            "downv": max(np.std(neg) if len(neg) > 2 else np.std(r[-20:]), .003),
            "volratio": np.mean(np.abs(r[-3:])) / max(np.mean(np.abs(r[-20:])), .003),
        }
    if len(data) < 10:
        return
    market10 = np.mean([d["t10"] for d in data.values()])
    raw = {}
    for s, d in data.items():
        agreement = np.mean([d["t10"] > 0, d["t20"] > 0, d["t60"] > 0])
        resid = -(d["t3"] - np.mean([x["t3"] for x in data.values()])) / (d["v20"] + .01)
        # Positive residual score means a controlled short-term bounce candidate.
        raw[s] = {
            "quiet": d["t20"] * abs(d["t20"]) / (d["v60"] + .01),
            "relative": d["t20"] / (d["v20"] + .01),
            "downtrend": d["t60"] / (d["downv"] + .01),
            "accel": (d["t10"] - d["t30"] / 3.0) / (d["v20"] + .01),
            "compression": -d["v10"] / (d["v60"] + .003),
            "breadth": d["t30"] / (d["v60"] + .01),
            "stress": (d["t10"] if d["t30"] >= 0 else -d["t10"]) / (d["v20"] + .01),
            "agreement_rev": resid * (.5 + .5 * agreement),
            "volume_rev": resid * (1.0 + min(d["volratio"], 2.0) * .25),
        }
    ranks = {f: cs_rank({s: raw[s][f] for s in raw}) for f in FACTOR_W}
    score = {s: sum(FACTOR_W[f] * ranks[f][s] for f in FACTOR_W) for s in ASSETS}

    # Sideways/bearish, elevated-risk overlay: redirect risk to tradable defensives.
    breadth = np.mean([d["t30"] > 0 for d in data.values()])
    spx30 = data.get("SPX", {}).get("t30", 0.0)
    bear = breadth < .50 or spx30 < -.03
    intensity = min(1.0, max(0.0, (.50 - breadth) * 1.8) + max(0.0, (-spx30 - .01) / .15)) if bear else 0.0
    for s in ("XAU", "US10Y", "CN10Y"):
        score[s] *= 1.0 + 1.25 * intensity
    for s in ("BTC", "ETH", "WTI", "SOX", "NDX", "COPPER"):
        score[s] *= max(.15, 1.0 - .75 * intensity)
    medv = np.median([d["v20"] for d in data.values()])
    for s in ASSETS:
        score[s] = max(.01, score[s]) * np.clip(medv / max(data.get(s, {}).get("v20", medv), .003), .75, 1.20)
    target = bounded_weights(score)
    if set(target) == set(ASSETS) and all(np.isfinite(v) and v >= 0 for v in target.values()) and abs(sum(target.values()) - 1) < 1e-8:
        rebalance_to_weights(target)

def strategy():
    cross_asset_strategy()
