import numpy as np
from alphacrafter.sim.utils import register_hook, get_stock_daily_data, rebalance_to_weights

ASSETS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
# Exact current screener ensemble; capped at ten active factors.
FACTOR_W = {
    "quiet": 0.18, "relative": 0.14, "downside": 0.12,
    "compression": 0.12, "breadth": 0.10, "risk_rev": 0.10,
    "stress": 0.09, "volume_rev": 0.06, "accel": 0.05,
    "agreement_rev": 0.04,
}
CADENCE = 10
_day = 0

def rank_cs(values):
    good = [(s, v) for s, v in values.items() if np.isfinite(v)]
    out = {s: 0.5 for s in ASSETS}
    if len(good) > 1:
        good.sort(key=lambda z: z[1])
        for i, (s, _) in enumerate(good):
            out[s] = i / (len(good) - 1)
    return out

def make_weights(score):
    # Full investment, 2%-20% bounds, suitable for exactly 15 assets.
    floor, cap = 0.02, 0.20
    x = {s: max(float(score.get(s, 0.5)), 1e-8) for s in ASSETS}
    w = {s: floor + (1 - floor * len(ASSETS)) * x[s] / sum(x.values()) for s in ASSETS}
    for _ in range(30):
        high = [s for s in ASSETS if w[s] > cap]
        if not high:
            break
        excess = sum(w[s] - cap for s in high)
        for s in high:
            w[s] = cap
        rest = [s for s in ASSETS if s not in high]
        denom = sum(x[s] for s in rest)
        for s in rest:
            w[s] += excess * x[s] / max(denom, 1e-12)
    total = sum(w.values())
    return {s: max(0.0, w[s] / total) for s in ASSETS}

@register_hook
def cross_asset_strategy():
    global _day
    _day += 1
    if (_day - 1) % CADENCE != 0:
        return
    features = {}
    for s in ASSETS:
        df = get_stock_daily_data(symbol=s, days=280)
        if df is None or len(df) < 100:
            continue
        close = np.asarray(df.sort_values("date").iloc[:-1]["close"], dtype=float)
        close = close[np.isfinite(close) & (close > 0)]
        if len(close) < 80:
            continue
        ret = close[1:] / close[:-1] - 1.0
        neg = ret[-30:][ret[-30:] < 0]
        features[s] = {
            "t3": close[-1] / close[-4] - 1.0,
            "t10": close[-1] / close[-11] - 1.0,
            "t20": close[-1] / close[-21] - 1.0,
            "t30": close[-1] / close[-31] - 1.0,
            "t60": close[-1] / close[-61] - 1.0,
            "v10": max(np.std(ret[-10:]), .003),
            "v20": max(np.std(ret[-20:]), .003),
            "v60": max(np.std(ret[-60:]), .003),
            "downv": max(np.std(neg) if len(neg) > 2 else np.std(ret[-20:]), .003),
            "activity": np.mean(np.abs(ret[-3:])) / max(np.mean(np.abs(ret[-20:])), .003),
        }
    if len(features) < 10:
        return
    mean3 = np.mean([v["t3"] for v in features.values()])
    mean60 = np.mean([v["t60"] for v in features.values()])
    raw = {}
    for s, d in features.items():
        agreement = np.mean([d["t10"] > 0, d["t20"] > 0, d["t60"] > 0])
        short_resid = -(d["t3"] - mean3) / (d["v20"] + .01)
        # Conservative 60-day risk-adjusted contrarian sleeve.
        risk_rev = -(d["t60"] - mean60) / (d["v60"] + .01)
        raw[s] = {
            "quiet": d["t20"] * abs(d["t20"]) / (d["v60"] + .01),
            "relative": d["t20"] / (d["v20"] + .01),
            "downside": d["t60"] / (d["downv"] + .01),
            "compression": -d["v10"] / (d["v60"] + .003),
            "breadth": d["t30"] / (d["v60"] + .01),
            "risk_rev": risk_rev,
            "stress": (d["t10"] if d["t30"] >= 0 else -d["t10"]) / (d["v20"] + .01),
            "volume_rev": short_resid * (1.0 + .20 * min(d["activity"], 2.0)),
            "accel": (d["t10"] - d["t30"] / 3.0) / (d["v20"] + .01),
            "agreement_rev": short_resid * (.5 + .5 * agreement),
        }
    ranks = {f: rank_cs({s: raw[s][f] for s in raw}) for f in FACTOR_W}
    score = {s: sum(FACTOR_W[f] * ranks[f].get(s, .5) for f in FACTOR_W) for s in ASSETS}
    breadth = np.mean([d["t30"] > 0 for d in features.values()])
    spx = features.get("SPX", {}).get("t30", 0.0)
    # High-risk sideways/bearish posture: shift, never reduce, gross exposure.
    intensity = min(1.0, max(0.0, (.50 - breadth) * 1.8) + max(0.0, (-spx - .01) / .15))
    for s in ("XAU", "US10Y", "CN10Y"):
        score[s] *= 1.0 + 1.25 * intensity
    for s in ("BTC", "ETH", "WTI", "SOX", "NDX", "COPPER"):
        score[s] *= max(.15, 1.0 - .75 * intensity)
    median_vol = np.median([d["v20"] for d in features.values()])
    for s in ASSETS:
        score[s] = max(.01, score[s]) * np.clip(median_vol / max(features.get(s, {}).get("v20", median_vol), .003), .75, 1.20)
    target = make_weights(score)
    if set(target) == set(ASSETS) and all(np.isfinite(v) and v >= 0 for v in target.values()) and abs(sum(target.values()) - 1.0) < 1e-8:
        rebalance_to_weights(target)

def strategy():
    cross_asset_strategy()
