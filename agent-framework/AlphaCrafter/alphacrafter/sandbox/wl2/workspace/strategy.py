import numpy as np
from alphacrafter.sim.utils import register_hook, get_stock_daily_data, rebalance_to_weights

ASSETS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
# Screener ensemble: nine factors, deliberately below the ten-factor live cap.
FACTOR_W = {
    "quiet": 0.17, "relative": 0.14, "downtrend": 0.13,
    "compression": 0.13, "stress": 0.12, "breadth": 0.10,
    "residual": 0.08, "agreement": 0.07, "asymmetry": 0.06,
}
CADENCE = 10
_day = 0

def rank_cs(vals):
    good = sorted((s, float(v)) for s, v in vals.items() if np.isfinite(v))
    out = {s: 0.5 for s in ASSETS}
    n = len(good)
    if n:
        for i, (s, _) in enumerate(good):
            out[s] = (i + 1.0) / n
    return out

def make_weights(score):
    # 2% floor and 22% cap provide diversification and leave no cash sleeve.
    floor, cap = 0.02, 0.22
    x = {s: max(float(score.get(s, 0.5)), 1e-6) for s in ASSETS}
    w = {s: floor + (1 - floor * len(ASSETS)) * x[s] / sum(x.values()) for s in ASSETS}
    for _ in range(20):
        high = [s for s in ASSETS if w[s] > cap + 1e-12]
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
    return {s: w[s] / total for s in ASSETS}

@register_hook
def cross_asset_strategy():
    global _day
    _day += 1
    if (_day - 1) % CADENCE != 0:
        return

    data = {}
    for s in ASSETS:
        df = get_stock_daily_data(symbol=s, days=270)
        if df is None or len(df) < 100:
            continue
        close = np.asarray(df.sort_values("date").iloc[:-1]["close"], dtype=float)
        close = close[np.isfinite(close) & (close > 0)]
        if len(close) < 80:
            continue
        ret = close[1:] / close[:-1] - 1.0
        neg = ret[-30:][ret[-30:] < 0]
        data[s] = {
            "t10": close[-1] / close[-11] - 1, "t20": close[-1] / close[-21] - 1,
            "t30": close[-1] / close[-31] - 1, "t60": close[-1] / close[-61] - 1,
            "v10": max(np.std(ret[-10:]), .003), "v20": max(np.std(ret[-20:]), .003),
            "v60": max(np.std(ret[-60:]), .003),
            "downv": max(np.std(neg) if len(neg) > 2 else np.std(ret[-20:]), .003),
        }
    if len(data) < 10:
        return

    market10 = np.mean([d["t10"] for d in data.values()])
    raw = {}
    for s, d in data.items():
        agree = np.mean([d["t10"] > 0, d["t20"] > 0, d["t60"] > 0])
        residual = -(d["t10"] - market10) / (d["v20"] + .01)
        raw[s] = {
            "quiet": d["t20"] * abs(d["t20"]) / (d["v60"] + .01),
            "relative": d["t20"] / (d["v20"] + .01),
            "downtrend": d["t60"] / (d["downv"] + .01),
            "compression": -d["v10"] / (d["v60"] + .003),
            "stress": (d["t10"] if d["t30"] >= 0 else -d["t10"]) / (d["v20"] + .01),
            "breadth": d["t30"] / (d["v60"] + .01),
            "residual": residual,
            "agreement": residual * (.5 + .5 * agree),
            "asymmetry": -d["downv"] / (d["v20"] + .01),
        }
    ranks = {f: rank_cs({s: raw[s][f] for s in raw}) for f in FACTOR_W}
    score = {s: sum(FACTOR_W[f] * ranks[f][s] for f in FACTOR_W) for s in ASSETS}

    # High-risk sideways/bearish posture: full investment, defensive tradable tilt.
    breadth = np.mean([d["t30"] > 0 for d in data.values()])
    spx30 = data.get("SPX", {}).get("t30", 0.0)
    bear = breadth < .45 or spx30 < -.05
    intensity = min(1.0, max(0.0, (.45 - breadth) * 1.8) + max(0.0, (-spx30 - .02) / .18)) if bear else 0.0
    for s in ("XAU", "US10Y", "CN10Y"):
        score[s] *= 1.0 + .9 * intensity
    for s in ("BTC", "ETH", "WTI", "SOX", "NDX", "COPPER"):
        score[s] *= max(.18, 1.0 - .70 * intensity)

    median_v = np.median([d["v20"] for d in data.values()])
    for s in ASSETS:
        score[s] = max(.01, score[s]) * np.clip(median_v / max(data.get(s, {}).get("v20", median_v), .003), .70, 1.25)
    target = make_weights(score)
    if set(target) == set(ASSETS) and all(np.isfinite(v) and v >= 0 for v in target.values()):
        rebalance_to_weights(target)

def strategy():
    cross_asset_strategy()
