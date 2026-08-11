import numpy as np
from alphacrafter.sim.utils import register_hook, get_stock_daily_data, rebalance_to_weights

UNIVERSE = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
DEFENSIVE = {"XAU", "US10Y", "CN10Y"}
CADENCE = 10
MIN_W, MAX_W = 0.04, 0.14
# Screener ensemble, with overlapping fast reversal signals deliberately moderated.
FACTOR_W = {"cluster": .18, "medium_cluster": .18, "volume_shock": .13,
            "vol_scaled": .12, "intraday": .10, "dispersion": .10,
            "smooth_reversal": .10, "momentum": .09}
CLUSTERS = {
    "equity": {"000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX"},
    "commodity": {"XAU", "COPPER", "WTI"}, "crypto": {"BTC", "ETH"},
    "rates": {"US10Y", "CN10Y"}}
_day = 0
_previous = None


def rank_cs(vals):
    out = {s: .5 for s in UNIVERSE}
    good = sorted((s, float(v)) for s, v in vals.items() if np.isfinite(v))
    for i, (s, _) in enumerate(good):
        out[s] = (i + 1.0) / len(good)
    return out


def bounded(raw):
    # Iterative capped simplex projection; preserves a complete, cash-free target.
    w = {s: max(float(raw.get(s, 1.0)), 1e-12) for s in UNIVERSE}
    fixed = set()
    for _ in range(40):
        free = [s for s in UNIVERSE if s not in fixed]
        rem = 1.0 - sum(w[s] for s in fixed)
        z = sum(w[s] for s in free)
        for s in free:
            w[s] = rem * w[s] / max(z, 1e-12)
        hit = {s for s in free if w[s] < MIN_W or w[s] > MAX_W}
        if not hit:
            break
        for s in hit:
            w[s] = MIN_W if w[s] < MIN_W else MAX_W
        fixed |= hit
    z = sum(w.values())
    return {s: w[s] / z for s in UNIVERSE}


@register_hook
def cross_asset_strategy():
    global _day, _previous
    _day += 1
    if _day != 1 and (_day - 1) % CADENCE != 0:
        return

    data = {}
    for s in UNIVERSE:
        df = get_stock_daily_data(symbol=s, days=270)
        if df is None or len(df) < 160:
            continue
        c = np.asarray(df.sort_values("date")["close"], dtype=float)
        o = np.asarray(df.sort_values("date")["open"], dtype=float)
        vol = np.nan_to_num(np.asarray(df.sort_values("date")["volume"], dtype=float), nan=0.0)
        if np.any(~np.isfinite(c)) or np.any(c <= 0):
            continue
        r = c[1:] / c[:-1] - 1.0
        on = c[1:] / np.maximum(o[1:], 1e-12) - 1.0
        data[s] = {"r": r, "on": on, "h3": np.prod(1+r[-3:])-1,
                   "h5": np.prod(1+r[-5:])-1, "h10": np.prod(1+r[-10:])-1,
                   "h20": np.prod(1+r[-20:])-1, "h60": np.prod(1+r[-60:])-1,
                   "h120": np.prod(1+r[-120:])-1, "v5": max(np.std(r[-5:]), .006),
                   "v20": max(np.std(r[-20:]), .006), "v60": max(np.std(r[-60:]), .006),
                   "vr": np.mean(vol[-5:]) / max(np.mean(vol[-20:]), 1e-12)}
    if len(data) < 10:
        return

    syms = list(data)
    med = {k: np.median([data[s][k] for s in syms]) for k in ("h3", "h5", "h10", "h20")}
    f = {k: {} for k in FACTOR_W}
    for s in syms:
        x = data[s]
        cl = next((k for k, v in CLUSTERS.items() if s in v), None)
        peers = [data[t]["r"] for t in syms if t in CLUSTERS.get(cl, set())]
        n = 5
        peer = np.median([a[-n:] for a in peers], axis=0) if peers else np.zeros(n)
        f["cluster"][s] = -np.sum(x["r"][-n:] - peer) / (x["v20"] + .01)
        f["medium_cluster"][s] = -(x["h10"] - med["h10"]) / (x["v20"] * np.sqrt(10) + .02)
        f["volume_shock"][s] = (med["h3"] - x["h3"]) / (x["v5"] + .01) * np.clip(x["vr"], .5, 2.)
        f["vol_scaled"][s] = (med["h5"] - x["h5"]) / (x["v20"] * np.sqrt(5) + .02)
        f["intraday"][s] = -np.mean(x["on"][-3:]) / (x["v5"] + .01)
        f["dispersion"][s] = (med["h20"] - x["h20"]) / (x["v20"] * np.sqrt(20) + .02) * np.clip(x["vr"], .6, 1.6)
        f["smooth_reversal"][s] = -(0.5*x["h5"] + 0.3*x["h10"] + 0.2*x["h20"]) / (x["v20"] * np.sqrt(20) + .02)
        f["momentum"][s] = x["h20"] / (x["v20"] * np.sqrt(20) + .02)
    ranks = {k: rank_cs(v) for k, v in f.items()}
    breadth = np.mean([data[s]["h120"] > 0 for s in syms])
    if breadth < .45:
        ranks["momentum"] = {s: .5 for s in UNIVERSE}
    score = {s: sum(FACTOR_W[k] * ranks[k].get(s, .5) for k in FACTOR_W) for s in UNIVERSE}
    if _previous is not None:
        score = {s: .55*score[s] + .45*_previous[s] for s in UNIVERSE}
    _previous = score.copy()

    high_risk = np.median([data[s]["v20"] for s in syms]) > .018 or breadth < .50
    invmean = np.mean([1.0 / data[s]["v20"] for s in syms])
    raw = {}
    for s in UNIVERSE:
        x = data.get(s, {"v20": .02, "h120": 0.0})
        invvol = (1.0/x["v20"]) / max(invmean, 1e-12) if s in data else 1.0
        raw[s] = max(score[s], .15) * (.92 + .08*np.clip(invvol, .7, 1.3))
        if high_risk and s in DEFENSIVE:
            raw[s] *= 2.4
        if high_risk and x["h120"] < -.12:
            raw[s] *= .80
    rebalance_to_weights(bounded(raw))
