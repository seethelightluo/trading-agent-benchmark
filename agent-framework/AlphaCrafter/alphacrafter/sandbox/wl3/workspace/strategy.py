import numpy as np
from alphacrafter.sim.utils import register_hook, get_stock_daily_data, rebalance_to_weights

UNIVERSE = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
DEFENSIVE = {"XAU", "US10Y", "CN10Y"}
RISKY = {"BTC", "ETH", "WTI", "COPPER"}
FACTOR_IDS = [
    "miner_2_20280907_breakout_failure_reversal",
    "miner_3_20280601_beta_residual_momentum20",
    "miner_2_20280615_volmanaged_consistency30",
    "miner_1_20280601_relative_strength20",
    "miner_3_20280504_downside_adjusted_momentum20",
    "miner_2_20270603_medium_dispersion_reversal",
]
FACTOR_WEIGHTS = [0.27, 0.20, 0.18, 0.15, 0.14, 0.06]
CADENCE = 10
_day = 0
_previous = None


def rank_cs(values):
    good = sorted((s, v) for s, v in values.items() if np.isfinite(v))
    out = {s: 0.5 for s in UNIVERSE}
    for i, (s, _) in enumerate(good):
        out[s] = (i + 1.0) / len(good)
    return out


def full_weights(raw, floor=0.02, ceiling=0.25):
    w = np.array([max(float(raw.get(s, 0.0)), 1e-8) for s in UNIVERSE])
    for _ in range(50):
        w /= w.sum()
        over = w > ceiling
        if not np.any(over):
            break
        excess = float(np.sum(w[over] - ceiling))
        w[over] = ceiling
        under = ~over
        if np.any(under):
            w[under] += excess * w[under] / max(float(w[under].sum()), 1e-12)
    w = np.maximum(w, floor)
    w /= w.sum()
    return {s: float(x) for s, x in zip(UNIVERSE, w)}


@register_hook
def cross_asset_strategy():
    global _day, _previous
    _day += 1
    if _day != 1 and (_day - 1) % CADENCE != 0:
        return
    data = {}
    for symbol in UNIVERSE:
        df = get_stock_daily_data(symbol=symbol, days=280)
        if df is None or len(df) < 140:
            continue
        close = np.asarray(df.sort_values("date")["close"], dtype=float)[:-1]
        if len(close) < 125 or np.any(~np.isfinite(close)) or np.any(close <= 0):
            continue
        r = close[1:] / close[:-1] - 1.0
        v15 = max(float(np.std(r[-15:])), 0.006)
        v20 = max(float(np.std(r[-20:])), 0.006)
        v30 = max(float(np.std(r[-30:])), 0.006)
        v60 = max(float(np.std(r[-60:])), 0.006)
        ret20 = float(np.prod(1 + r[-20:]) - 1)
        ret60 = float(np.prod(1 + r[-60:]) - 1)
        ret120 = float(np.prod(1 + r[-120:]) - 1)
        neg = r[-20:][r[-20:] < 0]
        downside = max(float(np.std(neg)) if len(neg) > 1 else v20, 0.006)
        data[symbol] = {"r": r, "v": v20,
            "failure": -(ret120 - ret60) / v60,
            "consistency": (float(np.mean(r[-30:] > 0)) - 0.5) / v30,
            "beta": 0.0, "relative": ret20 / v20,
            "downside": ret20 / downside, "dispersion": -ret20 / v30}
    if len(data) < 10:
        return
    market = data.get("000300.SH", data.get("SPX"))
    if market is None:
        return
    mr = market["r"][-20:]
    var = max(float(np.var(mr)), 1e-8)
    for x in data.values():
        ar = x["r"][-20:]
        beta = float(np.cov(ar, mr, ddof=0)[0, 1]) / var if len(ar) > 2 else 0.0
        x["beta"] = float(np.mean(ar - beta * mr) / x["v"])
    names = ["failure", "beta", "consistency", "relative", "downside", "dispersion"]
    ranks = {n: rank_cs({s: x[n] for s, x in data.items()}) for n in names}
    score = {s: sum(w * ranks[n][s] for w, n in zip(FACTOR_WEIGHTS, names)) for s in UNIVERSE}
    if _previous is not None:
        score = {s: 0.80 * score[s] + 0.20 * _previous[s] for s in UNIVERSE}
    _previous = score.copy()
    vols = [x["v"] for x in data.values()]
    medvol = float(np.median(vols))
    breadth = float(np.mean([x["consistency"] > 0 for x in data.values()]))
    market20 = float(np.prod(1 + mr) - 1)
    stressed = medvol > 0.015 or market20 < -0.06 or breadth < 0.40
    invmean = float(np.mean([1 / max(v, 0.006) for v in vols]))
    raw = {}
    for s in UNIVERSE:
        x = data.get(s)
        vol = x["v"] if x else medvol
        damp = np.clip((1 / max(vol, 0.006)) / max(invmean, 1e-12), 0.70, 1.15)
        tilt = (2.60 if s in DEFENSIVE else (0.18 if s in RISKY else 0.60)) if stressed else (1.15 if s in DEFENSIVE else (0.82 if s in RISKY else 1.0))
        raw[s] = max(score[s], 0.03) * (0.85 + 0.15 * damp) * tilt
    target = full_weights(raw)
    forecast_returns = {s: float(np.clip((score[s] - 0.5) * 0.06, -0.03, 0.03)) for s in UNIVERSE}
    rebalance_to_weights(target, forecast_returns=forecast_returns, factor_ids=FACTOR_IDS)
