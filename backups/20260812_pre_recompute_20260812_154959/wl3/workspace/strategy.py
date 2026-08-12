import numpy as np
from alphacrafter.sim.utils import register_hook, get_stock_daily_data, rebalance_to_weights

UNIVERSE = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
DEFENSIVE = {"XAU", "US10Y", "CN10Y"}
RISKY = {"BTC", "ETH", "WTI", "COPPER"}
# Exact active screener ensemble, with persisted IDs and directions.
FACTOR_IDS = [
    "miner_3_20270211_volstate_reversal_3d",
    "miner_2_20280907_breakout_failure_reversal",
    "miner_2_20270603_medium_dispersion_reversal",
    "miner_3_20280601_beta_residual_momentum20",
    "miner_2_20280615_volmanaged_consistency30",
    "miner_1_20280601_relative_strength20",
    "miner_2_20280727_breakout_distance120",
    "miner_3_20280504_downside_adjusted_momentum20",
]
FACTOR_WEIGHTS = [0.20, 0.15, 0.12, 0.18, 0.15, 0.12, 0.05, 0.03]
CADENCE = 10
_day = 0
_previous = None


def pct_rank(values):
    out = {s: 0.5 for s in UNIVERSE}
    good = sorted((s, float(v)) for s, v in values.items() if np.isfinite(v))
    n = max(1, len(good))
    for i, (s, _) in enumerate(good):
        out[s] = (i + 1.0) / n
    return out


def bounded(raw):
    # Full-investment, long-only weights; bounds reduce cross-asset concentration.
    lo, hi = 0.035, 0.17
    free, fixed = set(UNIVERSE), {}
    vals = {s: max(float(raw.get(s, 1.0)), 1e-12) for s in UNIVERSE}
    for _ in range(40):
        if not free:
            break
        rem = 1.0 - sum(fixed.values())
        z = sum(vals[s] for s in free)
        p = {s: rem * vals[s] / max(z, 1e-12) for s in free}
        low = [s for s in free if p[s] < lo]
        high = [s for s in free if p[s] > hi]
        if not low and not high:
            fixed.update(p)
            break
        for s in low:
            fixed[s] = lo
            free.remove(s)
        for s in high:
            fixed[s] = hi
            free.remove(s)
    if free:
        rem = 1.0 - sum(fixed.values())
        z = sum(vals[s] for s in free)
        fixed.update({s: rem * vals[s] / max(z, 1e-12) for s in free})
    total = sum(fixed.values())
    return {s: fixed.get(s, 0.0) / max(total, 1e-12) for s in UNIVERSE}


@register_hook
def cross_asset_strategy():
    global _day, _previous
    _day += 1
    if _day != 1 and (_day - 1) % CADENCE != 0:
        return
    series = {}
    for symbol in UNIVERSE:
        df = get_stock_daily_data(symbol=symbol, days=270)
        if df is None or len(df) < 140:
            continue
        close = np.asarray(df.sort_values("date")["close"], dtype=float)[:-1]
        if len(close) < 125 or np.any(~np.isfinite(close)) or np.any(close <= 0):
            continue
        r = close[1:] / close[:-1] - 1.0
        v20 = max(float(np.std(r[-20:])), 0.006)
        v30 = max(float(np.std(r[-30:])), 0.006)
        v60 = max(float(np.std(r[-60:])), 0.006)
        neg = r[-30:][r[-30:] < 0]
        downside = max(float(np.std(neg)) if len(neg) else 0.006, 0.006)
        r3 = float(np.sum(r[-3:]))
        r20 = float(np.prod(1.0 + r[-20:]) - 1.0)
        r30 = float(np.prod(1.0 + r[-30:]) - 1.0)
        series[symbol] = {"r": r, "v20": v20, "consistency": (float(np.mean(r[-30:] > 0)) - .5) / v30,
            "volstate": -r3 * (v20 / v60) / v20,
            "failure": -r20 / v60, "dispersion": -r20 / v30,
            "relative": r20 / v20, "downside": r30 / downside,
            "breakout": close[-1] / max(close[-121:-1]) - 1.0}
    if len(series) < 10:
        return
    market = series.get("000300.SH", series.get("SPX"))
    if market is None:
        return
    mr = market["r"][-20:]
    variance = max(float(np.var(mr)), 1e-8)
    for symbol, x in series.items():
        ar = x["r"][-20:]
        beta = float(np.cov(ar, mr, ddof=0)[0, 1] / variance)
        x["beta_mom"] = float(np.mean(ar - beta * mr)) / x["v20"]
    names = ["volstate", "failure", "dispersion", "beta_mom", "consistency", "relative", "breakout", "downside"]
    rr = {n: pct_rank({s: x[n] for s, x in series.items()}) for n in names}
    score = {s: sum(w * rr[n].get(s, .5) for n, w in zip(names, FACTOR_WEIGHTS)) for s in UNIVERSE}
    if _previous is not None:
        score = {s: .70 * score[s] + .30 * _previous[s] for s in UNIVERSE}
    _previous = score.copy()
    vols = [x["v20"] for x in series.values()]
    medvol = float(np.median(vols))
    breadth = float(np.mean([x["consistency"] > 0 for x in series.values()]))
    stressed = medvol > .015 or float(np.prod(1.0 + mr) - 1.0) < -.06 or breadth < .40
    invmean = float(np.mean([1.0 / max(x["v20"], .006) for x in series.values()]))
    raw = {}
    for s in UNIVERSE:
        x = series.get(s)
        vol = x["v20"] if x else medvol
        damp = np.clip((1.0 / max(vol, .006)) / max(invmean, 1e-12), .78, 1.10)
        raw[s] = max(score[s], .02) * (.88 + .12 * damp)
        raw[s] *= (2.15 if s in DEFENSIVE else .30 if s in RISKY else .75) if stressed else (1.10 if s in DEFENSIVE else .86 if s in RISKY else 1.0)
    target = bounded(raw)
    # Deterministic 10-day forecast is supplied for every tradable asset.
    forecast_returns = {s: float(np.clip((score[s] - .5) * .08, -.04, .04)) for s in UNIVERSE}
    if all(np.isfinite(target[s]) and target[s] >= 0 for s in UNIVERSE) and abs(sum(target.values()) - 1.0) < 1e-8:
        rebalance_to_weights(target, forecast_returns=forecast_returns, factor_ids=FACTOR_IDS)
