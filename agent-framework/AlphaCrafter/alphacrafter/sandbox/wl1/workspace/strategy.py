import numpy as np
from alphacrafter.sim.utils import (register_hook, get_stock_daily_data,
    get_index_daily_data, rebalance_to_weights)

UNIVERSE = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
FACTOR_IDS = [
    "miner_1_20270325_vix_beta_penalized_momentum_20d",
    "miner_3_20261119_stress_conditioned_residual_20d",
    "miner_3_20261105_downside_risk_adjusted_momentum",
    "miner_1_20260730_vix_state_reversal_5d",
    "miner_2_20261217_residual_beta_volscaled_20d",
    "miner_2_20261203_drawdown_recovery_20d",
    "miner_3_20270128_downside_asymmetry_30d",
]
FACTOR_WEIGHTS = np.array([.18, .18, .16, .15, .14, .10, .09])
_gate = 0


def rank_factor(vals, syms):
    good = sorted((float(vals[s]), s) for s in syms if np.isfinite(vals.get(s, np.nan)))
    out = {s: .5 for s in syms}
    n = len(good)
    for i, (_, s) in enumerate(good):
        out[s] = (i + 1.0) / n if n else .5
    return out


def make_weights(score, vol, defensive):
    raw = {}
    for s in UNIVERSE:
        tilt = 1.0
        if defensive and s in ("XAU", "US10Y", "CN10Y"): tilt = 2.0
        if defensive and s in ("BTC", "ETH", "WTI"): tilt = .35
        raw[s] = max(.002, (.30 + score.get(s, .5)) * tilt / max(vol.get(s, .02), .012))
    w = {s: raw[s] / sum(raw.values()) for s in UNIVERSE}
    cap = .12 if defensive else .14
    for _ in range(20):
        over = [s for s in UNIVERSE if w[s] > cap]
        if not over: break
        excess = sum(w[s] - cap for s in over)
        for s in over: w[s] = cap
        free = [s for s in UNIVERSE if s not in over]
        base = sum(w[s] for s in free)
        for s in free: w[s] += excess * w[s] / max(base, 1e-12)
    z = sum(w.values())
    return {s: max(0.0, w[s] / z) for s in UNIVERSE}


@register_hook
def cross_asset_strategy():
    global _gate
    if _gate:
        _gate -= 1
        return
    prices, rets, vols = {}, {}, {}
    for s in UNIVERSE:
        df = get_stock_daily_data(symbol=s, days=230)
        if df is None or len(df) < 80: continue
        c = np.asarray(df.sort_values("date")["close"], dtype=float)
        if len(c) >= 80 and np.all(np.isfinite(c)) and np.all(c > 0):
            prices[s] = c
            rets[s] = c[1:] / c[:-1] - 1.0
            vols[s] = max(float(np.std(rets[s][-30:])), .008)
    syms = [s for s in UNIVERSE if s in prices]
    if len(syms) < 12: return
    r5 = {s: prices[s][-1] / prices[s][-6] - 1 for s in syms}
    r20 = {s: prices[s][-1] / prices[s][-21] - 1 for s in syms}
    r60 = {s: prices[s][-1] / prices[s][-61] - 1 for s in syms}
    peer = np.mean([rets[s][-20:] for s in syms], axis=0)
    vd = get_index_daily_data(symbol="VIX", days=35)
    vix_level, vix_jump = 0.0, 0.0
    if vd is not None and len(vd) >= 22:
        v = np.asarray(vd.sort_values("date")["close"], dtype=float)
        vix_level = max(0.0, v[-1] / max(np.median(v[-21:]), 1e-9) - 1.0)
        vix_jump = max(0.0, v[-1] / max(v[-6], 1e-9) - 1.0)
    breadth = np.mean([r20[s] > 0 for s in syms])
    defensive = (r20.get("SPX", 0) < 0 and r5.get("SPX", 0) < 0) or breadth < .45 or np.mean(list(vols.values())) > .022 or vix_level > .15
    fs = [dict() for _ in FACTOR_WEIGHTS]
    for s in syms:
        d = rets[s]
        downside = max(float(np.std(np.minimum(d[-40:], 0))), .006)
        beta = float(np.cov(d[-20:], peer)[0, 1] / max(np.var(peer), 1e-8))
        residual = r20[s] - beta * np.median([r20[x] for x in syms if x != s])
        fs[0][s] = r20[s] / vols[s] / (1.0 + max(vix_level, 0.0) * max(beta, 0.0))
        fs[1][s] = residual / vols[s]
        fs[2][s] = r20[s] / downside
        fs[3][s] = -r5[s] * (1.0 + vix_level) / vols[s]
        fs[4][s] = residual / max(vols[s], .01)
        fs[5][s] = (r20[s] - r60[s]) / vols[s] - max(r60[s], 0.0) / 2.0
        fs[6][s] = (np.mean(np.maximum(d[-30:], 0)) - np.mean(np.minimum(d[-30:], 0))) / downside
    score = {s: sum(w * rank_factor(f, syms)[s] for w, f in zip(FACTOR_WEIGHTS, fs)) for s in syms}
    if defensive:
        for s in ("XAU", "US10Y", "CN10Y"): score[s] = score.get(s, .5) + .15
        for s in ("BTC", "ETH", "WTI"): score[s] = score.get(s, .5) - .10
    target = make_weights(score, vols, defensive)
    forecast = {s: max(.0001, .03 * (score.get(s, .5) - .5)) for s in UNIVERSE}
    rebalance_to_weights(target, forecast_returns=forecast, factor_ids=FACTOR_IDS, horizon_days=10)
    _gate = 9
