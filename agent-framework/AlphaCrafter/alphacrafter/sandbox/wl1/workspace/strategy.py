import numpy as np
from alphacrafter.sim.utils import register_hook, get_stock_daily_data, get_index_daily_data, rebalance_to_weights

UNIVERSE = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
# Current Screener ensemble: capped at ten active factors.
FACTOR_IDS = [
    "miner_2_20261231_corr_adjusted_trend_20d",
    "miner_3_20261105_downside_risk_adjusted_momentum",
    "miner_3_20261119_stress_conditioned_residual_20d",
    "miner_1_20260730_long_recovery_20d",
    "miner_1_20270325_vix_beta_penalized_momentum_20d",
    "miner_3_20270128_downside_asymmetry_30d",
    "miner_2_20261217_residual_beta_volscaled_20d",
    "miner_3_20270603_common_beta_40d",
    "miner_2_volatility_scaled_momentum_20d",
    "miner_1_20260730_vix_state_reversal_5d",
]
FACTOR_WEIGHTS = np.array([.14, .12, .12, .12, .10, .10, .08, .07, .05, .10])
_days_to_rebalance = 0


def ranks(x):
    out = {s: .5 for s in UNIVERSE}
    good = sorted((v, s) for s, v in x.items() if np.isfinite(v))
    for i, (_, s) in enumerate(good):
        out[s] = (i + 1.) / len(good)
    return out


def make_weights(score, vol, defensive):
    raw = {}
    for s in UNIVERSE:
        tilt = 1.
        if defensive and s in ("XAU", "US10Y", "CN10Y"):
            tilt = 1.45
        if defensive and s in ("BTC", "ETH", "WTI"):
            tilt = .55
        raw[s] = max(.006, (.65 + score.get(s, .5)) * tilt / max(vol.get(s, .02), .012))
    w = {s: raw[s] / sum(raw.values()) for s in UNIVERSE}
    cap = .12 if defensive else .15
    # Iterative capped redistribution preserves a complete, invested vector.
    for _ in range(50):
        over = [s for s in UNIVERSE if w[s] > cap + 1e-10]
        if not over: break
        excess = sum(w[s] - cap for s in over)
        for s in over: w[s] = cap
        free = [s for s in UNIVERSE if s not in over]
        base = sum(w[s] for s in free)
        for s in free: w[s] += excess * w[s] / max(base, 1e-12)
    z = sum(w.values())
    return {s: w[s] / z for s in UNIVERSE}


@register_hook
def cross_asset_strategy():
    global _days_to_rebalance
    if _days_to_rebalance:
        _days_to_rebalance -= 1
        return
    px, ret, vol = {}, {}, {}
    for s in UNIVERSE:
        df = get_stock_daily_data(symbol=s, days=230)
        if df is None or len(df) < 80: continue
        c = np.asarray(df.sort_values("date")["close"], dtype=float)
        if len(c) >= 80 and np.all(np.isfinite(c)) and np.all(c > 0):
            px[s] = c; ret[s] = c[1:] / c[:-1] - 1.
            vol[s] = max(float(np.std(ret[s][-30:])), .008)
    symbols = [s for s in UNIVERSE if s in px]
    if len(symbols) < 12: return
    r5 = {s: px[s][-1] / px[s][-6] - 1. for s in symbols}
    r20 = {s: px[s][-1] / px[s][-21] - 1. for s in symbols}
    r40 = {s: px[s][-1] / px[s][-41] - 1. for s in symbols}
    r60 = {s: px[s][-1] / px[s][-61] - 1. for s in symbols}
    peer = np.mean([ret[s][-20:] for s in symbols], axis=0)
    vix_state = 0.
    vd = get_index_daily_data(symbol="VIX", days=40)
    if vd is not None and len(vd) >= 22:
        vc = np.asarray(vd.sort_values("date")["close"], dtype=float)
        vix_state = vc[-1] / max(np.median(vc[-21:]), 1e-9) - 1.
    breadth = np.mean([r20[s] > 0 for s in symbols])
    defensive = (r20.get("SPX", 0.) < 0 or breadth < .45 or
                 np.mean(list(vol.values())) > .022 or vix_state > .15)
    fs = [dict() for _ in range(10)]
    for s in symbols:
        d = ret[s]; downside = max(float(np.std(np.minimum(d[-40:], 0))), .006)
        beta = float(np.cov(d[-20:], peer)[0, 1] / max(np.var(peer), 1e-8))
        residual = r20[s] - beta * np.median([r20[x] for x in symbols if x != s])
        recovery = r40[s] - .5 * min(px[s][-1] / max(np.max(px[s][-61:-1]), 1e-9) - 1., 0.)
        asym = -float(np.mean(np.minimum(d[-30:], 0))) / max(float(np.mean(np.maximum(d[-30:], 0))), .003)
        fs[0][s] = (r20[s] - .5 * r60[s]) / max(vol[s] * (1 + abs(beta)), .01)
        fs[1][s] = r20[s] / downside
        fs[2][s] = residual / max(vol[s], .01) * (1 + .5 * max(vix_state, 0))
        fs[3][s] = recovery
        fs[4][s] = r20[s] / vol[s] / (1 + max(beta, 0) * max(vix_state, 0))
        fs[5][s] = asym
        fs[6][s] = residual / max(vol[s] * (1 + abs(beta)), .01)
        fs[7][s] = -beta if defensive else beta
        fs[8][s] = r20[s] / max(vol[s], .01)
        fs[9][s] = -r5[s] * (1 + max(vix_state, 0))
    rr = [ranks(f) for f in fs]
    score = {s: sum(w * q[s] for w, q in zip(FACTOR_WEIGHTS, rr)) for s in UNIVERSE}
    if defensive:
        for s in ("XAU", "US10Y", "CN10Y"): score[s] += .06
        for s in ("BTC", "ETH", "WTI"): score[s] -= .04
    target = make_weights(score, vol, defensive)
    forecast = {s: .015 * (score[s] - .5) for s in UNIVERSE}
    rebalance_to_weights(target, forecast_returns=forecast, factor_ids=FACTOR_IDS, horizon_days=10)
    _days_to_rebalance = 9
