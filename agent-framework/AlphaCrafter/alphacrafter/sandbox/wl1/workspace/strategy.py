import numpy as np
from alphacrafter.sim.utils import register_hook, get_stock_daily_data, get_index_daily_data, rebalance_to_weights

UNIVERSE = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
FACTOR_IDS = [
    "miner_1_20271202_trend_acceleration_20d",
    "miner_3_20280113_momentum_acceleration_volscaled20",
    "miner_2_20271007_reversal5_vol40",
    "miner_2_20261217_residual_beta_volscaled_20d",
    "miner_3_20261105_downside_risk_adjusted_momentum",
    "miner_3_20261119_stress_conditioned_residual_20d",
    "miner_1_20270325_vix_beta_penalized_momentum_20d",
    "miner_3_20270603_common_beta_40d",
]
FACTOR_WEIGHTS = np.array([.18, .16, .16, .14, .13, .10, .08, .05])
_days_to_rebalance = 0


def rank(values, names):
    good = sorted((float(values[s]), s) for s in names if np.isfinite(values.get(s, np.nan)))
    out = {s: .5 for s in UNIVERSE}
    for i, (_, s) in enumerate(good):
        out[s] = (i + 1.) / len(good)
    return out


def weights(score, vol, defensive):
    raw = {}
    for s in UNIVERSE:
        tilt = 1.0
        if defensive and s in ("XAU", "US10Y", "CN10Y"): tilt = 1.85
        if defensive and s in ("BTC", "ETH", "WTI"): tilt = .25
        raw[s] = max(.010, (.70 + score[s]) ** 1.15 * tilt / max(vol.get(s, .02), .012) ** .35)
    cap = .11 if defensive else .15
    w = {s: raw[s] / sum(raw.values()) for s in UNIVERSE}
    for _ in range(40):
        over = [s for s in UNIVERSE if w[s] > cap]
        if not over: break
        excess = sum(w[s] - cap for s in over)
        for s in over: w[s] = cap
        free = [s for s in UNIVERSE if s not in over]
        den = sum(w[s] for s in free)
        if not free or den <= 0: break
        for s in free: w[s] += excess * w[s] / den
    total = sum(w.values())
    return {s: max(0., w[s] / total) for s in UNIVERSE}


@register_hook
def cross_asset_strategy():
    global _days_to_rebalance
    if _days_to_rebalance:
        _days_to_rebalance -= 1
        return
    closes, rets, vol = {}, {}, {}
    for s in UNIVERSE:
        df = get_stock_daily_data(symbol=s, days=230)
        if df is None or len(df) < 80: continue
        c = np.asarray(df.sort_values("date")["close"], dtype=float)
        if len(c) >= 65 and np.all(np.isfinite(c)) and np.all(c > 0):
            closes[s] = c
            rets[s] = c[1:] / c[:-1] - 1.
            vol[s] = max(float(np.std(rets[s][-30:])), .008)
    names = [s for s in UNIVERSE if s in closes]
    if len(names) < 12: return
    r5 = {s: closes[s][-1] / closes[s][-6] - 1. for s in names}
    r20 = {s: closes[s][-1] / closes[s][-21] - 1. for s in names}
    r40 = {s: closes[s][-1] / closes[s][-41] - 1. for s in names}
    peer = np.mean([rets[s][-20:] for s in names], axis=0)
    vix_state = 0.
    vd = get_index_daily_data(symbol="VIX", days=45)
    if vd is not None and len(vd) >= 22:
        vc = np.asarray(vd.sort_values("date")["close"], dtype=float)
        vix_state = vc[-1] / max(float(np.median(vc[-21:])), 1e-9) - 1.
    breadth = np.mean([r20[s] > 0 for s in names])
    market = r20.get("SPX", np.median(list(r20.values())))
    defensive = market < 0 or breadth < .45 or np.mean(list(vol.values())) > .022 or vix_state > .15
    f = [dict() for _ in FACTOR_IDS]
    for s in names:
        d = rets[s]
        down = max(float(np.std(np.minimum(d[-40:], 0.))), .006)
        beta = float(np.cov(d[-20:], peer)[0, 1] / max(np.var(peer), 1e-8))
        peer20 = np.median([r20[x] for x in names if x != s])
        residual = r20[s] - beta * peer20
        # Trend acceleration and its volatility-scaled version.
        accel = r20[s] - r40[s] / 2.
        f[0][s] = accel / max(vol[s], .01)
        f[1][s] = accel / max(vol[s] * np.sqrt(20.), .02)
        # Keep reversal deliberately modest in the final sizing via its ensemble weight.
        f[2][s] = -r5[s] / max(float(np.std(d[-10:])), .006)
        f[3][s] = residual / max(vol[s] * (1. + abs(beta)), .01)
        f[4][s] = r20[s] / down
        f[5][s] = residual * (1. + .5 * max(vix_state, 0.))
        f[6][s] = r20[s] / max(vol[s] * (1. + max(beta, 0.) * max(vix_state, 0.)), .01)
        f[7][s] = (-beta if defensive else beta) + .20 * r40[s] / max(vol[s], .01)
    ranks = [rank(x, names) for x in f]
    score = {s: sum(w * q[s] for w, q in zip(FACTOR_WEIGHTS, ranks)) for s in UNIVERSE}
    if defensive:
        for s in ("XAU", "US10Y", "CN10Y"): score[s] += .10
        for s in ("BTC", "ETH", "WTI"): score[s] -= .09
    target = weights(score, vol, defensive)
    forecast = {s: .008 * (score[s] - .5) for s in UNIVERSE}
    rebalance_to_weights(target, forecast_returns=forecast, factor_ids=FACTOR_IDS, horizon_days=10)
    _days_to_rebalance = 9

assert len(UNIVERSE) == 15 and len(FACTOR_IDS) <= 10
assert not set(UNIVERSE).intersection({"DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"})
assert abs(sum(weights({s:.5 for s in UNIVERSE}, {}, True).values()) - 1.) < 1e-8
