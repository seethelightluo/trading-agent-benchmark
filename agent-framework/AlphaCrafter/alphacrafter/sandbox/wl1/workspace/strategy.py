import numpy as np
from alphacrafter.sim.utils import (
    register_hook, get_stock_daily_data, get_index_daily_data,
    rebalance_to_weights,
)

UNIVERSE = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
# Current screener ensemble: nine active factors, all positive direction.
FACTOR_IDS = [
    "miner_3_20261105_downside_risk_adjusted_momentum",
    "miner_3_20261119_stress_conditioned_residual_20d",
    "miner_2_20261217_residual_beta_volscaled_20d",
    "miner_1_20270325_vix_beta_penalized_momentum_20d",
    "miner_1_20270729_compression_confirmed_trend_20d",
    "miner_2_20261231_corr_adjusted_trend_20d",
    "miner_3_20270603_common_beta_40d",
    "miner_1_20260730_vix_state_reversal_5d",
    "miner_2_20270923_short_reversal_volscaled",
]
FACTOR_WEIGHTS = np.array([.20, .17, .15, .13, .10, .09, .04, .07, .05])
_days_to_rebalance = 0


def rank_factor(values, symbols):
    valid = sorted((float(values[s]), s) for s in symbols if np.isfinite(values.get(s, np.nan)))
    out = {s: .5 for s in UNIVERSE}
    for i, (_, s) in enumerate(valid):
        out[s] = (i + 1.0) / max(len(valid), 1)
    return out


def make_weights(score, vol, defensive):
    raw = {}
    for s in UNIVERSE:
        tilt = 1.0
        if defensive and s in ("XAU", "US10Y", "CN10Y"):
            tilt = 1.45
        elif defensive and s in ("BTC", "ETH", "WTI"):
            tilt = .45
        raw[s] = max(.01, (.58 + score[s]) * tilt / max(vol.get(s, .02), .012))
    cap = .12 if defensive else .15
    w = {s: raw[s] / sum(raw.values()) for s in UNIVERSE}
    for _ in range(30):
        over = [s for s in UNIVERSE if w[s] > cap]
        if not over:
            break
        excess = sum(w[s] - cap for s in over)
        for s in over:
            w[s] = cap
        free = [s for s in UNIVERSE if s not in over]
        denom = sum(w[s] for s in free)
        for s in free:
            w[s] += excess * w[s] / max(denom, 1e-12)
    total = sum(w.values())
    return {s: w[s] / total for s in UNIVERSE}


@register_hook
def cross_asset_strategy():
    global _days_to_rebalance
    if _days_to_rebalance:
        _days_to_rebalance -= 1
        return
    close, returns, vol = {}, {}, {}
    for s in UNIVERSE:
        df = get_stock_daily_data(symbol=s, days=230)
        if df is None or len(df) < 80:
            continue
        c = np.asarray(df.sort_values("date")["close"], dtype=float)
        if np.all(np.isfinite(c)) and np.all(c > 0):
            close[s] = c
            returns[s] = c[1:] / c[:-1] - 1.0
            vol[s] = max(float(np.std(returns[s][-30:])), .008)
    symbols = [s for s in UNIVERSE if s in close]
    if len(symbols) < 12:
        return
    r5 = {s: close[s][-1] / close[s][-6] - 1 for s in symbols}
    r20 = {s: close[s][-1] / close[s][-21] - 1 for s in symbols}
    r40 = {s: close[s][-1] / close[s][-41] - 1 for s in symbols}
    r60 = {s: close[s][-1] / close[s][-61] - 1 for s in symbols}
    peer = np.mean([returns[s][-20:] for s in symbols], axis=0)
    vix_state = 0.0
    vd = get_index_daily_data(symbol="VIX", days=45)
    if vd is not None and len(vd) >= 22:
        vc = np.asarray(vd.sort_values("date")["close"], dtype=float)
        vix_state = vc[-1] / max(float(np.median(vc[-21:])), 1e-9) - 1
    breadth = np.mean([r20[s] > 0 for s in symbols])
    defensive = (r20.get("SPX", 0) < 0 or breadth < .45 or np.mean(list(vol.values())) > .022 or vix_state > .15)
    f = [dict() for _ in FACTOR_IDS]
    for s in symbols:
        d = returns[s]
        downside = max(float(np.std(np.minimum(d[-40:], 0))), .006)
        beta = float(np.cov(d[-20:], peer)[0, 1] / max(np.var(peer), 1e-8))
        peer_r20 = np.median([r20[x] for x in symbols if x != s])
        residual = r20[s] - beta * peer_r20
        f[0][s] = r20[s] / downside
        f[1][s] = residual * (1 + .5 * max(vix_state, 0))
        f[2][s] = residual / max(vol[s] * (1 + abs(beta)), .01)
        f[3][s] = r20[s] / max(vol[s] * (1 + max(beta, 0) * max(vix_state, 0)), .01)
        short_vol = max(float(np.std(d[-10:])), .006)
        f[4][s] = r20[s] / max(vol[s], .01) * (1 + .35 * (short_vol < vol[s]))
        f[5][s] = (r20[s] - .5 * r60[s]) / max(vol[s] * (1 + abs(beta)), .01)
        f[6][s] = (-beta if defensive else beta) + .25 * r40[s] / max(vol[s], .01)
        f[7][s] = -r5[s] * (1 + max(vix_state, 0))
        # Newly admitted short reversal is deliberately capped at 5%.
        f[8][s] = -r5[s] / max(vol[s], .01)
    ranks = [rank_factor(x, symbols) for x in f]
    score = {s: sum(w * q[s] for w, q in zip(FACTOR_WEIGHTS, ranks)) for s in UNIVERSE}
    if defensive:
        for s in ("XAU", "US10Y", "CN10Y"):
            score[s] += .06
        for s in ("BTC", "ETH", "WTI"):
            score[s] -= .05
    target = make_weights(score, vol, defensive)
    forecast = {s: .012 * (score[s] - .5) for s in UNIVERSE}
    rebalance_to_weights(target, forecast_returns=forecast, factor_ids=FACTOR_IDS, horizon_days=10)
    _days_to_rebalance = 9
