import numpy as np
from alphacrafter.sim.utils import (register_hook, get_stock_daily_data,
    get_index_daily_data, get_account_dict, rebalance_to_weights)

UNIVERSE = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
# 2028-11-30 screener ensemble: nine active, positive-direction factors.
FACTOR_IDS = [
    "miner_1_20280323_vix_conditioned_efficiency_trend",
    "miner_1_20280629_stable_volatility_trend_20d",
    "miner_1_20260730_vix_state_reversal_5d",
    "miner_3_20261119_stress_conditioned_residual_20d",
    "miner_3_20280907_downside_asymmetry_quality_30d",
    "miner_2_20261217_residual_beta_volscaled_20d",
    "miner_1_20281116_defensive_relative_lead_20d",
    "miner_1_20281130_residual_volscaled_trend_20d",
    "directional_payoff_efficiency_30d",
]
FACTOR_WEIGHTS = np.array([.18, .16, .14, .12, .12, .10, .08, .05, .05])
_gate = 0


def ranks(x, names):
    good = sorted((float(x.get(s, np.nan)), s) for s in names if np.isfinite(x.get(s, np.nan)))
    out = {s: .5 for s in UNIVERSE}
    for i, (_, s) in enumerate(good):
        out[s] = (i + 1) / max(len(good), 1)
    return out


def weights(score, vol, stress):
    # Full investment; defensive benchmarks absorb risk-off exposure.
    tilt = {s: 1.0 for s in UNIVERSE}
    if stress:
        tilt.update({"XAU": 4.0, "US10Y": 3.2, "CN10Y": 2.7,
                     "BTC": .08, "ETH": .08, "WTI": .20, "COPPER": .35})
    raw = {s: tilt[s] * (.18 + score[s]) / max(vol.get(s, .02), .008) ** .30 for s in UNIVERSE}
    w = {s: max(raw[s], .001) for s in UNIVERSE}
    cap = .16 if stress else .18
    for _ in range(40):
        over = [s for s in UNIVERSE if w[s] > cap]
        if not over: break
        excess = sum(w[s] - cap for s in over)
        for s in over: w[s] = cap
        free = [s for s in UNIVERSE if s not in over]
        z = sum(w[s] for s in free)
        for s in free: w[s] += excess * w[s] / max(z, 1e-12)
    z = sum(w.values())
    return {s: w[s] / z for s in UNIVERSE}


@register_hook
def cross_asset_strategy():
    global _gate
    if _gate:
        _gate -= 1
        return
    close, daily, vol = {}, {}, {}
    for s in UNIVERSE:
        df = get_stock_daily_data(symbol=s, days=250)
        if df is None or len(df) < 95: continue
        c = np.asarray(df.sort_values("date")["close"], dtype=float)[:-1]
        if len(c) >= 75 and np.all(np.isfinite(c)) and np.all(c > 0):
            close[s] = c; daily[s] = c[1:] / c[:-1] - 1.; vol[s] = max(float(np.std(daily[s][-30:])), .008)
    names = [s for s in UNIVERSE if s in close]
    if len(names) < 12: return
    ret = {s: {k: close[s][-1] / close[s][-(k + 1)] - 1. for k in (5, 10, 20, 60)} for s in names}
    peer = np.mean([daily[s][-20:] for s in names], axis=0)
    breadth = np.mean([ret[s][20] > 0 for s in names])
    market = ret.get("SPX", {20: np.median([ret[s][20] for s in names])})[20]
    vx = 0.
    vdf = get_index_daily_data(symbol="VIX", days=50)
    if vdf is not None and len(vdf) >= 23:
        vc = np.asarray(vdf.sort_values("date")["close"], dtype=float)[:-1]
        vx = float(vc[-1] / max(np.median(vc[-21:]), 1e-9) - 1.)
    stress = market < -.05 or breadth < .40 or np.mean(list(vol.values())) > .022 or vx > .15

    f = [{} for _ in FACTOR_IDS]
    for s in names:
        d = daily[s]; beta = float(np.cov(d[-20:], peer)[0, 1] / max(np.var(peer), 1e-8))
        peer20 = np.median([ret[t][20] for t in names if t != s])
        residual = ret[s][20] - beta * peer20
        recovery = (close[s][-1] - min(close[s][-61:-1])) / max(max(close[s][-61:-1]) - min(close[s][-61:-1]), 1e-9)
        efficiency = abs(ret[s][20]) / max(float(np.sum(abs(d[-20:]))), .01)
        downside = max(float(np.std(np.minimum(d[-40:], 0))), .006)
        # Factor proxies use lagged prices only and preserve screener directions.
        # Proxies are deliberately lagged and mapped one-for-one to the active ensemble.
        f[0][s] = efficiency * (1. + max(vx, 0.)) * np.sign(ret[s][20])
        f[1][s] = (.35 * ret[s][10] + .65 * ret[s][20]) / max(vol[s], .01)
        f[2][s] = (ret[s][5] - ret[s][20] / 4.) / max(vol[s], .01)
        f[3][s] = residual * (1. + .7 * max(vx, 0.))
        f[4][s] = residual / max(vol[s] * (1. + max(beta, 0.)), .01)
        # Downside-asymmetry quality rewards positive payoff with controlled losses.
        f[5][s] = (.45 * ret[s][20] + .35 * ret[s][60] + .20 * ret[s][10]) / max(vol[s], .01) - .50 * downside
        # Residual-beta factor: residual strength, penalized by common beta risk.
        f[6][s] = residual / max(vol[s] * (1. + abs(beta)), .01)
        # Defensive relative leadership combines recovery and relative trend.
        f[7][s] = .6 * recovery + .4 * (ret[s][20] - peer20) - .15 * downside
        # Newly admitted residual-volscaled trend is deliberately kept at 5%.
        f[8][s] = residual / max(vol[s], .01) + .35 * efficiency - .20 * downside
    rr = [ranks(x, names) for x in f]
    score = {s: sum(float(w) * r[s] for w, r in zip(FACTOR_WEIGHTS, rr)) for s in UNIVERSE}
    if stress:
        for s, add in (("XAU", .18), ("US10Y", .14), ("CN10Y", .11)): score[s] += add
        for s in ("BTC", "ETH", "WTI", "COPPER"): score[s] -= .12
    target = weights(score, vol, stress)
    account = get_account_dict(); assets = max(float(account.get("total_assets", 0)), 1.)
    old = {s: 0. for s in UNIVERSE}
    for p in account.get("positions", []):
        s = p.get("symbol")
        if s in old and float(p.get("quantity", 0)) > 0: old[s] = max(float(p.get("market_value", 0)), 0.) / assets
    if sum(old.values()) > .001:
        target = {s: .40 * target[s] + .60 * old[s] for s in UNIVERSE}
        z = sum(target.values()); target = {s: target[s] / z for s in UNIVERSE}
    forecast = {s: .01 * (score[s] - .5) for s in UNIVERSE}
    rebalance_to_weights(target, forecast_returns=forecast, factor_ids=FACTOR_IDS, horizon_days=10)
    _gate = 9
