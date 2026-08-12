import numpy as np
from alphacrafter.sim.utils import (register_hook, get_stock_daily_data,
    get_index_daily_data, get_account_dict, rebalance_to_weights)

UNIVERSE = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
FACTOR_IDS = [
    "miner_1_20281102_defensive_relative_strength_20d",
    "miner_1_20281116_defensive_relative_lead_20d",
    "miner_3_20280907_downside_asymmetry_quality_30d",
    "breadth_vol_quality_40d", "macro_stress_resilience_20d",
    "miner_1_20280629_stable_volatility_trend_20d",
    "miner_2_20290503_path_efficiency_30d",
    "miner_1_20281005_vix_shock_resilient_momentum_20d",
    "miner_2_20271007_reversal5_vol40"]
FACTOR_WEIGHTS = np.array([.18, .16, .16, .14, .10, .10, .08, .05, .03])
_gate = 0


def cs_rank(values, names):
    good = sorted((float(values[s]), s) for s in names if s in values and np.isfinite(values[s]))
    out = {s: .5 for s in UNIVERSE}
    for i, (_, s) in enumerate(good):
        out[s] = (i + 1.) / max(len(good), 1)
    return out


def make_weights(score, vol, stressed):
    tilt = {s: 1.0 for s in UNIVERSE}
    if stressed:
        tilt.update({"XAU": 2.8, "US10Y": 2.3, "CN10Y": 2.1,
                     "BTC": .20, "ETH": .20, "WTI": .50, "COPPER": .55})
    raw = {s: tilt[s] * (.20 + score[s]) / max(vol[s], .008) ** .25 for s in UNIVERSE}
    cap = .16 if stressed else .19
    w = {s: max(raw[s], .001) for s in UNIVERSE}
    # Iterative cap-and-redistribute keeps the target diversified and normalized.
    for _ in range(30):
        over = [s for s in UNIVERSE if w[s] > cap]
        if not over:
            break
        excess = sum(w[s] - cap for s in over)
        for s in over:
            w[s] = cap
        free = [s for s in UNIVERSE if s not in over]
        den = sum(w[s] for s in free)
        if not free or den <= 0:
            break
        for s in free:
            w[s] += excess * w[s] / den
    total = sum(w.values())
    return {s: w[s] / total for s in UNIVERSE}


@register_hook
def cross_asset_strategy():
    global _gate
    if _gate:
        _gate -= 1
        return
    prices, returns, vols = {}, {}, {}
    for s in UNIVERSE:
        df = get_stock_daily_data(symbol=s, days=280)
        if df is None or len(df) < 100:
            continue
        c = np.asarray(df.sort_values("date")["close"], dtype=float)[:-1]  # lag one completed day
        if len(c) >= 80 and np.all(np.isfinite(c)) and np.all(c > 0):
            prices[s] = c
            returns[s] = c[1:] / c[:-1] - 1.
            vols[s] = max(float(np.std(returns[s][-40:])), .008)
    names = [s for s in UNIVERSE if s in prices]
    if len(names) < 12:
        return
    horizons = (5, 20, 30, 40, 60)
    ret = {s: {h: prices[s][-1] / prices[s][-(h + 1)] - 1. for h in horizons} for s in names}
    breadth = np.mean([ret[s][20] > 0 for s in names])
    market20 = ret.get("SPX", {20: np.median([ret[s][20] for s in names])})[20]
    vlevel, vshock = 0., 0.
    vd = get_index_daily_data(symbol="VIX", days=55)
    if vd is not None and len(vd) >= 23:
        vc = np.asarray(vd.sort_values("date")["close"], dtype=float)[:-1]
        if len(vc) >= 22 and np.all(np.isfinite(vc)):
            vlevel = float(vc[-1]); vshock = vlevel / max(float(np.median(vc[-21:])), 1e-9) - 1.
    stressed = market20 < -.05 or breadth < .40 or np.mean(list(vols.values())) > .022 or vlevel >= 20 or vshock > .15
    peer = np.mean([returns[s][-20:] for s in names], axis=0)
    fac = [{} for _ in FACTOR_IDS]
    for s in names:
        peer20 = np.median([ret[t][20] for t in names if t != s])
        beta = float(np.cov(returns[s][-20:], peer)[0, 1] / max(np.var(peer), 1e-8))
        residual = (ret[s][20] - beta * peer20) / max(vols[s], .01)
        neg = returns[s][-30:][returns[s][-30:] < 0]
        downvol = max(float(np.std(neg)) if len(neg) > 2 else .01, .006)
        downmean = float(np.mean(neg)) if len(neg) else 0.
        # Ordered to match the screener ensemble: relative defense, downside quality,
        # breadth/low risk, stress, stable trend, path efficiency, conditional momentum, reversal.
        fac[0][s] = ret[s][20] - peer20 - .35 * max(-ret[s][20], 0.)
        fac[1][s] = (ret[s][20] - peer20) + .35 * (ret[s][40] - peer20)
        fac[2][s] = (.55 * ret[s][30] + .45 * ret[s][60]) / downvol - downmean / downvol
        fac[3][s] = (.45 * ret[s][40] + .35 * ret[s][20] + .20 * ret[s][5]) / max(vols[s], .01)
        fac[4][s] = residual - .55 * max(-ret[s][20], 0.) / downvol
        fac[5][s] = (.50 * ret[s][20] + .30 * ret[s][40] + .20 * ret[s][60]) / max(vols[s], .01)
        path = np.mean(np.abs(returns[s][-30:]))
        fac[6][s] = ret[s][30] / max(path, .005)
        fac[7][s] = residual * (1. + max(vshock, 0.))
        fac[8][s] = -ret[s][5] / max(vols[s], .01)
    ranks = [cs_rank(f, names) for f in fac]
    score = {s: sum(w * r[s] for w, r in zip(FACTOR_WEIGHTS, ranks)) for s in UNIVERSE}
    if stressed:
        for s, a in (("XAU", .10), ("US10Y", .08), ("CN10Y", .06)):
            score[s] += a
        for s in ("BTC", "ETH", "WTI", "COPPER"):
            score[s] -= .05
    target = make_weights(score, {s: vols.get(s, .02) for s in UNIVERSE}, stressed)
    account = get_account_dict(); assets = max(float(account.get("total_assets", 0.)), 1.)
    old = {s: 0. for s in UNIVERSE}
    for p in account.get("positions", []):
        s = p.get("symbol")
        if s in old and float(p.get("quantity", 0)) > 0:
            old[s] = max(float(p.get("market_value", 0.)), 0.) / assets
    if sum(old.values()) > .001:
        target = {s: .25 * target[s] + .75 * old[s] for s in UNIVERSE}
        z = sum(target.values()); target = {s: target[s] / z for s in UNIVERSE}
    forecast = {s: .01 * (score[s] - .5) for s in UNIVERSE}
    rebalance_to_weights(target, forecast_returns=forecast, factor_ids=FACTOR_IDS, horizon_days=10)
    _gate = 9

cross_asset_strategy
