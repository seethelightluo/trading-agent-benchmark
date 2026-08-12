import numpy as np
from alphacrafter.sim.utils import (register_hook, get_stock_daily_data,
    get_index_daily_data, get_account_dict, rebalance_to_weights)

UNIVERSE = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
FACTOR_IDS = [
    "miner_3_20280907_downside_asymmetry_quality_30d",
    "stable_asymmetry_40_60",
    "miner_1_20281116_defensive_relative_lead_20d",
    "miner_1_20281102_defensive_relative_strength_20d",
    "macro_stress_resilience_20d",
    "breadth_vol_quality_40d",
    "miner_1_20281005_vix_shock_resilient_momentum_20d",
    "miner_1_20280629_stable_volatility_trend_20d"]
FACTOR_WEIGHTS = np.array([.18, .15, .15, .12, .12, .11, .09, .08])
_gate = 0

def _rank(values, names):
    good = sorted((float(values.get(s, np.nan)), s) for s in names if np.isfinite(values.get(s, np.nan)))
    out = {s: .5 for s in UNIVERSE}
    for i, (_, s) in enumerate(good):
        out[s] = (i + 1.) / max(len(good), 1)
    return out

def _weights(score, vol, stressed):
    tilt = {s: 1. for s in UNIVERSE}
    if stressed:
        tilt.update({"XAU": 2.8, "US10Y": 2.3, "CN10Y": 2.1,
                     "BTC": .20, "ETH": .20, "WTI": .50, "COPPER": .55})
    raw = {s: tilt[s] * (.20 + score[s]) / max(vol.get(s, .02), .008) ** .25 for s in UNIVERSE}
    cap = .16 if stressed else .19
    w = {s: max(raw[s], .001) for s in UNIVERSE}
    for _ in range(20):
        over = [s for s in UNIVERSE if w[s] > cap]
        if not over: break
        excess = sum(w[s] - cap for s in over)
        for s in over: w[s] = cap
        free = [s for s in UNIVERSE if s not in over]
        den = sum(w[s] for s in free)
        if not free or den <= 0: break
        for s in free: w[s] += excess * w[s] / den
    z = sum(w.values())
    return {s: w[s] / z for s in UNIVERSE}

@register_hook
def cross_asset_strategy():
    global _gate
    if _gate:
        _gate -= 1
        return
    prices, returns, vol = {}, {}, {}
    for s in UNIVERSE:
        d = get_stock_daily_data(symbol=s, days=280)
        if d is None or len(d) < 100: continue
        c = np.asarray(d.sort_values("date")["close"], dtype=float)[:-1]
        if len(c) >= 80 and np.all(np.isfinite(c)) and np.all(c > 0):
            prices[s] = c; returns[s] = c[1:] / c[:-1] - 1.
            vol[s] = max(float(np.std(returns[s][-40:])), .008)
    names = [s for s in UNIVERSE if s in prices]
    if len(names) < 12: return
    horizons = (5, 20, 30, 40, 60)
    r = {s: {h: prices[s][-1] / prices[s][-(h + 1)] - 1. for h in horizons} for s in names}
    breadth = np.mean([r[s][20] > 0 for s in names])
    market20 = r["SPX"][20] if "SPX" in r else float(np.median([r[s][20] for s in names]))
    vlevel, shock = 0., 0.
    vd = get_index_daily_data(symbol="VIX", days=55)
    if vd is not None and len(vd) >= 23:
        vc = np.asarray(vd.sort_values("date")["close"], dtype=float)[:-1]
        if len(vc) >= 22 and np.all(np.isfinite(vc)):
            vlevel = float(vc[-1]); shock = vlevel / max(float(np.median(vc[-21:])), 1e-9) - 1.
    stressed = market20 < -.05 or breadth < .40 or np.mean(list(vol.values())) > .022 or vlevel >= 20 or shock > .15
    peer = np.mean([returns[s][-20:] for s in names], axis=0)
    factors = [dict() for _ in FACTOR_IDS]
    for s in names:
        peers20 = [r[t][20] for t in names if t != s]
        peers40 = [r[t][40] for t in names if t != s]
        peer20, peer40 = np.median(peers20), np.median(peers40)
        beta = float(np.cov(returns[s][-20:], peer)[0, 1] / max(np.var(peer), 1e-8))
        residual = (r[s][20] - beta * peer20) / max(vol[s], .01)
        neg = returns[s][-30:][returns[s][-30:] < 0]
        downvol = max(float(np.std(neg)) if len(neg) > 2 else .01, .006)
        downmean = float(np.mean(neg)) if len(neg) else 0.
        path = max(float(np.mean(np.abs(returns[s][-30:]))), .005)
        factors[0][s] = (.55*r[s][30] + .45*r[s][60] - downmean) / downvol
        factors[1][s] = (.55*r[s][40] + .45*r[s][60]) / max(vol[s], .01)
        factors[2][s] = (r[s][20]-peer20) + .35*(r[s][40]-peer40)
        factors[3][s] = r[s][20]-peer20 - .35*max(-r[s][20], 0.)
        factors[4][s] = residual - .55*max(-r[s][20], 0.)/downvol
        factors[5][s] = (.45*r[s][40] + .35*r[s][20] + .20*r[s][5]) / max(vol[s], .01)
        factors[6][s] = residual * (1. + max(shock, 0.))
        factors[7][s] = (.50*r[s][20] + .30*r[s][40] + .20*r[s][60]) / max(vol[s], .01)
    ranks = [_rank(f, names) for f in factors]
    score = {s: sum(w*ranked[s] for w, ranked in zip(FACTOR_WEIGHTS, ranks)) for s in UNIVERSE}
    if stressed:
        for s, bonus in (("XAU", .10), ("US10Y", .08), ("CN10Y", .06)): score[s] += bonus
        for s in ("BTC", "ETH", "WTI", "COPPER"): score[s] -= .05
    target = _weights(score, vol, stressed)
    account = get_account_dict(); assets = max(float(account.get("total_assets", 0.)), 1.)
    old = {s: 0. for s in UNIVERSE}
    for p in account.get("positions", []):
        s = p.get("symbol")
        if s in old and float(p.get("quantity", 0)) > 0:
            old[s] = max(float(p.get("market_value", 0.)), 0.) / assets
    if sum(old.values()) > .001:
        target = {s: .25*target[s] + .75*old[s] for s in UNIVERSE}
        z = sum(target.values()); target = {s: target[s]/z for s in UNIVERSE}
    forecast = {s: .01*(score[s]-.5) for s in UNIVERSE}
    rebalance_to_weights(target, forecast_returns=forecast, factor_ids=FACTOR_IDS, horizon_days=10)
    _gate = 9

cross_asset_strategy
