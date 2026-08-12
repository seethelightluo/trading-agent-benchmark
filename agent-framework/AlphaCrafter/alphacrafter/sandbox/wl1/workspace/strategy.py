import numpy as np
from alphacrafter.sim.utils import (register_hook, get_stock_daily_data,
    get_index_daily_data, get_account_dict, rebalance_to_weights)

U = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
F = [
    "miner_3_20280907_downside_asymmetry_quality_30d",
    "breadth_vol_quality_40d",
    "miner_1_20281116_defensive_relative_lead_20d",
    "miner_1_20281102_defensive_relative_strength_20d",
    "miner_1_20291227_short_reversal_volscaled_20d",
    "miner_1_20300124_drawdown_recovery_velocity",
    "miner_1_20281005_vix_shock_resilient_momentum_20d",
    "stable_asymmetry_40_60",
    "macro_stress_resilience_20d",
    "miner_3_20280727_multihorizon_consistency_20d",
]
FW = np.array([.18, .14, .11, .07, .11, .05, .07, .11, .05, .11])
_gate = 0

def rank_map(x, names):
    vals = sorted((v, s) for s, v in x.items() if s in names and np.isfinite(v))
    out = {s: .5 for s in U}
    for i, (_, s) in enumerate(vals):
        out[s] = (i + 1) / max(1, len(vals))
    return out

def full_weights(score, vol, stressed):
    tilt = {s: 1.0 for s in U}
    if stressed:
        tilt.update({"XAU": 2.8, "US10Y": 2.3, "CN10Y": 2.1,
                     "BTC": .20, "ETH": .20, "WTI": .50, "COPPER": .55})
    raw = {s: tilt[s] * (.20 + score[s]) / max(vol.get(s, .02), .008) ** .25 for s in U}
    cap = .16 if stressed else .19
    w, free, left = {s: 0.0 for s in U}, set(U), 1.0
    while free:
        den = sum(max(raw[s], .001) for s in free)
        hit = [s for s in free if left * max(raw[s], .001) / den > cap]
        if not hit:
            for s in free: w[s] = left * max(raw[s], .001) / den
            break
        for s in hit:
            w[s], left = cap, left - cap
            free.remove(s)
    z = sum(w.values())
    return {s: w[s] / z for s in U}

@register_hook
def cross_asset_strategy():
    global _gate
    if _gate:
        _gate -= 1
        return
    px, ret, vol = {}, {}, {}
    for s in U:
        d = get_stock_daily_data(symbol=s, days=280)
        if d is None or len(d) < 100: continue
        c = np.asarray(d.sort_values("date")["close"], float)[:-1]  # one-day lag
        if len(c) >= 80 and np.all(np.isfinite(c)) and np.all(c > 0):
            px[s] = c; ret[s] = c[1:] / c[:-1] - 1
            vol[s] = max(float(np.std(ret[s][-40:])), .008)
    names = [s for s in U if s in px]
    if len(names) < 12: return
    R = {s: {h: px[s][-1] / px[s][-(h + 1)] - 1 for h in (5, 20, 30, 40, 60)} for s in names}
    breadth = np.mean([R[s][20] > 0 for s in names])
    market20 = R.get("SPX", {20: np.median([R[s][20] for s in names])})[20]
    vix, shock = 0., 0.
    vd = get_index_daily_data(symbol="VIX", days=55)
    if vd is not None and len(vd) >= 23:
        vc = np.asarray(vd.sort_values("date")["close"], float)[:-1]
        if len(vc) >= 22 and np.all(np.isfinite(vc)):
            vix = float(vc[-1]); shock = vix / max(np.median(vc[-21:]), 1e-9) - 1
    stressed = market20 < -.05 or breadth < .40 or np.mean(list(vol.values())) > .022 or vix >= 20 or shock > .15
    peer = np.mean([ret[s][-20:] for s in names], axis=0)
    fs = [{} for _ in F]
    for s in names:
        p20 = np.median([R[t][20] for t in names if t != s]); p40 = np.median([R[t][40] for t in names if t != s])
        beta = np.cov(ret[s][-20:], peer)[0, 1] / max(np.var(peer), 1e-8)
        residual = (R[s][20] - beta * p20) / max(vol[s], .01)
        neg = ret[s][-30:][ret[s][-30:] < 0]
        dv = max(float(np.std(neg)) if len(neg) > 2 else .01, .006)
        dm = float(np.mean(neg)) if len(neg) else 0.
        fs[0][s] = (.55*R[s][30] + .45*R[s][60] - dm) / dv
        fs[1][s] = (.5*R[s][40] + .3*R[s][20] + .2*R[s][5]) / max(vol[s], .01)
        fs[2][s] = residual - .55*max(-R[s][20], 0)/dv
        fs[3][s] = (R[s][20]-p20) + .35*(R[s][40]-p40)
        fs[4][s] = -R[s][20] / max(vol[s], .01)
        fs[5][s] = max(R[s][20], 0) / max(1 + max(-R[s][60], 0), .2)
        fs[6][s] = residual * (1 + max(shock, 0))
        fs[7][s] = (.55*R[s][40] + .45*R[s][60]) / max(vol[s], .01)
        fs[8][s] = residual - .4*max(-R[s][20], 0)/dv
        fs[9][s] = (.5*R[s][20] + .3*R[s][40] + .2*R[s][60]) / max(vol[s], .01)
    rr = [rank_map(x, names) for x in fs]
    score = {s: sum(w*q[s] for w, q in zip(FW, rr)) for s in U}
    if stressed:
        for s, b in (("XAU", .10), ("US10Y", .08), ("CN10Y", .06)): score[s] += b
        for s in ("BTC", "ETH", "WTI", "COPPER"): score[s] -= .05
    target = full_weights(score, vol, stressed)
    a = get_account_dict(); assets = max(float(a.get("total_assets", 0)), 1.)
    old = {s: 0. for s in U}
    for p in a.get("positions", []):
        s = p.get("symbol")
        if s in old and float(p.get("quantity", 0)) > 0:
            old[s] = max(float(p.get("market_value", 0)), 0.) / assets
    if sum(old.values()) > .001:
        target = {s: (.20*target[s] + .80*old[s]) for s in U}
        z = sum(target.values()); target = {s: target[s]/z for s in U}
    forecast = {s: .01*(score[s]-.5) for s in U}
    rebalance_to_weights(target, forecast_returns=forecast, factor_ids=F, horizon_days=10)
    _gate = 9

cross_asset_strategy
