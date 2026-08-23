import json
from pathlib import Path
import numpy as np
from alphacrafter.sim.utils import register_hook, get_stock_daily_data, rebalance_to_weights

U = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
DEF = {"XAU", "US10Y", "CN10Y"}
_skip = 0

def ensemble():
    try:
        d = json.loads((Path(__file__).parent / "factors" / "factor_ensemble.json").read_text())
        fs = d.get("selected_factors", [])
        return fs if 0 < len(fs) <= 10 and abs(sum(float(f["weight"]) for f in fs)-1) < 1e-6 else []
    except Exception:
        return []

def rank(v):
    a = sorted(U, key=lambda s: (float(v[s]), s))
    return {s: (i + 1) / len(U) - .5 for i, s in enumerate(a)}

def allocate(raw):
    lo, hi = .03, .18
    w = {s: lo for s in U}; free = set(U); left = 1.0 - lo * len(U)
    while free:
        z = sum(max(float(raw[s]), 1e-10) for s in free)
        add = {s: left * max(float(raw[s]), 1e-10) / z for s in free}
        cap = {s for s in free if add[s] > hi - lo}
        if not cap:
            for s in free: w[s] += add[s]
            break
        for s in cap: w[s] = hi
        left -= (hi - lo) * len(cap); free -= cap
    z = sum(w.values())
    return {s: w[s] / z for s in U}

@register_hook
def cross_asset_strategy():
    global _skip
    if _skip:
        _skip -= 1
        return
    fs = ensemble()
    if not fs: return
    p = {}
    for s in U:
        d = get_stock_daily_data(symbol=s, days=280)
        if d is None or len(d) < 125: return
        p[s] = np.asarray(d.sort_values("date")["close"], dtype=float)
    def ret(s, n): return p[s][-1] / max(p[s][-n-1], 1e-12) - 1.0
    r5 = {s: ret(s, 5) for s in U}; r20 = {s: ret(s, 20) for s in U}
    r60 = {s: ret(s, 60) for s in U}; r120 = {s: ret(s, 120) for s in U}
    lr = {s: np.diff(np.log(np.maximum(p[s], 1e-12))) for s in U}
    v20 = {s: max(float(np.std(lr[s][-20:])), .008) for s in U}
    down40 = {s: max(float(np.sqrt(np.mean(np.minimum(lr[s][-40:], 0.0)**2)) * np.sqrt(40)), .01) for s in U}
    med20 = float(np.median(list(r20.values()))); breadth = float(np.mean([r20[s] > 0 for s in U]))
    gate = 1.35 if breadth < .5 else (.75 if breadth > .7 else 1.0)
    market = np.mean(np.array([lr[s][-120:] for s in U]), axis=0)
    m20 = float(np.sum(market[-20:])); mstd = max(float(np.std(market[-60:])), .004)
    sig = {}
    sig["miner_2_20300207_defensive_relative_strength_20d"] = {s:(r20[s]-med20)/v20[s]*gate*(1 if r60[s]>0 else .55) for s in U}
    sig["miner_3_20290712_inverse_risk_adjusted_trend_10d"] = {s:-(r120[s] / max(v20[s] * np.sqrt(252), .01)) for s in U}
    sig["miner_1_20271118_macro_relative_strength_reversal"] = {s:-(r20[s]-float(np.mean(list(r20.values()))))/v20[s] for s in U}
    sig["miner_1_20281019_downside_risk_adjusted_reversal"] = {s:-(.7*r60[s]+.3*r20[s])/down40[s] for s in U}
    sig["miner_2_20270923_volnorm_reversal"] = {s:-r5[s]/max(v20[s]*np.sqrt(20), .01) for s in U}
    residual = {}
    for s in U:
        y, x = lr[s][-80:], market[-80:]
        beta = float(np.cov(y, x, ddof=1)[0, 1] / max(np.var(x, ddof=1), 1e-10))
        e = y - beta*x
        residual[s] = -sum(e[-20:]) / max(float(np.std(e[-60:])), .006)
    sig["miner_2_20320805_bearish_state_residual_reversal"] = {s:residual[s]*(1+.5*np.tanh(-m20/(mstd*np.sqrt(20)))) for s in U}
    score = {s: 0.0 for s in U}; ids = []
    for f in fs:
        fid = str(f["factor_id"])
        if fid not in sig: return
        ids.append(fid); rr = rank(sig[fid])
        for s in U: score[s] += float(f["weight"]) * float(f.get("direction", 1)) * rr[s]
    risk = min(1.0, float(np.std(list(r20.values()))) / max(float(np.median(list(v20.values()))), .008) / 4.0)
    raw = {s:max(score[s] + .5, 1e-8) / max(v20[s] ** .60, 1e-6) for s in U}
    if risk > .45 or breadth < .5:
        for s in DEF: raw[s] *= 1.7 + 1.3 * risk
    target = allocate(raw)
    sd = max(float(np.std(list(score.values()))), 1e-9); avg = float(np.mean(list(score.values())))
    forecast = {s: float(.025 * (score[s] - avg) / sd) for s in U}
    for s in DEF: forecast[s] += .005 * (1 + risk)
    rebalance_to_weights(target, forecast_returns=forecast, factor_ids=ids, horizon_days=10)
    _skip = 9

# End: complete 15-asset, long-only target is always submitted on decision days.
