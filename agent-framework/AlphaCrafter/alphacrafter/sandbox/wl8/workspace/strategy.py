import json
from pathlib import Path
import numpy as np
from alphacrafter.sim.utils import (register_hook, get_account_dict,
    get_stock_daily_data, get_index_daily_data, rebalance_to_weights)

ASSETS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
ENSEMBLE = Path(__file__).parent / "factors" / "factor_ensemble.json"
last_decision = None

def ranks(values):
    good = sorted((s, float(v)) for s, v in values.items() if np.isfinite(v))
    out = {s: .5 for s in ASSETS}
    for i, (s, _) in enumerate(good):
        out[s] = (i + 1) / max(1, len(good))
    return out

def bounded(raw):
    floor, cap = .015, .16
    z = {s: max(float(raw.get(s, .01)), 1e-9) for s in ASSETS}
    w = {s: floor + (1 - 15 * floor) * z[s] / sum(z.values()) for s in ASSETS}
    for _ in range(40):
        high = [s for s in ASSETS if w[s] > cap + 1e-12]
        if not high: break
        excess = sum(w[s] - cap for s in high)
        for s in high: w[s] = cap
        free = [s for s in ASSETS if s not in high]
        den = sum(z[s] for s in free)
        for s in free: w[s] += excess * z[s] / max(den, 1e-12)
    total = sum(w.values())
    return {s: float(w[s] / total) for s in ASSETS}

def vix_stress():
    d = get_index_daily_data(symbol="VIX", days=130)
    if d is None or len(d) < 61: return 0.0
    c = np.asarray(d.sort_values("date")["close"], float)
    base = c[:-1]
    med = np.median(base)
    return float(np.clip((c[-1] - med) / max(np.percentile(base, 90) - med, 1e-9), 0, 1))

@register_hook
def strategy():
    global last_decision
    try:
        selected = json.loads(ENSEMBLE.read_text()).get("selected_factors", [])[:10]
    except Exception:
        return
    if not selected or abs(sum(float(f.get("weight", 0)) for f in selected) - 1) > 1e-6:
        return
    account = get_account_dict()
    if set(account.get("watch_list", [])) != set(ASSETS): return
    data = {}
    for s in ASSETS:
        d = get_stock_daily_data(symbol=s, days=240)
        if d is None or len(d) < 125: return
        data[s] = d.sort_values("date").reset_index(drop=True)
    day = str(data[ASSETS[0]].iloc[-1]["date"])[:10]
    if last_decision is not None and np.busday_count(last_decision, day) < 10: return
    close = {s: np.asarray(data[s]["close"], float) for s in ASSETS}
    rets = {s: np.diff(close[s]) / np.maximum(close[s][:-1], 1e-12) for s in ASSETS}
    vol = {s: max(float(np.std(rets[s][-20:])), .008) for s in ASSETS}
    market = np.mean([rets[s][-65:] for s in ASSETS], axis=0)
    stress = vix_stress()
    dispersion = float(np.std([close[s][-1] / close[s][-6] - 1 for s in ASSETS]))
    raw = {s: {} for s in ASSETS}
    for s in ASSETS:
        c, r, v = close[s], rets[s], vol[s]
        m5, m10, m20 = c[-1]/c[-6]-1, c[-1]/c[-11]-1, c[-1]/c[-21]-1
        path = max(np.sum(np.abs(r[-20:])), 1e-9)
        persistence = np.mean(r[-20:] > 0) - .5
        beta = np.cov(r[-60:], market[-60:])[0, 1] / max(np.var(market[-60:]), 1e-10)
        residual5 = m5 - beta * np.sum(market[-5:])
        lo, hi = np.min(c[-41:]), np.max(c[-41:])
        range_pos = (c[-1] - lo) / max(hi - lo, 1e-12)
        own_breadth = np.mean(r[-33:] > 0) - .5
        raw[s] = {
            "miner_2_20281005_trend_persistence_10d": persistence * m20 / v,
            "miner_1_20310626_breadth33_conditioned_momentum_10d": m10 / v * (1 + own_breadth),
            "miner_1_20310710_efficiency_weighted_momentum_10d": (m20 / path) / v,
            "miner_2_20290920_stress_residual_reversal_10d": -residual5 / v * (1 + stress),
            "miner_2_20290322_range_extreme_reversal_5d": -(range_pos - .5) / v,
            "miner_3_20280921_beta_neutral_reversal_5d": -residual5 / v,
            "miner_1_20281005_smoothed_reversal_10d": -(m5 + m10) / (2 * v),
            "miner_3_20280824_dispersion_conditioned_reversal_5d": -m5 / v if dispersion > .025 else 0.0,
        }
    score = {s: 0.0 for s in ASSETS}
    for f in selected:
        rr = ranks({s: int(f.get("direction", 1)) * raw[s].get(str(f["factor_id"]), 0.0) for s in ASSETS})
        for s in ASSETS: score[s] += float(f["weight"]) * rr[s]
    breadth = np.mean([close[s][-1] > close[s][-21] for s in ASSETS])
    if stress > .05 or breadth < .45:
        for s in ("XAU", "US10Y", "CN10Y"): score[s] *= 1.8
        for s in ("BTC", "ETH", "WTI", "SOX", "NDX", "COPPER"): score[s] *= .65
    med = np.median(list(vol.values()))
    raw_score = {s: max(score[s], .01) / (1 + .20 * vol[s] / max(med, 1e-12)) for s in ASSETS}
    weights = bounded(raw_score)
    avg, sd = np.mean(list(raw_score.values())), max(np.std(list(raw_score.values())), 1e-9)
    forecast = {s: float(.01 * (raw_score[s] - avg) / sd) for s in ASSETS}
    rebalance_to_weights(weights, forecast_returns=forecast,
        factor_ids=[str(f["factor_id"]) for f in selected], horizon_days=10)
    last_decision = day
