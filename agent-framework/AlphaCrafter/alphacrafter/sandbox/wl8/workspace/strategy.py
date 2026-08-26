import json
from pathlib import Path
import numpy as np
from alphacrafter.sim.utils import (
    register_hook, get_account_dict, get_stock_daily_data,
    get_index_daily_data, rebalance_to_weights,
)

ASSETS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
ENSEMBLE_PATHS = [Path(__file__).parent / "factor_ensemble.json", Path(__file__).parent / "factors" / "factor_ensemble.json"]
last_decision = None


def rank_map(vals):
    good = sorted((s, v) for s, v in vals.items() if np.isfinite(v))
    out = {s: 0.5 for s in ASSETS}
    n = len(good)
    for i, (s, _) in enumerate(good):
        out[s] = (i + 1) / max(n, 1)
    return out


def make_weights(score):
    # Diversified full investment: 1.5% floor and 14% cap.
    floor, cap = 0.015, 0.14
    q = {s: max(float(score[s]), 1e-8) for s in ASSETS}
    w = {s: floor + (1 - len(ASSETS) * floor) * q[s] / sum(q.values()) for s in ASSETS}
    for _ in range(20):
        over = [s for s in ASSETS if w[s] > cap]
        if not over:
            break
        extra = sum(w[s] - cap for s in over)
        for s in over:
            w[s] = cap
        rest = [s for s in ASSETS if s not in over]
        den = sum(q[s] for s in rest)
        for s in rest:
            w[s] += extra * q[s] / max(den, 1e-12)
    z = sum(w.values())
    return {s: float(w[s] / z) for s in ASSETS}


def load_factors():
    for p in ENSEMBLE_PATHS:
        if p.exists():
            fs = json.loads(p.read_text()).get("selected_factors", [])[:10]
            if fs and abs(sum(float(x.get("weight", 0)) for x in fs) - 1.0) < 1e-6:
                return fs
    return []


def vix_risk():
    d = get_index_daily_data("VIX", 100)
    if d is None or len(d) < 30:
        return False
    c = np.asarray(d.sort_values("date").close, float)
    return bool(c[-1] > np.mean(c[-21:-1]) * 1.08)


@register_hook
def strategy():
    global last_decision
    factors = load_factors()
    if not factors:
        return
    account = get_account_dict()
    if set(account.get("watch_list", [])) != set(ASSETS):
        return
    data = {}
    for s in ASSETS:
        d = get_stock_daily_data(s, 270)
        if d is None or len(d) < 190:
            return
        data[s] = d.sort_values("date").reset_index(drop=True)
    day = str(data[ASSETS[0]].iloc[-1].date)[:10]
    if last_decision is not None and np.busday_count(last_decision, day) < 10:
        return

    raw = {s: {} for s in ASSETS}
    vols = {}
    for s, d in data.items():
        c = np.asarray(d.close, float)
        r = np.diff(c) / np.maximum(c[:-1], 1e-12)
        vols[s] = max(float(np.std(r[-20:])), 0.008)
        m10 = c[-1] / c[-11] - 1
        prior10 = c[-11] / c[-21] - 1
        m20 = c[-1] / c[-21] - 1
        downside = np.std(np.minimum(r[-10:], 0.0))
        peak = max(np.max(c[-181:-1]), 1e-12)
        raw[s] = {
            "downside": -downside / vols[s],
            "drawdown": (1 - c[-1] / peak) / vols[s],
            "inverse30": -(c[-2] / max(c[-32], 1e-12) - 1) / vols[s],
            "accel": (m10 - prior10) / vols[s],
            "eff": m20 * (abs(np.sum(np.log(np.maximum(1 + r[-20:], 1e-8)))) /
                    max(np.sum(abs(np.log(np.maximum(1 + r[-20:], 1e-8)))), 1e-9)) / vols[s],
        }

    score = {s: 0.0 for s in ASSETS}
    for f in factors:
        fid = str(f.get("factor_id", ""))
        if "downside_risk" in fid: key = "downside"
        elif "drawdown180" in fid: key = "drawdown"
        elif "inverse_trend_30d" in fid: key = "inverse30"
        elif "return_acceleration" in fid: key = "accel"
        elif "efficiency_weighted_momentum" in fid: key = "eff"
        else: return
        rr = rank_map({s: int(f.get("direction", 1)) * raw[s][key] for s in ASSETS})
        for s in ASSETS:
            score[s] += float(f["weight"]) * rr[s]

    # Bullish but fragile: remain fully invested, favor defensive tradable assets.
    if vix_risk():
        for s in ("XAU", "US10Y", "CN10Y"):
            score[s] *= 1.55
        for s in ("BTC", "ETH", "WTI", "SOX", "NDX", "COPPER"):
            score[s] *= 0.75
    medvol = max(float(np.median(list(vols.values()))), 1e-9)
    adjusted = {s: score[s] / (1 + 0.20 * vols[s] / medvol) for s in ASSETS}
    target = make_weights(adjusted)
    mean = float(np.mean(list(adjusted.values())))
    sd = max(float(np.std(list(adjusted.values()))), 1e-9)
    forecast = {s: float(0.01 * (adjusted[s] - mean) / sd) for s in ASSETS}
    rebalance_to_weights(target, forecast_returns=forecast,
                         factor_ids=[str(f["factor_id"]) for f in factors],
                         horizon_days=10)
    last_decision = day
