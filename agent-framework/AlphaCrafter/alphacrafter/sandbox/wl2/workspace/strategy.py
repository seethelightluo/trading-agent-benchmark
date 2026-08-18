"""Screener-synchronized, lagged, fully invested long-only cross-asset strategy."""
import json, math
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import (get_account_dict, get_stock_daily_data,
                                     rebalance_to_weights, register_hook)

ASSETS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
DEFENSIVE = {"XAU", "US10Y", "CN10Y"}
with open("factors/factor_ensemble.json", encoding="utf-8") as f:
    FACTORS = [(str(x["factor_id"]), float(x["weight"]), int(x.get("direction", 1)))
               for x in json.load(f).get("selected_factors", [])[:10]]

def rank(s):
    return pd.Series(s, index=ASSETS, dtype=float).replace([np.inf, -np.inf], np.nan).rank(pct=True).fillna(.5)

def build_target(frames):
    close = pd.concat({a: frames[a]["close"] for a in ASSETS}, axis=1).sort_index()
    close.columns = ASSETS
    ret = close.pct_change()
    v20 = ret.rolling(20, min_periods=10).std()
    v60 = ret.rolling(60, min_periods=30).std()
    breadth = ret.gt(0).rolling(5, min_periods=5).mean().mean(axis=1)
    score = pd.Series(0., index=ASSETS)
    for fid, wt, direction in FACTORS:
        n = fid.lower()
        if "macro_breadth_shock" in n:
            raw = (-ret.rolling(5, min_periods=5).sum()).where(breadth < .45, 0.)
        elif "quiet_trend" in n:
            dn = ret.where(ret < 0).rolling(20, min_periods=10).std()
            raw = close.pct_change(15) / (dn + 1e-9)
            raw /= 1 + 5 * ret.abs().rolling(15, min_periods=10).mean()
        elif "relative_risk_trend" in n:
            m = close.pct_change(20)
            raw = m.sub(m.median(axis=1), axis=0) / (v20 + 1e-9)
        elif "range_location_reversal" in n:
            w = 180 if "180d" in n else 120
            hi, lo = close.rolling(w, min_periods=max(40, w//3)).max(), close.rolling(w, min_periods=max(40, w//3)).min()
            raw = -(close-lo) / (hi-lo+1e-9)
        elif "stress_strength_residual_reversal" in n or "persistent_stress_residual_reversal" in n:
            m = -close.pct_change(40)
            raw = m.sub(m.median(axis=1), axis=0).where(breadth < .50, 0.)
        elif "compression_reversal" in n:
            raw = (-close.pct_change(20)).where(v20/(v60+1e-9) < .85, 0.)
        elif "dispersion_gated_reversal" in n:
            m = -close.pct_change(10)
            raw = m.sub(m.median(axis=1), axis=0).where(v20.gt(v20.median(axis=1), axis=1), 0.)
        else:
            raw = close.pct_change(10)/(v20+1e-9)
        score += wt * (rank(raw.shift(1).iloc[-1]) if direction > 0 else 1-rank(raw.shift(1).iloc[-1]))
    total = float(score.sum()) or 1.
    b = float(breadth.iloc[-2]) if len(breadth) > 1 and pd.notna(breadth.iloc[-2]) else .5
    stress = min(.25, max(0., (.52-b)*.75))
    weights = {a: (.025 + .625*float(score[a])/total)*(1-stress) for a in ASSETS}
    for a in DEFENSIVE: weights[a] += stress/3
    for _ in range(20):
        over = [a for a in ASSETS if weights[a] > .16]
        if not over: break
        excess = sum(weights[a]-.16 for a in over)
        for a in over: weights[a] = .16
        rest = [a for a in ASSETS if a not in over]
        denom = sum(weights[a] for a in rest) or 1.
        for a in rest: weights[a] += excess*weights[a]/denom
    z = sum(weights.values())
    weights = {a: max(0., weights[a]/z) for a in ASSETS}
    scale = max(float(score.max()-score.min())/2, 1e-8)
    center = float(score.mean())
    forecast = {a: float(np.clip(.04*(float(score[a])-center)/scale, -.05, .05)) for a in ASSETS}
    return weights, forecast

@register_hook
def strategy_hook():
    try:
        with open("../persistent/date.json", encoding="utf-8") as f: info = json.load(f)
        date, cal = str(info.get("current_date")), info.get("trading_days", [])
        if date < "2026-07-16" or date not in cal or not FACTORS: return
        try:
            with open("trader_state.json", encoding="utf-8") as f: last = json.load(f).get("last_proposal_date")
        except Exception: last = None
        if last in cal and cal.index(date)-cal.index(last) < 10: return
        if set(get_account_dict().get("watch_list", [])) != set(ASSETS): return
        frames = {}
        for a in ASSETS:
            d = get_stock_daily_data(symbol=a, days=240)
            if d is None or len(d) < 190: return
            frames[a] = d.sort_values("date").set_index("date")
        target, forecast = build_target(frames)
        if set(target) != set(ASSETS) or any(not math.isfinite(x) or x < 0 for x in target.values()): return
        if abs(sum(target.values())-1.) > 1e-8: return
        rebalance_to_weights(target, forecast_returns=forecast,
                             factor_ids=[x[0] for x in FACTORS], horizon_days=10)
        with open("trader_state.json", "w", encoding="utf-8") as f: json.dump({"last_proposal_date": date}, f)
    except Exception:
        return
