"""Long-only, full-investment cross-sectional factor ensemble strategy."""
import json, math
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, rebalance_to_weights, register_hook

ASSETS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
START = "2026-07-16"
DATE_FILE = "../persistent/date.json"
STATE_FILE = "trader_state.json"
FLOOR, CAP = 0.025, 0.16
with open("factor_ensemble.json", encoding="utf-8") as f:
    _ensemble = json.load(f)
FACTORS = [(x["factor_id"], float(x["weight"]), int(x.get("direction", 1)))
           for x in _ensemble.get("selected_factors", [])[:10]]


def signal(df, fid, peers):
    c = df["close"].astype(float)
    r = c.pct_change()
    v10, v20, v60 = r.rolling(10).std(), r.rolling(20).std(), r.rolling(60).std()
    m5, m10, m20, m30, m60 = (c.pct_change(k) for k in (5, 10, 20, 30, 60))
    down = r.where(r < 0).rolling(60).std()
    breadth = peers.gt(0).mean(axis=1)
    dispersion = peers.std(axis=1)
    n = fid.lower()
    if "macro_breadth_shock" in n:
        stress = (breadth.shift(1) < .40) | (dispersion.shift(1) > dispersion.rolling(60).median().shift(1))
        x = (m10/(v20+1e-8)).where(stress, m30/(v60+1e-8))
    elif "vix_breadth_capitulation" in n:
        stress = (breadth.shift(1) < .40) & (dispersion.shift(1) > dispersion.rolling(60).median().shift(1))
        x = (-m5/(v20+1e-8)).where(stress, 0.)
    elif "dispersion_capitulation" in n:
        stress = dispersion.shift(1) > dispersion.rolling(60).quantile(.70).shift(1)
        x = (-m5/(v20+1e-8)).where(stress, 0.)
    elif "vix_stress_reversal" in n:
        stress = (breadth.shift(1) < .50) | (dispersion.shift(1) > dispersion.rolling(60).median().shift(1))
        x = (-m5/(v20+1e-8)).where(stress, 0.)
    elif "volatility_compression" in n:
        x = -(v10/(v60+1e-8))
    elif "risk_adjusted_trend_reversal" in n:
        x = -(m60/(down+1e-8))
    elif "downside_adjusted_breadth_trend" in n:
        x = (m30/(down+1e-8)) * (0.5 + breadth)
    elif "quiet_trend" in n:
        x = (m20/(v20+1e-8)).where(v20 < v60, 0.)
    elif "relative_risk_trend" in n:
        x = m20/(v20+1e-8)
    elif "trend_agreement" in n:
        x = ((m20 > 0).astype(float) + (m60 > 0).astype(float) + m20/(v20+1e-8))/3.
    else:
        x = m20/(v60+1e-8)
    return x.replace([np.inf, -np.inf], np.nan).shift(1).iloc[-1]


def cross_rank(values):
    good = [float(v) for v in values.values() if v is not None and pd.notna(v) and math.isfinite(float(v))]
    den = max(1, len(good)-1)
    return {a: ((sum(z <= float(v) for z in good)-1)/den if v is not None and pd.notna(v) and math.isfinite(float(v)) else .5) for a,v in values.items()}


def construct(frames):
    peers = pd.DataFrame({a: frames[a]["close"].pct_change() for a in ASSETS})
    score = {a: 0.0 for a in ASSETS}
    for fid, fw, direction in FACTORS:
        rr = cross_rank({a: signal(frames[a], fid, peers) for a in ASSETS})
        for a in ASSETS:
            score[a] += fw * (rr[a] if direction > 0 else 1.-rr[a])
    weights = {a: FLOOR + .10*score[a] for a in ASSETS}
    weak = sum(frames[a]["close"].iloc[-1] < frames[a]["close"].rolling(20).mean().iloc[-1] for a in ASSETS)
    if weak >= 8:
        for a in ("XAU", "US10Y", "CN10Y"): weights[a] += .035
    for _ in range(50):
        total = sum(weights.values()); weights = {a:v/total for a,v in weights.items()}
        over = [a for a in ASSETS if weights[a] > CAP]
        if not over: break
        excess = sum(weights[a]-CAP for a in over)
        for a in over: weights[a] = CAP
        room = [a for a in ASSETS if a not in over]; rt = sum(weights[a] for a in room)
        for a in room: weights[a] += excess*weights[a]/max(rt,1e-12)
    weights = {a:v/sum(weights.values()) for a,v in weights.items()}
    center = sum(score.values())/len(ASSETS)
    scale = max((max(score.values())-min(score.values()))/2, 1e-8)
    forecast = {a:float(np.clip(.04*(score[a]-center)/scale,-.05,.05)) for a in ASSETS}
    return weights, forecast


@register_hook
def strategy_hook():
    with open(DATE_FILE, encoding="utf-8") as f: info = json.load(f)
    date, days = str(info["current_date"]), info.get("trading_days", [])
    if date < START or date not in days: return
    try:
        with open(STATE_FILE, encoding="utf-8") as f: previous = json.load(f).get("last_proposal_date")
    except Exception: previous = None
    if previous and previous in days and days.index(date)-days.index(previous) < 10: return
    if set(get_account_dict().get("watch_list", [])) != set(ASSETS): return
    frames = {}
    for a in ASSETS:
        d = get_stock_daily_data(symbol=a, days=180)
        if d is None or len(d) < 100: return
        frames[a] = d.sort_values("date").set_index("date")
    target, forecast = construct(frames)
    if set(target) != set(ASSETS) or any((not math.isfinite(v) or v < 0) for v in target.values()): return
    rebalance_to_weights(target, forecast_returns=forecast, factor_ids=[x[0] for x in FACTORS], horizon_days=10)
    with open(STATE_FILE, "w", encoding="utf-8") as f: json.dump({"last_proposal_date": date}, f)

strategy_hook
