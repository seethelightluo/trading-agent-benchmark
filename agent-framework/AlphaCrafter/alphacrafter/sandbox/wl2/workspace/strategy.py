"""Full-investment lagged cross-sectional ensemble for the 15-asset worldline."""
import json, math
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import (
    get_account_dict, get_stock_daily_data, rebalance_to_weights, register_hook
)

ASSETS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
START = "2026-07-16"
DATE_FILE, STATE_FILE = "../persistent/date.json", "trader_state.json"
FLOOR, CAP = 0.025, 0.16
with open("factors/factor_ensemble.json", encoding="utf-8") as f:
    selected = json.load(f).get("selected_factors", [])[:10]
FACTORS = [(x["factor_id"], float(x["weight"]), int(x.get("direction", 1))) for x in selected]


def signal(df, fid, peer):
    c = df["close"].astype(float)
    r = c.pct_change()
    n = fid.lower()
    v10, v20, v30, v60 = r.rolling(10).std(), r.rolling(20).std(), r.rolling(30).std(), r.rolling(60).std()
    mom20, mom30, mom60 = c.pct_change(20), c.pct_change(30), c.pct_change(60)
    if "quiet_trend" in n:
        x = (mom20 / (v20 + 1e-8)).where(r.rolling(10).std() < v60, 0.0)
    elif "relative_risk_trend" in n:
        x = mom20 / (v20 + 1e-8)
    elif "volatility_compression" in n:
        x = -(v10 / (v60 + 1e-8))
    elif "downside_adjusted_breadth" in n:
        dn = r.where(r < 0).rolling(30).std()
        x = (r.gt(0).rolling(30).mean() - .5) / (dn + 1e-8)
    elif "risk_adjusted_trend_reversal" in n:
        x = -mom60 / (v60 + 1e-8)
    elif "momentum_acceleration" in n:
        x = (mom10 := c.pct_change(10)) - mom20
        x = x / (v20 + 1e-8)
    elif "dispersion_capitulation_reversal" in n:
        breadth = peer.gt(0).mean(axis=1) - .5
        dispersion = peer.std(axis=1)
        shock = peer.mean(axis=1).rolling(5).sum()
        active = (breadth < breadth.rolling(60).quantile(.35)) & (dispersion > dispersion.rolling(60).median())
        x = (-c.pct_change(5)).where(active, 0.0)
    elif "macro_breadth_shock" in n:
        breadth = peer.gt(0).mean(axis=1) - .5
        shock = peer.mean(axis=1).rolling(5).sum()
        x = -(shock - shock.rolling(60).mean()) * breadth
    else:
        x = mom20 / (v60 + 1e-8)
    return x.replace([np.inf, -np.inf], np.nan).shift(1).iloc[-1]


def cross_rank(vals):
    good = [float(v) for v in vals.values() if v is not None and pd.notna(v) and math.isfinite(float(v))]
    return {a: ((sum(z <= float(v) for z in good)-1) / max(1, len(good)-1)
                if v is not None and pd.notna(v) and math.isfinite(float(v)) else .5)
            for a, v in vals.items()}


def construct(frames):
    peer = pd.DataFrame({a: frames[a]["close"].pct_change() for a in ASSETS})
    score = {a: 0.0 for a in ASSETS}
    for fid, fw, direction in FACTORS:
        q = cross_rank({a: signal(frames[a], fid, peer) for a in ASSETS})
        for a in ASSETS:
            score[a] += fw * (q[a] if direction > 0 else 1.0 - q[a])
    weights = {a: FLOOR + 0.10 * score[a] for a in ASSETS}
    # High-risk overlay: ensure defensive tradable assets absorb broad weakness.
    weak = sum(frames[a]["close"].iloc[-1] < frames[a]["close"].rolling(20).mean().iloc[-1] for a in ASSETS)
    if weak >= 9:
        for a in ("XAU", "US10Y", "CN10Y"):
            weights[a] += .025
    for _ in range(50):
        total = sum(weights.values())
        weights = {a: v / total for a, v in weights.items()}
        over = [a for a in ASSETS if weights[a] > CAP]
        if not over:
            break
        excess = sum(weights[a] - CAP for a in over)
        for a in over:
            weights[a] = CAP
        room = [a for a in ASSETS if a not in over]
        base = sum(weights[a] for a in room)
        for a in room:
            weights[a] += excess * weights[a] / max(base, 1e-12)
    weights = {a: v / sum(weights.values()) for a, v in weights.items()}
    mid = sum(score.values()) / len(ASSETS)
    scale = max((max(score.values()) - min(score.values())) / 2, 1e-8)
    forecast = {a: float(np.clip(.04 * (score[a] - mid) / scale, -.05, .05)) for a in ASSETS}
    return weights, forecast


@register_hook
def strategy_hook():
    with open(DATE_FILE, encoding="utf-8") as f:
        info = json.load(f)
    date, days = str(info["current_date"]), info.get("trading_days", [])
    if date < START or date not in days:
        return
    try:
        with open(STATE_FILE, encoding="utf-8") as f:
            prior = json.load(f).get("last_proposal_date")
    except Exception:
        prior = None
    if prior and prior in days and days.index(date) - days.index(prior) < 10:
        return
    if set(get_account_dict().get("watch_list", [])) != set(ASSETS):
        return
    frames = {}
    for a in ASSETS:
        d = get_stock_daily_data(symbol=a, days=180)
        if d is None or len(d) < 100:
            return
        frames[a] = d.sort_values("date").set_index("date")
    target, forecast = construct(frames)
    rebalance_to_weights(target, forecast_returns=forecast,
                         factor_ids=[x[0] for x in FACTORS], horizon_days=10)
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump({"last_proposal_date": date}, f)

strategy_hook
