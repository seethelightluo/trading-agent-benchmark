"""Long-only 15-asset cross-sectional ensemble with full-investment targets."""
import json, math
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import (get_account_dict, get_stock_daily_data,
    rebalance_to_weights, register_hook)

ONLINE_START = "2026-07-16"
DATE_FILE = "../persistent/date.json"
STATE_FILE = "trader_state.json"
ASSET_COUNT = 15
LOOKBACK = 170
CAP = 0.16

with open("factor_ensemble.json", encoding="utf-8") as fh:
    _ensemble = json.load(fh).get("selected_factors", [])
FACTORS = [(x["factor_id"], float(x["weight"]), int(x.get("direction", 1)))
           for x in _ensemble[:10]]


def _window(fid):
    f = fid.lower()
    if "compression" in f: return 10, 60
    if "breadth" in f: return 30, 30
    if "stress" in f: return 5, 20
    if "leadlag" in f: return 5, 5
    if "agreement" in f: return 20, 60
    if "relative_risk" in f: return 20, 60
    return 20, 20


def _signal(df, fid):
    c = df["close"].astype(float); r = c.pct_change(); f = fid.lower(); a, b = _window(f)
    lr = np.log(c).diff()
    if "compression" in f or "stress" in f:
        x = -(lr.rolling(a).std() / lr.rolling(b).std())
    elif "breadth" in f:
        x = r.gt(0).rolling(a).mean() - .5
    elif "leadlag" in f:
        x = c.pct_change(a) - c.pct_change(a).rolling(b).mean()
    elif "agreement" in f:
        x = c.pct_change(a) * (c > c.rolling(b).mean()).astype(float)
    else:
        x = c.pct_change(a)
    return x.replace([np.inf, -np.inf], np.nan).shift(1)


def _rank(vals, assets):
    good = sorted((v, a) for a, v in vals.items() if v is not None and math.isfinite(v))
    out = {a: .5 for a in assets}
    for i, (_, a) in enumerate(good): out[a] = i / max(1, len(good) - 1)
    return out


def _data(assets):
    out = {}
    for a in assets:
        try:
            d = get_stock_daily_data(symbol=a, days=LOOKBACK)
            out[a] = None if d is None or len(d) < 100 else d.copy().sort_values("date").set_index("date")
        except Exception: out[a] = None
    return out


def _targets(frames, assets):
    score = {a: 0.0 for a in assets}; used = 0
    for fid, weight, direction in FACTORS:
        vals = {}
        for a in assets:
            try:
                s = _signal(frames[a], fid) if frames[a] is not None else None
                v = float(s.iloc[-1]) if s is not None else float("nan")
                vals[a] = v if math.isfinite(v) else None
            except Exception: vals[a] = None
        if sum(v is not None for v in vals.values()) < 8: continue
        rr = _rank(vals, assets)
        for a in assets: score[a] += weight * (rr[a] if direction > 0 else 1 - rr[a])
        used += 1
    if not used: return {a: 1 / len(assets) for a in assets}, score
    raw = {a: .025 + .10 * score[a] for a in assets}
    w = {a: x / sum(raw.values()) for a, x in raw.items()}
    # Iterative cap redistribution keeps every tradable asset in the target.
    for _ in range(20):
        over = sum(max(0., x - CAP) for x in w.values())
        if over < 1e-12: break
        room = [a for a in assets if w[a] < CAP - 1e-12]
        for a in assets: w[a] = min(w[a], CAP)
        den = sum(w[a] for a in room) or 1.
        for a in room: w[a] += over * w[a] / den
    z = sum(w.values()); w = {a: max(0., x / z) for a, x in w.items()}
    weak = sum(frames[a] is not None and float(frames[a].close.iloc[-1]) <
               float(frames[a].close.rolling(20).mean().iloc[-1]) for a in assets)
    if weak >= 10:
        for a in ("XAU", "US10Y", "CN10Y"):
            if a in w: w[a] += .025
        z = sum(w.values()); w = {a: x / z for a, x in w.items()}
    return w, score


def _dates():
    with open(DATE_FILE, encoding="utf-8") as fh: d = json.load(fh)
    return str(d["current_date"]), d.get("trading_days", [])


def _due(cur, days):
    if cur < ONLINE_START or cur not in days: return False
    try:
        with open(STATE_FILE, encoding="utf-8") as fh: last = json.load(fh).get("last_proposal_date")
    except Exception: last = None
    if last in days: return days.index(cur) - days.index(last) >= 10
    return (days.index(cur) - days.index(ONLINE_START)) % 10 == 0 or last is None


@register_hook
def strategy_hook():
    cur, days = _dates()
    if not _due(cur, days): return
    account = get_account_dict(); assets = list(account.get("watch_list", []))
    if len(assets) != ASSET_COUNT or len(FACTORS) == 0: return
    frames = _data(assets); weights, scores = _targets(frames, assets)
    mean = float(np.mean(list(scores.values())))
    span = max(1e-9, (max(scores.values()) - min(scores.values())) / 2)
    forecasts = {a: float(np.clip(.04 * (scores[a] - mean) / span, -.05, .05)) for a in assets}
    rebalance_to_weights(weights, forecast_returns=forecasts,
                         factor_ids=[x[0] for x in FACTORS], horizon_days=10)
    with open(STATE_FILE, "w", encoding="utf-8") as fh: json.dump({"last_proposal_date": cur}, fh)

strategy_hook
