"""Trader strategy v1 - Screener 7-factor quality_ic_tilt ensemble (reversal regime).

Cross-sectional rank composite over the 15-name tradable panel. Fully invested,
non-negative weights summing to 1, no cash sleeve. One atomic rebalance per
10-trading-day block (first day only) via rebalance_to_weights with aligned
forecast returns so the gate (gross edge > one-way turnover * 3bp) can decide.
Bear regime adds a modest defensive tilt (XAU/US10Y/CN10Y).
"""
import json
import math
import numpy as np
from alphacrafter.sim.utils import (get_account_dict, get_stock_daily_data,
                                    rebalance_to_weights, register_hook)

ONLINE_START = "2026-07-16"
DATE_FILE = "../persistent/date.json"
DATA_DAYS = 70
DEFENSIVE = {"XAU", "US10Y", "CN10Y"}
CAP_W = 0.14


def _load_ensemble():
    with open("factor_ensemble.json") as f:
        ens = json.load(f)
    return [(x["factor_id"], float(x["weight"]), int(x["direction"]))
            for x in ens["selected_factors"]]


FACTORS = _load_ensemble()


def _today_and_calendar():
    with open(DATE_FILE) as f:
        d = json.load(f)
    return str(d["current_date"]), d.get("trading_days", [])


def _is_rebalance_day(cur, tds):
    if cur < ONLINE_START or cur not in tds or ONLINE_START not in tds:
        return False
    return (tds.index(cur) - tds.index(ONLINE_START)) % 10 == 0


def _fetch(assets):
    frames = {}
    for a in assets:
        try:
            df = get_stock_daily_data(symbol=a, days=DATA_DAYS)
            frames[a] = df if df is not None and len(df) >= 30 else None
        except Exception:
            frames[a] = None
    return frames


def _factor_values(frames, fid):
    """Compute the factor on the last completed bar for every asset."""
    out = {}
    for a, df in frames.items():
        if df is None:
            out[a] = None
            continue
        try:
            o = df["open"].astype(float)
            h = df["high"].astype(float)
            l = df["low"].astype(float)
            c = df["close"].astype(float)
            if fid.endswith("nclv_1d"):
                x = -(c - l) / (h - l)
            elif fid.endswith("nclv_2d"):
                x = -(c - l.rolling(2).min()) / (h.rolling(2).max() - l.rolling(2).min())
            elif fid.endswith("nclv_3d"):
                x = -(c - l.rolling(3).min()) / (h.rolling(3).max() - l.rolling(3).min())
            elif fid.endswith("rev_1d"):
                x = -np.log(c / c.shift(1))
            elif fid.endswith("rev_2d"):
                x = -np.log(c / c.shift(2))
            elif fid.endswith("nbody_1d"):
                x = -(c - o) / (h - l)
            elif "mom_10d_skip5" in fid:
                x = np.log(c.shift(5) / c.shift(15))
            else:
                out[a] = None
                continue
            x = x.replace([np.inf, -np.inf], np.nan)
            v = float(x.iloc[-1])
            out[a] = v if math.isfinite(v) else None
        except Exception:
            out[a] = None
    return out


def _ranks(values, assets):
    valid = sorted((float(v), a) for a, v in values.items()
                   if v is not None and math.isfinite(float(v)))
    out = {a: 0.5 for a in assets}
    n = len(valid)
    for i, (_, a) in enumerate(valid):
        out[a] = i / max(1, n - 1)
    return out


def _scores(frames, assets):
    score = {a: 0.0 for a in assets}
    used = 0
    for fid, w, direction in FACTORS:
        vals = _factor_values(frames, fid)
        if sum(1 for v in vals.values() if v is not None) < 8:
            continue
        r = _ranks(vals, assets)
        for a in assets:
            score[a] += w * (r[a] if direction > 0 else 1.0 - r[a])
        used += 1
    return score, used


def _regime(frames, assets):
    rets = []
    for a in assets:
        df = frames.get(a)
        if df is not None and len(df) >= 25:
            rets.append(float(df["close"].astype(float).pct_change().tail(20).mean()))
    if not rets:
        return "side"
    m = float(np.mean(rets))
    return "bull" if m > 0.015 else ("bear" if m < -0.015 else "side")


def _weights(scores, assets, regime):
    order = sorted(assets, key=lambda a: (scores[a], a))
    n = len(assets)
    raw = {}
    for i, a in enumerate(order):
        r = i / max(1, n - 1)
        raw[a] = 0.02 + 0.10 * r  # rank-linear 2%..12% pre-normalization
    if regime == "bear":
        tilt = 0.045
        defs = [a for a in DEFENSIVE if a in assets]
        nd = [a for a in assets if a not in DEFENSIVE]
        dsum = sum(raw[a] for a in defs)
        nsum = sum(raw[a] for a in nd)
        for a in defs:
            raw[a] += tilt * (raw[a] / dsum if dsum > 0 else 1.0 / len(defs))
        for a in nd:
            raw[a] -= tilt * (raw[a] / nsum if nsum > 0 else 1.0 / len(nd))
    tot = sum(raw.values())
    w = {a: max(0.0, x / tot) for a, x in raw.items()}
    for _ in range(60):
        excess = sum(max(0.0, x - CAP_W) for x in w.values())
        if excess < 1e-12:
            break
        w = {a: min(CAP_W, x) for a, x in w.items()}
        room = [a for a in w if w[a] < CAP_W - 1e-12]
        if not room:
            break
        den = sum(w[a] for a in room) + 1e-12
        for a in room:
            w[a] += excess * w[a] / den
    tot = sum(w.values())
    w = {a: x / tot for a, x in w.items()}
    w[assets[-1]] += 1.0 - sum(w.values())
    return w


def _forecasts(scores, assets):
    vals = [scores[a] for a in assets]
    mean = float(np.mean(vals))
    half = max(1e-9, (max(vals) - min(vals)) / 2.0)
    f = {}
    for a in assets:
        z = (scores[a] - mean) / half
        f[a] = float(np.clip(0.04 * z, -0.05, 0.05))
    return f


@register_hook
def strategy_hook():
    cur, tds = _today_and_calendar()
    if not _is_rebalance_day(cur, tds):
        return  # non-rebalance day: simulator marks positions / processes orders
    if not FACTORS:
        return  # no Screener ensemble -> skip this cycle
    account = get_account_dict()
    assets = list(account.get("watch_list", []))
    if len(assets) != 15:
        return
    frames = _fetch(assets)
    scores, used = _scores(frames, assets)
    if used < 5:
        w = {a: 1.0 / len(assets) for a in assets}
        w[assets[-1]] += 1.0 - sum(w.values())
        rebalance_to_weights(w)
        return
    regime = _regime(frames, assets)
    w = _weights(scores, assets, regime)
    f = _forecasts(scores, assets)
    rebalance_to_weights(
        w,
        forecast_returns=f,
        factor_ids=[fid for fid, _, _ in FACTORS],
        horizon_days=10,
    )
