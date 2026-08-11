"""Trader strategy v2 - Screener 6-factor quality_ic_tilt ensemble.

Ensemble (2026-10-30): mom_120d_skip5 (.36,+) | miner2_nclv_1d (.18,+)
| miner2_nclv_2d (.14,+) | miner2_rev_2d (.12,+) | vol_of_vol20x60 (.12,+)
| vix_beta_cond_60x20 (.08,-).

Momentum anchor + short-term reversal + vol-of-vol regime + VIX-beta risk
guard. Cross-sectional rank composite over the 15-name tradable panel; fully
invested, non-negative weights sum to 1, no cash sleeve. One atomic rebalance
proposal per 10-trading-day block (first day only) via rebalance_to_weights
with aligned forecast returns so the execution gate (gross edge > one-way
turnover * 3bp) decides. Bear regime adds a modest defensive tilt
(XAU/US10Y/CN10Y).
"""
import json
import math
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import (get_account_dict, get_stock_daily_data,
                                    rebalance_to_weights, register_hook)

ONLINE_START = "2026-07-16"
DATE_FILE = "../persistent/date.json"
VIX_FILE = "../persistent/index_data/VIX.csv"
DATA_DAYS = 170          # enough for mom_120d_skip5 (shift(125)) + buffers
MIN_ROWS = 140
DEFENSIVE = {"XAU", "US10Y", "CN10Y"}
CAP_W = 0.16

_VIX_CACHE = {}


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
            if df is None or len(df) < MIN_ROWS:
                frames[a] = None
                continue
            df = df.copy()
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
            frames[a] = df
        except Exception:
            frames[a] = None
    return frames


def _vix_close(cur):
    if cur in _VIX_CACHE:
        return _VIX_CACHE[cur]
    try:
        vix = pd.read_csv(VIX_FILE)
        vix["date"] = pd.to_datetime(vix["date"])
        vix = vix[vix["date"] <= pd.Timestamp(cur)].sort_values("date")
        s = vix.set_index("date")["close"].astype(float)
        _VIX_CACHE[cur] = s
        return s
    except Exception:
        _VIX_CACHE[cur] = None
        return None


def _asset_factor(df, fid, cur):
    """Return the factor Series on df's date index (or None if unsupported)."""
    o = df["open"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)
    if fid.endswith("nclv_1d"):
        return -(c - l) / (h - l)
    if fid.endswith("nclv_2d"):
        return -(c - l.rolling(2).min()) / (h.rolling(2).max() - l.rolling(2).min())
    if fid.endswith("nclv_3d"):
        return -(c - l.rolling(3).min()) / (h.rolling(3).max() - l.rolling(3).min())
    if fid.endswith("rev_1d"):
        return -np.log(c / c.shift(1))
    if fid.endswith("rev_2d"):
        return -np.log(c / c.shift(2))
    if fid.endswith("nbody_1d"):
        return -(c - o) / (h - l)
    if "mom_120d_skip5" in fid:
        return c.shift(5) / c.shift(125) - 1.0
    if fid == "vol_of_vol20x60":
        return c.pct_change().rolling(20).std().rolling(60).std()
    if fid == "vix_beta_cond_60x20":
        vixc = _vix_close(cur)
        if vixc is None or len(vixc) < 90:
            return None
        v = vixc.reindex(df.index).ffill()
        ar = c.pct_change()
        vr = v.pct_change()
        beta = ar.rolling(60).cov(vr) / vr.rolling(60).var()
        vm = v / v.shift(20) - 1.0
        return -beta * vm
    return None


def _factor_values(frames, fid, cur):
    out = {}
    for a, df in frames.items():
        if df is None:
            out[a] = None
            continue
        try:
            s = _asset_factor(df, fid, cur)
            if s is None:
                out[a] = None
                continue
            s = s.replace([np.inf, -np.inf], np.nan)
            v = float(s.iloc[-1])
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


def _scores(frames, assets, cur):
    score = {a: 0.0 for a in assets}
    used = 0
    for fid, w, direction in FACTORS:
        vals = _factor_values(frames, fid, cur)
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
            rets.append(float(df["close"].pct_change().tail(20).mean()))
    if not rets:
        return "side"
    m = float(np.mean(rets))
    return "bull" if m > 0.010 else ("bear" if m < -0.010 else "side")


def _weights(scores, assets, regime):
    order = sorted(assets, key=lambda a: (scores[a], a))
    n = len(assets)
    raw = {}
    for i, a in enumerate(order):
        r = i / max(1, n - 1)
        raw[a] = 0.02 + 0.10 * r          # rank-linear 2%..12% pre-normalization
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
    for _ in range(80):                   # cap-and-redistribute
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
    w[assets[-1]] += 1.0 - sum(w.values())   # exact sum-to-1 fix
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
        return   # non-rebalance day: simulator marks positions / processes orders
    if not FACTORS:
        return   # no Screener ensemble -> skip this cycle
    account = get_account_dict()
    assets = list(account.get("watch_list", []))
    if len(assets) != 15:
        return
    frames = _fetch(assets)
    scores, used = _scores(frames, assets, cur)
    if used < 5:                            # degraded fallback: equal weight
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
