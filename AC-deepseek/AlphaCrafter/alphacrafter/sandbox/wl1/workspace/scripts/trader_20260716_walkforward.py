"""Trader walk-forward validation of the 7-factor ensemble over recent regimes.

Simulates 10-trading-day rebalance blocks exactly like strategy.py (same factor
formulas, rank-linear weights, defensive tilt on bear regime, cap at 0.14),
using only data visible at each block start. Charges 3bp on migrated notional.
"""
import json
import math
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

ASSETS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
          "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
DEFENSIVE = {"XAU", "US10Y", "CN10Y"}
CAP_W = 0.14

ens = json.load(open("factor_ensemble.json"))["selected_factors"]
FACTORS = [(x["factor_id"], float(x["weight"]), int(x["direction"])) for x in ens]


def factor_values(df, fid):
    if df is None or len(df) < 30:
        return None
    o = df["open"].astype(float); h = df["high"].astype(float)
    l = df["low"].astype(float); c = df["close"].astype(float)
    try:
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
            return None
        x = x.replace([np.inf, -np.inf], np.nan)
        v = float(x.iloc[-1])
        return v if math.isfinite(v) else None
    except Exception:
        return None


def ranks(values):
    valid = sorted((float(v), a) for a, v in values.items() if v is not None and math.isfinite(float(v)))
    out = {a: 0.5 for a in ASSETS}
    n = len(valid)
    for i, (_, a) in enumerate(valid):
        out[a] = i / max(1, n - 1)
    return out


def weights(scores, regime):
    order = sorted(ASSETS, key=lambda a: (scores[a], a))
    n = len(ASSETS)
    raw = {}
    for i, a in enumerate(order):
        r = i / max(1, n - 1)
        raw[a] = 0.02 + 0.10 * r
    if regime == "bear":
        tilt = 0.045
        defs = [a for a in DEFENSIVE if a in ASSETS]
        nd = [a for a in ASSETS if a not in DEFENSIVE]
        dsum = sum(raw[a] for a in defs); nsum = sum(raw[a] for a in nd)
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
    w[ASSETS[-1]] += 1.0 - sum(w.values())
    return w


# --- fetch full history ---
hist = {}
for a in ASSETS:
    df = get_stock_daily_data(symbol=a, days=400)
    hist[a] = df.reset_index(drop=True) if df is not None else None

# common trading calendar (union of dates where all assets have data)
dates = None
for a in ASSETS:
    if hist[a] is not None:
        dts = set(pd.to_datetime(hist[a]["date"]))
        dates = dts if dates is None else (dates & dts)
dates = sorted(dates)
print("common dates:", len(dates), dates[0].date(), "->", dates[-1].date())

by_date = {a: {pd.to_datetime(r["date"]): r for _, r in hist[a].iterrows()} if hist[a] is not None else {} for a in ASSETS}

def frame_until(d):
    """Build a DataFrame for each asset with rows <= d (in order), plus next-10d fwd return."""
    rows = {}
    idx = dates.index(d)
    fwd = {}
    for a in ASSETS:
        recs = []
        for dd in dates[:idx + 1]:
            r = by_date[a].get(dd)
            if r is not None:
                recs.append({"open": float(r["open"]), "high": float(r["high"]),
                             "low": float(r["low"]), "close": float(r["close"])})
        if len(recs) < 30:
            rows[a] = None
            fwd[a] = None
            continue
        rows[a] = pd.DataFrame(recs)
        c0 = float(by_date[a][d]["close"])
        j = idx + 10
        c10 = float(by_date[a][dates[j]]["close"]) if j < len(dates) else None
        fwd[a] = c10 / c0 - 1.0 if c10 else None
    return rows, fwd

# --- run walk-forward over the last ~150 trading days in 10-day blocks ---
start_idx = len(dates) - 150
rebal_idx = list(range(start_idx, len(dates) - 10, 10))
print("rebalance blocks:", len(rebal_idx), "| first:", dates[rebal_idx[0]].date(), "last:", dates[rebal_idx[-1]].date())

equity = 1.0
prev_w = None
block_rets = []
details = []
for bi in rebal_idx:
    d = dates[bi]
    frames, fwd = frame_until(d)
    scores = {a: 0.0 for a in ASSETS}
    used = 0
    for fid, wf, direction in FACTORS:
        vals = {a: factor_values(frames[a], fid) for a in ASSETS}
        if sum(1 for v in vals.values() if v is not None) < 8:
            continue
        r = ranks(vals)
        for a in ASSETS:
            scores[a] += wf * (r[a] if direction > 0 else 1.0 - r[a])
        used += 1
    if used < 5:
        print(d.date(), "insufficient factors", used)
        continue
    rets = []
    for a in ASSETS:
        if frames[a] is not None and len(frames[a]) >= 25:
            rets.append(float(frames[a]["close"].pct_change().tail(20).mean()))
    regime = "bull" if np.mean(rets) > 0.015 else ("bear" if np.mean(rets) < -0.015 else "side")
    w = weights(scores, regime)
    # portfolio fwd return over next 10 days
    pr = sum(w[a] * fwd[a] for a in ASSETS if fwd[a] is not None)
    if prev_w is not None:
        to = sum(abs(w[a] - prev_w[a]) for a in ASSETS) / 2.0
        cost = to * 0.0003
    else:
        cost = 0.0
    net = pr - cost
    equity *= (1.0 + net)
    block_rets.append(net)
    details.append((str(d.date()), regime, round(pr * 100, 2), round(cost * 100, 3), round(net * 100, 2), round(equity * 100, 1)))
    prev_w = w

# metrics
rets = np.array(block_rets)
ann = (1 + rets.mean()) ** (252 / 10) - 1
sharpe = rets.mean() / rets.std() * np.sqrt(252 / 10) if rets.std() > 0 else 0.0
eq = np.cumprod(1 + rets)
dd = np.maximum.accumulate(eq) - eq
mdd = dd.max()
print("\nblocks:", len(rets), "| mean/block: %.3f%%" % (rets.mean() * 100))
print("ann ret: %.2f%% | sharpe(ann): %.2f | maxDD: %.2f%% | calmar: %.2f" % (ann * 100, sharpe, mdd * 100, ann / mdd if mdd > 0 else 0))
print("\nblock details (date, regime, gross%, cost%, net%, equity%):")
for drow in details:
    print(drow)
