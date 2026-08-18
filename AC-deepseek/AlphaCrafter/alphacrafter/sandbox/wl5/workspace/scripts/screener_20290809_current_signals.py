"""Screener: compute current (2029-08-08) factor signals on the 15-asset universe.
Read-only analysis; does NOT call rebalance_to_weights or step anything.
Reimplements strategy.py factor math locally to avoid importing the live module.
"""
import json
from math import isfinite, copysign
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

WATCH = ["000300.SH", "000688.SH", "BTC", "CN10Y", "COPPER", "ETH", "HSI",
         "N225", "NDX", "SOX", "SPX", "SX5E", "US10Y", "WTI", "XAU"]
FETCH = 220


def _closes(assets):
    out = {}
    for a in assets:
        df = None
        try:
            df = get_stock_daily_data(a, days=FETCH)
        except Exception:
            df = None
        if df is None or len(df) < 140:
            try:
                df = get_index_daily_data(a, days=FETCH)
            except Exception:
                df = None
        if df is not None and len(df) >= 140 and "close" in df:
            s = df[["date", "close"]].copy()
            s["date"] = pd.to_datetime(s["date"])
            out[a] = s.set_index("date")["close"].astype(float)
    return out


def _macro_close(symbol):
    df = None
    try:
        df = get_index_daily_data(symbol, days=150)
    except Exception:
        df = None
    if df is None or "close" not in df or len(df) < 80:
        return None
    s = df[["date", "close"]].copy()
    s["date"] = pd.to_datetime(s["date"])
    return s.set_index("date")["close"].astype(float)


def _trend_r2(c):
    s = c.dropna().tail(30)
    if len(s) < 18:
        return None
    y = np.log(s.values.astype(float))
    x = np.arange(len(y))
    cov = float(np.cov(y, x)[0, 1])
    vy, vx = float(np.var(y)), float(np.var(x))
    if vy <= 0 or vx <= 0:
        return None
    return copysign(cov * cov / (vy * vx), cov)


def _semi_down_ratio(r):
    s = r.dropna().tail(20)
    if len(s) < 10:
        return None
    down = float((s.clip(upper=0) ** 2).mean() ** 0.5)
    up = float((s.clip(lower=0) ** 2).mean() ** 0.5)
    if up < 1e-12:
        return None
    return down / up - 1.0


def _mom_120(c):
    if len(c) < 126:
        return None
    p0 = float(c.iloc[-126])
    if p0 <= 0:
        return None
    return float(c.iloc[-6]) / p0 - 1.0


def _mom_10(c):
    if len(c) < 17:
        return None
    p0 = float(c.iloc[-16])
    if p0 <= 0:
        return None
    return float(c.iloc[-6]) / p0 - 1.0


def _underwater(c):
    s = c.dropna().tail(125)
    if len(s) < 60:
        return None
    w = s.tail(120).values.astype(float)
    roll = np.maximum.accumulate(w)
    mask = w == roll
    idx = np.flatnonzero(mask)
    return float(len(w) - 1 - idx[-1]) if len(idx) else float(len(w))


def _vol_of_vol(r):
    s = r.dropna().tail(120)
    if len(s) < 90:
        return None
    v = s.rolling(20).std()
    out = v.rolling(60).std().iloc[-1]
    return None if not isfinite(out) else float(out)


def _kurt_20(r):
    s = r.dropna().tail(40)
    if len(s) < 20:
        return None
    k = s.rolling(20, min_periods=8).kurt().iloc[-1]
    return None if not isfinite(k) else float(k)


def _tail_ratio(r):
    s = r.dropna().tail(20)
    if len(s) < 10:
        return None
    q95 = float(np.percentile(s.values, 95))
    q05 = float(np.percentile(s.values, 5))
    if abs(q05) < 1e-12:
        return None
    return q95 / abs(q05)


def _dxy_beta(r, dxy_r):
    z = pd.concat([r.rename("a"), dxy_r.rename("d")], axis=1).dropna().tail(60)
    if len(z) < 30:
        return None
    vd = float(z["d"].var())
    if vd < 1e-14:
        return None
    return float(z["a"].cov(z["d"]) / vd)


def _vix_beta_cond(r, vix_r, vix_c):
    z = pd.concat([r.rename("a"), vix_r.rename("v")], axis=1).dropna().tail(60)
    if len(z) < 30:
        return None
    vv = float(z["v"].var())
    if vv < 1e-14:
        return None
    beta = float(z["a"].cov(z["v"]) / vv)
    if vix_c is None or len(vix_c) < 22:
        return None
    v0 = float(vix_c.iloc[-21])
    if v0 <= 0:
        return None
    vmove = float(vix_c.iloc[-1]) / v0 - 1.0
    return -beta * vmove


def _rank_map(values, assets):
    valid = sorted((float(v), a) for a, v in values.items()
                   if v is not None and isfinite(float(v)))
    out = {a: 0.5 for a in assets}
    n = len(valid)
    if n >= 2:
        for i, (_, a) in enumerate(valid):
            out[a] = i / (n - 1)
    return out


closes = _closes(WATCH)
panel = pd.DataFrame(closes).sort_index()
rets = panel.pct_change()
dxy_r = _macro_close("DXY").pct_change()
vix_c = _macro_close("VIX")
vix_r = vix_c.pct_change()

print("last close date:", panel.index[-1].date(), "| n assets:", len(closes))
print("\n=== 10d forward-ish context: last 10d & 60d asset returns ===")
for a in WATCH:
    c = closes[a]
    r10 = c.iloc[-1] / c.iloc[-11] - 1 if len(c) > 11 else float("nan")
    r60 = c.iloc[-1] / c.iloc[-61] - 1 if len(c) > 61 else float("nan")
    print(f"{a:10s} 10d={r10*100:+6.2f}%  60d={r60*100:+7.2f}%  px={c.iloc[-1]:12.2f}")

FUNCS = {
    "trend_r2_30_signed": lambda c, r: _trend_r2(c),
    "semi_down_ratio_20": lambda c, r: _semi_down_ratio(r),
    "mom_120d_skip5": lambda c, r: _mom_120(c),
    "mom_10d_skip5": lambda c, r: _mom_10(c),
    "vol_of_vol20x60": lambda c, r: _vol_of_vol(r),
    "time_under_water_120": lambda c, r: _underwater(c),
    "tail_ratio_20": lambda c, r: _tail_ratio(r),
    "kurt_20": lambda c, r: _kurt_20(r),
    "dxy_beta_60": lambda c, r: _dxy_beta(r, dxy_r) if dxy_r is not None else None,
    "vix_beta_cond_60x20": lambda c, r: _vix_beta_cond(r, vix_r, vix_c) if vix_r is not None else None,
}

fvals = {}
print("\n=== Current factor raw values (as-of 2029-08-08) ===")
for fid, fn in FUNCS.items():
    fvals[fid] = {}
    for a in WATCH:
        c, r = closes[a], rets[a]
        try:
            fvals[fid][a] = fn(c, r)
        except Exception:
            fvals[fid][a] = None
    rk = _rank_map(fvals[fid], WATCH)
    order = sorted(WATCH, key=lambda a: -rk[a])
    top3 = ", ".join(f"{a}({fvals[fid][a]:.3f})" for a in order[:3])
    bot3 = ", ".join(f"{a}({fvals[fid][a]:.3f})" for a in order[-3:])
    print(f"{fid:24s} top: {top3}")
    print(f"{'':24s} bot: {bot3}")

# ensemble score under v10 weights and a candidate v11 tilt
ENS = {
    "trend_r2_30_signed": (0.1908, 1),
    "semi_down_ratio_20": (0.1759, -1),
    "vol_of_vol20x60": (0.1142, 1),
    "mom_120d_skip5": (0.1022, 1),
    "time_under_water_120": (0.0989, -1),
    "vix_beta_cond_60x20": (0.0863, -1),
    "dxy_beta_60": (0.0830, 1),
    "mom_10d_skip5": (0.0786, 1),
    "tail_ratio_20": (0.0701, 1),
}
print("\n=== v10 ensemble implied asset scores (rank-only) ===")
score = {a: 0.0 for a in WATCH}
for fid, (w, d) in ENS.items():
    rk = _rank_map(fvals[fid], WATCH)
    for a in WATCH:
        score[a] += w * d * rk[a]
for a in sorted(WATCH, key=lambda a: -score[a]):
    print(f"{a:10s} score={score[a]:+.4f}")
