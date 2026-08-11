"""Screener 2027-03-05: independent factor IC check via simulator API (same source as strategy.py).

Computes cross-sectional Spearman IC (1d forward) for each library factor over:
  - w26: 2026-08-01..2027-03-04 (post-warmup live window)
  - recent: last 250 trading days
  - full: 2021-01-01..2027-03-04
Uses get_stock_daily_data (API) so calendar/data match the live strategy.
Read-only; does not touch account/date or the step/backtest tools.
"""
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data

WATCH = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
         "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MIN_VALID = 8
DAYS = 600

frames = {}
for a in WATCH:
    df = get_stock_daily_data(symbol=a, days=DAYS)
    if df is None or len(df) < 200:
        continue
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    frames[a] = df

idx = sorted(set().union(*[set(f.index) for f in frames.values()]))
idx = pd.DatetimeIndex([d for d in idx if pd.Timestamp("2020-01-01") <= d <= pd.Timestamp("2027-03-04")])

C = pd.DataFrame(index=idx, columns=WATCH, dtype=float)
O = pd.DataFrame(index=idx, columns=WATCH, dtype=float)
H = pd.DataFrame(index=idx, columns=WATCH, dtype=float)
L = pd.DataFrame(index=idx, columns=WATCH, dtype=float)
for a in WATCH:
    if a in frames:
        f = frames[a]
        C.loc[f.index, a] = f["close"].values
        O.loc[f.index, a] = f["open"].values
        H.loc[f.index, a] = f["high"].values
        L.loc[f.index, a] = f["low"].values
C = C.ffill(); O = O.ffill(); H = H.ffill(); L = L.ffill()
R = C.pct_change()
lnC = np.log(C)

def build(fid):
    fdf = pd.DataFrame(index=C.index, columns=WATCH, dtype=float)
    for a in WATCH:
        c, o, h, l = C[a], O[a], H[a], L[a]
        if fid == "nclv_1d":
            fdf[a] = -(c - l) / (h - l)
        elif fid == "nclv_2d":
            fdf[a] = -(c - l.rolling(2).min()) / (h.rolling(2).max() - l.rolling(2).min())
        elif fid == "nclv_3d":
            fdf[a] = -(c - l.rolling(3).min()) / (h.rolling(3).max() - l.rolling(3).min())
        elif fid == "nclv_5d":
            fdf[a] = -(c - l.rolling(5).min()) / (h.rolling(5).max() - l.rolling(5).min())
        elif fid == "rev_1d":
            fdf[a] = -lnC[a].diff(1)
        elif fid == "rev_2d":
            fdf[a] = -lnC[a].diff(2)
        elif fid == "rev_3d":
            fdf[a] = -lnC[a].diff(3)
        elif fid == "rev_5d":
            fdf[a] = -lnC[a].diff(5)
        elif fid == "nbody_1d":
            fdf[a] = -(c - o) / (h - l)
        elif fid == "id_rev_1d":
            fdf[a] = -(c / o - 1.0)
        elif fid == "rev_1d_vs":
            fdf[a] = -lnC[a].diff(1) / R[a].rolling(20).std()
        elif fid == "mom_120d_skip5":
            fdf[a] = c.shift(5) / c.shift(125) - 1.0
        elif fid == "vol_of_vol20x60":
            fdf[a] = R[a].rolling(20).std().rolling(60).std()
        else:
            return None
    return fdf.replace([np.inf, -np.inf], np.nan)

def daily_ic(fdf, fwd=1):
    fr = C.shift(-fwd) / C - 1.0
    out = []
    for i in range(len(fdf)):
        fv = fdf.iloc[i].values
        rv = fr.iloc[i].values
        m = np.isfinite(fv) & np.isfinite(rv)
        if m.sum() < MIN_VALID:
            continue
        rho = spearmanr(fv[m], rv[m]).correlation
        out.append(rho if np.isfinite(rho) else np.nan)
    return np.array(out)

def stats(ic):
    ok = np.isfinite(ic)
    if ok.sum() < 20:
        return dict(n=int(ok.sum()), ic=np.nan, icir=np.nan)
    ic = ic[ok]
    return dict(n=int(ok.sum()), ic=float(np.nanmean(ic)),
                icir=float(np.nanmean(ic) / np.nanstd(ic)))

fids = ["rev_1d", "rev_2d", "rev_3d", "rev_5d", "nclv_1d", "nclv_2d", "nclv_3d",
        "nclv_5d", "nbody_1d", "id_rev_1d", "rev_1d_vs", "mom_120d_skip5", "vol_of_vol20x60"]
print(f"{'factor':<16}{'full_ic':>9}{'full_icir':>10}{'rec_ic':>9}{'rec_icir':>10}{'w26_ic':>9}{'w26_icir':>10}{'w26_n':>7}")
for fid in fids:
    fdf = build(fid)
    if fdf is None:
        continue
    full = fdf.loc["2021-01-01":]
    ic_f = daily_ic(full)
    ic_r = daily_ic(full.iloc[-250:])
    ic_w = daily_ic(full.loc["2026-08-01":])
    s_f, s_r, s_w = stats(ic_f), stats(ic_r), stats(ic_w)
    print(f"{fid:<16}{s_f['ic']:>+9.4f}{s_f['icir']:>+10.3f}{s_r['ic']:>+9.4f}{s_r['icir']:>+10.3f}{s_w['ic']:>+9.4f}{s_w['icir']:>+10.3f}{s_w['n']:>7d}")
