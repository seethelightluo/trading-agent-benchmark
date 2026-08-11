"""Screener 2027-03-19: recent IC windows (120d/60d) + current factor exposures."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data

WATCH = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
         "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MIN_VALID = 8
DAYS = 700
END = pd.Timestamp("2027-03-18")

frames = {}
for a in WATCH:
    df = get_stock_daily_data(symbol=a, days=DAYS)
    if df is None or len(df) < 200:
        continue
    df = df.copy(); df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    frames[a] = df

idx = sorted(set().union(*[set(f.index) for f in frames.values()]))
idx = pd.DatetimeIndex([d for d in idx if pd.Timestamp("2020-01-01") <= d <= END])
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

VIX = pd.read_csv("../persistent/index_data/VIX.csv")
VIX["date"] = pd.to_datetime(VIX["date"])
VIX = VIX[VIX["date"] <= END].set_index("date")["close"].astype(float)

def build(fid):
    fdf = pd.DataFrame(index=C.index, columns=WATCH, dtype=float)
    for a in WATCH:
        c, o, h, l = C[a], O[a], H[a], L[a]
        if fid == "nclv_1d": fdf[a] = -(c - l) / (h - l)
        elif fid == "nclv_2d": fdf[a] = -(c - l.rolling(2).min()) / (h.rolling(2).max() - l.rolling(2).min())
        elif fid == "nclv_3d": fdf[a] = -(c - l.rolling(3).min()) / (h.rolling(3).max() - l.rolling(3).min())
        elif fid == "nclv_5d": fdf[a] = -(c - l.rolling(5).min()) / (h.rolling(5).max() - l.rolling(5).min())
        elif fid == "rev_1d": fdf[a] = -lnC[a].diff(1)
        elif fid == "rev_2d": fdf[a] = -lnC[a].diff(2)
        elif fid == "rev_3d": fdf[a] = -lnC[a].diff(3)
        elif fid == "rev_5d": fdf[a] = -lnC[a].diff(5)
        elif fid == "nbody_1d": fdf[a] = -(c - o) / (h - l)
        elif fid == "id_rev_1d": fdf[a] = -(c / o - 1.0)
        elif fid == "rev_1d_vs": fdf[a] = -lnC[a].diff(1) / R[a].rolling(20).std()
        elif fid == "mom_120d_skip5": fdf[a] = c.shift(5) / c.shift(125) - 1.0
        elif fid == "vol_of_vol20x60": fdf[a] = R[a].rolling(20).std().rolling(60).std()
        elif fid == "vix_beta_cond_60x20":
            v = VIX.reindex(C.index).ffill()
            ar = c.pct_change(); vr = v.pct_change()
            beta = ar.rolling(60).cov(vr) / vr.rolling(60).var()
            vm = v / v.shift(20) - 1.0
            fdf[a] = -beta * vm
        else: return None
    return fdf.replace([np.inf, -np.inf], np.nan)

def daily_ic(fdf, fwd):
    fr = C.shift(-fwd) / C - 1.0
    out = []
    for i in range(len(fdf)):
        fv = fdf.iloc[i].values; rv = fr.iloc[i].values
        m = np.isfinite(fv) & np.isfinite(rv)
        if m.sum() < MIN_VALID: continue
        if np.all(fv[m] == fv[m][0]): continue
        rho = spearmanr(fv[m], rv[m]).correlation
        out.append(rho if np.isfinite(rho) else np.nan)
    return np.array(out)

def stats(ic):
    ok = np.isfinite(ic)
    if ok.sum() < 15: return dict(n=int(ok.sum()), ic=np.nan, icir=np.nan)
    ic = ic[ok]
    return dict(n=int(ok.sum()), ic=float(np.nanmean(ic)), icir=float(np.nanmean(ic) / np.nanstd(ic)))

fids = ["rev_1d","rev_2d","rev_3d","rev_5d","nclv_1d","nclv_2d","nclv_3d","nclv_5d",
        "nbody_1d","id_rev_1d","rev_1d_vs","mom_120d_skip5","vol_of_vol20x60","vix_beta_cond_60x20"]

print("=== IC10 by recent window (10d forward, matches block horizon) ===")
print(f"{'factor':<20}{'last60':>10}{'last60_ir':>10}{'last120':>10}{'last120_ir':>11}")
for fid in fids:
    fdf = build(fid)
    if fdf is None: continue
    s60 = stats(daily_ic(fdf.iloc[-60:], 10))
    s120 = stats(daily_ic(fdf.iloc[-120:], 10))
    print(f"{fid:<20}{s60['ic']:>+10.4f}{s60['icir']:>+10.3f}{s120['ic']:>+10.4f}{s120['icir']:>+11.3f}")

print("\n=== Current factor exposures (2027-03-18): rank of raw signal across 15 assets (0=low,1=high) ===")
fids_sel = ["nclv_5d","nclv_3d","nclv_1d","rev_1d","nbody_1d","rev_1d_vs",
            "mom_120d_skip5","vol_of_vol20x60","vix_beta_cond_60x20"]
sig = {}
for fid in fids_sel:
    fdf = build(fid)
    sig[fid] = fdf.iloc[-1]
S = pd.DataFrame(sig)
ranked = S.rank()
print(f"{'asset':<12}" + "".join(f"{f:>16}" for f in fids_sel))
for a in WATCH:
    print(f"{a:<12}" + "".join(f"{ranked.loc[a,f]:>16.2f}" for f in fids_sel))
