"""Screener 2027-03-19 cycle: API-based factor IC at 1/5/10d horizons (matches live holding cadence),
factor correlation, and market regime assessment. Data through last completed trading day 2027-03-18."""
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
    if ok.sum() < 20: return dict(n=int(ok.sum()), ic=np.nan, icir=np.nan)
    ic = ic[ok]
    return dict(n=int(ok.sum()), ic=float(np.nanmean(ic)), icir=float(np.nanmean(ic) / np.nanstd(ic)))

fids = ["rev_1d","rev_2d","rev_3d","rev_5d","nclv_1d","nclv_2d","nclv_3d","nclv_5d",
        "nbody_1d","id_rev_1d","rev_1d_vs","mom_120d_skip5","vol_of_vol20x60","vix_beta_cond_60x20"]

print("=== IC by horizon: since-2026-08 window (post-warmup live period) ===")
print(f"{'factor':<20}{'ic1':>8}{'ic5':>8}{'ic10':>8}{'ic1_icir':>10}{'ic5_icir':>10}{'ic10_icir':>11}")
rows = {}
for fid in fids:
    fdf = build(fid)
    if fdf is None: continue
    w = fdf.loc["2026-08-01":]
    s1, s5, s10 = stats(daily_ic(w,1)), stats(daily_ic(w,5)), stats(daily_ic(w,10))
    rows[fid] = (s1, s5, s10)
    print(f"{fid:<20}{s1['ic']:>+8.4f}{s5['ic']:>+8.4f}{s10['ic']:>+8.4f}{s1['icir']:>+10.3f}{s5['icir']:>+10.3f}{s10['icir']:>+11.3f}")

print("\n=== IC by horizon: recent 250 trading days ===")
print(f"{'factor':<20}{'ic1':>8}{'ic5':>8}{'ic10':>8}{'ic1_icir':>10}{'ic5_icir':>10}{'ic10_icir':>11}")
rows250 = {}
for fid in fids:
    fdf = build(fid)
    if fdf is None: continue
    w = fdf.iloc[-250:]
    s1, s5, s10 = stats(daily_ic(w,1)), stats(daily_ic(w,5)), stats(daily_ic(w,10))
    rows250[fid] = (s1, s5, s10)
    print(f"{fid:<20}{s1['ic']:>+8.4f}{s5['ic']:>+8.4f}{s10['ic']:>+8.4f}{s1['icir']:>+10.3f}{s5['icir']:>+10.3f}{s10['icir']:>+11.3f}")

print("\n=== Quality q = |IC10|*|ICIR10| on since-2026-08 window (with sign) ===")
q = {}
for fid in fids:
    if fid in rows:
        s10 = rows[fid][2]
        q[fid] = (np.sign(s10['ic']) if np.isfinite(s10['ic']) else 0.0) * abs(s10['ic']) * abs(s10['icir'])
for fid in sorted(q, key=lambda k: -abs(q[k])):
    print(f"{fid:<20}{q[fid]:>+10.4f}")

# Factor correlation (average pairwise across recent window)
print("\n=== Factor signal correlation (recent 250d, daily cross-section mean) ===")
sig = {}
for fid in fids:
    fdf = build(fid)
    if fdf is None: continue
    w = fdf.iloc[-250:]
    sig[fid] = w.mean(axis=1)
S = pd.DataFrame(sig)
corr = S.corr()
fids_sorted = sorted(fids)
for i, a in enumerate(fids_sorted):
    line = []
    for b in fids_sorted:
        v = corr.loc[a, b]
        line.append(f"{v:+.2f}" if np.isfinite(v) else "  NA")
    print(f"{a:<20}" + " ".join(line))

# Regime assessment
print("\n=== Regime assessment (through 2027-03-18) ===")
for a in WATCH:
    c = C[a].dropna()
    if len(c) < 250: continue
    r = c.pct_change().dropna()
    ma50 = c.rolling(50).mean().iloc[-1]
    ma200 = c.rolling(200).mean().iloc[-1]
    last = c.iloc[-1]
    ret20 = c.iloc[-1]/c.iloc[-21] - 1
    ret60 = c.iloc[-1]/c.iloc[-61] - 1
    vol20 = r.iloc[-20:].std()*np.sqrt(252)
    dd = (c/c.cummax()-1).iloc[-1]
    print(f"{a:<12} close={last:>10.2f} ma50={ma50:>10.2f} ma200={ma200:>10.2f} "
          f"r20={ret20:>+7.1%} r60={ret60:>+7.1%} vol20={vol20:>6.1%} dd={dd:>7.1%}")

# Cross-asset correlation regime
rc = R.iloc[-60:]
cc = rc.corr()
vals = cc.values[np.triu_indices(len(cc), 1)]
print(f"\nAvg pairwise corr (60d, 15 assets): {np.nanmean(vals):.3f}")
rc250 = R.iloc[-250:]
cc250 = rc250.corr()
vals250 = cc250.values[np.triu_indices(len(cc250), 1)]
print(f"Avg pairwise corr (250d, 15 assets): {np.nanmean(vals250):.3f}")

# VIX level
vix_last = VIX.iloc[-1]
print(f"VIX last={vix_last:.1f} 20d ago={VIX.iloc[-21]:.1f} chg20={(vix_last/VIX.iloc[-21]-1):+.1%}")
