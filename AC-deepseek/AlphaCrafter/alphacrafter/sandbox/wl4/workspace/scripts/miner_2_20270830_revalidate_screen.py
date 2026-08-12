"""miner_2 2027-08-30: revalidate library factors + screen candidates.
Data filtered to visible_through (2027-08-27). Rank IC h=10 on 15 assets.
Gates: |IC|>=0.0070, |ICIR|>=0.0840.
"""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import time

ASSETS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
          "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MACRO = {"DXY": "DXY.csv", "USDCNY": "USDCNY.csv", "USDJPY": "USDJPY.csv",
         "EURUSD": "EURUSD.csv", "VIX": "VIX.csv"}
VISIBLE_THROUGH = "2027-08-27"
T0 = time.time()

def load_asset(a):
    df = pd.read_csv(f"../persistent/stock_data/{a}.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df[df["date"] <= pd.Timestamp(VISIBLE_THROUGH)].reset_index(drop=True)

def load_macro(m):
    df = pd.read_csv(f"../persistent/index_data/{MACRO[m]}")
    df["date"] = pd.to_datetime(df["date"])
    return df[df["date"] <= pd.Timestamp(VISIBLE_THROUGH)].reset_index(drop=True)

PX = {a: load_asset(a) for a in ASSETS}
CLOSE = {a: dict(zip(px["date"], px["close"])) for a, px in PX.items()}
VOL = {a: dict(zip(px["date"], px["volume"])) for a, px in PX.items()}
MAC = {m: load_macro(m) for m in MACRO}
MAC_CLOSE = {m: dict(zip(df["date"], df["close"])) for m, df in MAC.items()}

DIDX = {}
for a in ASSETS:
    ds = sorted(CLOSE[a])
    DIDX[a] = {d: i for i, d in enumerate(ds)}

RETS = {}
for a in ASSETS:
    ds = sorted(CLOSE[a])
    r = {}
    for i in range(1, len(ds)):
        r[ds[i]] = CLOSE[a][ds[i]] / CLOSE[a][ds[i-1]] - 1.0
    RETS[a] = r

# ---------- factor implementations ----------
def f_vol_price_corr(a, win=20):
    ds = sorted(CLOSE[a]); out = {}
    for i in range(win, len(ds)):
        seg = ds[i-win:i+1]
        rs = np.array([RETS[a].get(d, np.nan) for d in seg[1:]])
        vs = np.array([VOL[a].get(d, np.nan) for d in seg[1:]])
        m = np.isfinite(rs) & np.isfinite(vs)
        if m.sum() >= 10 and np.std(rs[m]) > 0 and np.std(vs[m]) > 0:
            out[ds[i]] = np.corrcoef(rs[m], vs[m])[0, 1]
    return out

def f_rolling_beta(a, bench_ret, win=60):
    ds = sorted(CLOSE[a]); out = {}
    for i in range(win, len(ds)):
        seg = ds[i-win:i+1]
        rs = np.array([RETS[a].get(d, np.nan) for d in seg[1:]])
        bs = np.array([bench_ret.get(d, np.nan) for d in seg[1:]])
        m = np.isfinite(rs) & np.isfinite(bs)
        if m.sum() >= 30 and np.std(rs[m]) > 0 and np.std(bs[m]) > 0:
            out[ds[i]] = np.cov(rs[m], bs[m])[0, 1] / np.var(bs[m])
    return out

def f_risk_adj_mom(a, win=60):
    ds = sorted(CLOSE[a]); out = {}
    for i in range(win, len(ds)):
        seg = ds[i-win:i+1]
        rs = np.array([RETS[a].get(d, np.nan) for d in seg[1:]])
        rs = rs[np.isfinite(rs)]
        sd = np.std(rs)
        if sd > 0:
            out[ds[i]] = (CLOSE[a][ds[i]] / CLOSE[a][ds[i-win]] - 1.0) / sd
    return out

def f_downside_vol(a, win=20):
    ds = sorted(CLOSE[a]); out = {}
    for i in range(win, len(ds)):
        rs = np.array([RETS[a].get(d, np.nan) for d in ds[i-win:i]])
        rs = rs[np.isfinite(rs)]
        neg = rs[rs < 0]
        out[ds[i]] = np.std(neg) if len(neg) >= 3 else np.nan
    return out

def f_dist_high(a, win=60):
    ds = sorted(CLOSE[a]); out = {}
    for i in range(win, len(ds)):
        seg = ds[i-win:i+1]
        mx = max(CLOSE[a][d] for d in seg)
        out[ds[i]] = CLOSE[a][ds[i]] / mx - 1.0
    return out

def f_vol_ratio_20_60(a):
    ds = sorted(CLOSE[a]); out = {}
    for i in range(60, len(ds)):
        s20 = np.std([RETS[a].get(d, np.nan) for d in ds[i-20:i]])
        s60 = np.std([RETS[a].get(d, np.nan) for d in ds[i-60:i]])
        if np.isfinite(s60) and s60 > 0 and np.isfinite(s20):
            out[ds[i]] = s20 / s60
    return out

def f_skew_20(a):
    ds = sorted(CLOSE[a]); out = {}
    for i in range(20, len(ds)):
        rs = np.array([RETS[a].get(d, np.nan) for d in ds[i-20:i]])
        rs = rs[np.isfinite(rs)]
        if len(rs) >= 10 and np.std(rs) > 0:
            out[ds[i]] = float(pd.Series(rs).skew())
    return out

def f_zscore_close_60(a):
    ds = sorted(CLOSE[a]); out = {}
    for i in range(60, len(ds)):
        seg = np.array([CLOSE[a][d] for d in ds[i-60:i+1]])
        mu, sd = np.mean(seg), np.std(seg)
        if sd > 0:
            out[ds[i]] = (seg[-1] - mu) / sd
    return out

def f_mom_30_skip5(a):
    ds = sorted(CLOSE[a]); out = {}
    for i in range(5, len(ds)):
        j = i - 5
        if j - 30 >= 0:
            out[ds[i]] = CLOSE[a][ds[i]] / CLOSE[a][ds[j-30]] - 1.0
    return out

def f_volume_z_20(a):
    ds = sorted(CLOSE[a]); out = {}
    for i in range(20, len(ds)):
        vs = np.array([VOL[a].get(d, np.nan) for d in ds[i-20:i]])
        if np.isfinite(vs).all() and np.std(vs) > 0 and np.mean(vs) > 0:
            out[ds[i]] = (VOL[a][ds[i]] - np.mean(vs)) / np.std(vs)
    return out

def f_range_pos_20(a):
    ds = sorted(CLOSE[a]); out = {}
    for i in range(20, len(ds)):
        seg = ds[i-20:i+1]
        hi = max(CLOSE[a][d] for d in seg); lo = min(CLOSE[a][d] for d in seg)
        if hi > lo:
            out[ds[i]] = (CLOSE[a][ds[i]] - lo) / (hi - lo)
    return out

mkt_ret = {}
for d in sorted(set().union(*[set(RETS[a]) for a in ASSETS])):
    fv = [RETS[a].get(d, np.nan) for a in ASSETS]
    fv = [v for v in fv if np.isfinite(v)]
    if len(fv) >= 8:
        mkt_ret[d] = np.mean(fv)

def macro_ret_series(m):
    out = {}
    ds = sorted(MAC_CLOSE[m])
    for i in range(1, len(ds)):
        out[ds[i]] = MAC_CLOSE[m][ds[i]] / MAC_CLOSE[m][ds[i-1]] - 1.0
    return out

eur_ret = macro_ret_series("EURUSD")
vix_ret = macro_ret_series("VIX")
cn10y_ret = RETS["CN10Y"]

def ic_series(fvals, h=10):
    rows = []
    base_dates = sorted(fvals[ASSETS[0]].keys())
    for d in base_dates:
        xs, ys = [], []
        for a in ASSETS:
            fv = fvals[a].get(d)
            if fv is None or not np.isfinite(fv):
                continue
            j = DIDX[a].get(d)
            if j is None:
                continue
            ds = sorted(CLOSE[a])
            k = j + h
            if k >= len(ds):
                continue
            xs.append(fv); ys.append(CLOSE[a][ds[k]] / CLOSE[a][d] - 1.0)
        if len(xs) >= 8:
            ic = spearmanr(xs, ys).correlation
            if np.isfinite(ic):
                rows.append((d, ic))
    return rows

def summarize(name, fvals, h=10):
    rows = ic_series(fvals, h)
    ics = [r[1] for r in rows]
    if len(ics) < 30:
        return f"{name:24s} INSUFFICIENT dates n={len(ics)}"
    m, s = np.mean(ics), np.std(ics)
    icir = m / s if s > 0 else 0.0
    tot = sum(len(v) for v in fvals.values())
    valid = sum(int(np.isfinite(v)) for vv in fvals.values() for v in vv.values())
    cov = valid / tot if tot else 0
    # recent 250 obs
    rec = rows[-250:]
    rics = [r[1] for r in rec]
    rm, rs = np.mean(rics), np.std(rics)
    ricir = rm / rs if rs > 0 else 0.0
    gate = "PASS" if (abs(m) >= 0.0070 and abs(icir) >= 0.0840) else "fail"
    return (f"{name:24s} {gate:4s} full IC={m:+.4f} ICIR={icir:+.4f} n={len(ics):4d} hit={np.mean([1 if i>0 else 0 for i in ics]):.2f} "
            f"cov={cov:.2f} | recent250 IC={rm:+.4f} ICIR={ricir:+.4f} n={len(rec)}")

factors = {
    "vol_price_corr_20": {a: f_vol_price_corr(a) for a in ASSETS},
    "dn_mkt_beta_60d": {a: f_rolling_beta(a, mkt_ret, 60) for a in ASSETS},
    "eurusd_beta_60d": {a: f_rolling_beta(a, eur_ret, 60) for a in ASSETS},
    "rate_beta_cn10y_60d": {a: f_rolling_beta(a, cn10y_ret, 60) for a in ASSETS},
    "risk_adj_mom_60": {a: f_risk_adj_mom(a, 60) for a in ASSETS},
    "downside_vol_20": {a: f_downside_vol(a, 20) for a in ASSETS},
    "dist_high_60": {a: f_dist_high(a, 60) for a in ASSETS},
    "vol_ratio_20_60": {a: f_vol_ratio_20_60(a) for a in ASSETS},
    "skew_20": {a: f_skew_20(a) for a in ASSETS},
    "zscore_close_60": {a: f_zscore_close_60(a) for a in ASSETS},
    "mom_30_skip5": {a: f_mom_30_skip5(a) for a in ASSETS},
    "volume_z_20": {a: f_volume_z_20(a) for a in ASSETS},
    "range_pos_20": {a: f_range_pos_20(a) for a in ASSETS},
}

print(f"visible_through={VISIBLE_THROUGH} n_dates={len(RETS[ASSETS[0]])} n_assets={len(ASSETS)} elapsed={time.time()-T0:.0f}s", flush=True)
print("=" * 140)
for name, fv in factors.items():
    print(summarize(name, fv), flush=True)

print("\n=== max |corr| of candidates vs effective library (pooled) ===", flush=True)
def pooled(fvals):
    return {(d, a): v for a in ASSETS for d, v in fvals[a].items() if np.isfinite(v)}

lib = ["vol_price_corr_20", "dn_mkt_beta_60d", "eurusd_beta_60d", "rate_beta_cn10y_60d"]
cands = ["risk_adj_mom_60", "downside_vol_20", "dist_high_60", "vol_ratio_20_60",
         "skew_20", "zscore_close_60", "mom_30_skip5", "volume_z_20", "range_pos_20"]
pool = {k: pooled(factors[k]) for k in list(lib) + cands}
for c in cands:
    best, bestr = None, 0.0
    for l in lib:
        keys = sorted(set(pool[c]) & set(pool[l]))
        if len(keys) < 30:
            continue
        va = np.array([pool[c][k] for k in keys]); vb = np.array([pool[l][k] for k in keys])
        if np.std(va) == 0 or np.std(vb) == 0:
            continue
        r = abs(np.corrcoef(va, vb)[0, 1])
        if r > bestr:
            bestr, best = r, l
    print(f"  {c:22s} max_abs_lib_corr={bestr:.3f} vs {best}", flush=True)
print(f"done elapsed={time.time()-T0:.0f}s", flush=True)
