#!/usr/bin/env python
"""SCREENER cycle 2028-01-13: regime-aware factor refresh.
Computes rank IC / ICIR for the active factor library on the 15-asset
cross-asset universe and rebuilds factor_ensemble.json via quality_ic_tilt
(q = |IC|*|ICIR|, direction = sign(IC)). Data visible only through 2028-01-12.
No live-account/backtest/step interaction.
"""
import json
import numpy as np
import pandas as pd

ASSETS = ['000300.SH','000688.SH','SPX','HSI','N225','SX5E','SOX','NDX',
          'XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END = '2028-01-12'

def load_close(sym, base='../persistent/stock_data'):
    df = pd.read_csv(f'{base}/{sym}.csv')
    df.columns = [c.strip() for c in df.columns]
    dc = 'date' if 'date' in df.columns else df.columns[0]
    df[dc] = df[dc].astype(str)
    df = df[df[dc] <= END].sort_values(dc).reset_index(drop=True)
    cc = 'close' if 'close' in df.columns else df.columns[3]
    s = pd.Series(df[cc].to_numpy(), index=pd.to_datetime(df[dc]), name=sym)
    return s

def load_macro(sym, base='../persistent/index_data'):
    df = pd.read_csv(f'{base}/{sym}.csv')
    df.columns = [c.strip() for c in df.columns]
    dc = 'date' if 'date' in df.columns else df.columns[0]
    df[dc] = df[dc].astype(str)
    df = df[df[dc] <= END].sort_values(dc).reset_index(drop=True)
    cc = 'close' if 'close' in df.columns else df.columns[3]
    return pd.Series(df[cc].to_numpy(), index=pd.to_datetime(df[dc]), name=sym)

px = pd.DataFrame({a: load_close(a) for a in ASSETS})
dxy = load_macro('DXY')
vix = load_macro('VIX')
wti = px['WTI']

ret = px.pct_change()
fwd = px.shift(-10) / px - 1.0  # 10d forward return

# ---- factor builders (must match factor JSON specs) ----
def trend_r2_30(p):
    lp = np.log(p)
    t = np.arange(len(lp))
    def rol(x):
        x = x.dropna()
        if len(x) < 18: return np.nan
        tt = t[-len(x):]
        c = np.cov(tt, x)[0, 1]
        v = np.var(x)
        if v <= 0: return np.nan
        return np.sign(c) * c**2 / (np.var(tt) * v)
    return lp.rolling(30, min_periods=18).apply(rol, raw=False)

def semi_down_ratio_20(r):
    dn = r.clip(upper=0) ** 2
    up = r.clip(lower=0) ** 2
    d = dn.rolling(20).mean().apply(np.sqrt)
    u = up.rolling(20).mean().apply(np.sqrt)
    return d / u - 1.0

def mom_120(p):
    return p.shift(5) / p.shift(125) - 1.0

def dxy_beta_60(r, dxy):
    dr = dxy.pct_change()
    cov = r.rolling(60).cov(dr)
    var = dr.rolling(60).var()
    return cov / var

def vol_of_vol(r):
    v20 = r.rolling(20).std()
    return v20.rolling(60).std()

def mom_10(p):
    return p.shift(5) / p.shift(15) - 1.0

def time_under_water_120(p):
    rm = p.rolling(120, min_periods=60).max()
    out = pd.DataFrame(np.nan, index=p.index, columns=p.columns)
    for c in p.columns:
        s = p[c]; m = rm[c]
        cnt = 0
        vals = []
        for i in range(len(s)):
            if np.isnan(m.iloc[i]):
                vals.append(np.nan); continue
            if s.iloc[i] >= m.iloc[i] - 1e-12:
                cnt = 0
            else:
                cnt += 1
            vals.append(cnt)
        out[c] = vals
    return out

def vix_beta_cond(r, vix):
    vr = vix.pct_change()
    b = r.rolling(60).cov(vr) / vr.rolling(60).var()
    vix20 = (vix / vix.shift(20) - 1.0)
    return -b * vix20

def tail_ratio_20(r):
    out = pd.DataFrame(np.nan, index=r.index, columns=r.columns)
    for c in r.columns:
        out[c] = r[c].rolling(20, min_periods=10).apply(
            lambda x: np.nanpercentile(x, 95) / abs(np.nanpercentile(x, 5)), raw=True)
    return out

def kurt_20(r):
    def kurt(x):
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        if len(x) < 8 or np.std(x) == 0: return np.nan
        m2 = np.mean((x - x.mean())**2)
        m4 = np.mean((x - x.mean())**4)
        return m4 / m2**2 - 3.0
    return r.rolling(20, min_periods=8).apply(kurt, raw=True)

def wti_beta_60(r, wti):
    wr = wti.pct_change()
    cov = r.rolling(60).cov(wr)
    var = wr.rolling(60).var()
    return cov / var

FACTORS = {
    'trend_r2_30_signed': trend_r2_30(px),
    'semi_down_ratio_20': semi_down_ratio_20(ret),
    'mom_120d_skip5': mom_120(px),
    'dxy_beta_60': dxy_beta_60(ret, dxy),
    'vol_of_vol20x60': vol_of_vol(ret),
    'mom_10d_skip5': mom_10(px),
    'time_under_water_120': time_under_water_120(px),
    'vix_beta_cond_60x20': vix_beta_cond(ret, vix),
    'tail_ratio_20': tail_ratio_20(ret),
    'kurt_20': kurt_20(ret),
    'WTI_BETA_60': wti_beta_60(ret, wti),
}

# ---- rank IC on dates with >= 8 valid assets ----
def rank_ic_series(fv, fwd_ret, min_assets=8):
    idx = fv.index.intersection(fwd_ret.index)
    ics = []
    for dt in idx:
        f = fv.loc[dt]; r = fwd_ret.loc[dt]
        m = f.notna() & r.notna()
        if m.sum() < min_assets: continue
        if f[m].nunique() < 3: continue
        ic = f[m].rank().corr(r[m].rank())
        if np.isfinite(ic): ics.append((dt, ic))
    if not ics: return pd.Series(dtype=float)
    return pd.Series([v for _, v in ics], index=[d for d, _ in ics])

print(f"{'factor':24s} {'IC_all':>8s} {'ICIR_all':>9s} {'IC_120':>8s} {'ICIR_120':>9s} {'IC_60':>8s} {'ICIR_60':>9s} {'q120':>7s} {'dir':>4s}")
results = {}
for fid, fv in FACTORS.items():
    ic_series = rank_ic_series(fv, fwd)
    if len(ic_series) == 0:
        print(f"{fid:24s}  no data"); continue
    ic_all = ic_series.mean()
    icir_all = ic_series.mean() / ic_series.std() * np.sqrt(len(ic_series)) if ic_series.std() > 0 else 0
    ic_120 = ic_series.tail(120).mean()
    icir_120 = ic_series.tail(120).mean() / ic_series.tail(120).std() * np.sqrt(120) if ic_series.tail(120).std() > 0 else 0
    ic_60 = ic_series.tail(60).mean()
    icir_60 = ic_series.tail(60).mean() / ic_series.tail(60).std() * np.sqrt(60) if ic_series.tail(60).std() > 0 else 0
    results[fid] = dict(ic_all=ic_all, icir_all=icir_all, ic_120=ic_120, icir_120=icir_120,
                        ic_60=ic_60, icir_60=icir_60, n=len(ic_series))
    print(f"{fid:24s} {ic_all:8.4f} {icir_all:9.3f} {ic_120:8.4f} {icir_120:9.3f} {ic_60:8.4f} {icir_60:9.3f} {abs(ic_120)*abs(icir_120):7.4f} {np.sign(ic_120):+.0f}")

# ---- factor cross-sectional correlation (avg pairwise corr of rank exposures, last 120d) ----
print('\n=== factor exposure correlation (avg abs pairwise, last 120d) ===')
corrs = {}
fids = list(FACTORS.keys())
for i in range(len(fids)):
    for j in range(i+1, len(fids)):
        fi, fj = FACTORS[fids[i]], FACTORS[fids[j]]
        ri = fi.tail(120).rank(axis=1)
        rj = fj.tail(120).rank(axis=1)
        common = ri.index.intersection(rj.index)
        if len(common) < 30: continue
        cs = []
        for dt in common:
            a = ri.loc[dt]; b = rj.loc[dt]
            m = a.notna() & b.notna()
            if m.sum() >= 8:
                c = a[m].corr(b[m])
                if np.isfinite(c): cs.append(c)
        if cs:
            corrs[(fids[i], fids[j])] = np.mean(cs)
top = sorted(corrs.items(), key=lambda kv: -abs(kv[1]))[:15]
for (a, b), c in top:
    print(f'{a:24s} vs {b:24s} corr={c:+.3f}')

with open('scripts/screener_diag_20280113.json', 'w') as fh:
    json.dump({k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()}, fh, indent=2)
print('\nsaved scripts/screener_diag_20280113.json')
