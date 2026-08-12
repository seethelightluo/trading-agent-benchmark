"""
miner_1 cycle 2028-04-20 (v3): fix trend_r2 partial-window coverage; re-evaluate
regime-hybrid candidates; compute max_abs_library_correlation vs library artifacts.
"""
import numpy as np
import pandas as pd
from scipy import stats
import glob, os, json

SYMS = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU',
        'COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END = '2028-04-20'

px = {}
for s in SYMS:
    df = pd.read_csv(f'../persistent/stock_data/{s}.csv', parse_dates=['date'])
    df = df[df['date'] <= END].set_index('date')['close']
    px[s] = df
px = pd.DataFrame(px).sort_index()
px = px.dropna(how='all')

vix = pd.read_csv('../persistent/index_data/VIX.csv', parse_dates=['date'])
vix = vix[vix['date'] <= END].set_index('date')['close']
vix = vix.reindex(px.index).ffill()

ret = px.pct_change()
logpx = np.log(px)

def rolling_vol(s, w):
    return s.rolling(w, min_periods=max(10, w // 2)).std() * np.sqrt(252)

fA = -(px / px.shift(10) - 1.0) / rolling_vol(ret, 60)
fB = -(px / px.shift(5) - 1.0) / rolling_vol(ret, 20)
mom20 = px.shift(5) / px.shift(25) - 1.0
rev5 = -(px / px.shift(5) - 1.0) / rolling_vol(ret, 20)

def trend_r2(series, w=60, min_pts=45):
    """R^2 of linear fit on log price over trailing w rows, tolerating NaN gaps."""
    out = pd.Series(np.nan, index=series.index)
    vals = series.values
    for i in range(w - 1, len(series)):
        y = vals[i - w + 1:i + 1]
        m = ~np.isnan(y)
        if m.sum() < min_pts:
            continue
        x = np.arange(w)[m]
        _, _, r, _, _ = stats.linregress(x, y[m])
        out.iloc[i] = r ** 2
    return out

fD = logpx.apply(trend_r2, w=60)

def regime_hybrid(lo_factor, hi_factor, thr=22.0):
    out = pd.DataFrame(np.nan, index=px.index, columns=px.columns)
    lo = vix < thr
    out[lo] = lo_factor[lo]
    out[~lo] = hi_factor[~lo]
    return out

THR = 22.0
fC = regime_hybrid(mom20, rev5, THR)          # vixreg_mom20
fF = regime_hybrid(fD, fA, THR)               # vixreg_r2rev
fG = fD                                        # trend_r2_60 standalone

factors = {
    'vixreg_mom20': (fC, +1),
    'trend_r2_60':  (fG, +1),
    'vixreg_r2rev': (fF, +1),
}

def ic_series(factor, fwd, min_valid=8):
    dates, ics = [], []
    for d in factor.index:
        fv = factor.loc[d]
        fr = fwd.loc[d]
        mask = fv.notna() & fr.notna()
        if mask.sum() < min_valid:
            continue
        ic, _ = stats.spearmanr(fv[mask], fr[mask])
        if np.isnan(ic):
            continue
        dates.append(d); ics.append(ic)
    return pd.Series(ics, index=pd.DatetimeIndex(dates))

def report(name, direction, fwd, min_valid=8):
    ic = ic_series(factors[name][0], fwd, min_valid)
    if len(ic) < 60:
        print(f'{name}: too few IC dates ({len(ic)})'); return None
    adj = ic * direction
    mu, sd = adj.mean(), adj.std(ddof=1)
    icir = mu / sd if sd > 0 else 0.0
    hit = (adj > 0).mean()
    f = factors[name][0]
    fz = f.rank(axis=1, pct=True)
    turn = (fz - fz.shift(10)).abs().mean().mean()
    cov = f.notna().mean().mean()
    cov8 = (f.notna().sum(axis=1) >= 8).mean()
    decay = {}
    for h in [1, 2, 3, 5, 10, 20]:
        fwd_h = px.shift(-h) / px - 1.0
        ic_h = ic_series(factors[name][0], fwd_h, min_valid)
        if len(ic_h) > 60:
            decay[h] = round((ic_h.mean() * direction), 4)
    last250 = ic[ic.index >= ic.index[-1] - pd.Timedelta(days=365)]
    print(f'--- {name} dir={direction:+d} n={len(ic)} {ic.index.min().date()}..{ic.index.max().date()}')
    print(f'    IC={mu:.4f} ICIR={icir:.3f} hit={hit:.3f} turn10={turn:.3f} cov_ad={cov:.3f} '
          f'cov_d8={cov8:.3f} last250IC={last250.mean():.4f} decay{decay}')
    return {'name': name, 'ic': mu, 'icir': icir, 'hit': hit, 'turn': turn,
            'cov': cov, 'cov8': cov8, 'last250': last250.mean(), 'decay': decay, 'n': len(ic)}

fwd10 = px.shift(-10) / px - 1.0
print('================ HORIZON 10d FULL SAMPLE (fixed coverage) ================')
results = {}
for name in factors:
    r = report(name, factors[name][1], fwd10)
    if r:
        results[name] = r

print('\n================ REGIME SPLITS (10d, dir-adj) ================')
for name, (f, direction) in factors.items():
    ic = ic_series(f, fwd10)
    if len(ic) < 60:
        continue
    adj = ic * direction
    yr = adj.groupby(adj.index.year).mean()
    vix_r = vix.reindex(ic.index).ffill()
    lo = adj[vix_r < 20]; hi = adj[vix_r >= 20]
    print(f'{name}: yearly {dict(round(yr, 4))}')
    if len(lo) > 40:
        print(f'   VIX<20 : n={len(lo)} IC={lo.mean():.4f} ICIR={lo.mean()/lo.std(ddof=1):.3f}')
    if len(hi) > 40:
        print(f'   VIX>=20: n={len(hi)} IC={hi.mean():.4f} ICIR={hi.mean()/hi.std(ddof=1):.3f}')

# ---------------- library correlation (pooled spearman, overlapping dates) ----------------
print('\n================ LIBRARY CORRELATION ================')
npy_files = sorted(glob.glob('factors/*.signal.npy'))
print('library artifacts found:', [os.path.basename(f) for f in npy_files])

# reconstruct the artifact date axis: assume rows align with px index <= 2026-07-29
art_dates = px.index[px.index <= '2026-07-29']
print('px rows through 2026-07-29:', len(art_dates))

def pooled_spearman(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 100:
        return np.nan
    rho, _ = stats.spearmanr(a[m], b[m])
    return rho

for name, (f, direction) in factors.items():
    fv = f.loc[art_dates].rank(axis=1).values.astype(float)
    corrs = {}
    for npy in npy_files:
        lib_id = os.path.basename(npy).replace('.signal.npy', '')
        arr = np.load(npy)
        if arr.shape != fv.shape:
            continue
        lib = np.nan_to_num(arr, nan=np.nan)
        # rank-transform library for comparability
        libr = pd.DataFrame(lib).rank(axis=1).values.astype(float)
        corrs[lib_id] = pooled_spearman(fv, libr)
    cs = {k: round(v, 3) for k, v in corrs.items() if v == v}
    mx = max(cs.values(), default=np.nan)
    print(f'{name}: max_abs_lib_corr={mx:.3f}  top: {sorted(cs.items(), key=lambda kv: -abs(kv[1]))[:6]}')
