"""miner_2 deep validation of risk_mom_20x60 (2026-11-19 cycle).
Gate: |IC|>=0.007, |ICIR|>=0.084 at h=10. Includes regime breakdown,
decay, turnover, coverage, and max abs correlation vs live library factors.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

CUTOFF = pd.Timestamp('2026-11-04')
ASSETS = ['000300.SH','000688.SH','BTC','CN10Y','COPPER','ETH','HSI','N225',
          'NDX','SOX','SPX','SX5E','US10Y','WTI','XAU']
HORIZONS = [1, 2, 3, 5, 10, 20]
MIN_VALID = 8

def load_close(sym, base):
    df = pd.read_csv(base / f'{sym}.csv', parse_dates=['date'])
    df = df[df['date'] <= CUTOFF].set_index('date').sort_index()
    return df['close'].astype(float)

px = pd.DataFrame({a: load_close(a, Path('../persistent/stock_data')) for a in ASSETS}).sort_index()
vix = load_close('VIX', Path('../persistent/index_data'))
ret = px.pct_change()
vix_ret = vix.pct_change()

def rolling_beta(x_df, y_s, w, mp=None):
    mp = mp if mp is not None else w
    my = y_s.rolling(w, min_periods=mp).mean()
    cov = x_df.mul(y_s, axis=0).rolling(w, min_periods=mp).mean() - x_df.rolling(w, min_periods=mp).mean().mul(my, axis=0)
    var = (y_s ** 2).rolling(w, min_periods=mp).mean() - my ** 2
    return cov.div(var, axis=0)

vol60 = ret.rolling(60, min_periods=30).std(ddof=0)
fac = (px.shift(5) / px.shift(25) - 1.0) / vol60   # risk_mom_20x60

# library panels (strategy definitions)
lib = {}
lib['mom_10d_skip5'] = px.shift(5) / px.shift(15) - 1.0
lib['mom_120d_skip5'] = px.shift(5) / px.shift(125) - 1.0
lib['vol_of_vol20x60'] = ret.rolling(20, min_periods=10).std(ddof=0).rolling(60, min_periods=20).std(ddof=0)
v20 = rolling_beta(ret, vix_ret, 60, mp=30)
vix_move = vix / vix.shift(20) - 1.0
lib['vix_beta_cond_60x20'] = -v20.mul(vix_move, axis=0)

idx = px.index
fwd_rank = {h: (px.shift(-h) / px - 1.0).rank(axis=1).values for h in HORIZONS}

def ic_series(fac_panel):
    fac_r = fac_panel.reindex(idx).rank(axis=1).values
    pos = {h: [] for h in HORIZONS}
    val = {h: [] for h in HORIZONS}
    for i in range(len(idx)):
        xi = fac_r[i]
        for h in HORIZONS:
            yi = fwd_rank[h][i]
            m = ~(np.isnan(xi) | np.isnan(yi))
            if m.sum() < MIN_VALID:
                continue
            xm, ym = xi[m], yi[m]
            xm = xm - xm.mean(); ym = ym - ym.mean()
            denom = np.sqrt((xm * xm).sum() * (ym * ym).sum())
            if denom > 0:
                pos[h].append(i); val[h].append(float((xm * ym).sum() / denom))
    return {h: pd.Series(val[h], index=idx[pos[h]]) for h in HORIZONS}

def stats(ic):
    if len(ic) < 30:
        return None
    m, s = ic.mean(), ic.std(ddof=0)
    return dict(n=int(len(ic)), ic=float(m), icir=float(m / s) if s > 0 else 0.0,
                hit=float((np.sign(ic) == np.sign(m)).mean()))

ic_all = ic_series(fac)
ic10 = ic_all[10]
print('=== risk_mom_20x60 @ h=10 ===')
st_full = stats(ic10.values)
print('FULL  :', st_full)
for lo, hi, lab in [('2020-01-01','2022-12-31','2020-2022'), ('2023-01-01','2024-12-31','2023-2024'),
                    ('2025-01-01','2026-11-04','2025-2026')]:
    sub = ic10.loc[(ic10.index >= lo) & (ic10.index <= hi)].values
    print(f'{lab}:', stats(sub))
recent = ic10.loc[ic10.index >= '2025-06-01'].values
print('RECENT(>=2025-06):', stats(recent))
print('decay (mean IC per horizon):', {str(h): round(float(ic_all[h].mean()), 4) for h in HORIZONS})
print('decay (ICIR per horizon)   :', {str(h): round(float(stats(ic_all[h].values)['icir']), 3) if stats(ic_all[h].values) else None for h in HORIZONS})

cov_ad = float(fac.notna().mean().mean())
cov_d8 = float((fac.notna().sum(axis=1) >= MIN_VALID).mean())
r = fac.rank(axis=1, pct=True)
to = float(r.diff(10).abs().mean().mean())
print(f'coverage asset-day={cov_ad:.3f} dates_ge8={cov_d8:.3f} turnover10d={to:.3f}')

# correlation vs library (rank signals, cross-sectional, paired dates)
def rank_panel(p):
    return p.reindex(idx).rank(axis=1)
fr = rank_panel(fac)
corrs = {}
for fid, p in lib.items():
    lr = rank_panel(p)
    both = fr.notna() & lr.notna()
    vals = []
    for d in idx[both.sum(axis=1) >= MIN_VALID]:
        a = fr.loc[d].values; b = lr.loc[d].values
        m = ~(np.isnan(a) | np.isnan(b))
        if m.sum() < MIN_VALID:
            continue
        x, y = a[m], b[m]
        x = x - x.mean(); y = y - y.mean()
        den = np.sqrt((x*x).sum()*(y*y).sum())
        if den > 0:
            vals.append(float((x*y).sum()/den))
    corrs[fid] = round(float(np.mean(vals)), 3) if vals else None
print('mean daily rank-rho vs library:', corrs)
print('max_abs_library_correlation:', round(max(abs(v) for v in corrs.values() if v is not None), 3))

json.dump({
    'factor': 'risk_mom_20x60', 'h': 10, 'full': st_full,
    'regime': {'2020-2022': stats(ic10.loc[(ic10.index>='2020-01-01')&(ic10.index<='2022-12-31')].values),
               '2023-2024': stats(ic10.loc[(ic10.index>='2023-01-01')&(ic10.index<='2024-12-31')].values),
               '2025-2026': stats(ic10.loc[(ic10.index>='2025-01-01')&(ic10.index<=CUTOFF)].values),
               'recent2025H2': stats(recent)},
    'decay_ic': {str(h): round(float(ic_all[h].mean()), 4) for h in HORIZONS},
    'coverage_asset_day': round(cov_ad, 3), 'coverage_dates_ge8': round(cov_d8, 3),
    'turnover_10d': round(to, 3),
    'rank_rho_library': corrs,
    'max_abs_library_correlation': round(max(abs(v) for v in corrs.values() if v is not None), 3),
}, open('scripts/miner_2_20261119_risk_mom_deep.json', 'w'), indent=1)
print('saved scripts/miner_2_20261119_risk_mom_deep.json')
