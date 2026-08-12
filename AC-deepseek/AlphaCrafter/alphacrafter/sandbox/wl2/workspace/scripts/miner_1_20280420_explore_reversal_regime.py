"""
miner_1 cycle 2028-04-20 (v2): fixed coverage (min_periods), direction-adjusted ICIR,
added regime-hybrid candidates + drawdown-depth + library correlation for passers.
Candidates:
  A rev_10d_vol60   : -ret10d/vol60            (dir -1, vol-scaled reversal)
  B rev_5d_vol20    : -ret5d/vol20             (dir -1, short reversal)
  C vixreg_mom20    : VIX<thr -> +mom20_skip5 ; else -> -ret5d/vol20 (dir +1)
  D trend_r2_60     : R^2 of 60d linear fit on log close (dir +1)
  E ar1_60          : AR(1) coeff of daily returns 60d (dir +1)
  F vixreg_r2rev    : VIX<thr -> trend_r2_60 ; else -> -ret10d/vol60 (dir +1)
  H dd_from_high_120: close/rollmax(close,120)-1 (distance below high, dir +1 dip-buy)
"""
import numpy as np
import pandas as pd
from scipy import stats

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

def trend_r2(series, w=60):
    out = pd.Series(np.nan, index=series.index)
    vals = series.values
    x = np.arange(w)
    for i in range(w - 1, len(series)):
        y = vals[i - w + 1:i + 1]
        if np.isnan(y).any():
            continue
        _, _, r, _, _ = stats.linregress(x, y)
        out.iloc[i] = r ** 2
    return out

fD = logpx.apply(trend_r2, w=60)

def ar1_coef(s, w=60):
    out = pd.Series(np.nan, index=s.index)
    v = s.values
    for i in range(w, len(s)):
        y = v[i - w + 1:i + 1]
        x = v[i - w:i]
        m = ~(np.isnan(x) | np.isnan(y))
        if m.sum() < 20:
            continue
        slope, _, _, _, _ = stats.linregress(x[m], y[m])
        out.iloc[i] = slope
    return out

fE = ret.apply(ar1_coef, w=60)

def regime_hybrid(lo_factor, hi_factor, thr=22.0):
    out = pd.DataFrame(np.nan, index=px.index, columns=px.columns)
    lo = vix < thr
    out[lo] = lo_factor[lo]
    out[~lo] = hi_factor[~lo]
    return out

THR = 22.0
fC = regime_hybrid(mom20, rev5, THR)
fF = regime_hybrid(fD, fA, THR)
fH = px / px.rolling(120, min_periods=60).max() - 1.0

factors = {
    'rev_10d_vol60': (fA, -1),
    'rev_5d_vol20':  (fB, -1),
    'vixreg_mom20':  (fC, +1),
    'trend_r2_60':   (fD, +1),
    'ar1_60':        (fE, +1),
    'vixreg_r2rev':  (fF, +1),
    'dd_from_high_120': (fH, +1),
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
print('================ HORIZON 10d FULL SAMPLE ================')
results = {}
for name in factors:
    r = report(name, factors[name][1], fwd10)
    if r:
        results[name] = r

print('\n================ REGIME SPLITS (10d) ================')
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

# sensitivity of threshold
print('\n================ VIX THRESHOLD SENSITIVITY (dir-adj IC, 10d) ================')
for thr in [18.0, 20.0, 22.0, 25.0]:
    c = regime_hybrid(mom20, rev5, thr)
    ic = ic_series(c, fwd10)
    adj = ic * 1.0
    print(f'vixreg_mom20 thr={thr}: n={len(ic)} IC={adj.mean():.4f} ICIR={adj.mean()/adj.std(ddof=1):.3f}')
    c2 = regime_hybrid(fD, fA, thr)
    ic2 = ic_series(c2, fwd10)
    adj2 = ic2 * 1.0
    print(f'vixreg_r2rev thr={thr}: n={len(ic2)} IC={adj2.mean():.4f} ICIR={adj2.mean()/adj2.std(ddof=1):.3f}')
