"""miner1 2031-08-15: Re-validate trend-conditioned momentum family on fresh panel (cutoff 2031-08-14).
Idea (re-validation from 2031-07-04): plain momentum whipsaws in crypto/commodities
(BTC/ETH/WTI repeated drags per trader memory). Signal = mom_k * sign(close - MA_long),
neutral 0 when |z| < flat_z. Report across full/recent/last3y windows, with and without
flat-artifact series (HSI, CN10Y).
"""
import numpy as np
import pandas as pd
import json

PANEL = 'scripts/panel_cache_20310815.pkl'
GATE_IC = 0.0070
GATE_ICIR = 0.0840

with open(PANEL, 'rb') as f:
    panel = pd.read_pickle(f)
close = panel['close']
r = close.pct_change()

def forward_returns(close, horizons=(1, 2, 3, 5, 10, 15, 20)):
    out = {}
    for h in horizons:
        out[h] = close.shift(-h) / close - 1.0
    return out

def daily_ic(factor_df, fwd_ret, min_valid=8):
    ic, dates = [], []
    for dt in factor_df.index:
        f, rv = factor_df.loc[dt], fwd_ret.loc[dt]
        m = f.notna() & rv.notna()
        if m.sum() >= min_valid:
            ic.append(f[m].rank().corr(rv[m].rank()))
            dates.append(dt)
    return pd.Series(ic, index=pd.DatetimeIndex(dates))

def summarize(ic_s):
    ic = ic_s.dropna()
    if len(ic) == 0:
        return {'n_dates': 0, 'ic': np.nan, 'icir': np.nan, 'hit': np.nan}
    mean, std = ic.mean(), ic.std(ddof=1)
    return {'n_dates': len(ic), 'ic': float(mean),
            'icir': float(mean / std) if std > 0 else np.nan,
            'hit': float((ic > 0).mean())}

def coverage(factor_df):
    cov = factor_df.notna().mean(axis=1)
    return float(cov.mean()), float(cov.min())

def turnover_rank(factor_df):
    ranks = factor_df.rank(axis=1) / factor_df.notna().sum(axis=1)
    return float(ranks.diff().abs().mean().mean())

def library_corr_max(factor_df, close):
    lib = {}
    c = close
    lib['rev_2d'] = -(c.pct_change(2).shift(1))
    lib['nclv_1d'] = -(c / c.rolling(1).min() - 1.0)
    lib['vol_of_vol20x60'] = r.rolling(20).std() / r.rolling(60).std()
    lib['mom_120d_skip5'] = c.shift(5) / c.shift(125) - 1.0
    lib['vol_20d'] = r.rolling(20).std()
    best = 0.0
    for name, lf in lib.items():
        a = factor_df.stack()
        b = lf.stack()
        df = pd.concat([a.rename('f'), b.rename('l')], axis=1).dropna()
        if len(df) < 50:
            continue
        rho = df['f'].corr(df['l'])
        best = max(best, abs(rho))
    return float(best)

def report(name, factor_df, close, horizons=(1, 2, 3, 5, 10), window=None, exclude=None):
    if window is not None:
        factor_df = factor_df.loc[window[0]:window[1]]
    if exclude:
        factor_df = factor_df.drop(columns=[e for e in exclude if e in factor_df.columns])
    fwd = forward_returns(close.loc[factor_df.index.min():], horizons)
    print(f"\n=== {name}  [{factor_df.index.min().date()} .. {factor_df.index.max().date()}] "
          f"n_dates={len(factor_df)} n_assets={factor_df.shape[1]} excl={exclude or '-'}")
    res = {}
    for h in horizons:
        ic_s = daily_ic(factor_df, fwd[h])
        s = summarize(ic_s)
        res[h] = s
        passed = abs(s['ic']) >= GATE_IC and abs(s['icir']) >= GATE_ICIR
        print(f"  h={h:2d} IC={s['ic']:+.4f} ICIR={s['icir']:+.3f} hit={s['hit']:.3f} n={s['n_dates']} "
              f"{'PASS' if passed and s['n_dates'] > 200 else 'fail'}")
    cov_avg, cov_min = coverage(factor_df)
    to = turnover_rank(factor_df)
    maxrho = library_corr_max(factor_df, close)
    print(f"  coverage_avg={cov_avg:.3f} cov_min={cov_min:.3f} turnover_rank={to:.3f} max_lib_rho={maxrho:.3f}")
    return res, {'coverage_avg': cov_avg, 'coverage_min': cov_min, 'turnover_rank': to, 'max_lib_rho': maxrho}

# ---------- factor definitions ----------
def trend_cond_mom(close, k=20, long=60, flat_z=0.5):
    mom = close / close.shift(k) - 1.0
    ma_long = close.rolling(long).mean()
    z = (close - ma_long) / close.rolling(long).std()
    trend = np.sign(z).where(z.abs() >= flat_z, 0.0)
    return mom * trend

def plain_mom(close, k=20):
    return close / close.shift(k) - 1.0

cands = {
    'tcm_20x60': trend_cond_mom(close, 20, 60, 0.5),
    'tcm_10x120': trend_cond_mom(close, 10, 120, 0.5),
    'mom_20d_plain': plain_mom(close, 20),
}

windows = {
    'full_2020_2031': ('2020-06-01', '2031-08-14'),
    'recent_2026_2031': ('2026-01-01', '2031-08-14'),
    'last3y_2028_2031': ('2028-08-15', '2031-08-14'),
}

for name, f in cands.items():
    for wname, w in windows.items():
        report(name, f, close, horizons=(1, 5, 10), window=w)
    # robustness without flat artifacts
    report(name, f, close, horizons=(5, 10), window=('2026-01-01', '2031-08-14'),
           exclude=['HSI', 'CN10Y'])

print("\n=== Decay profile tcm_20x60 (recent 2026-2031) ===")
f = cands['tcm_20x60'].loc['2026-01-01':'2031-08-14']
for h in [1, 2, 3, 5, 8, 10, 15, 20]:
    ic_s = daily_ic(f, forward_returns(close, (h,))[h])
    s = summarize(ic_s)
    print(f"  h={h:2d} IC={s['ic']:+.4f} ICIR={s['icir']:+.3f} hit={s['hit']:.3f} n={s['n_dates']}")

print("\n=== Decay profile tcm_10x120 (last3y) ===")
f2 = cands['tcm_10x120'].loc['2028-08-15':'2031-08-14']
for h in [1, 2, 3, 5, 8, 10, 15, 20]:
    ic_s = daily_ic(f2, forward_returns(close, (h,))[h])
    s = summarize(ic_s)
    print(f"  h={h:2d} IC={s['ic']:+.4f} ICIR={s['icir']:+.3f} hit={s['hit']:.3f} n={s['n_dates']}")
