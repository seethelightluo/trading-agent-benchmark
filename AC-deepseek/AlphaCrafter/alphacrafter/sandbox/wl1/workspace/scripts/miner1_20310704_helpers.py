"""miner1 2031-07-04: shared validation helpers (panel 2031-07-03 cutoff)."""
import numpy as np
import pandas as pd
import json

PANEL = 'scripts/panel_cache_20310704.pkl'
GATE_IC = 0.0070
GATE_ICIR = 0.0840


def load_panel():
    with open(PANEL, 'rb') as f:
        panel = pd.read_pickle(f)
    return panel


def forward_returns(close, horizons=(1, 2, 3, 5, 10)):
    out = {}
    for h in horizons:
        out[h] = close.shift(-h) / close - 1.0
    return out


def daily_ic(factor_df, fwd_ret, min_valid=8):
    """Cross-sectional Spearman IC per date (require >=8 valid instruments)."""
    ic, dates = [], []
    for dt in factor_df.index:
        f, r = factor_df.loc[dt], fwd_ret.loc[dt]
        m = f.notna() & r.notna()
        if m.sum() >= min_valid:
            ic.append(f[m].rank().corr(r[m].rank()))
            dates.append(dt)
    return pd.Series(ic, index=pd.DatetimeIndex(dates))


def summarize(ic_s):
    ic = ic_s.dropna()
    if len(ic) == 0:
        return {'n_dates': 0, 'ic': np.nan, 'icir': np.nan, 'hit': np.nan}
    mean, std = ic.mean(), ic.std(ddof=1)
    return {'n_dates': len(ic), 'ic': float(mean), 'icir': float(mean / std) if std > 0 else np.nan,
            'hit': float((ic > 0).mean())}


def coverage(factor_df):
    cov = factor_df.notna().mean(axis=1)
    return float(cov.mean()), float(cov.min())


def turnover_rank(factor_df):
    ranks = factor_df.rank(axis=1) / factor_df.notna().sum(axis=1)
    return float(ranks.diff().abs().mean().mean())


def library_corr_max(factor_df, close):
    """Max |rho| of factor vs persisted library factors reconstructed from close."""
    lib = {}
    c = close
    lib['rev_2d'] = -(c.pct_change(2).shift(1))
    lib['nclv_1d'] = -(c / c.rolling(1).min() - 1.0)
    r = c.pct_change()
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


def report(name, factor_df, close, horizons=(1, 2, 3, 5, 10), window=None, label=None):
    if window is not None:
        factor_df = factor_df.loc[window[0]:window[1]]
    fwd = forward_returns(close.loc[factor_df.index.min():], horizons)
    print(f"\n=== {label or name}  [{factor_df.index.min().date()} .. {factor_df.index.max().date()}] "
          f"n_dates={len(factor_df)} n_assets={factor_df.shape[1]}")
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
