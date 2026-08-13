"""miner2 2031-08-15: shared factor validation helpers.
Computes daily rank IC / ICIR / hit / coverage / turnover / decay on the
15-name cross-asset panel through 2031-08-14 (visible data).
"""
import pandas as pd
import numpy as np

PANEL_PATH = 'scripts/panel_cache_20310815.pkl'


def load_panel():
    panel = pd.read_pickle(PANEL_PATH)
    cp = panel['close']
    hi = panel['high']
    lo = panel['low']
    op = panel['open']
    vo = panel['vol']
    macro = panel['macro']
    return cp, hi, lo, op, vo, macro


def forward_returns(cp, horizons=(1, 2, 3, 5, 10, 20)):
    fr = {}
    for h in horizons:
        fr[h] = cp.shift(-h) / cp - 1.0
    return fr


def daily_rank_ic(factor_df, fwd):
    """factor_df: dates x assets. fwd: dates x assets (same index)."""
    ics = []
    dates = factor_df.index.intersection(fwd.index)
    for d in dates:
        x = factor_df.loc[d]
        y = fwd.loc[d]
        m = x.notna() & y.notna()
        if m.sum() >= 8:
            ic = pd.Series(x[m]).rank().corr(pd.Series(y[m]).rank())
            if pd.notna(ic):
                ics.append((d, ic, m.sum()))
    if not ics:
        return None
    idx = pd.Index([t[0] for t in ics], name='date')
    ic_series = pd.Series([t[1] for t in ics], index=idx)
    n_obs = pd.Series([t[2] for t in ics], index=idx)
    return ic_series, n_obs


def summarize(ic_series, n_obs, direction=1):
    ic = ic_series.mean()
    icir = ic_series.mean() / ic_series.std(ddof=1) if ic_series.std(ddof=1) > 0 else 0.0
    hit = (ic_series * direction > 0).mean()
    return {
        'ic': float(ic),
        'icir': float(icir),
        'hit': float(hit),
        'n_dates': int(len(ic_series)),
        'n_obs_mean': float(n_obs.mean()),
        'ic_std': float(ic_series.std(ddof=1)),
    }


def coverage_turnover(factor_df):
    cov = float(factor_df.notna().mean().mean())
    # turnover: mean abs change of cross-sectional z-score between consecutive days
    z = factor_df.sub(factor_df.mean(axis=1), axis=0).div(factor_df.std(axis=1), axis=0)
    dz = z.diff().abs().mean().mean()
    return cov, float(dz) if pd.notna(dz) else float('nan')


def full_eval(factor_df, fwd_map, direction=1, label=''):
    ic_series, n_obs = daily_rank_ic(factor_df, fwd_map[1])
    if ic_series is None:
        print(f'{label}: NO VALID DATES')
        return None
    s1 = summarize(ic_series, n_obs, direction)
    cov, to = coverage_turnover(factor_df)
    out = {'horizon1': s1, 'coverage': cov, 'turnover_z': to}
    for h in (2, 3, 5, 10, 20):
        ic2, n2 = daily_rank_ic(factor_df, fwd_map[h])
        if ic2 is not None:
            out[f'ic{h}'] = float(ic2.mean())
            out[f'icir{h}'] = float(ic2.mean() / ic2.std(ddof=1)) if ic2.std(ddof=1) > 0 else 0.0
    return out


def by_year(ic_series):
    yr = ic_series.groupby(ic_series.index.year)
    return {
        str(y): {'ic': float(g.mean()), 'icir': float(g.mean() / g.std(ddof=1)) if g.std(ddof=1) > 0 else 0.0,
                 'n': int(len(g))}
        for y, g in yr
    }


def max_lib_corr(factor_df, lib_signals, exclude_ids=()):
    """Max absolute pairwise Pearson corr of factor signal vs library signal panels."""
    best = None
    for fid, sig in lib_signals.items():
        if fid in exclude_ids:
            continue
        m = factor_df.notna() & sig.notna()
        if m.sum().sum() < 500:
            continue
        a = factor_df[m].values.ravel()
        b = sig[m].values.ravel()
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            continue
        r = float(np.corrcoef(a, b)[0, 1])
        if not np.isnan(r) and (best is None or abs(r) > abs(best[1])):
            best = (fid, r)
    return best
