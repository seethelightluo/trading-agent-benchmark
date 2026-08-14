"""Shared loader + IC harness for miner_3 factor work, cycle 2034-10-26.

Data cutoff: visible_through = 2034-10-25 (previous completed trading day).
CSV files in ../persistent/stock_data contain future rows; we MUST clip to the
visible date to avoid look-ahead in factor validation.
"""
import numpy as np
import pandas as pd

WATCH = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
         'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']
CUTOFF = pd.Timestamp('2034-10-25')


def load_all():
    out = {}
    for s in WATCH:
        df = pd.read_csv(f'../persistent/stock_data/{s}.csv')
        df['date'] = pd.to_datetime(df['date'])
        df = df[df['date'] <= CUTOFF].sort_values('date').reset_index(drop=True)
        out[s] = df
    return out


def load_macro(name):
    df = pd.read_csv(f'../persistent/index_data/{name}.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] <= CUTOFF].sort_values('date').reset_index(drop=True)
    return df


def close_panel(data):
    return pd.DataFrame({s: d.set_index('date')['close'] for s, d in data.items()}).sort_index()


def ret_panel(data):
    return close_panel(data).pct_change()


def forward_ret(px, h):
    return px.shift(-h) / px - 1.0


def daily_spearman_ic(factor_series, fwd, min_valid=8):
    dates = factor_series.index.intersection(fwd.index)
    ic_recs = []
    for dt in dates:
        f = factor_series.loc[dt]
        r = fwd.loc[dt]
        mask = f.notna() & r.notna() & np.isfinite(f.astype(float)) & np.isfinite(r.astype(float))
        n = int(mask.sum())
        if n < min_valid:
            continue
        ic = f[mask].astype(float).corr(r[mask].astype(float), method='spearman')
        if np.isfinite(ic):
            ic_recs.append((dt, ic, n))
    out = pd.DataFrame(ic_recs, columns=['date', 'ic', 'n']).set_index('date')
    return out


def ic_stats(ic_df):
    if len(ic_df) == 0:
        return None
    ic = ic_df['ic']
    sd = ic.std(ddof=1)
    return {
        'n_ic_dates': int(len(ic)),
        'ic': float(ic.mean()),
        'icir': float(ic.mean() / sd) if sd > 0 else np.nan,
        'ic_hit_ratio': float((ic > 0).mean()),
        'ic_std': float(sd),
        'median_n': float(ic_df['n'].median()),
    }


def coverage_stats(factor_series):
    valid = factor_series.notna().sum(axis=1)
    return {
        'coverage_asset_days': float(factor_series.notna().mean().mean()),
        'coverage_dates_ge8': float((valid >= 8).mean()),
    }


def rank_turnover(factor_series, h=10):
    ranks = factor_series.rank(axis=1, pct=True)
    diffs = (ranks - ranks.shift(h)).abs()
    valid = ranks.notna().sum(axis=1)
    sub = diffs[valid >= 8]
    if len(sub) == 0:
        return np.nan
    return float(sub.mean().mean())


def decay_ics(factor_series, px, horizons=(1, 2, 3, 5, 10, 20)):
    out = {}
    for h in horizons:
        fwd = forward_ret(px, h)
        m = ic_stats(daily_spearman_ic(factor_series, fwd))
        out[h] = m['ic'] if m else np.nan
    return out


def print_stats(label, m, cov=None, to=None, decay=None, flags=None):
    if m is None:
        print(f'--- {label}: NO IC DATES ---')
        return
    print(f'--- {label} ---')
    print(f"  n_dates={m['n_ic_dates']} ic={m['ic']:.4f} icir={m['icir']:.3f} "
          f"hit={m['ic_hit_ratio']:.3f} median_n={m['median_n']:.1f}")
    if cov:
        print(f"  coverage_asset_days={cov['coverage_asset_days']:.3f} "
              f"coverage_dates_ge8={cov['coverage_dates_ge8']:.3f}")
    if to is not None:
        print(f'  turnover_10d_rank={to:.4f}')
    if decay:
        print('  decay_ic_by_horizon:', {k: round(v, 4) for k, v in decay.items()})
    if flags:
        print('  flags:', flags)


def gate_pass(m, ic_thr=0.007, icir_thr=0.084):
    if m is None:
        return False
    return abs(m['ic']) >= ic_thr and abs(m['icir']) >= icir_thr
