"""Shared loader + validation utilities for miner_3 factor research (2034-10-02 cycle).

Data hygiene: only data through the last completed trading day (2034-09-29) is used.
Macro index CSVs under ../persistent/index_data/ extend beyond the current date; clip them.
"""
import os
import json
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

WATCH = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
         'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']
MACRO = ['DXY', 'USDCNY', 'USDJPY', 'EURUSD', 'VIX']
CUTOFF = pd.Timestamp('2034-09-29')  # last completed trading day


def load_assets(days=10000):
    closes, rets = {}, {}
    for s in WATCH:
        df = get_stock_daily_data(symbol=s, days=days)
        if df is None:
            continue
        df = df[df['date'] <= CUTOFF].copy()
        df = df.set_index('date').sort_index()
        closes[s] = df['close']
        rets[s] = df['close'].pct_change()
    px = pd.DataFrame(closes).dropna(how='all')
    rt = pd.DataFrame(rets)
    return px, rt


def load_macro():
    out = {}
    for m in MACRO:
        p = f'../persistent/index_data/{m}.csv'
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p)
        df['date'] = pd.to_datetime(df['date'])
        df = df[df['date'] <= CUTOFF].set_index('date').sort_index()
        out[m] = df['close']
    return pd.DataFrame(out)


def build_forward_returns(px, horizons=(1, 2, 3, 5, 10, 20)):
    fwd = {}
    for h in horizons:
        fwd[h] = px.shift(-h) / px - 1.0
    return fwd


def rank_ic_panel(factor_df, fwd_ret_df, min_valid=8):
    """Cross-sectional Spearman IC per date. factor_df/fwd_ret_df aligned (date x symbol)."""
    dates, ics = [], []
    common = factor_df.index.intersection(fwd_ret_df.index)
    for d in common:
        f = factor_df.loc[d].dropna()
        r = fwd_ret_df.loc[d].reindex(f.index).dropna()
        f = f.reindex(r.index)
        if len(f) < min_valid:
            continue
        ic = f.rank().corr(r.rank())
        if np.isfinite(ic):
            dates.append(d)
            ics.append(ic)
    return pd.Series(ics, index=dates)


def factor_stats(factor_df, px, horizons=(1, 2, 3, 5, 10, 20), min_valid=8):
    """Full validation stats for a factor panel."""
    fwd = build_forward_returns(px, horizons)
    stats = {}
    ic_series = {}
    for h in horizons:
        ic = rank_ic_panel(factor_df, fwd[h], min_valid)
        ic_series[h] = ic
    h_main = 10
    ic = ic_series[h_main]
    stats['ic'] = float(ic.mean())
    stats['icir'] = float(ic.mean() / ic.std()) if ic.std() > 0 else 0.0
    stats['ic_std'] = float(ic.std())
    stats['ic_hit_ratio'] = float((ic > 0).mean()) if len(ic) else np.nan
    stats['n_ic_dates'] = int(len(ic))
    stats['decay_ic_by_horizon'] = {str(h): float(ic_series[h].mean()) for h in horizons}
    # coverage
    total_cells = len(factor_df) * factor_df.shape[1]
    stats['coverage_asset_days'] = float(factor_df.notna().sum().sum() / max(total_cells, 1))
    n_dates_ge8 = sum(1 for d in factor_df.index if factor_df.loc[d].notna().sum() >= min_valid)
    stats['coverage_dates_ge8'] = float(n_dates_ge8 / max(len(factor_df), 1))
    # turnover: mean abs rank change over 10d
    rk = factor_df.rank(axis=1)
    rk10 = rk.sub(rk.shift(10))
    stats['turnover_10d_rank'] = float(rk10.abs().mean().mean())
    # recent-window checks
    for w in (250, 500):
        sub = ic[ic.index >= ic.index[-1] - pd.Timedelta(days=w * 1.6)]
        if len(sub) > 20:
            stats[f'ic_r{w}'] = float(sub.mean())
            stats[f'icir_r{w}'] = float(sub.mean() / sub.std()) if sub.std() > 0 else 0.0
    return stats, ic_series


def max_library_corr(factor_df, factor_id, lib_dir='factors'):
    """Max |pearson rho| between z-scored signal panels of new factor and library factors."""
    sig = factor_df.stack().rename('value').reset_index()
    sig.columns = ['date', 'symbol', 'value']
    best = 0.0
    best_name = None
    for f in sorted(os.listdir(lib_dir)):
        if not f.endswith('_signal.csv'):
            continue
        if f.startswith(factor_id):
            continue
        try:
            lib = pd.read_csv(os.path.join(lib_dir, f))
        except Exception:
            continue
        m = sig.merge(lib, on=['date', 'symbol'], suffixes=('_a', '_b'))
        if len(m) < 200:
            continue
        rho = float(np.corrcoef(m['value_a'], m['value_b'])[0, 1])
        if np.isfinite(rho) and abs(rho) > best:
            best = abs(rho)
            best_name = f
    return best, best_name


def write_signal_artifact(factor_id, factor_df, out_dir='factors'):
    sig = factor_df.stack().rename('value').reset_index()
    sig.columns = ['date', 'symbol', 'value']
    sig = sig.dropna()
    path = os.path.join(out_dir, f'{factor_id}_signal.csv')
    sig.to_csv(path, index=False)
    return path


def recent_summary(ic_series, label):
    """Yearly IC breakdown to check regime robustness (rough calendar-year split)."""
    ic = ic_series[10]
    ic = ic[ic.index >= '2024-01-01']
    out = {}
    for y in sorted(set(ic.index.year)):
        sub = ic[ic.index.year == y]
        if len(sub) >= 20:
            out[str(y)] = {'ic': round(float(sub.mean()), 4),
                           'icir': round(float(sub.mean() / sub.std()), 3),
                           'n': int(len(sub))}
    return out
