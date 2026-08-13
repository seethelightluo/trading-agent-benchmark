"""Shared library for miner_1 factor research (current date 2031-05-19).
Loads daily data for the 15 tradable watchlist instruments through the
visible date (2031-05-16) and provides factor computation + IC utilities.
Data source: alphacrafter.sim.utils.get_stock_daily_data (respects simulator visibility).
"""
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

WATCH = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
         'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']

def load_prices(days=5000):
    closes, vols = {}, {}
    for s in WATCH:
        df = get_stock_daily_data(symbol=s, days=days)
        if df is None or len(df) < 100:
            print(f'WARN: {s} short data {0 if df is None else len(df)}')
            continue
        df = df.copy()
        df['date'] = pd.to_datetime(df['date']).dt.normalize()
        df = df.set_index('date').sort_index()
        closes[s] = df['close']
        vols[s] = df['volume'] if 'volume' in df.columns else pd.Series(np.nan, index=df.index)
    px = pd.DataFrame(closes).sort_index()
    vo = pd.DataFrame(vols).sort_index()
    return px, vo

def fwd_ret(px, h=10):
    """Forward h-day return (t -> t+h) using close-to-close."""
    return px.shift(-h) / px - 1.0

def ic_series(factor_df, ret_df):
    """Daily cross-sectional Spearman IC between factor values and forward returns.
    Returns Series indexed by date; dates with <8 valid instruments are dropped."""
    dates, vals = [], []
    idx = factor_df.index.intersection(ret_df.index)
    for d in idx:
        f = factor_df.loc[d].dropna()
        r = ret_df.loc[d].reindex(f.index).dropna()
        common = f.index.intersection(r.index)
        if len(common) < 8:
            continue
        ic = f.loc[common].corr(r.loc[common], method='spearman')
        if np.isfinite(ic):
            dates.append(d)
            vals.append(ic)
    return pd.Series(vals, index=pd.DatetimeIndex(dates), name='ic')

def summarize_ic(ic, label='', horizon=10, verbose=True):
    if len(ic) == 0:
        print(f'{label}: NO IC OBS')
        return {}
    ic = ic.dropna()
    n = len(ic)
    mean_ic = ic.mean()
    std_ic = ic.std(ddof=1)
    icir = mean_ic / std_ic * np.sqrt(n) if std_ic > 0 else 0.0
    hit = (ic > 0).mean()
    res = dict(n_ic_dates=n, ic=float(mean_ic), icir=float(icir), ic_std=float(std_ic),
               ic_hit_ratio=float(hit))
    if verbose:
        print(f'{label} h={horizon}: n={n} IC={mean_ic:+.4f} ICIR={icir:+.3f} '
              f'hit={hit:.3f} std={std_ic:.3f}  GATE: |IC|>=0.007 & |ICIR|>=0.084 -> '
              f'{"PASS" if abs(mean_ic)>=0.007 and abs(icir)>=0.084 else "FAIL"}')
    return res

def decay_profile(factor_df, px, horizons=(1, 2, 3, 5, 10, 20), verbose=True):
    out = {}
    for h in horizons:
        r = fwd_ret(px, h)
        ic = ic_series(factor_df, r)
        out[str(h)] = float(ic.mean()) if len(ic) else np.nan
    if verbose:
        print('  decay IC by horizon:', {k: round(v, 4) for k, v in out.items()})
    return out

def coverage_stats(factor_df):
    valid = factor_df.notna()
    asset_days = float(valid.values.mean())
    ge8 = float((valid.sum(axis=1) >= 8).mean())
    return dict(coverage_asset_days=asset_days, coverage_dates_ge8=ge8)

def turnover_10d_rank(factor_df):
    """Mean absolute cross-sectional rank change over 10 trading days."""
    ranks = factor_df.rank(axis=1)
    r10 = ranks.shift(10)
    d = (ranks - r10).abs().dropna(how='all')
    return float(d.values[~np.isnan(d.values)].mean()) if d.size else np.nan

def library_corr(factor_df, lib_signals):
    """Max |Spearman rho| vs library factor signals (daily cross-sectional concat)."""
    best, best_id = 0.0, None
    my = factor_df.stack().rename('value').reset_index()
    my['date'] = pd.to_datetime(my['date'])
    for fid, sig in lib_signals.items():
        s = sig.copy()
        s['date'] = pd.to_datetime(s['date'])
        m = my.merge(s, on=['date', 'symbol'], suffixes=('_a', '_b'))
        if len(m) < 100:
            continue
        rho = m['value_a'].corr(m['value_b'], method='spearman')
        if np.isfinite(rho) and abs(rho) > best:
            best, best_id = abs(rho), fid
    return best, best_id
