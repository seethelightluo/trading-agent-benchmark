"""miner_1 2033-12-08 validation library (API-only, no future data).

Loads the 15 tradable assets via get_stock_daily_data (data visible through
current sim date = 2033-12-07), builds factor panels, forward-return panels,
and computes IC / ICIR / hit / coverage / turnover / decay / regime splits.

Usage: import as module, or run to demo data coverage.
"""
import json
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

ASSETS = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
          'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']
HORIZON = 10
MIN_ASSETS = 8
GATE_IC = 0.0070
GATE_ICIR = 0.0840

# observation-only macro signals (never traded)
MACRO_ASSETS = ['DXY', 'USDCNY', 'USDJPY', 'EURUSD', 'VIX']


def load_close(symbols=ASSETS, macro=False):
    """Return dict symbol -> Series(close) with common union of dates."""
    out = {}
    for s in symbols:
        df = get_stock_daily_data(symbol=s, days=4000)
        if df is None or len(df) < 300:
            print('WARN: %s no/insufficient data (%s)' % (s, None if df is None else len(df)))
            continue
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        out[s] = df['close'].astype(float)
    idx = None
    for s, ser in out.items():
        idx = ser.index if idx is None else idx.union(ser.index)
    idx = idx.sort_values()
    for s in out:
        out[s] = out[s].reindex(idx)
    return out


def to_grid(panel):
    """dict symbol->Series -> DataFrame(columns=symbols)."""
    return pd.DataFrame(panel)


def ic_series(factor_df, fwd_df):
    """Per-date spearman IC between factor and forward return (>=MIN_ASSETS valid)."""
    ics, dates = [], []
    for dt in factor_df.index:
        x = factor_df.loc[dt]
        y = fwd_df.loc[dt]
        m = x.notna() & y.notna()
        if m.sum() >= MIN_ASSETS:
            ics.append(x[m].rank().corr(y[m].rank()))
            dates.append(dt)
    return pd.Series(ics, index=pd.DatetimeIndex(dates))


def summarize(ic_s, factor_name, turnover=None, coverage=None, decay=None,
              max_corr=None, max_corr_with=None):
    ic = float(ic_s.mean()) if len(ic_s) else np.nan
    icir = float(ic_s.mean() / ic_s.std()) if len(ic_s) > 2 and ic_s.std() > 0 else np.nan
    hit = float((ic_s > 0).mean()) if len(ic_s) else np.nan
    n = len(ic_s)
    # regime splits
    regime = {}
    if n:
        years = ic_s.index.year
        for lo, hi, lab in [(2020, 2021, '2020-2021'), (2022, 2022, '2022'),
                            (2023, 2024, '2023-2024'), (2025, 2033, '2025-2033')]:
            m = (years >= lo) & (years <= hi)
            if m.sum() > 5:
                sub = ic_s[m]
                regime[lab] = {'ic': round(float(sub.mean()), 4),
                               'icir': round(float(sub.mean() / sub.std()), 3) if sub.std() > 0 else None,
                               'n': int(m.sum())}
        m = ic_s.index >= ic_s.index[-1] - pd.Timedelta(days=365)
        if m.sum() > 5:
            sub = ic_s[m]
            regime['last365'] = {'ic': round(float(sub.mean()), 4),
                                 'icir': round(float(sub.mean() / sub.std()), 3) if sub.std() > 0 else None,
                                 'n': int(m.sum())}
    return {'ic': round(ic, 4) if np.isfinite(ic) else None,
            'icir': round(icir, 4) if np.isfinite(icir) else None,
            'hit': round(hit, 4) if np.isfinite(hit) else None,
            'n_ic_dates': n,
            'regime': regime,
            'turnover_10d_rank': turnover,
            'coverage': coverage,
            'decay_ic_by_horizon': decay,
            'max_abs_library_correlation': max_corr,
            'max_corr_with': max_corr_with,
            'factor_name': factor_name,
            'pass_gate': bool(np.isfinite(ic) and np.isfinite(icir)
                              and abs(ic) >= GATE_IC and abs(icir) >= GATE_ICIR)}


def build_panels(close):
    """Return (factor_df, fwd_df) for a given horizon."""
    fwd = {s: ser.shift(-HORIZON) / ser - 1.0 for s, ser in close.items()}
    return pd.DataFrame(close), pd.DataFrame(fwd)


def load_macro():
    """Load observation-only macro closes from persistent index_data (full file)."""
    out = {}
    for s in MACRO_ASSETS:
        try:
            df = pd.read_csv('../persistent/index_data/%s.csv' % s)
        except Exception as e:
            print('macro load fail', s, e)
            continue
        df.columns = [c.strip().lower() for c in df.columns]
        datecol = 'date' if 'date' in df.columns else df.columns[0]
        df[datecol] = pd.to_datetime(df[datecol])
        df = df.set_index(datecol).sort_index()
        ccol = 'close' if 'close' in df.columns else df.columns[1]
        out[s] = df[ccol].astype(float)
    return out


def decay_profile(close, factor_df, horizons=(1, 2, 3, 5, 10, 20)):
    """IC by forward horizon for one factor panel."""
    out = {}
    for h in horizons:
        fwd = {s: ser.shift(-h) / ser - 1.0 for s, ser in close.items()}
        fwd_df = pd.DataFrame(fwd)
        ic_s = ic_series(factor_df, fwd_df)
        out[h] = round(float(ic_s.mean()), 4) if len(ic_s) else None
    return out


def turnover_rank(factor_df, lookback=10):
    """Mean cross-sectional rank-change turnover (0..1)."""
    ranks = factor_df.rank(axis=1)
    rc = ranks.diff(lookback).abs().mean(axis=1)
    return round(float(rc.mean()), 4) if len(rc) else None


if __name__ == '__main__':
    close = load_close()
    print('assets loaded:', len(close))
    grid = to_grid(close)
    print('grid shape:', grid.shape, 'range:', grid.index.min(), '->', grid.index.max())
    cov = 1 - grid.isna().mean().mean()
    print('coverage:', round(float(cov), 4))
