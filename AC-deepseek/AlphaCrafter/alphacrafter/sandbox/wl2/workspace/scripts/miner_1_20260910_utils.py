"""Shared factor validation utilities for miner_1 cycle 2026-09-10."""
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

WATCH = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
         'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']


def load_panel(days=2500):
    """Return dict symbol -> DataFrame sorted by date."""
    panel = {}
    for s in WATCH:
        df = get_stock_daily_data(symbol=s, days=days)
        if df is None or len(df) == 0:
            continue
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        panel[s] = df
    return panel


def align_close(panel):
    """Align close prices on the union of dates; return DataFrame symbol x date."""
    closes = {}
    for s, df in panel.items():
        closes[s] = df.set_index('date')['close']
    cdf = pd.DataFrame(closes)
    # keep dates where at least 8 instruments have prices
    return cdf


def forward_returns(close_df, horizon=10):
    """Forward (horizon-day) return per asset, aligned on same dates. NaN at tail."""
    fwd = close_df.shift(-horizon) / close_df - 1.0
    return fwd


def daily_ic(factor_df, fwd_df, min_assets=8):
    """Spearman IC per date between factor values and forward returns."""
    dates, ics = [], []
    for dt in factor_df.index:
        f = factor_df.loc[dt]
        r = fwd_df.loc[dt]
        mask = f.notna() & r.notna()
        if mask.sum() < min_assets:
            continue
        ics.append(f[mask].rank().corr(r[mask].rank()))
        dates.append(dt)
    return pd.Series(ics, index=pd.DatetimeIndex(dates))


def summarize_ic(ics, label=''):
    """Compute IC stats: mean, std, ICIR, hit ratio, t-stat, yearly breakdown."""
    ics = ics.dropna()
    if len(ics) == 0:
        print(f'{label}: NO IC DATES')
        return None
    mean_ic = ics.mean()
    std_ic = ics.std(ddof=1)
    icir = mean_ic / std_ic if std_ic > 0 else np.nan
    hit = (ics > 0).mean()
    tstat = mean_ic / (std_ic / np.sqrt(len(ics))) if std_ic > 0 else np.nan
    print(f'--- {label} ---')
    print(f'n_ic_dates={len(ics)}  mean_ic={mean_ic:.4f}  std={std_ic:.4f}  '
          f'icir={icir:.4f}  hit={hit:.3f}  tstat={tstat:.3f}')
    yr = ics.groupby(ics.index.year).agg(['mean', 'count'])
    for y, row in yr.iterrows():
        print(f'  {y}: ic={row["mean"]:.4f} n={int(row["count"])}')
    # gates
    gate_ic = abs(mean_ic) >= 0.0070
    gate_icir = abs(icir) >= 0.0840
    print(f'GATE |IC|>=0.007: {gate_ic} (|IC|={abs(mean_ic):.4f})   '
          f'GATE |ICIR|>=0.084: {gate_icir} (|ICIR|={abs(icir):.4f})')
    return {'mean_ic': mean_ic, 'icir': icir, 'hit': hit, 'n': len(ics),
            'gate_ic': gate_ic, 'gate_icir': gate_icir, 'tstat': tstat}


def decay_profile(factor_df, fwd_close, max_h=20, min_assets=8):
    """IC at horizons 1..max_h."""
    out = {}
    for h in range(1, max_h + 1):
        fwd = forward_returns(fwd_close, h)
        ics = daily_ic(factor_df, fwd, min_assets)
        out[h] = ics.mean() if len(ics) else np.nan
    return out


def turnover_rank(factor_df, horizon=10):
    """Mean fraction of assets whose cross-sectional rank changes > 1 decile-equivalent
    between factor dates spaced `horizon` days apart. Simpler: mean abs rank change / 14."""
    valid = factor_df.dropna(how='all')
    ranks = valid.rank(axis=1)
    r = ranks.iloc[::horizon]
    if len(r) < 2:
        return np.nan
    d = r.diff().abs().dropna()
    return float(d.mean().mean() / 14.0)


def coverage(factor_df, close_df):
    """Coverage = fraction of (asset,date) cells with valid factor among price-valid cells."""
    valid_price = close_df.notna()
    valid_factor = factor_df.notna() & valid_price
    cov = valid_factor.sum().sum() / max(valid_price.sum().sum(), 1)
    dates_ge8 = (valid_factor.sum(axis=1) >= 8).mean()
    return cov, dates_ge8


def rolling_ic_series(factor_df, fwd_df, window=63, min_assets=8):
    """Rolling 63-day mean IC for stability visualization."""
    ics = daily_ic(factor_df, fwd_df, min_assets)
    return ics.rolling(window, min_periods=30).mean()
