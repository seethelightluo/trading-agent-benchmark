"""Shared factor validation utilities for miner_1 cycle 2026-09-24.

Data visible through 2026-09-23 (previous completed trading day).
"""
import numpy as np
import pandas as pd
import glob, os, json
from alphacrafter.sim.utils import get_stock_daily_data

WATCH = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
         'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']
START = pd.Timestamp('2020-01-01')


def load_panel(days=2500):
    """Return dict symbol -> DataFrame sorted by date (through visible date)."""
    panel = {}
    for s in WATCH:
        df = get_stock_daily_data(symbol=s, days=days)
        if df is None or len(df) == 0:
            continue
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        df = df[df['date'] >= START]
        panel[s] = df
    return panel


def align_close(panel):
    """Align close prices on the union of dates; return DataFrame symbol x date."""
    closes = {s: df.set_index('date')['close'] for s, df in panel.items()}
    return pd.DataFrame(closes)


def forward_returns(close_df, horizon=10):
    """Forward (horizon-day) return per asset, aligned on same dates."""
    return close_df.shift(-horizon) / close_df - 1.0


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
    gate_ic = abs(mean_ic) >= 0.0070
    gate_icir = abs(icir) >= 0.0840
    print(f'GATE |IC|>=0.007: {gate_ic} (|IC|={abs(mean_ic):.4f})   '
          f'GATE |ICIR|>=0.084: {gate_icir} (|ICIR|={abs(icir):.4f})')
    return {'mean_ic': mean_ic, 'icir': icir, 'hit': hit, 'n': len(ics),
            'gate_ic': gate_ic, 'gate_icir': gate_icir, 'tstat': tstat}


def decay_profile(factor_df, close_df, max_h=20, min_assets=8):
    out = {}
    for h in range(1, max_h + 1):
        fwd = forward_returns(close_df, h)
        ics = daily_ic(factor_df, fwd, min_assets)
        out[h] = ics.mean() if len(ics) else np.nan
    return out


def turnover_rank(factor_df, horizon=10):
    valid = factor_df.dropna(how='all')
    if len(valid) < 2:
        return np.nan
    ranks = valid.rank(axis=1)
    r = ranks.iloc[::horizon]
    d = r.diff().abs().dropna()
    if len(d) == 0:
        return np.nan
    return float(d.mean().mean() / 14.0)


def coverage(factor_df, close_df):
    valid_price = close_df.notna()
    valid_factor = factor_df.notna() & valid_price
    cov = valid_factor.sum().sum() / max(valid_price.sum().sum(), 1)
    dates_ge8 = (valid_factor.sum(axis=1) >= 8).mean()
    return float(cov), float(dates_ge8)


def library_corr(factor_df):
    """Max |Pearson corr| per asset vs every row-aligned *.signal.npy in factors/."""
    out = {}
    for npy in sorted(glob.glob(os.path.join('factors', '*.signal.npy'))):
        fid = os.path.basename(npy).replace('.signal.npy', '')
        try:
            arr = np.load(npy)
        except Exception:
            continue
        N = arr.shape[0]
        if arr.shape[1] != 15 or N < 500 or N > factor_df.shape[0]:
            continue
        lib = pd.DataFrame(arr, index=factor_df.index[:N], columns=factor_df.columns)
        a = factor_df.iloc[:N]
        corrs = []
        for c in factor_df.columns:
            x = a[c].astype(float)
            y = lib[c].astype(float)
            m = x.notna() & y.notna()
            if m.sum() >= 60:
                r = np.corrcoef(x[m], y[m])[0, 1]
                if np.isfinite(r):
                    corrs.append(r)
        if corrs:
            out[fid] = {'maxabs': float(max(abs(r) for r in corrs)),
                        'meanabs': float(np.mean([abs(r) for r in corrs])),
                        'n_pairs': len(corrs)}
    return out


def save_signal_artifact(factor_df, factor_id, outdir='factors'):
    """Save row-aligned signal matrix to factors/<id>.signal.npy (float32)."""
    mat = factor_df.astype(np.float32).fillna(np.nan).values
    path = os.path.join(outdir, f'{factor_id}.signal.npy')
    np.save(path, mat)
    print(f'saved artifact {path} shape={mat.shape}')
    return path
