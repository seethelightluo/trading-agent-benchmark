"""Shared framework for factor mining (miner_1, cycle 2026-08-13).

Data visible through 2026-08-12 (index 2407 of trading_days grid).
Provides:
  - grid / asset OHLCV / macro loading aligned to canonical date.json grid
  - cross-sectional rank IC / ICIR / hit ratio / turnover / coverage
  - decay analysis by horizon
  - gate-style mean-daily cross-sectional Spearman vs library signal artifacts
    (npy or embedded daily_panel), matching the post-Miner deterministic gate.
"""
import json, os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

UNIVERSE = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX',
            'NDX', 'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']
MACRO = ['DXY', 'USDCNY', 'USDJPY', 'EURUSD', 'VIX']

DATA_DIR = '../persistent'


def load_grid():
    with open(f'{DATA_DIR}/date.json') as f:
        d = json.load(f)
    tds = d['trading_days']
    vt = d['visible_through']
    idx = tds.index(vt)
    grid = tds[:idx + 1]
    return grid, vt


def _read_ts(path, cols, grid):
    df = pd.read_csv(path, parse_dates=['date'])
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')
    df = df.set_index('date')
    df = df.reindex(grid)
    out = pd.DataFrame(index=grid)
    for c in cols:
        out[c] = df[c].astype(float)
    return out


def load_assets():
    grid, vt = load_grid()
    closes = pd.DataFrame(index=grid)
    ohlcv = {}
    for s in UNIVERSE:
        df = _read_ts(f'{DATA_DIR}/stock_data/{s}.csv',
                      ['open', 'close', 'high', 'low', 'volume'], grid)
        ohlcv[s] = df
        closes[s] = df['close']
    return closes, ohlcv, grid, vt


def load_macro():
    grid, _ = load_grid()
    out = {}
    for s in MACRO:
        out[s] = _read_ts(f'{DATA_DIR}/index_data/{s}.csv', ['close'], grid)['close']
    return pd.DataFrame(out), grid


def load_library_signals():
    """Return {factor_id: (matrix (n_days,15), dates(list))} for all signal
    artifacts present in factors/ root (npy preferred, else embedded panel)."""
    lib = {}
    for fn in sorted(os.listdir('factors')):
        if not fn.endswith('.json') or 'bak' in fn or fn == 'factor_ensemble.json':
            continue
        try:
            with open(f'factors/{fn}') as f:
                d = json.load(f)
        except Exception:
            continue
        fid = d.get('factor_id')
        if not fid:
            continue
        sa = d.get('signal_artifact')
        if isinstance(sa, str) and os.path.exists(f'factors/{sa}') and sa.endswith('.npy'):
            m = np.load(f'factors/{sa}')
            dates = None
            prov = d.get('artifact_provenance', {})
            if prov.get('dates_first') and prov.get('dates_last'):
                dates = None  # keep matrix as-is; align by overlap later
            lib[fid] = {'matrix': m, 'src': sa}
        elif isinstance(sa, dict) and sa.get('format') == 'daily_panel':
            vals = np.array(sa.get('values'), dtype=float)
            dates = sa.get('dates')
            lib[fid] = {'matrix': vals, 'src': f'{fn}#embedded', 'dates': dates}
    return lib


def daily_rank_ic_matrix(factor, fwd, min_obs=8):
    """factor, fwd: (n_days, n_assets) matrices aligned.
    Returns per-date cross-sectional Spearman IC series."""
    n = factor.shape[0]
    ics = np.full(n, np.nan)
    for t in range(n):
        x = factor[t]
        y = fwd[t]
        m = ~(np.isnan(x) | np.isnan(y))
        if m.sum() >= min_obs:
            if np.nanstd(x[m]) == 0 or np.nanstd(y[m]) == 0:
                continue
            r = spearmanr(x[m], y[m])[0]
            if not np.isnan(r):
                ics[t] = r
    return ics


def ic_stats(ics):
    v = ics[~np.isnan(ics)]
    if len(v) == 0:
        return dict(ic=np.nan, icir=np.nan, hit=np.nan, n=0)
    ic = float(np.mean(v))
    sd = float(np.std(v, ddof=1))
    return dict(ic=ic, icir=float(ic / sd) if sd > 0 else np.nan,
                hit=float(np.mean(np.sign(v) == np.sign(ic))) if ic != 0 else np.nan,
                n=int(len(v)))


def turnover_rank(factor, horizon=10):
    """Mean cross-sectional rank turnover over `horizon`-day steps."""
    n = factor.shape[0]
    rk = np.full_like(factor, np.nan, dtype=float)
    for t in range(n):
        row = factor[t]
        m = ~np.isnan(row)
        if m.sum() >= 8:
            rk[t, m] = pd.Series(row[m]).rank(pct=True).values
    tos = []
    for t in range(horizon, n):
        a, b = rk[t - horizon], rk[t]
        m = ~(np.isnan(a) | np.isnan(b))
        if m.sum() >= 8:
            tos.append(float(np.mean(np.abs(a[m] - b[m]))))
    return float(np.mean(tos)) if tos else np.nan


def coverage(factor, min_obs=8):
    n = factor.shape[0]
    valid_dates = 0
    asset_days = 0
    for t in range(n):
        m = ~np.isnan(factor[t])
        asset_days += m.sum()
        if m.sum() >= min_obs:
            valid_dates += 1
    return dict(coverage_asset_days=float(asset_days) / (n * factor.shape[1]),
                coverage_dates_ge8=float(valid_dates) / n,
                n_dates_total=int(n), n_dates_ge8=int(valid_dates))


def mean_daily_spearman(a, b, min_obs=8):
    """Gate-style: mean of daily cross-sectional Spearman over overlapping
    rows with >= min_obs valid pairs. a,b same row count."""
    n = min(a.shape[0], b.shape[0])
    rhos = []
    for t in range(n):
        x, y = a[t], b[t]
        m = ~(np.isnan(x) | np.isnan(y))
        if m.sum() >= min_obs:
            if np.nanstd(x[m]) == 0 or np.nanstd(y[m]) == 0:
                continue
            r = spearmanr(x[m], y[m])[0]
            if not np.isnan(r):
                rhos.append(r)
    return float(np.mean(rhos)) if rhos else np.nan


def library_corr_sweep(factor, lib, label='candidate'):
    """factor: (n_days,15). Returns sorted list of (abs_rho, fid, rho)."""
    res = []
    for fid, d in lib.items():
        m = d['matrix']
        r = mean_daily_spearman(factor, m)
        if not np.isnan(r):
            res.append((abs(r), fid, r))
    res.sort(reverse=True)
    return res


def zscore_winsor(factor, sigma=3.0):
    """Cross-sectional rank -> z-score -> winsorize (ensemble transform)."""
    out = np.full_like(factor, np.nan, dtype=float)
    for t in range(factor.shape[0]):
        row = factor[t]
        m = ~np.isnan(row)
        if m.sum() >= 8:
            r = pd.Series(row[m]).rank(pct=True).values
            z = (r - r.mean()) / (r.std(ddof=0) + 1e-12)
            z = np.clip(z, -sigma, sigma)
            out[t, m] = z
    return out
