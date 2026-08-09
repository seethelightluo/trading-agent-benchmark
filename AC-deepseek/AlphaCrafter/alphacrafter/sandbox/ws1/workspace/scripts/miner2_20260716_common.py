"""Shared validation framework for miner_2 (cross-asset 15-instrument universe).

Strictly uses data through 2026-07-15 (visible_through) to avoid lookahead.
Panel is aligned on the weekday calendar (BTC/ETH weekend prints dropped) so that
rolling/forward computations are clean; per-date cross-sections use whatever
instruments are valid (>= MIN_INSTR).
"""
import json
import numpy as np
import pandas as pd

SYMBOLS = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
           'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']
INDEX_SYMBOLS = ['DXY', 'USDCNY', 'USDJPY', 'EURUSD', 'VIX']
CUTOFF = pd.Timestamp('2026-07-15')
MIN_INSTR = 8  # min instruments for a valid cross-sectional IC observation

DATA_DIR = '../persistent/stock_data'
INDEX_DIR = '../persistent/index_data'


def load_close(symbol, cutoff=CUTOFF):
    df = pd.read_csv(f'{DATA_DIR}/{symbol}.csv', parse_dates=['date'])
    df = df[df['date'] <= cutoff].sort_values('date').reset_index(drop=True)
    return df


def load_panel(symbols=SYMBOLS, cutoff=CUTOFF, weekday_only=True):
    """DataFrame of closes indexed by date, columns = symbols (weekday calendar)."""
    out = {}
    for s in symbols:
        df = load_close(s, cutoff)
        out[s] = pd.Series(df['close'].astype(float).values, index=pd.to_datetime(df['date']), name=s)
    panel = pd.concat(out, axis=1).sort_index()
    if weekday_only:
        panel = panel[panel.index.dayofweek < 5]
    return panel


def load_index_panel(cutoff=CUTOFF, weekday_only=True):
    out = {}
    for s in INDEX_SYMBOLS:
        df = pd.read_csv(f'{INDEX_DIR}/{s}.csv', parse_dates=['date'])
        df = df[df['date'] <= cutoff].sort_values('date').reset_index(drop=True)
        out[s] = pd.Series(df['close'].astype(float).values, index=pd.to_datetime(df['date']), name=s)
    panel = pd.concat(out, axis=1).sort_index()
    if weekday_only:
        panel = panel[panel.index.dayofweek < 5]
    return panel


def forward_returns(panel, horizon):
    return panel.shift(-horizon) / panel - 1.0


def rank_ic(factor, fwd, min_instr=MIN_INSTR):
    ics, dates, counts = [], [], []
    common = factor.join(fwd, lsuffix='_f', rsuffix='_r')
    for dt, row in common.iterrows():
        f = row[[c for c in common.columns if c.endswith('_f')]].values
        r = row[[c for c in common.columns if c.endswith('_r')]].values
        mask = np.isfinite(f) & np.isfinite(r)
        if mask.sum() < min_instr:
            continue
        if np.all(f[mask] == f[mask][0]):
            continue
        ic = pd.Series(f[mask]).corr(pd.Series(r[mask]), method='spearman')
        if np.isfinite(ic):
            ics.append(ic)
            dates.append(dt)
            counts.append(int(mask.sum()))
    return pd.Series(ics, index=pd.DatetimeIndex(dates)), counts


def summarize_ic(ic_series, sign=1.0):
    if len(ic_series) == 0:
        return None
    ic = ic_series * sign
    mean = float(ic.mean())
    std = float(ic.std(ddof=1)) if len(ic) > 1 else float('nan')
    icir = mean / std if std and std > 0 else float('nan')
    hit = float((ic > 0).mean())
    tstat = mean / (std / np.sqrt(len(ic))) if std and std > 0 else float('nan')
    return {
        'n_dates': int(len(ic)),
        'mean_ic': mean,
        'std_ic': std,
        'icir': icir,
        'hit_ratio': hit,
        't_stat': tstat,
    }


def turnover_rank(factor, min_instr=MIN_INSTR):
    diffs, counts = [], []
    prev = None
    for dt in factor.index:
        row = factor.loc[dt].dropna()
        if len(row) < min_instr:
            prev = None
            continue
        cur = row.rank(method='average')
        cur = (cur - 1.0) / max(1, len(cur) - 1)
        if prev is not None and len(cur) == len(prev):
            diffs.append(float((cur - prev).abs().mean()))
            counts.append(len(cur))
        prev = cur
    return float(np.mean(diffs)) if diffs else float('nan'), int(np.mean(counts)) if counts else 0


def coverage_stats(factor, min_instr=MIN_INSTR):
    valid = factor.notna().sum().sum()
    total = factor.shape[0] * factor.shape[1]
    per_date = factor.notna().sum(axis=1)
    return {
        'coverage': valid / total,
        'dates_with_min8': int((per_date >= min_instr).sum()),
        'avg_instr_per_date': float(per_date.mean()),
    }


def evaluate_factor(factor, panel, horizons=(5, 10, 20, 40), sign=1.0):
    res = {}
    for h in horizons:
        fwd = forward_returns(panel, h)
        ic_series, counts = rank_ic(factor, fwd)
        summ = summarize_ic(ic_series, sign=sign)
        if summ is not None:
            summ['mean_instr'] = float(np.mean(counts))
        res[h] = summ
    to, tc = turnover_rank(factor)
    cov = coverage_stats(factor)
    return res, to, tc, cov
