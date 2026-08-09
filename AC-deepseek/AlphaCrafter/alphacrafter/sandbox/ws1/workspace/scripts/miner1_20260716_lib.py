"""Shared data + validation infrastructure for factor mining (warm-up 2020-01-01..2026-07-15).

Usage:
    from miner1_20260716_lib import build_panel, validate_factor
"""
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

WATCH = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
         'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']
MACRO = ['DXY', 'USDCNY', 'USDJPY', 'EURUSD', 'VIX']
WARMUP_END = pd.Timestamp('2026-07-15')
DATA_START = pd.Timestamp('2020-01-01')


def _fetch(sym, days=4000):
    try:
        df = get_stock_daily_data(symbol=sym, days=days)
    except Exception:
        df = get_index_daily_data(symbol=sym, days=days)
    if df is None or len(df) == 0:
        return None
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df[(df['date'] >= DATA_START) & (df['date'] <= WARMUP_END)].sort_values('date')
    df = df.set_index('date')
    return df


def load_macro():
    out = {}
    for m in MACRO:
        df = pd.read_csv(f'../persistent/index_data/{m}.csv')
        df['date'] = pd.to_datetime(df['date'])
        df = df[(df['date'] >= DATA_START) & (df['date'] <= WARMUP_END)].sort_values('date')
        out[m] = df.set_index('date')['close'].astype(float)
    return out


def build_panel():
    """Return dict of per-asset DataFrames (close/volume on native calendars) and union grid."""
    closes, volumes = {}, {}
    for sym in WATCH:
        df = _fetch(sym)
        if df is None:
            continue
        closes[sym] = df['close'].astype(float)
        if 'volume' in df.columns:
            volumes[sym] = df['volume'].astype(float)
    macro = load_macro()
    grid = pd.DatetimeIndex(sorted(set().union(*[set(c.index) for c in closes.values()])))
    return {'closes': closes, 'volumes': volumes, 'macro': macro, 'grid': grid}


def factor_values(closes, volumes, grid, fn):
    """fn(sym, close_series, volume_series) -> factor series on native calendar."""
    out = pd.DataFrame(index=grid)
    for sym in closes:
        try:
            v = fn(sym, closes[sym], volumes.get(sym))
            if v is not None and len(v) > 0:
                out[sym] = v.reindex(grid)
        except Exception:
            continue
    return out


def forward_returns(closes, grid, horizon):
    """Forward h-day return per asset on native calendar, reindexed to grid."""
    out = pd.DataFrame(index=grid)
    for sym, s in closes.items():
        fwd = s.shift(-horizon) / s - 1.0
        out[sym] = fwd.reindex(grid)
    return out


def daily_ic(factor_frame, ret_frame, min_valid=8, method='spearman'):
    """Cross-sectional IC per date."""
    idx = factor_frame.index.intersection(ret_frame.index)
    ics = []
    for d in idx:
        f = factor_frame.loc[d]
        r = ret_frame.loc[d]
        mask = f.notna() & r.notna()
        n = int(mask.sum())
        if n < min_valid:
            continue
        fv, rv = f[mask].astype(float), r[mask].astype(float)
        if fv.nunique() < 2 or rv.nunique() < 2:
            continue
        if method == 'spearman':
            ic = fv.rank().corr(rv.rank())
        else:
            ic = fv.corr(rv)
        if np.isfinite(ic):
            ics.append((d, ic, n))
    return pd.DataFrame(ics, columns=['date', 'ic', 'n']).set_index('date')


def summarize(ics, label, horizon, turnover=None, coverage=None):
    if len(ics) == 0:
        print(f"[{label}] NO VALID IC DATES")
        return None
    ic = ics['ic']
    mean_ic = float(ic.mean())
    std_ic = float(ic.std(ddof=1)) if len(ic) > 1 else float('nan')
    icir = mean_ic / std_ic if std_ic and np.isfinite(std_ic) and std_ic > 0 else float('nan')
    hit = float((ic > 0).mean())
    med_n = int(ics['n'].median())
    turn_s = f"{turnover:.3f}" if turnover is not None else "nan"
    cov_s = f"{coverage:.3f}" if coverage is not None else "nan"
    print(f"[{label}] h={horizon} dates={len(ic)} med_n={med_n} "
          f"IC={mean_ic:+.4f} ICIR={icir:+.3f} hit={hit:.2f} "
          f"turn={turn_s} cov={cov_s}")
    return {'label': label, 'horizon': horizon, 'dates': len(ic), 'median_n': med_n,
            'ic': mean_ic, 'icir': icir, 'hit': hit, 'std': std_ic}


def validate_factor(label, fn, horizons=(1, 2, 3, 5, 10, 20), min_valid=8):
    """fn(sym, close, volume) -> factor series. Validates across horizons, reports decay."""
    panel = build_panel()
    closes, volumes, grid = panel['closes'], panel['volumes'], panel['grid']
    fac = factor_values(closes, volumes, grid, fn)
    cov = float(fac.notna().mean().mean())
    rank = fac.rank(axis=1)
    turn = float(rank.diff().abs().mean().mean()) if len(rank) > 1 else float('nan')
    print(f"[{label}] factor frame shape={fac.shape} coverage={cov:.3f} avg_turnover={turn:.3f}")
    results = {}
    for h in horizons:
        ret = forward_returns(closes, grid, h)
        ics = daily_ic(fac, ret, min_valid=min_valid)
        res = summarize(ics, label, h, turnover=turn, coverage=cov)
        results[h] = res
    return panel, fac, results


def decay_table(results):
    print("\nDECAY: ", " | ".join(f"h={h}: IC={r['ic']:+.4f} ICIR={r['icir']:+.3f}" for h, r in results.items()))


def regime_breakdown(ics, panel, label):
    if len(ics) == 0:
        return
    t = ics.copy()
    t['year'] = t.index.year
    print(f"\n[{label}] IC by year:")
    for y, g in t.groupby('year'):
        s = g['ic'].std(ddof=1)
        icir = g['ic'].mean() / s if s and np.isfinite(s) and s > 0 else float('nan')
        print(f"  {y}: dates={len(g)} IC={g['ic'].mean():+.4f} ICIR={icir:+.3f}")
    vix = panel['macro'].get('VIX')
    if vix is not None:
        v = vix.reindex(t.index)
        q = v.quantile([0.33, 0.66])
        lo, hi = q.iloc[0], q.iloc[1]
        for nm, mask in [('VIX_low', v <= lo), ('VIX_mid', (v > lo) & (v <= hi)), ('VIX_high', v > hi)]:
            g = t[mask.values]
            if len(g):
                s = g['ic'].std(ddof=1)
                icir = g['ic'].mean() / s if s and np.isfinite(s) and s > 0 else float('nan')
                print(f"  {nm}: dates={len(g)} IC={g['ic'].mean():+.4f} ICIR={icir:+.3f}")
