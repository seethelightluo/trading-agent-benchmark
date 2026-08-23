"""miner_2 shared validation framework (2029-06-28 cycle).

Computes, for a candidate factor signal panel:
  - per-date cross-sectional IC vs forward return at admission horizon (10d)
  - mean IC, ICIR, hit ratio, n dates, coverage
  - decay IC by horizons [1,2,3,5,10,20]
  - turnover at 10d rebalance cadence
  - max abs correlation vs existing library factor signal panels
No lookahead: factor uses data up to date t; forward return uses close at t+h.
"""
import numpy as np
import pandas as pd
import json, os, base64, zlib, io


def load_asset_panels(watchlist, min_days=60):
    from alphacrafter.sim.utils import get_stock_daily_data
    panels = {}
    for sym in watchlist:
        df = get_stock_daily_data(symbol=sym, days=2500)
        if df is None or len(df) < min_days:
            continue
        df = df.set_index('date')
        panels[sym] = {
            'close': df['close'].astype(float),
            'high': df['high'].astype(float),
            'low': df['low'].astype(float),
            'volume': df['volume'].astype(float) if 'volume' in df.columns else None,
        }
    return panels


def load_macro(series_name, kind='index', days=2500):
    if kind == 'index':
        from alphacrafter.sim.utils import get_index_daily_data
        df = get_index_daily_data(symbol=series_name, days=days)
    else:
        df = pd.read_csv(f'../persistent/index_data/{series_name}.csv')
        df['date'] = pd.to_datetime(df['date'])
    if df is None or len(df) == 0:
        return None
    s = df.set_index('date')['close'].astype(float)
    return s


def build_panel_from_scorer(watchlist, panels, scorer, drop_first_n=0):
    """scorer(date_idx_dict, sym) -> factor value at that date (time t)."""
    # union index (dates) across panels
    all_idx = None
    for sym in watchlist:
        if sym in panels:
            all_idx = panels[sym]['close'].index.union(all_idx) if all_idx is not None else panels[sym]['close'].index
    all_idx = pd.DatetimeIndex(sorted(all_idx))
    rows = {}
    for sym in watchlist:
        if sym not in panels:
            continue
        close = panels[sym]['close']
        s_vals = {}
        for t in all_idx:
            if t not in close.index:
                continue
            try:
                v = scorer(t, sym)
            except Exception:
                v = np.nan
            if v is not None and np.isfinite(v):
                s_vals[t] = v
        rows[sym] = pd.Series(s_vals)
    panel = pd.DataFrame(rows, index=all_idx).sort_index()
    return panel


def compute_ic_metrics(panel, panels, watchlist, horizon=10, min_assets=8,
                       start=None, end=None, max_lookback=2500):
    dates = panel.index
    if start is not None:
        dates = dates[dates >= start]
    if end is not None:
        dates = dates[dates <= end]
    ic_recs = []
    for t in dates:
        vals = {}
        for sym in panel.columns:
            v = panel.loc[t, sym]
            if not np.isfinite(v):
                continue
            cp = panels[sym]['close']
            if t not in cp.index:
                continue
            loc = cp.index.get_loc(t)
            if loc + horizon >= len(cp):
                continue
            fwd = cp.iloc[loc + horizon] / cp.iloc[loc] - 1.0
            if np.isfinite(fwd):
                vals[sym] = (v, fwd)
        if len(vals) < min_assets:
            continue
        syms = list(vals.keys())
        f = np.array([vals[s][0] for s in syms])
        r = np.array([vals[s][1] for s in syms])
        if np.std(f) < 1e-12 or np.std(r) < 1e-12:
            continue
        ic = np.corrcoef(f, r)[0, 1]
        if np.isfinite(ic):
            ic_recs.append({'date': t, 'ic': ic, 'n': len(syms)})
    if len(ic_recs) == 0:
        return {'ic': None, 'icir': None, 'n_ic_dates': 0}
    ics = np.array([x['ic'] for x in ic_recs])
    mean_ic = ics.mean()
    std_ic = ics.std(ddof=1) if len(ics) > 1 else 0.0
    icir = mean_ic / std_ic if std_ic > 0 else 0.0
    hit = (ics > 0).mean()
    return {
        'ic': float(mean_ic), 'icir': float(icir),
        'ic_hit_ratio': float(hit), 'n_ic_dates': len(ic_recs),
        'std_ic': float(std_ic),
        'dates': [x['date'].strftime('%Y-%m-%d') for x in ic_recs],
    }


def decay_ic(panel, panels, watchlist, horizons=(1, 2, 3, 5, 10, 20), min_assets=8):
    out = {}
    for h in horizons:
        m = compute_ic_metrics(panel, panels, watchlist, horizon=h, min_assets=min_assets)
        out[str(h)] = m['ic']
    return out


def turnover_10d(panel):
    """Mean absolute daily rank change; scaled by coverage; reported per rebalance (10d window avg)."""
    ranks = panel.rank(axis=1)
    d = ranks.diff().abs().mean(axis=1)
    cov = panel.notna().sum(axis=1).clip(lower=1)
    daily = (d / cov).mean()
    return float(daily)  # mean daily rank-move fraction


def coverage_report(panel, watchlist):
    n_assets = len(watchlist)
    total_cells = len(panel) * n_assets
    valid = int(panel.notna().sum().sum())
    cov_assets_days = valid / total_cells if total_cells else 0
    ge8 = (panel.notna().sum(axis=1) >= 8).mean()
    return {'coverage_asset_days': float(cov_assets_days),
            'coverage_dates_ge8': float(ge8),
            'n_assets': n_assets,
            'panel_shape': list(panel.shape)}


def load_library_panels(factor_dir='factors/'):
    """Load persisted signal artifact panels from factor JSONs."""
    lib = {}
    for fp in sorted(os.listdir(factor_dir)):
        if not fp.endswith('.json') or fp.startswith('factor_ensemble') or 'bak' in fp:
            continue
        try:
            d = json.load(open(os.path.join(factor_dir, fp), encoding='utf-8'))
        except Exception:
            continue
        sa = d.get('validation', {}).get('signal_artifact', {})
        data = sa.get('data')
        if not data:
            continue
        try:
            raw = zlib.decompress(base64.b64decode(data))
            df = pd.read_csv(io.BytesIO(raw), index_col=0)
            df.index = pd.to_datetime(df.index)
            lib[d.get('factor_id', fp)] = df
        except Exception:
            try:
                df = pd.read_csv(io.BytesIO(base64.b64decode(data)), index_col=0)
                df.index = pd.to_datetime(df.index)
                lib[d.get('factor_id', fp)] = df
            except Exception:
                continue
    return lib


def max_lib_corr(panel, lib, align_dates=True):
    """Max abs correlation between candidate panel and library panels (Pearson, all cells)."""
    best = 0.0
    best_id = None
    cand = panel.astype(float)
    for fid, lp in lib.items():
        # align on common dates & columns
        common_idx = cand.index.intersection(lp.index)
        common_cols = cand.columns.intersection(lp.columns)
        if len(common_idx) < 60 or len(common_cols) < 5:
            continue
        a = cand.loc[common_idx, common_cols]
        b = lp.loc[common_idx, common_cols]
        av = a.values.ravel()
        bv