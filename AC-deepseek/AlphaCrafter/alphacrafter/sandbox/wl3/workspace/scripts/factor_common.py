"""Shared data loading + IC validation framework for the 15-asset cross-asset universe.

All data comes from the simulator API (no future data). Validation window is
restricted to research warm-up 2020-01-01..2026-07-15 to match library factors.

Also provides signal-artifact persistence: every persisted factor JSON must be
accompanied by a recoverable 2D signal matrix (n_dates x 15) stored on a shared
canonical date grid so the deterministic gate can recompute pairwise rho.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

WATCHLIST = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
             'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']
INDEX_SIGNALS = ['DXY', 'USDCNY', 'USDJPY', 'EURUSD', 'VIX']
VAL_END = pd.Timestamp('2026-07-15')
VAL_START = pd.Timestamp('2020-01-01')
LIBRARY_FACTORS = ['mom_10d_skip5', 'mom_120d_skip5', 'vix_beta_cond_60x20', 'vol_of_vol20x60']

_CANON_GRID = None  # cached canonical date grid


def load_prices(days=2000):
    """Return dict {symbol: df} of daily OHLCV via API (capped at current date)."""
    out = {}
    for s in WATCHLIST:
        try:
            df = get_stock_daily_data(symbol=s, days=days)
            if df is not None and len(df) >= 30:
                df = df.copy()
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
                for c in ['open', 'close', 'high', 'low', 'volume']:
                    df[c] = pd.to_numeric(df[c], errors='coerce')
                out[s] = df
        except Exception:
            pass
    return out


def load_index(symbol, days=2000, prices=None):
    """Load observation-only index signal; falls back to persistent CSV capped at
    the max date visible in the tradable price history (no future data)."""
    try:
        df = get_index_daily_data(symbol=symbol, days=days)
        if df is not None and len(df) >= 30:
            df = df.copy()
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            return df
    except Exception:
        pass
    # CSV fallback (persistent/index_data), capped at visible horizon
    try:
        path = Path('../persistent/index_data') / f'{symbol}.csv'
        df = pd.read_csv(path, parse_dates=['date'])
        df = df.set_index('date').sort_index()
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        if prices is not None:
            visible = max(dd.index.max() for dd in prices.values())
            df = df[df.index <= visible]
        return df
    except Exception:
        return None


def canonical_grid(prices):
    """Sorted union of all trading dates within the validation window.

    Every persisted factor artifact must use this exact grid (n_dates x 15) so
    the gate's pairwise Spearman comparison sees identical shapes.
    """
    global _CANON_GRID
    if _CANON_GRID is not None:
        return _CANON_GRID
    idx = set()
    for s, df in prices.items():
        idx.update(df.index)
    grid = pd.DatetimeIndex(sorted(idx))
    grid = grid[(grid >= VAL_START) & (grid <= VAL_END)]
    _CANON_GRID = grid
    return grid


def signal_matrix(panel, grid=None, prices=None):
    """Reindex factor panel to the canonical grid -> (n_dates, 15) float matrix.

    Missing values become NaN; column order is fixed to WATCHLIST order so every
    factor artifact is directly comparable shape-wise.
    """
    if grid is None:
        grid = canonical_grid(prices)
    m = panel.reindex(grid)
    for c in WATCHLIST:
        if c not in m.columns:
            m[c] = np.nan
    return m[WATCHLIST].values.astype(float)


def save_signal_artifact(panel, grid, path):
    """Persist factor signal matrix as .npy; returns the matrix."""
    arr = signal_matrix(panel, grid)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
    return arr


def factor_to_panel(factor_fn, prices, index_dict=None):
    """Apply factor_fn(df, symbol) to each asset, return wide panel aligned on dates."""
    cols = {}
    for s, df in prices.items():
        try:
            ser = factor_fn(df, s)
            if ser is not None and len(ser) > 0:
                cols[s] = ser.astype(float)
        except Exception:
            pass
    if not cols:
        return pd.DataFrame()
    panel = pd.DataFrame(cols)
    panel = panel[~panel.index.duplicated(keep='last')].sort_index()
    return panel


def forward_returns(prices, horizon):
    """Wide panel of forward h-day simple returns aligned at t."""
    cols = {}
    for s, df in prices.items():
        if 'close' in df:
            cols[s] = df['close'].shift(-horizon) / df['close'] - 1.0
    return pd.DataFrame(cols).sort_index()


def rank_ic_series(factor_panel, fwd_ret, min_valid=8):
    """Daily cross-sectional Spearman IC between factor and forward return."""
    common = factor_panel.index.intersection(fwd_ret.index)
    ic = {}
    for d in common:
        x = factor_panel.loc[d]
        y = fwd_ret.loc[d]
        m = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
        if m.sum() >= min_valid:
            ic[d] = x[m].rank().corr(y[m].rank())
    return pd.Series(ic).sort_index()


def validate_factor(factor_id, factor_panel, prices, horizons=(1, 2, 3, 5, 10, 20),
                    min_valid=8, icir_scale=True):
    """Full validation battery. Returns dict of metrics."""
    fwd = {h: forward_returns(prices, h) for h in horizons}
    ic_series = {h: rank_ic_series(factor_panel, fwd[h], min_valid) for h in horizons}

    ic10 = ic_series[10]
    ic10 = ic10[(ic10.index >= VAL_START) & (ic10.index <= VAL_END)]
    if len(ic10) < 100:
        return None
    ic_mean = float(ic10.mean())
    ic_std = float(ic10.std(ddof=1))
    icir = ic_mean / ic_std if ic_std > 0 else 0.0
    hit = float((ic10 > 0).mean()) if ic_mean >= 0 else float((ic10 < 0).mean())

    # coverage
    fac = factor_panel[(factor_panel.index >= VAL_START) & (factor_panel.index <= VAL_END)]
    total_cells = fac.shape[0] * fac.shape[1]
    valid_cells = int(fac.notna().sum().sum())
    coverage = valid_cells / total_cells if total_cells else 0.0
    ge8 = float((fac.notna().sum(axis=1) >= min_valid).mean())

    # turnover: mean absolute rank change over 10-day steps
    ranked = fac.rank(axis=1)
    if len(ranked) > 10:
        turn = float(ranked.diff(10).abs().mean().mean())
    else:
        turn = float('nan')

    decay = {str(h): (float(ic_series[h].mean()) if len(ic_series[h]) else float('nan'))
             for h in horizons}

    return {
        'ic': ic_mean, 'icir': icir, 'ic_hit_ratio': hit,
        'n_ic_dates': int(len(ic10)), 'coverage_asset_days': coverage,
        'coverage_dates_ge8': ge8, 'turnover_10d_rank': turn,
        'decay_ic_by_horizon': decay,
    }


def max_library_correlation(factor_panel, library_panels):
    """Max absolute mean daily cross-sectional Spearman corr with library factors."""
    best = 0.0
    best_id = None
    common = factor_panel.index
    for fid, lp in library_panels.items():
        if lp is None or len(lp) == 0:
            continue
        idx = common.intersection(lp.index)
        corrs = []
        for d in idx:
            x = factor_panel.loc[d]
            y = lp.loc[d]
            m = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
            if m.sum() >= 8:
                c = x[m].rank().corr(y[m].rank())
                if np.isfinite(c):
                    corrs.append(c)
        if corrs:
            r = float(np.mean(corrs))
            if abs(r) > best:
                best = abs(r)
                best_id = fid
    return best, best_id


def build_library_panels(prices):
    """Recompute the 4 currently-effective library factor signals (for correlation audit)."""
    vix = load_index('VIX', prices=prices)
    out = {}
    def f_mom10(df, s): return df['close'].shift(5) / df['close'].shift(15) - 1.0
    def f_mom120(df, s): return df['close'].shift(5) / df['close'].shift(125) - 1.0
    def f_vixbeta(df, s):
        if vix is None:
            return None
        r = df['close'].pct_change(); vr = vix['close'].pct_change()
        z = pd.concat([r.rename('r'), vr.rename('v')], axis=1).dropna()
        b = z['r'].rolling(60).cov(z['v']) / z['v'].rolling(60).var()
        return (-b * (vix['close'] / vix['close'].shift(20) - 1.0)).reindex(z.index)
    def f_vov(df, s): return df['close'].pct_change().rolling(20).std().rolling(60).std()
    out['mom_10d_skip5'] = factor_to_panel(f_mom10, prices)
    out['mom_120d_skip5'] = factor_to_panel(f_mom120, prices)
    out['vix_beta_cond_60x20'] = factor_to_panel(f_vixbeta, prices)
    out['vol_of_vol20x60'] = factor_to_panel(f_vov, prices)
    return out


def evaluate_candidate(factor_id, factor_fn, prices, library_panels=None, print_out=True):
    """One-idea validation battery. Returns metrics dict or None."""
    panel = factor_to_panel(factor_fn, prices)
    m = validate_factor(factor_id, panel, prices)
    if m is None:
        if print_out:
            print(f"{factor_id}: insufficient data -> None")
        return None, panel
    if library_panels is None:
        library_panels = build_library_panels(prices)
    rho, fid = max_library_correlation(panel, library_panels)
    m['max_abs_library_correlation'] = rho
    m['max_corr_library_id'] = fid
    if print_out:
        print(f"Factor {factor_id}: panel {panel.shape} range {panel.index.min()}..{panel.index.max()}")
        print(json.dumps({k: v for k, v in m.items() if k != 'decay_ic_by_horizon'}, indent=2, default=str))
        print("decay:", json.dumps(m['decay_ic_by_horizon'], default=str))
        ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084
        print(f"ADMISSION: |IC|={abs(m['ic']):.4f}>=0.007 {abs(m['ic'])>=0.007} | |ICIR|={abs(m['icir']):.4f}>=0.084 {abs(m['icir'])>=0.084} -> {'PASS' if ok else 'FAIL'}")
    return m, panel


def persist_factor(factor_id, factor_name, expression, description, dependencies,
                   parameters, expected_direction, panel, metrics, tags,
                   grid=None, prices=None, version='1.0.0', status='EFFECTIVE',
                   regime_notes='', extra=None):
    """Persist a factor JSON + .npy signal artifact to factors/.

    Returns the path written. The artifact is mandatory for the gate's pairwise
    rho computation.
    """
    if grid is None:
        grid = canonical_grid(prices)
    art_path = Path('factors') / f'{factor_id}_signal.npy'
    arr = save_signal_artifact(panel, grid, art_path)
    payload = {
        'factor_id': factor_id,
        'factor_name': factor_name,
        'version': version,
        'calculation': {
            'expression': expression,
            'description': description,
        },
        'dependencies': dependencies,
        'parameters': parameters,
        'expected_direction': expected_direction,
        'signal_artifact': art_path.name,
        'signal_artifact_format': 'npy',
        'signal_artifact_shape': list(arr.shape),
        'signal_artifact_grid': {
            'start': str(grid.min().date()),
            'end': str(grid.max().date()),
            'n_dates': int(len(grid)),
            'columns': WATCHLIST,
            'note': 'canonical grid shared by all library factors (see factor_common.canonical_grid)',
        },
        'validation': {
            'status': status,
            'period': f'{VAL_START.date()}..{VAL_END.date()}',
            'last_validated': '2026-07-30',
            'admission_horizon': 10,
            'regime_notes': regime_notes,
            'metrics': metrics,
        },
        'tags': tags,
        'benchmark_admission': {
            'contract': {
                'ic_threshold': 0.007,
                'icir_threshold': 0.084,
                'correlation_threshold': 0.5,
                'library_capacity': 30,
                'active_top_k': 10,
            },
            'selected_metrics': {
                'ic': metrics['ic'],
                'icir': metrics['icir'],
                'metric_path': 'validation.metrics',
                'max_abs_library_correlation': metrics.get('max_abs_library_correlation'),
                'correlation_path': 'validation.metrics.max_abs_library_correlation',
            },
        },
    }
    if extra:
        payload.update(extra)
    path = Path('factors') / f'{factor_id}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding='utf-8')
    return path, arr


def load_artifact_matrix(factor_json_path):
    """Reconstruct the 2D signal matrix referenced by a persisted factor JSON."""
    payload = json.loads(Path(factor_json_path).read_text(encoding='utf-8'))
    art = payload.get('signal_artifact')
    if not art:
        return None
    p = Path(factor_json_path).parent / str(art)
    if not p.exists():
        return None
    return np.load(p, allow_pickle=False)
