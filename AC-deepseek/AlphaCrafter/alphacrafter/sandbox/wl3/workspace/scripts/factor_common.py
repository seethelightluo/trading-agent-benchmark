"""Shared data loading + IC validation framework for the 15-asset cross-asset universe.

All data comes from the simulator API (no future data). Validation window is
restricted to research warm-up 2020-01-01..2026-07-15 to match library factors.
"""
import json
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

WATCHLIST = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
             'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']
INDEX_SIGNALS = ['DXY', 'USDCNY', 'USDJPY', 'EURUSD', 'VIX']
VAL_END = pd.Timestamp('2026-07-15')
VAL_START = pd.Timestamp('2020-01-01')
LIBRARY_FACTORS = ['mom_10d_skip5', 'mom_120d_skip5', 'vix_beta_cond_60x20', 'vol_of_vol20x60']


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


def load_index(symbol, days=2000):
    try:
        df = get_index_daily_data(symbol=symbol, days=days)
        if df is None or len(df) < 30:
            return None
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        return df
    except Exception:
        return None


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
    vix = load_index('VIX')
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
        return None
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
    return m
