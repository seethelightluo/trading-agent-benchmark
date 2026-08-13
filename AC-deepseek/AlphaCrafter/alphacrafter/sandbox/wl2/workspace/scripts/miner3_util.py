"""Shared harness for miner_3 factor validation (data through 2032-10-13)."""
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

WATCH = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
         'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']
# Instruments that are flat-feed (zero returns) for recent period -> degenerate for IC
FLAT_FEED = {'HSI', 'SX5E', 'BTC', 'US10Y', 'CN10Y', '000300.SH'}  # last nonzero move pre-2032


def load_all(days=3000):
    """Return dict symbol -> DataFrame (date, close, open, high, low, pct_change) sorted old->new."""
    out = {}
    for s in WATCH:
        df = get_stock_daily_data(symbol=s, days=days)
        if df is None:
            print('NO DATA', s)
            continue
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        out[s] = df
    return out


def load_macro(name, max_date='2032-10-13'):
    """Load observation-only macro CSV truncated to current sim date."""
    df = pd.read_csv(f'../persistent/index_data/{name}.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] <= pd.Timestamp(max_date)].sort_values('date').reset_index(drop=True)
    return df


def close_panel(data):
    """Symbol x date close panel (wide)."""
    return pd.DataFrame({s: d.set_index('date')['close'] for s, d in data.items()}).sort_index()


def ret_panel(data):
    """Daily pct returns (fraction) panel."""
    px = close_panel(data)
    return px.pct_change()


def forward_ret(px, h):
    """Forward h-day return per asset (fraction). fwd_ret_t = px_{t+h}/px_t - 1, last h NaN."""
    return px.shift(-h) / px - 1.0


def daily_spearman_ic(factor_series, fwd, min_valid=8):
    """factor_series: DataFrame (date x symbol) of factor values.
    fwd: DataFrame (date x symbol) of forward returns.
    Returns DataFrame with ic per date (only dates with >=min_valid non-nan pairs)."""
    dates = factor_series.index.intersection(fwd.index)
    ic_recs = []
    for dt in dates:
        f = factor_series.loc[dt]
        r = fwd.loc[dt]
        mask = f.notna() & r.notna() & np.isfinite(f) & np.isfinite(r)
        n = int(mask.sum())
        if n < min_valid:
            continue
        ic = f[mask].corr(r[mask], method='spearman')
        if np.isfinite(ic):
            ic_recs.append((dt, ic, n))
    out = pd.DataFrame(ic_recs, columns=['date', 'ic', 'n']).set_index('date')
    return out


def summarize_ic(ic_df, label='', horizons=None, factor_rank_df=None):
    """Aggregate daily IC stats. horizons: dict h->fwd already computed -> decay table.
    factor_rank_df: date x symbol cross-sectional rank (for turnover)."""
    if len(ic_df) == 0:
        print(f'{label}: NO IC DATES'); return {}
    ic = ic_df['ic']
    m = {
        'n_ic_dates': int(len(ic)),
        'ic': float(ic.mean()),
        'icir': float(ic.mean() / ic.std(ddof=1)) if ic.std(ddof=1) > 0 else np.nan,
        'ic_hit_ratio': float((ic > 0).mean()),
        'ic_std': float(ic.std(ddof=1)),
        'median_n': float(ic_df['n'].median()),
    }
    print(f'--- {label} ---')
    print(f"  n_dates={m['n_ic_dates']} ic={m['ic']:.4f} icir={m['icir']:.3f} hit={m['ic_hit_ratio']:.3f} "
          f"median_n={m['median_n']:.1f}")
    if horizons:
        dec = {}
        for h, fwd in horizons.items():
            ic_h = daily_spearman_ic(factor_rank_df if factor_rank_df is not None else factor, fwd)
            # NOTE: use raw factor for decay IC, rank df only for turnover
            dec[str(h)] = float(ic_h['ic'].mean()) if len(ic_h) else np.nan
        print('  decay_ic_by_horizon:', {k: round(v, 4) for k, v in dec.items()})
        m['decay_ic_by_horizon'] = dec
    return m


def rank_turnover(factor_series, h=10):
    """Mean abs change of cross-sectional rank over h days (dates with >=8 valid)."""
    ranks = factor_series.rank(axis=1, pct=True)
    diffs = (ranks - ranks.shift(h)).abs()
    valid = ranks.notna().sum(axis=1)
    sub = diffs[valid >= 8]
    if len(sub) == 0:
        return np.nan
    return float(sub.mean().mean())


def library_corr(factor_series, lib_dir='factors/', self_id=None):
    """Max abs pairwise Spearman corr (date x symbol) vs existing effective library signal files.
    Uses persisted .signal.npy artifacts if present, else computes from json expression is skipped.
    Returns (max_abs, dict)."""
    import glob, os
    best = 0.0; pairs = {}
    fvals = factor_series.rank(axis=1)
    for npy in sorted(glob.glob(os.path.join(lib_dir, '*.signal.npy'))):
        try:
            sig = np.load(npy)
            if sig.shape != fvals.shape:
                # try to align by shape only if matches; else skip
                continue
            arr = pd.DataFrame(sig, index=fvals.index, columns=fvals.columns).astype(float)
            corr = fvals.corrwith(arr.rank(axis=1), axis=1).dropna()
            rho = float(corr.mean())
            if np.isfinite(rho):
                pairs[os.path.basename(npy)] = rho
                best = max(best, abs(rho))
        except Exception as e:
            pass
    return best, pairs
