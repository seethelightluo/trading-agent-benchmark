"""miner1 2028-05-26: shared panel loader + IC evaluation helpers."""
import pandas as pd, numpy as np, json, os

WATCH = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
         'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']
MACRO = ['DXY', 'USDCNY', 'USDJPY', 'EURUSD', 'VIX']
END = '2028-05-25'  # most recent completed trading day at decision date 2028-05-26

def load_panel():
    with open('scripts/panel_cache.pkl', 'rb') as f:
        panel = pd.read_pickle(f)
    # safety: truncate macro to panel horizon (avoid any future data)
    for k in panel:
        if isinstance(panel[k], pd.DataFrame):
            panel[k] = panel[k].loc[panel[k].index <= END]
    return panel

def ic_series(factor_df, fwd_ret, min_valid=8):
    """Daily cross-sectional Spearman IC between factor (t) and forward return (t+1).
    factor_df: DataFrame indexed by date, columns = assets.
    fwd_ret: DataFrame of forward returns aligned to factor index (value at t = return t->t+1).
    Returns Series of daily IC indexed by date (only dates with >= min_valid valid pairs).
    """
    dates = factor_df.index
    out = {}
    for t in dates:
        f = factor_df.loc[t]
        r = fwd_ret.loc[t]
        m = f.notna() & r.notna()
        if m.sum() < min_valid:
            continue
        ic = f[m].rank().corr(r[m].rank())
        if np.isfinite(ic):
            out[t] = ic
    return pd.Series(out, dtype=float)

def fwd_returns(close, horizons=(1, 2, 3, 5, 10)):
    out = {}
    for h in horizons:
        out[h] = close.shift(-h) / close - 1.0
    return out

def summarize_ic(ic, label=''):
    if len(ic) == 0:
        return None
    mean = ic.mean()
    std = ic.std(ddof=1)
    icir = mean / std if std > 0 else np.nan
    hit = (ic > 0).mean()
    return {'label': label, 'n_dates': int(len(ic)),
            'mean_ic': float(mean), 'std': float(std), 'icir': float(icir),
            'hit_rate': float(hit), 't_stat': float(mean / (std / np.sqrt(len(ic)))) if std > 0 else np.nan}

def coverage_stats(factor_df, min_valid=8):
    valid_cnt = factor_df.notna().sum(axis=1)
    dates_ok = int((valid_cnt >= min_valid).sum())
    return {'dates_valid_ge8': dates_ok, 'total_dates': int(len(factor_df)),
            'avg_valid': float(valid_cnt.mean()), 'median_valid': float(valid_cnt.median()),
            'min_valid': int(valid_cnt.min()), 'max_valid': int(valid_cnt.max()),
            'coverage_frac': float((valid_cnt >= min_valid).mean())}

def turnover_rank(factor_df):
    """Mean absolute daily change in cross-sectional rank (normalized 0..1)."""
    ranks = factor_df.rank(axis=1) / factor_df.notna().sum(axis=1)
    d = ranks.diff().abs().mean().mean()
    return float(d)

def turnover_signal(factor_df):
    """Mean absolute daily change in standardized factor value."""
    z = factor_df.sub(factor_df.mean(axis=1), axis=0).div(factor_df.std(axis=1), axis=0)
    d = z.diff().abs().mean().mean()
    return float(d)
