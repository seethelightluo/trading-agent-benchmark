"""Shared factor validation helper for miner_1 (15-instrument cross-asset universe).
Computes factor signals, forward-return IC, ICIR, hit ratio, coverage, turnover,
decay, and max abs library correlation vs existing signal artifacts (.npy).

Data: warm-up 2020-01-01..2026-07-15 + live 2026-07-16..2027-09-27.
Admission gates (shared, 15-instrument universe): |IC| >= 0.0070, |ICIR| >= 0.0840.
IMPORTANT: factor values are computed per-asset on each asset's own trading calendar
(rolling windows must not be polluted by other markets' holidays), then reindexed
to the union date grid.
"""
import glob
import json
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

WATCH = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
         'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']
WARM_END = pd.Timestamp('2026-07-15')
LIVE_END = pd.Timestamp('2027-09-27')


def load_close_panel(days=2400):
    """Return DataFrame [date x asset] of closes, oldest->newest (union grid)."""
    closes = {}
    for s in WATCH:
        df = get_stock_daily_data(s, days)
        if df is None or len(df) < 100:
            print(f'WARN: {s} insufficient data')
            continue
        closes[s] = df.set_index('date')['close'].astype(float)
    panel = pd.DataFrame(closes).sort_index()
    panel = panel[~panel.index.duplicated(keep='last')]
    return panel


def apply_factor_per_asset(panel, func):
    """func(series) -> factor series computed on the asset's own calendar.
    Returns DataFrame on the union grid (NaN where asset absent)."""
    out = {}
    for col in panel.columns:
        s = panel[col].dropna()
        if len(s) < 60:
            out[col] = pd.Series(np.nan, index=panel.index)
            continue
        f = func(s)
        out[col] = f.reindex(panel.index)
    return pd.DataFrame(out, index=panel.index)


def fwd_returns(panel, horizon):
    """Forward return at horizon (days ahead), aligned to panel index."""
    return panel.shift(-horizon) / panel - 1.0


def cross_sectional_ic(factor_df, fwd_df, min_valid=8):
    """Daily Spearman IC between factor values and forward returns."""
    dates, ics = [], []
    for dt in factor_df.index:
        f = factor_df.loc[dt]
        r = fwd_df.loc[dt]
        mask = f.notna() & r.notna() & np.isfinite(f) & np.isfinite(r)
        if mask.sum() >= min_valid:
            ics.append(f[mask].corr(r[mask], method='spearman'))
            dates.append(dt)
    return pd.Series(ics, index=dates)


def summarize_ic(ic_series, label='all'):
    ic = ic_series.dropna()
    if len(ic) == 0:
        return {'label': label, 'n_dates': 0}
    mean_ic = ic.mean()
    std_ic = ic.std(ddof=1)
    icir = mean_ic / std_ic if std_ic > 0 else 0.0
    hit = (ic > 0).mean()
    return {
        'label': label,
        'n_dates': len(ic),
        'ic': float(mean_ic),
        'icir': float(icir),
        'ic_hit_ratio': float(hit),
        'ic_std': float(std_ic),
    }


def coverage_stats(factor_df):
    valid = factor_df.notna().sum().sum()
    total = factor_df.shape[0] * factor_df.shape[1]
    ge8 = (factor_df.notna().sum(axis=1) >= 8).mean()
    return {
        'coverage_asset_days': float(valid / total),
        'coverage_dates_ge8': float(ge8),
    }


def turnover_10d(factor_df):
    """Mean cross-sectional rank change over 10 trading days (scaled)."""
    ranks = factor_df.rank(axis=1)
    r10 = ranks.shift(10)
    chg = (ranks - r10).abs().mean(axis=1).dropna()
    return float(chg.mean()) if len(chg) else float('nan')


def decay_profile(factor_df, panel, horizons=(1, 2, 3, 5, 10, 20)):
    out = {}
    for h in horizons:
        fwd = fwd_returns(panel, h)
        ic = cross_sectional_ic(factor_df, fwd)
        out[str(h)] = round(float(ic.mean()), 4) if len(ic) else None
    return out


def max_library_corr(factor_df, artifact_dir='factors', exclude=()):
    """Max abs Spearman corr of factor signal vs existing .npy signal panels
    (aligned on common dates/assets)."""
    best = None
    for path in sorted(glob.glob(f'{artifact_dir}/*.signal.npy')):
        name = path.split('/')[-1].replace('.signal.npy', '')
        if name in exclude:
            continue
        try:
            arr = np.load(path)
            if arr.ndim != 2 or arr.shape[1] != len(WATCH):
                continue
            if arr.shape[0] != factor_df.shape[0]:
                continue
            lib = pd.DataFrame(arr, index=factor_df.index, columns=WATCH)
            corrs = []
            for dt in factor_df.index:
                a = factor_df.loc[dt]
                b = lib.loc[dt]
                m = a.notna() & b.notna() & np.isfinite(a) & np.isfinite(b)
                if m.sum() >= 8:
                    c = a[m].corr(b[m], method='spearman')
                    if np.isfinite(c):
                        corrs.append(c)
            if len(corrs) >= 30:
                rho = float(np.mean(np.abs(corrs)))
                if best is None or rho > best[1]:
                    best = (name, rho)
        except Exception as e:
            print(f'  lib {name}: skip ({e})')
    return best


def full_validation(factor_id, factor_df, panel, direction=1,
                    horizons=(1, 2, 3, 5, 10, 20), artifact_path=None,
                    exclude_lib=()):
    """Run full validation battery; factor_df index must match panel index."""
    print(f'=== {factor_id} ===')
    print(f'shape: {factor_df.shape}, dates {factor_df.index[0].date()}..{factor_df.index[-1].date()}')
    res = {'factor_id': factor_id}
    fwd10 = fwd_returns(panel, 10)
    ic_full = cross_sectional_ic(factor_df, fwd10)
    res['metrics'] = summarize_ic(ic_full, 'full')
    warm_mask = ic_full.index <= WARM_END
    live_mask = ic_full.index > WARM_END
    res['warmup'] = summarize_ic(ic_full[warm_mask], 'warmup')
    res['live'] = summarize_ic(ic_full[live_mask], 'live')
    adj = ic_full * direction
    res['metrics']['ic_directed'] = float(adj.mean())
    res['metrics']['icir_directed'] = float(adj.mean() / adj.std(ddof=1)) if adj.std(ddof=1) > 0 else 0.0
    res['metrics'].update(coverage_stats(factor_df))
    res['metrics']['turnover_10d_rank'] = turnover_10d(factor_df)
    res['metrics']['decay_ic_by_horizon'] = decay_profile(factor_df, panel, horizons)
    lib = max_library_corr(factor_df, exclude=exclude_lib)
    res['metrics']['max_abs_library_correlation'] = lib
    ic_abs = abs(res['metrics']['ic'])
    icir_abs = abs(res['metrics']['icir'])
    gate = (ic_abs >= 0.0070) and (icir_abs >= 0.0840)
    res['gate'] = {'ic_abs': round(ic_abs, 5), 'icir_abs': round(icir_abs, 5),
                   'passed': bool(gate)}
    print(f"IC={res['metrics']['ic']:.4f} ICIR={res['metrics']['icir']:.3f} "
          f"hit={res['metrics']['ic_hit_ratio']:.3f} n={res['metrics']['n_dates']}")
    print(f"warmup IC={res['warmup'].get('ic'):.4f} ICIR={res['warmup'].get('icir'):.3f} "
          f"live IC={res['live'].get('ic'):.4f} ICIR={res['live'].get('icir'):.3f}")
    print(f"coverage={res['metrics']['coverage_asset_days']:.3f} "
          f"ge8={res['metrics']['coverage_dates_ge8']:.3f} "
          f"turnover={res['metrics']['turnover_10d_rank']:.3f}")
    print('decay:', res['metrics']['decay_ic_by_horizon'])
    print('max_abs_library_corr:', lib)
    print('gate:', res['gate'])
    if artifact_path:
        arr = factor_df.values.astype(np.float64)
        np.save(artifact_path, arr)
        print(f'saved artifact {artifact_path} shape {arr.shape}')
    return res


def rank_demean(s):
    """Cross-sectional rank then demean (robust to outliers)."""
    return s.rank(pct=True) - 0.5


if __name__ == '__main__':
    print('helper loaded')
