"""Persist dd_duration_120 orthogonized against mom_120d_skip5."""
import sys, json
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.insert(0, 'scripts')
from factor_common import (load_prices, factor_to_panel, validate_factor,
                           build_library_panels, max_library_correlation, persist_factor,
                           WATCHLIST)

prices = load_prices(days=2100)
lib = build_library_panels(prices)
mom120 = lib['mom_120d_skip5']


def dd_duration(df, s, win=120, minp=60):
    c = df['close']
    h = c.rolling(win, min_periods=minp).max()
    is_high = (c >= h).fillna(False)
    idx_high = np.flatnonzero(is_high.values)
    pos = np.arange(len(c))
    last = np.searchsorted(idx_high, pos) - 1
    dur = np.where(last >= 0, pos - idx_high[np.maximum(last, 0)], np.nan)
    return pd.Series(np.log1p(dur), index=c.index)


def zscore_rows(panel):
    return panel.sub(panel.mean(axis=1), axis=0).div(panel.std(axis=1), axis=0)


def orthogonalize(panel, ref, min_valid=8):
    z, zr = zscore_rows(panel), zscore_rows(ref)
    out = z.copy()
    for d in z.index:
        if d not in zr.index:
            out.loc[d] = np.nan
            continue
        x, y = z.loc[d], zr.loc[d]
        m = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
        if m.sum() < min_valid:
            out.loc[d] = np.nan
            continue
        xv, yv = x[m].values, y[m].values
        xv = (xv - xv.mean()) / (xv.std() + 1e-12)
        yv = (yv - yv.mean()) / (yv.std() + 1e-12)
        beta = float(np.dot(xv, yv) / (len(xv) + 1e-12))  # correlation approx
        out.loc[d, m] = xv - beta * yv
    return out


panel_raw = factor_to_panel(lambda df, s: dd_duration(df, s, 120), prices)
panel = orthogonalize(panel_raw, mom120)

metrics = validate_factor('dd_duration_120_resid', panel, prices)
if metrics is None:
    print('FAILED: insufficient data')
    sys.exit(1)

rho, rid = max_library_correlation(panel, lib)
metrics['max_abs_library_correlation'] = rho
metrics['max_corr_library_id'] = rid

print('IC:', metrics['ic'], 'ICIR:', metrics['icir'], 'hit:', metrics['ic_hit_ratio'])
print('max_abs_library_correlation:', rho, rid)

passes = abs(metrics['ic']) >= 0.007 and abs(metrics['icir']) >= 0.084 and rho < 0.5
print('PASS:', passes)

if not passes:
    print('NOT PERSISTING - fails thresholds')
    sys.exit(1)

extra = {
    'mining_notes': 'Drawdown-duration (log-1p days since 120d high), cross-sectionally '
                    'orthogonalized against mom_120d_skip5 to reduce correlation. '
                    'Negative IC: assets longer in drawdown tend to underperform over 10d.',
}
path = persist_factor(
    factor_id='dd_duration_120_resid',
    factor_name='Drawdown Duration 120d (orthogonalized vs mom120)',
    expression='log1p(days_since_120d_high) - beta * zscore(mom_120d_skip5)',
    description=('Log1p of number of days since the last 120-day rolling high, '
                 'then cross-sectionally orthogonalized (per-date) against z-scored '
                 'mom_120d_skip5. Captures mean-reversion/recovery pressure after '
                 'prolonged drawdowns independent of raw momentum.'),
    dependencies=['close', 'mom_120d_skip5'],
    parameters={'lookback': 120, 'min_periods': 60, 'transform': 'log1p'},
    expected_direction='negative',
    panel=panel,
    metrics=metrics,
    tags=['drawdown', 'duration', 'mean-reversion', 'orthogonalized'],
    prices=prices,
    regime_notes='Validated 2020-01-01..2026-07-15 across 15-asset cross-asset universe.',
    extra=extra,
)
print('Persisted:', path)
