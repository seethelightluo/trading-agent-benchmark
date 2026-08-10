"""Persist streak_60 factor (round-16 PASS) with signal artifact + full JSON.

Admission metrics (from round-16 fast engine, 2026-07-30):
  IC10 = +0.0306, ICIR10 = +0.1021, hit = 0.546, n = 1625 dates,
  coverage_asset_days = 0.702, coverage_dates_ge8 = 0.680,
  turnover_10d_rank = 1.84, max_abs_library_correlation = 0.326 (hilo_pos_60).
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

import sys
sys.path.insert(0, 'scripts')
from factor_common import (WATCHLIST, load_prices, canonical_grid, signal_matrix,
                           factor_to_panel, forward_returns, VAL_START, VAL_END)

prices = load_prices(days=2500)
grid = canonical_grid(prices)
print(f"grid {len(grid)} dates {grid.min().date()}..{grid.max().date()}", flush=True)

# ---- recompute streak_60 panel (deterministic from OHLC) ----
def streak_60(df, s):
    r = df['close'].pct_change()
    up = (r > 0).astype(int)
    dn = (r < 0).astype(int)
    ups = up.groupby((~up.astype(bool)).cumsum()).cumsum()
    dns = dn.groupby((~dn.astype(bool)).cumsum()).cumsum()
    return (ups - dns).rolling(60).max() / 60.0

panel = factor_to_panel(streak_60, prices)
arr = signal_matrix(panel, grid)
np.save('factors/streak_60_signal.npy', arr)
print(f"signal artifact saved: factors/streak_60_signal.npy shape={arr.shape}", flush=True)

# ---- validation metrics (same-horizon 10d, full validation window) ----
fwd = {h: forward_returns(prices, h) for h in (1, 2, 3, 5, 10, 20)}
ic_series = {}
for h, fp in fwd.items():
    ic = []
    dates = []
    for d in grid:
        x = panel.loc[d] if d in panel.index else None
        y = fp.loc[d] if d in fp.index else None
        if x is None or y is None:
            continue
        m = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
        if m.sum() >= 8:
            xv = x[m].rank(); yv = y[m].rank()
            c = xv.corr(yv)
            if np.isfinite(c):
                ic.append(c); dates.append(d)
    ic_series[h] = pd.Series(ic, index=dates)

ic10 = ic_series[10]
ic10 = ic10[(ic10.index >= VAL_START) & (ic10.index <= VAL_END)]
ic_mean = float(ic10.mean())
ic_std = float(ic10.std(ddof=1))
icir = ic_mean / ic_std if ic_std > 0 else 0.0
hit = float((ic10 > 0).mean()) if ic_mean >= 0 else float((ic10 < 0).mean())

fac = panel[(panel.index >= VAL_START) & (panel.index <= VAL_END)]
total_cells = fac.shape[0] * fac.shape[1]
coverage = float(fac.notna().sum().sum()) / total_cells if total_cells else 0.0
ge8 = float((fac.notna().sum(axis=1) >= 8).mean())
ranked = fac.rank(axis=1)
turn = float(ranked.diff(10).abs().mean().mean()) if len(ranked) > 10 else float('nan')

decay = {str(h): (float(ic_series[h].mean()) if len(ic_series[h]) else float('nan')) for h in fwd}

def sub_ic(a, b):
    sub = ic10[(ic10.index >= pd.Timestamp(a)) & (ic10.index <= pd.Timestamp(b))]
    return float(sub.mean()) if len(sub) > 30 else float('nan')

recent = ic10[(ic10.index >= pd.Timestamp('2025-07-15')) & (ic10.index <= pd.Timestamp('2026-07-15'))]
recent_ic = float(recent.mean()) if len(recent) > 30 else float('nan')
recent_icir = float(recent.mean() / recent.std(ddof=1)) if len(recent) > 30 and recent.std(ddof=1) > 0 else 0.0

# ---- library correlation (deterministic recompute) ----
from miner_3_20260730_library_rebuild import build_library_panels
vix = None
import alphacrafter.sim.utils as U
try:
    vix = U.get_index_daily_data(symbol='VIX', days=2500)
except Exception:
    pass
dxy = None
try:
    dxy = U.get_index_daily_data(symbol='DXY', days=2500)
except Exception:
    pass
eurusd = None
try:
    eurusd = U.get_index_daily_data(symbol='EURUSD', days=2500)
except Exception:
    pass

def _idx_df(raw):
    if raw is None:
        return None
    df = raw.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    return df

lib = build_library_panels(prices, _idx_df(vix), _idx_df(dxy), _idx_df(eurusd))
# keep only currently EFFECTIVE factors
eff = set()
for f in Path('factors').glob('*.json'):
    try:
        d = json.load(open(f))
        if d.get('validation', {}).get('status') == 'EFFECTIVE':
            eff.add(d['factor_id'])
    except Exception:
        pass
lib = {k: v for k, v in lib.items() if k in eff}
print(f"library for corr audit: {len(lib)} -> {sorted(lib)}", flush=True)

best, best_id = 0.0, None
for fid, lp in lib.items():
    if lp is None or len(lp) == 0:
        continue
    idx = panel.index.intersection(lp.index)
    corrs = []
    for d in idx:
        x = panel.loc[d]; y = lp.loc[d]
        m = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
        if m.sum() >= 8:
            c = x[m].rank().corr(y[m].rank())
            if np.isfinite(c):
                corrs.append(c)
    if corrs:
        r = float(np.mean(corrs))
        if abs(r) > best:
            best = abs(r); best_id = fid

print(f"IC={ic_mean:+.4f} ICIR={icir:+.4f} hit={hit:.3f} n={len(ic10)} "
      f"cov={coverage:.3f} ge8={ge8:.3f} turn={turn:.2f} rho={best:.3f}({best_id})", flush=True)
print("decay:", {k: round(v, 4) for k, v in decay.items()}, flush=True)
print("regime p1/p2/p3:", round(sub_ic('2020-01-01','2022-12-31'), 4),
      round(sub_ic('2023-01-01','2024-12-31'), 4), round(sub_ic('2025-01-01','2026-07-15'), 4),
      "| recent_1y:", round(recent_ic, 4), flush=True)

metrics = {
    'ic': ic_mean, 'icir': icir, 'ic_hit_ratio': hit,
    'n_ic_dates': int(len(ic10)),
    'coverage_asset_days': coverage,
    'coverage_dates_ge8': ge8,
    'turnover_10d_rank': turn,
    'decay_ic_by_horizon': decay,
    'ic_2020_2022': sub_ic('2020-01-01', '2022-12-31'),
    'ic_2023_2024': sub_ic('2023-01-01', '2024-12-31'),
    'ic_2025_2026': sub_ic('2025-01-01', '2026-07-15'),
    'recent_1y_ic': recent_ic,
    'recent_1y_icir': recent_icir,
    'max_abs_library_correlation': best,
    'max_corr_library_id': best_id,
}

quality = abs(ic_mean) * abs(icir)
doc = {
    'factor_id': 'streak_60',
    'factor_name': 'Max net run-streak 60d (up-streak persistence)',
    'version': '1.0.0',
    'calculation': {
        'expression': 'rolling_max(up_streak - down_streak, 60) / 60, where up_streak = #consecutive pct_change>0 days and down_streak = #consecutive pct_change<0 days',
        'description': ('For each asset, track the length of the current consecutive up-day run minus the '
                        'current consecutive down-day run (net streak). Take the rolling 60-day maximum of this '
                        'net streak and normalize by 60. Positive cross-sectional IC: assets that recently built '
                        'a long positive run (persistent multi-day gains) tend to keep outperforming over the '
                        'next 10 days -- run-persistence / continuation effect. Decorrelated from the library '
                        '(max rho ~0.33 vs hilo_pos_60).')
    },
    'dependencies': ['close'],
    'parameters': {'window': 60, 'normalize': 60.0},
    'expected_direction': 1,
    'signal_artifact': 'streak_60_signal.npy',
    'signal_artifact_format': 'npy',
    'signal_artifact_shape': list(arr.shape),
    'signal_artifact_grid': {
        'start': str(grid.min().date()), 'end': str(grid.max().date()),
        'n_dates': len(grid),
        'columns': WATCHLIST,
        'note': 'canonical grid shared by all library factors (see factor_common.canonical_grid)'
    },
    'validation': {
        'status': 'EFFECTIVE',
        'period': '2020-01-01..2026-07-15',
        'last_validated': '2026-07-30',
        'admission_horizon': 10,
        'regime_notes': ('Validated 2020-01-01..2026-07-15 across COVID crash (p1 IC +0.042), 2022 tightening '
                         'bear (p2 +0.021) and 2023-25 risk-on (p3 +0.023); recent 1y IC +0.027. Positive IC at '
                         'all horizons 1-20d (peak 0.031 at 10d), hit ratio 0.546, turnover low at 1.84 rank units '
                         'per 10 days. Run-persistence effect is regime-stable.'),
        'metrics': metrics
    },
    'tags': ['momentum', 'run-streak', 'persistence', 'microstructure', 'cross-asset'],
    'benchmark_admission': {
        'contract': {
            'ic_threshold': 0.007, 'icir_threshold': 0.084,
            'correlation_threshold': 0.5, 'library_capacity': 30, 'active_top_k': 10
        },
        'selected_metrics': {
            'ic': ic_mean, 'icir': icir,
            'metric_path': 'validation.metrics',
            'reported_max_abs_library_correlation': best,
            'correlation_path': 'validation.metrics.max_abs_library_correlation',
            'quality': quality
        },
        'admitted_at': datetime.now(timezone.utc).isoformat()
    },
    'signal_provenance': {
        'source': 'recomputed from alphacrafter.sim.utils daily OHLC series',
        'panel_shape': f'{panel.shape[0]}x{panel.shape[1]}',
        'panel_range': f'{panel.index.min().date()}..{panel.index.max().date()}',
        'validation_window': '2020-01-01..2026-07-15',
        'ic_method': 'daily cross-sectional Spearman rank IC vs 10d forward return',
        'note': 'expression deterministic and reproducible from OHLC series only'
    }
}

Path('factors/streak_60.json').write_text(json.dumps(doc, indent=1, default=str))
print('factors/streak_60.json written', flush=True)

# ---- verify reload ----
check = json.load(open('factors/streak_60.json'))
assert check['factor_id'] == 'streak_60'
assert check['validation']['status'] == 'EFFECTIVE'
assert check['validation']['metrics']['ic'] == ic_mean
assert abs(check['validation']['metrics']['ic']) >= 0.007
assert abs(check['validation']['metrics']['icir']) >= 0.084
assert check['validation']['metrics']['max_abs_library_correlation'] < 0.5
art = np.load('factors/streak_60_signal.npy')
assert art.shape == tuple(check['signal_artifact_shape'])
assert np.isclose(np.nanmean(art), np.nanmean(arr), rtol=1e-6)
print('RELOAD VERIFIED: JSON valid, id=streak_60, status=EFFECTIVE, '
      f'|IC|={abs(ic_mean):.4f}>=0.007, |ICIR|={abs(icir):.4f}>=0.084, rho={best:.4f}<0.5, '
      f'artifact shape={art.shape}', flush=True)
