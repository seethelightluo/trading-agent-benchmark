"""Persist round-10 winner cn10y_beta_60 (passed |IC|>=0.007, |ICIR|>=0.084, rho<0.5).

Beta of each asset's daily return to CN10Y yield changes over 60d. Negative IC means
assets that fall when CN10Y yields rise tend to underperform -> defensive China-rate
sensitivity factor (expected_direction -1).
"""
import sys, json, glob
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.insert(0, 'scripts')
from factor_common import (WATCHLIST, load_prices, load_index, canonical_grid,
                           signal_matrix, factor_to_panel, validate_factor, persist_factor)

np.seterr(all='ignore')

prices = load_prices(days=2500)
grid = canonical_grid(prices)
print(f"prices: {len(prices)} assets; grid {len(grid)} dates ({grid.min().date()}..{grid.max().date()})")


def rolling_beta(r_asset, r_sig, win=60):
    return r_asset.rolling(win).cov(r_sig) / r_sig.rolling(win).var().replace(0, np.nan)


def f_cn10y_beta(df, s):
    u = prices['CN10Y']['close'].pct_change()
    r = df['close'].pct_change()
    z = pd.concat([r.rename('r'), u.rename('u')], axis=1).dropna()
    return rolling_beta(z['r'], z['u'], 60).reindex(z.index)


# ---------- library artifacts (real signal matrices) for rho audit ----------
lib = {}
for f in sorted(glob.glob('factors/*.json')):
    try:
        d = json.load(open(f))
        if d.get('validation', {}).get('status') == 'EFFECTIVE':
            art = d.get('signal_artifact')
            if art and Path('factors', art).exists():
                lib[d['factor_id']] = np.load(Path('factors', art))
    except Exception as e:
        print("lib skip", f, e)
print(f"library artifacts loaded: {len(lib)} -> {sorted(lib)}")


def max_lib_corr(panel):
    arr = signal_matrix(panel, grid)
    best, best_id = 0.0, None
    for fid, la in lib.items():
        arr_use = arr[-la.shape[0]:] if la.shape[0] < arr.shape[0] else arr
        corrs = []
        for i in range(arr_use.shape[0]):
            x, y = arr_use[i], la[i]
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() >= 8:
                r = pd.Series(x[m]).rank().corr(pd.Series(y[m]).rank())
                if np.isfinite(r):
                    corrs.append(r)
        if corrs:
            r = float(np.mean(corrs))
            if abs(r) > best:
                best, best_id = abs(r), fid
    return best, best_id


panel = factor_to_panel(f_cn10y_beta, prices)
m = validate_factor('cn10y_beta_60', panel, prices)
assert m is not None, "validation failed"
rho, fid = max_lib_corr(panel)
m['max_abs_library_correlation'] = rho
m['max_corr_library_id'] = fid
print(f"panel {panel.shape} range {panel.index.min().date()}..{panel.index.max().date()}")
print(f"IC={m['ic']:.4f} ICIR={m['icir']:.4f} hit={m['ic_hit_ratio']:.3f} n={m['n_ic_dates']} "
      f"cov={m['coverage_asset_days']:.3f} ge8={m['coverage_dates_ge8']:.3f} turn={m['turnover_10d_rank']:.2f} "
      f"maxlibrho={rho:.3f}({fid})")
print("decay:", {k: round(v, 4) for k, v in m['decay_ic_by_horizon'].items()})
ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084 and rho < 0.5
print("ADMISSION:", "PASS" if ok else "FAIL")
assert ok, "factor does not pass admission; do not persist"

path, arr = persist_factor(
    factor_id='cn10y_beta_60',
    factor_name='CN10Y yield-change beta (60d)',
    expression='rolling_beta(asset_daily_ret, cn10y_yield_daily_change, 60)',
    description=('60-day rolling regression beta of each asset\'s daily return on CN10Y '
                 '10-year yield daily changes. Assets negatively exposed to China rates '
                 '(fall when CN10Y yields rise) tend to underperform; expected_direction=-1.'),
    dependencies=['close', 'CN10Y close'],
    parameters={'window': 60, 'admission_horizon': 10},
    expected_direction=-1,
    panel=panel, metrics=m, tags=['beta', 'macro', 'rates', 'china', 'cross-asset'],
    grid=grid, prices=prices,
    regime_notes='2020-2026 multi-regime: China rates trend (2020-21 easing, 2022-24 repricing, 2025-26 cycle). '
                 'Negative IC stable across horizons; strongest at 10-20d.',
)
print("persisted:", path)
# verify reload
chk = json.loads(Path(path).read_text())
assert chk['factor_id'] == 'cn10y_beta_60'
assert chk['validation']['status'] == 'EFFECTIVE'
assert abs(chk['validation']['metrics']['ic']) >= 0.007
assert abs(chk['validation']['metrics']['icir']) >= 0.084
assert 'max_abs_library_correlation' in chk['validation']['metrics']
art = Path('factors', chk['signal_artifact'])
assert art.exists()
reloaded = np.load(art)
print("reload OK: json valid, status EFFECTIVE, IC/ICIR gates met, artifact",
      art.name, reloaded.shape)
