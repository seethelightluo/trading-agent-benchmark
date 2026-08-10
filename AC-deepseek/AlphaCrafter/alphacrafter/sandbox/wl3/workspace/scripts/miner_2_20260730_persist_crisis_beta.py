"""miner_2 2026-07-30: validate + persist crisis_beta_60 (and 120d variants).

Uses shared canonical grid via factor_common; computes max abs library
correlation against the 12 currently-effective library signal artifacts.
Persists JSON + .npy artifact only if admission gate AND rho<0.5 pass.
"""
import sys, json, glob, time
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.insert(0, 'scripts')
from factor_common import (load_prices, factor_to_panel, validate_factor,
                           canonical_grid, signal_matrix, save_signal_artifact,
                           WATCHLIST, VAL_START, VAL_END)

np.seterr(all='ignore')
t0 = time.time()
prices = load_prices(days=2000)
grid = canonical_grid(prices)
print(f'loaded {len(prices)} assets; grid {len(grid)} dates '
      f'{grid.min().date()}..{grid.max().date()}')

# ---- library artifacts (12 effective factors) ----
lib = {}
for jf in glob.glob('factors/*.json'):
    fid = Path(jf).stem
    if fid == 'factor_ensemble' or fid.endswith('_deprecated'):
        continue
    try:
        d = json.loads(Path(jf).read_text())
        if d.get('validation', {}).get('status') != 'EFFECTIVE':
            continue
        art = d.get('signal_artifact')
        if not art:
            continue
        arr = np.load(Path('factors') / art)
        if arr.shape == (len(grid), 15):
            lib[fid] = pd.DataFrame(arr, index=grid, columns=WATCHLIST)
    except Exception as e:
        print(f'  lib skip {jf}: {e}')
print('library panels loaded:', sorted(lib.keys()))


def lib_corr(panel):
    best, best_id = 0.0, None
    for fid, lp in lib.items():
        idx = panel.index.intersection(lp.index)
        corrs = []
        for d in idx:
            x, y = panel.loc[d], lp.loc[d]
            m = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
            if m.sum() >= 8:
                c = x[m].rank().corr(y[m].rank())
                if np.isfinite(c):
                    corrs.append(c)
        if corrs:
            r = float(np.mean(corrs))
            if abs(r) > best:
                best, best_id = abs(r), fid
    return best, best_id


def crisis_beta(df, s, lookback=60, spx_vol_win=20, med_win=252, min_obs=None):
    spx = prices.get('SPX')
    if spx is None:
        return None
    r = df['close'].pct_change()
    rs = spx['close'].reindex(df.index).pct_change()
    spx_vol = rs.rolling(spx_vol_win).std()
    med = spx_vol.rolling(med_win).median()
    crisis = (spx_vol > med).astype(float)
    z = pd.concat([r.rename('r'), rs.rename('s'), crisis.rename('c')], axis=1).dropna()
    out = pd.Series(np.nan, index=z.index)
    cr = z[z['c'] > 0][['r', 's']]
    b = cr['r'].rolling(lookback).cov(cr['s']) / cr['s'].rolling(lookback).var()
    out.loc[b.index] = b
    if min_obs is not None:
        cnt = cr['r'].rolling(lookback).count()
        out[cnt < min_obs] = np.nan
    return out


def check(fid, lookback, min_obs=None):
    panel = factor_to_panel(lambda df, s: crisis_beta(df, s, lookback=lookback, min_obs=min_obs), prices)
    m = validate_factor(fid, panel, prices)
    if m is None:
        print(f'{fid}: insufficient data')
        return None, panel
    rho, best = lib_corr(panel)
    m['max_abs_library_correlation'] = rho
    m['max_corr_library_id'] = best
    ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084 and rho < 0.5
    print(f'{fid}: ic={m["ic"]:+.4f} icir={m["icir"]:+.4f} hit={m["ic_hit_ratio"]:.3f} '
          f'n={m["n_ic_dates"]} cov={m["coverage_asset_days"]:.3f} ge8={m["coverage_dates_ge8"]:.3f} '
          f'turn={m["turnover_10d_rank"]:.2f} rho={rho:.3f} vs {best} '
          f'decay5={m["decay_ic_by_horizon"]["5"]:+.4f} decay10={m["decay_ic_by_horizon"]["10"]:+.4f} '
          f'decay20={m["decay_ic_by_horizon"]["20"]:+.4f} -> {"ADMIT" if ok else "skip"}')
    return m, panel


def persist(fid, name, expr, desc, params, panel, metrics, tags, regime_notes):
    art = Path('factors') / f'{fid}_signal.npy'
    arr = save_signal_artifact(panel, grid, art)
    payload = {
        'factor_id': fid,
        'factor_name': name,
        'version': '1.0.0',
        'calculation': {'expression': expr, 'description': desc},
        'dependencies': ['close'],
        'parameters': params,
        'expected_direction': 1,
        'signal_artifact': art.name,
        'signal_artifact_format': 'npy',
        'signal_artifact_shape': list(arr.shape),
        'signal_artifact_grid': {
            'start': str(grid.min().date()), 'end': str(grid.max().date()),
            'n_dates': int(len(grid)), 'columns': WATCHLIST,
            'note': 'canonical grid shared by all library factors (see factor_common.canonical_grid)'},
        'validation': {
            'status': 'EFFECTIVE',
            'period': f'{VAL_START.date()}..{VAL_END.date()}',
            'last_validated': '2026-07-30',
            'admission_horizon': 10,
            'regime_notes': regime_notes,
            'metrics': metrics},
        'tags': tags,
        'benchmark_admission': {
            'contract': {'ic_threshold': 0.007, 'icir_threshold': 0.084,
                         'correlation_threshold': 0.5, 'library_capacity': 30,
                         'active_top_k': 10},
            'selected_metrics': {
                'ic': metrics['ic'], 'icir': metrics['icir'],
                'metric_path': 'validation.metrics',
                'max_abs_library_correlation': metrics.get('max_abs_library_correlation'),
                'correlation_path': 'validation.metrics.max_abs_library_correlation'}}}
    p = Path('factors') / f'{fid}.json'
    p.write_text(json.dumps(payload, indent=2, default=str))
    print(f'PERSISTED {p} (+ {art.name})')
    return p


# ---- main ----
results = {}
for lookback, mo, fid in [(60, None, 'crisis_beta_60'),
                          (120, None, 'crisis_beta_120'),
                          (120, 25, 'crisis_beta_120_mo25')]:
    m, panel = check(fid, lookback, mo)
    if m is not None:
        results[fid] = (m, panel)

# persist the best variant that admits
for fid in ['crisis_beta_60', 'crisis_beta_120_mo25', 'crisis_beta_120']:
    if fid not in results:
        continue
    m, panel = results[fid]
    if abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084 and m['max_abs_library_correlation'] < 0.5:
        persist(
            fid,
            'Crisis Beta 60d',
            'rolling_beta(asset_ret, SPX_ret | SPX_20d_vol > SPX_252d_median_vol, window=60)',
            'Beta of asset to SPX estimated only on high-volatility (crisis) regime days, '
            'where SPX 20d realized vol exceeds its 1y rolling median; positive values mean '
            'the asset loads more on equities during stress. Higher beta -> higher forward return '
            'in the cross-asset universe.',
            {'lookback': 60, 'spx_vol_win': 20, 'med_win': 252},
            m, ['beta', 'crisis', 'conditional', 'equity-stress'],
            '2020-01..2026-07 warm-up; cross-asset regimes incl. COVID-19, 2022 tightening, '
            '2023-25 risk-on, crypto cycles. Conditional crisis-regime beta.')
        print(f'PERSIST OK: {fid}')
        break

print(f'\nTOTAL {time.time()-t0:.1f}s')
