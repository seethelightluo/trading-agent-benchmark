"""Persist round-11 passing candidates (mom_accel_60_120, range_pos_20).

Recomputes panels with factor_common, runs shared validation battery, audits max
library correlation against ALL effective library signal artifacts, persists
JSON + .npy artifact, and verifies round-trip.
"""
import sys, json, glob
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.insert(0, 'scripts')
from factor_common import (WATCHLIST, load_prices, canonical_grid,
                           signal_matrix, factor_to_panel, validate_factor,
                           persist_factor)

np.seterr(all='ignore')
prices = load_prices(days=2500)
grid = canonical_grid(prices)
print(f"prices {len(prices)} assets; canonical grid {len(grid)} dates "
      f"({grid.min().date()}..{grid.max().date()})", flush=True)

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
print(f"library artifacts: {len(lib)} -> {sorted(lib)}", flush=True)

MIN_V = 8


def rank_rows(M):
    T, n = M.shape
    R = np.full_like(M, np.nan)
    for t in range(T):
        v = M[t]
        m = np.isfinite(v)
        if m.sum() >= MIN_V:
            idx = np.where(m)[0]
            R[t, idx] = v[idx].argsort().argsort().astype(float)
    return R


def row_spearman(RA, RB):
    m = np.isfinite(RA) & np.isfinite(RB)
    A = np.where(m, RA, np.nan)
    B = np.where(m, RB, np.nan)
    cnt = m.sum(axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        Ac = A - np.nanmean(A, axis=1, keepdims=True)
        Bc = B - np.nanmean(B, axis=1, keepdims=True)
        num = np.nansum(Ac * Bc, axis=1)
        den = np.sqrt(np.nansum(Ac * Ac, axis=1) * np.nansum(Bc * Bc, axis=1))
        rho = num / den
    rho[~((cnt >= MIN_V) & (den > 0))] = np.nan
    return rho


def max_lib_corr(mat):
    Rc = rank_rows(mat)
    best, best_id = 0.0, None
    for fid, la in lib.items():
        if la.shape[0] < mat.shape[0]:
            Rc_use = Rc[-la.shape[0]:]
        else:
            Rc_use = Rc
        Rl = rank_rows(la)
        rho = row_spearman(Rc_use, Rl)
        r = float(np.nanmean(rho)) if np.isfinite(rho).any() else 0.0
        if abs(r) > best:
            best, best_id = abs(r), fid
    return best, best_id


def mom_accel_60_120(df, s):
    c = df['close']
    mom60 = c.shift(5) / c.shift(65) - 1.0
    mom120 = c.shift(5) / c.shift(125) - 1.0
    return mom60 - mom120


def range_pos_20(df, s):
    hi = df['high'].rolling(20).max()
    lo = df['low'].rolling(20).min()
    return (df['close'] - lo) / (hi - lo).replace(0, np.nan)


candidates = {
    'mom_accel_60_120': dict(
        fn=mom_accel_60_120,
        name='Momentum acceleration 60-120d (contrarian)',
        expr='mom60(skip5) - mom120(skip5)  [close.shift(5)/close.shift(65) - close.shift(5)/close.shift(125)]',
        desc=("Difference between 60d and 120d momentum (both skip 5 days). "
              "NEGATIVE cross-sectional IC: assets whose near-term momentum is "
              "accelerating relative to their longer-term momentum tend to "
              "UNDERPERFORM over the next 10 days (momentum-acceleration mean "
              "reversion / crowded-trend fade). Expected factor direction is "
              "-1 (low accel assets favored). Max library rho ~0.13 (vs "
              "cn10y_beta_60) - near-orthogonal to the existing library."),
        deps=['close'],
        params={'mom_short_win': 60, 'mom_long_win': 120, 'skip': 5},
        tags=['momentum', 'acceleration', 'contrarian', 'cross-asset'],
        direction=-1,
        regime="Validated 2020-01-01..2026-07-15 across COVID crash, 2022 tightening "
               "bear, 2023-25 risk-on, crypto/commodity cycles. Negative IC at all "
               "horizons 1-20d (strongest -0.035 at 5d and -0.037 at 20d); hit ratio 0.539."),
    'range_pos_20': dict(
        fn=range_pos_20,
        name='20d range position (close location)',
        expr='(close - min(low,20)) / (max(high,20) - min(low,20))',
        desc=("Position of close within the trailing 20-day high-low range. "
              "Assets trading near the top of their 20d range show positive "
              "10d forward continuation (breakout/trend persistence), distinct "
              "from magnitude momentum (max library rho ~0.37 vs "
              "vol_adj_mom_20_60, below the 0.5 gate)."),
        deps=['close', 'high', 'low'],
        params={'window': 20},
        tags=['range', 'breakout', 'trend', 'cross-asset'],
        direction=1,
        regime="Validated 2020-01-01..2026-07-15. Positive IC at 5-20d horizons "
               "(strongest 0.041 at 20d) with a mild short-horizon reversal "
               "(IC -0.019 at 1d); hit ratio 0.547."),
}

for fid, cfg in candidates.items():
    panel = factor_to_panel(cfg['fn'], prices)
    m = validate_factor(fid, panel, prices)
    if m is None:
        print(f"{fid}: INSUFFICIENT -> skip", flush=True)
        continue
    mat = signal_matrix(panel, grid)
    rho, lib_id = max_lib_corr(mat)
    m['max_abs_library_correlation'] = rho
    m['max_corr_library_id'] = lib_id
    ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084 and rho < 0.5
    print(f"\n=== {fid} === panel {panel.shape} | IC={m['ic']:+.4f} ICIR={m['icir']:+.4f} "
          f"hit={m['ic_hit_ratio']:.3f} n={m['n_ic_dates']} cov={m['coverage_asset_days']:.3f} "
          f"ge8={m['coverage_dates_ge8']:.3f} rho={rho:.3f}({lib_id}) -> {'PASS' if ok else 'FAIL'}", flush=True)
    if not ok:
        print(f"{fid}: FAIL -> not persisted", flush=True)
        continue
    path, arr = persist_factor(
        factor_id=fid,
        factor_name=cfg['name'],
        expression=cfg['expr'],
        description=cfg['desc'],
        dependencies=cfg['deps'],
        parameters=cfg['params'],
        expected_direction=cfg['direction'],
        panel=panel,
        metrics=m,
        tags=cfg['tags'],
        grid=grid,
        prices=prices,
        version='1.0.0',
        status='EFFECTIVE',
        regime_notes=cfg['regime'],
        extra={'signal_provenance': {
            'source': 'recomputed from alphacrafter.sim.utils daily OHLC series',
            'panel_shape': f"{panel.shape[0]}x{panel.shape[1]}",
            'panel_range': f"{panel.index.min().date()}..{panel.index.max().date()}",
            'validation_window': '2020-01-01..2026-07-15',
            'ic_method': 'daily cross-sectional Spearman rank IC vs 10d forward return',
            'note': 'expression deterministic and reproducible from OHLC series only'}},
    )
    print(f"{fid}: PERSISTED -> {path} artifact {arr.shape}", flush=True)

print("\n--- verify round-trip ---", flush=True)
for fid in candidates:
    p = Path('factors') / f'{fid}.json'
    if not p.exists():
        print(f"{fid}: MISSING", flush=True)
        continue
    d = json.loads(p.read_text(encoding='utf-8'))
    art = Path('factors') / d['signal_artifact']
    ok = (d['factor_id'] == fid and d['validation']['status'] == 'EFFECTIVE'
          and art.exists()
          and abs(d['validation']['metrics']['ic']) >= 0.007
          and abs(d['validation']['metrics']['icir']) >= 0.084
          and d['validation']['metrics']['max_abs_library_correlation'] < 0.5)
    print(f"{fid}: id={d['factor_id']} status={d['validation']['status']} "
          f"ic={d['validation']['metrics']['ic']:+.4f} icir={d['validation']['metrics']['icir']:+.4f} "
          f"rho={d['validation']['metrics']['max_abs_library_correlation']:.3f} "
          f"dir={d['expected_direction']} "
          f"artifact={d['signal_artifact']}({np.load(art).shape}) -> {'VERIFIED' if ok else 'CHECK'}", flush=True)
print("done", flush=True)
