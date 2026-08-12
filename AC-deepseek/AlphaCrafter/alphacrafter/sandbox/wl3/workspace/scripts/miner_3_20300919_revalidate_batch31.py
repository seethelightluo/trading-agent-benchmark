"""miner_3 2030-09-19: re-validate batch-31 PASS candidates (gap_freq_60,
range_amplitude_60, night_ret_share_20) with fresh data and persist if they
still pass the benchmark-wide admission gates.

Gates (warm-up 2020-01-01..2026-07-15, 15-instrument cross-asset universe):
  |IC10| >= 0.007, |ICIR10| >= 0.084

Methodology matches miner_3_20300725_screen_batch31.py for gate consistency.
"""
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, 'scripts')
from factor_common import (load_prices, WATCHLIST, VAL_START, VAL_END,
                           factor_to_panel, forward_returns, persist_factor)

t0 = time.time()
prices = load_prices(days=2800)
print(f"assets loaded: {len(prices)}; max date: {max(d.index.max() for d in prices.values())} ({time.time()-t0:.1f}s)")

idx = set()
for s, df in prices.items():
    idx.update(df.index)
gidx = pd.DatetimeIndex(sorted(idx))
gidx = gidx[gidx >= VAL_START]
print(f"trading grid: {len(gidx)} dates, {gidx.min().date()}..{gidx.max().date()}")

cal_grid = pd.date_range(VAL_START, VAL_END, freq='D')
print(f"library calendar grid: {len(cal_grid)} dates, {cal_grid.min().date()}..{cal_grid.max().date()}")


def row_spearman(X, Y, min_valid=8):
    X = pd.DataFrame(X, dtype=float)
    Y = pd.DataFrame(Y, dtype=float)
    m = X.notna() & Y.notna()
    n = m.sum(axis=1)
    X2 = X.where(m); Y2 = Y.where(m)
    rx = X2.rank(axis=1)
    ry = Y2.rank(axis=1)
    rxm = rx.sub(rx.mean(axis=1), axis=0)
    rym = ry.sub(ry.mean(axis=1), axis=0)
    num = (rxm * rym).sum(axis=1)
    den = np.sqrt((rxm ** 2).sum(axis=1) * (rym ** 2).sum(axis=1))
    rho = (num / den.replace(0, np.nan)).to_numpy(dtype=float).copy()
    rho[n < min_valid] = np.nan
    return rho


# ---- candidate factor functions (identical to batch-31) ----
def f_gap_freq_60(df, s):
    o = df['open']; c = df['close']
    gap = o / c.shift(1) - 1.0
    return (gap.abs() > 0.01).rolling(60, min_periods=30).mean().reindex(df.index)


def f_range_amplitude_60(df, s):
    c = df['close']
    rmax = c.rolling(60, min_periods=30).max()
    rmin = c.rolling(60, min_periods=30).min()
    rmean = c.rolling(60, min_periods=30).mean()
    return ((rmax - rmin) / rmean.replace(0, np.nan)).reindex(df.index)


def f_night_ret_share_20(df, s):
    o = df['open']; c = df['close']
    night = o / c.shift(1) - 1.0
    total = c / c.shift(20) - 1.0
    night_sum = night.rolling(20).sum()
    tot_sum = total.rolling(20).sum()
    return (night_sum / tot_sum.replace(0, np.nan)).reindex(df.index)


candidates = {
    'gap_freq_60': (f_gap_freq_60, 'frequency of |overnight gap| > 1% over 60d (gap risk)'),
    'range_amplitude_60': (f_range_amplitude_60, '(max-min)/mean of close over 60d (trading amplitude)'),
    'night_ret_share_20': (f_night_ret_share_20, 'overnight return share of 20d total'),
}

fwd_mats = {}
for h in (1, 2, 3, 5, 10, 20):
    fwd = forward_returns(prices, h).reindex(gidx)
    fwd_mats[h] = fwd[WATCHLIST].values.astype(float)

# ---- library artifacts with per-factor grids from JSON metadata ----
lib_artifacts = {}
for jp in sorted(Path('factors').glob('*.json')):
    try:
        p = json.loads(jp.read_text(encoding='utf-8'))
        art = p.get('signal_artifact')
        if not art:
            continue
        arr = np.load(Path('factors') / art, allow_pickle=False)
        if arr.ndim != 2 or arr.shape[1] != 15:
            continue
        g = p.get('signal_artifact_grid', {})
        grid = None
        try:
            cand = pd.date_range(pd.Timestamp(g['start']), pd.Timestamp(g['end']), freq='D')
            if len(cand) == g.get('n_dates') and len(cand) == arr.shape[0]:
                grid = cand
        except Exception:
            pass
        if grid is None:
            if arr.shape[0] == len(cal_grid):
                grid = cal_grid
        if grid is not None:
            lib_artifacts[p.get('factor_id', jp.stem)] = (grid, arr)
    except Exception:
        pass
print(f"library artifacts with usable grids: {len(lib_artifacts)} ({time.time()-t0:.1f}s)")


def max_lib_corr(panel):
    best, best_id = 0.0, None
    for fid, (grid, la) in lib_artifacts.items():
        mc = panel.reindex(grid)[WATCHLIST].values.astype(float)
        c = row_spearman(mc, la)
        c = c[np.isfinite(c)]
        if len(c):
            r = float(np.abs(c).mean())
            if r > best:
                best, best_id = r, fid
    return best, best_id


warm = (gidx >= VAL_START) & (gidx <= VAL_END)
rstart = VAL_END + pd.Timedelta(days=1)
recent = gidx >= rstart
recent = recent & (gidx <= gidx.max() - pd.Timedelta(days=15))

results = {}
for fid, (fn, desc) in candidates.items():
    t1 = time.time()
    panel = factor_to_panel(fn, prices)
    if panel.empty:
        print(f"{fid}: EMPTY panel"); continue
    mat = panel.reindex(gidx)[WATCHLIST].values.astype(float)
    ics = {}
    for h in (1, 2, 3, 5, 10, 20):
        ics[h] = row_spearman(mat, fwd_mats[h])
    ic10w = ics[10][warm]
    ic10w = ic10w[np.isfinite(ic10w)]
    if len(ic10w) < 100:
        print(f"{fid}: insufficient warm IC dates {len(ic10w)}"); continue
    ic = float(ic10w.mean()); sd = float(ic10w.std(ddof=1))
    icir = ic / sd if sd > 0 else 0.0
    hit = float((ic10w > 0).mean()) if ic >= 0 else float((ic10w < 0).mean())
    fac = panel[(panel.index >= VAL_START) & (panel.index <= VAL_END)]
    cov = float(fac.notna().sum().sum()) / (fac.shape[0] * fac.shape[1]) if fac.shape[0] else 0.0
    ge8 = float((fac.notna().sum(axis=1) >= 8).mean())
    turn = float(fac.rank(axis=1).diff(10).abs().mean().mean()) if len(fac) > 10 else float('nan')
    decay = {str(h): float(np.nanmean(ics[h][warm])) for h in (1, 2, 3, 5, 10, 20)}
    icr = ics[10][recent]
    icr = icr[np.isfinite(icr)]
    ic_rmean = float(icr.mean()) if len(icr) >= 30 else float('nan')
    ic_rsd = float(icr.std(ddof=1)) if len(icr) >= 30 else float('nan')
    ic_ricir = ic_rmean / ic_rsd if len(icr) >= 30 and ic_rsd > 0 else float('nan')
    rho, fid_rho = max_lib_corr(panel)
    ok = abs(ic) >= 0.007 and abs(icir) >= 0.084
    results[fid] = {
        'ic': ic, 'icir': icir, 'hit': hit, 'cov': cov, 'ge8': ge8, 'turn': turn,
        'decay': decay, 'rho': rho, 'rho_id': fid_rho,
        'ic_recent': ic_rmean, 'icir_recent': ic_ricir,
        'n_recent': int(len(icr)), 'n_warm': int(len(ic10w)), 'PASS': ok,
    }
    print(f"{fid}: ic={ic:+.4f} icir={icir:+.4f} hit={hit:.3f} cov={cov:.3f} ge8={ge8:.3f} "
          f"turn={turn:.2f} rho={rho:.3f}({fid_rho}) recent_ic={ic_rmean:+.4f} recent_icir={ic_ricir:+.4f} "
          f"n_rec={len(icr)} PASS={ok} ({time.time()-t1:.1f}s)")

with open('scripts/miner_3_20300919_results_batch31_reval.json', 'w') as f:
    json.dump(results, f, indent=1, default=float)

# ---- persist PASS candidates ----
print("\n=== PERSIST PHASE ===")
for fid, (fn, desc) in candidates.items():
    if fid not in results or not results[fid]['PASS']:
        print(f"{fid}: not persisting (no PASS)")
        continue
    panel = factor_to_panel(fn, prices)
    m = results[fid]
    metrics = {
        'ic': m['ic'], 'icir': m['icir'], 'ic_hit_ratio': m['hit'],
        'coverage_asset_days': m['cov'], 'coverage_dates_ge8': m['ge8'],
        'turnover_10d_rank': m['turn'],
        'decay_ic_by_horizon': m['decay'],
        'max_abs_library_correlation': m['rho'],
        'max_corr_library_id': m['rho_id'],
        'ic_recent_post_warmup': m['ic_recent'],
        'icir_recent_post_warmup': m['icir_recent'],
        'n_ic_dates_warm': m['n_warm'],
        'n_ic_dates_recent': m['n_recent'],
        'admission_horizon': 10,
    }
    params = {
        'gap_freq_60': {'window': 60, 'min_periods': 30, 'gap_threshold_pct': 1.0},
        'range_amplitude_60': {'window': 60, 'min_periods': 30},
        'night_ret_share_20': {'window': 20, 'min_periods': 20},
    }[fid]
    expr = {
        'gap_freq_60': "mean(|open/close_prev - 1| > 0.01) over trailing 60d",
        'range_amplitude_60': "(rolling_max(close,60) - rolling_min(close,60)) / rolling_mean(close,60)",
        'night_ret_share_20': "sum(overnight returns, 20d) / (close/close_20d_ago - 1)",
    }[fid]
    deps = {
        'gap_freq_60': ['open', 'close'],
        'range_amplitude_60': ['close'],
        'night_ret_share_20': ['open', 'close'],
    }[fid]
    path, arr = persist_factor(
        factor_id=fid,
        factor_name=desc,
        expression=expr,
        description=desc,
        dependencies=deps,
        parameters=params,
        expected_direction='positive (high factor -> higher forward 10d cross-sectional return)' if m['ic'] > 0 else 'negative',
        panel=panel,
        metrics=metrics,
        tags=['gap', 'volatility', 'amplitude', 'overnight', 'intraday', 'cross-asset'],
        grid=None, prices=prices, version='1.0.0', status='EFFECTIVE',
        regime_notes='Warm-up 2020-01..2026-07 validation; re-validated 2030-09-19 with fresh data. Recent (post-warm-up) IC shown in metrics for drift monitoring.',
        extra={'validation': {
            'status': 'EFFECTIVE',
            'period': f'{VAL_START.date()}..{VAL_END.date()}',
            'last_validated': '2030-09-19',
            'admission_horizon': 10,
            'regime_notes': 'Warm-up 2020-01..2026-07 validation; re-validated 2030-09-19 with fresh data. Recent (post-warm-up) IC shown in metrics for drift monitoring.',
            'metrics': metrics,
        }},
    )
    print(f"persisted -> {path} ({arr.shape})")
print(f"DONE {time.time()-t0:.1f}s")
