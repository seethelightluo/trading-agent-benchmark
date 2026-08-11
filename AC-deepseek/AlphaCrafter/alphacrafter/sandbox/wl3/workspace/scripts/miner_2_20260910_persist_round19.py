"""miner_2 2026-09-10: persist round19 passing factors (4 candidates).

Round19 screen (miner_2_20260730_screen_round19b_novel.py) found 4 candidates
passing the benchmark admission gates on the warm-up window 2020-01-01..2026-07-15:
  - sx5e_beta_60     IC 0.0659 ICIR 0.1652 rho_lib 0.485
  - gold_rel_mom_20  IC 0.0317 ICIR 0.0880 rho_lib 0.089
  - spx_rel_mom_20   IC 0.0312 ICIR 0.0868 rho_lib 0.089
  - amihud_z_20_60   IC 0.0383 ICIR 0.0882 rho_lib 0.116

This script re-executes validation (no fabricated metrics), computes library
correlation against the full EFFECTIVE library signal artifacts (vectorized),
persists JSON + .npy artifact via factor_common.persist_factor, then reads
back and verifies id/status/thresholds/artifact before finishing.
"""
import sys, time, json, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
from pathlib import Path
from factor_common import (load_prices, factor_to_panel, validate_factor,
                           canonical_grid, signal_matrix, WATCHLIST, persist_factor)

TODAY = '2026-09-10'
t0 = time.time()
prices = load_prices(days=2000)
grid = canonical_grid(prices)
print(f'prices={len(prices)} grid={len(grid)} {grid.min().date()}..{grid.max().date()}', flush=True)

# ---- library: effective factors with signal artifacts (same as round19b) ----
lib = {}
for p in sorted(Path('factors').glob('*.json')):
    if p.name.endswith('.bak') or 'deprecated' in p.name or 'ensemble' in p.name:
        continue
    try:
        payload = json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        continue
    if payload.get('validation', {}).get('status') != 'EFFECTIVE':
        continue
    art = payload.get('signal_artifact')
    art_path = p.parent / str(art) if art else None
    if art_path is not None and art_path.exists():
        lib[payload['factor_id']] = np.load(art_path, allow_pickle=False)
print(f'library factors: {len(lib)}', flush=True)


def rank_matrix(arr):
    out = np.full(arr.shape, np.nan)
    T = arr.shape[0]
    for t in range(T):
        row = arr[t]
        m = np.isfinite(row)
        n = int(m.sum())
        if n >= 8:
            r = np.full(arr.shape[1], np.nan)
            r[m] = pd.Series(row[m]).rank().values
            out[t] = r
    return out


lib_rank = {fid: rank_matrix(arr) for fid, arr in lib.items()}


def lib_max_corr_fast(panel):
    arr = signal_matrix(panel, grid)
    cand_rank = rank_matrix(arr)
    best, best_id = 0.0, None
    for fid, lr in lib_rank.items():
        cs = np.full(cand_rank.shape[0], np.nan)
        for t in range(cand_rank.shape[0]):
            a, b = cand_rank[t], lr[t]
            m = np.isfinite(a) & np.isfinite(b)
            n = int(m.sum())
            if n >= 8:
                a2, b2 = a[m], b[m]
                ma, mb = a2.mean(), b2.mean()
                sa, sb = a2.std(ddof=1), b2.std(ddof=1)
                if sa > 1e-12 and sb > 1e-12:
                    cs[t] = ((a2 - ma) * (b2 - mb)).mean() / (sa * sb)
        if np.isfinite(cs).any():
            r = float(np.nanmean(cs))
            if abs(r) > best:
                best, best_id = abs(r), fid
    return best, best_id


# ---- helpers (mirror round19b) ----
def bench_ret(sym):
    return prices[sym]['close'].pct_change()


def rolling_beta(asset_df, bench_series, window, min_obs=30, bench_is_yield=False):
    r = asset_df['close'].pct_change()
    b = bench_series.diff() if bench_is_yield else bench_series
    z = pd.concat([r.rename('r'), b.rename('b')], axis=1).dropna()
    cov = z['r'].rolling(window, min_periods=min_obs).cov(z['b'])
    var = z['b'].rolling(window, min_periods=min_obs).var()
    return (cov / var).reindex(z.index)


def f_sx5e_beta_60(df, s):
    return rolling_beta(df, bench_ret('SX5E'), 60)


def f_gold_rel_mom_20(df, s):
    mine = df['close'] / df['close'].shift(20) - 1.0
    gold = prices['XAU']['close'] / prices['XAU']['close'].shift(20) - 1.0
    return (mine - gold).reindex(mine.index)


def f_spx_rel_mom_20(df, s):
    mine = df['close'] / df['close'].shift(20) - 1.0
    spx = prices['SPX']['close'] / prices['SPX']['close'].shift(20) - 1.0
    return (mine - spx).reindex(mine.index)


def f_amihud_z_20_60(df, s):
    r = df['close'].pct_change().abs()
    v = df['volume'].astype(float).replace(0, np.nan)
    ami = (r / v).rolling(60, min_periods=30).mean()
    mu = ami.rolling(60, min_periods=30).mean()
    sd = ami.rolling(60, min_periods=30).std()
    z = (ami - mu) / sd
    return z.rolling(20, min_periods=10).mean()


FUNCS = {
    'sx5e_beta_60': f_sx5e_beta_60,
    'gold_rel_mom_20': f_gold_rel_mom_20,
    'spx_rel_mom_20': f_spx_rel_mom_20,
    'amihud_z_20_60': f_amihud_z_20_60,
}

REGIME = ('Validated 2020-01-01..2026-07-15 on the 15-asset cross-asset universe '
          'across COVID crash 2020, 2020-21 recovery bull, 2022 tightening bear, '
          '2023-24 AI-led equity rally, 2024-26 crypto/commodity cycles. Re-validated '
          f'{TODAY} on the frozen warm-up window; admission gates |IC|>=0.007 |ICIR|>=0.084.')

FACTORS = [
    {
        'fid': 'sx5e_beta_60',
        'name': 'SX5E-beta 60d',
        'expr': 'BETA(pct_change(close,1), pct_change(SX5E,1), 60)',
        'desc': 'Rolling 60d beta of each asset daily return on the Euro Stoxx 50 '
                '(SX5E) daily return. Assets positively exposed to European equity '
                'risk carry outperform over 10-20d horizons in this worldline: '
                'positive cross-sectional predictive power (IC +0.066, ICIR +0.165). '
                'High coverage (~93% asset-days, ~98% dates with >=8 valid).',
        'deps': ['close', 'SX5E close'],
        'params': {'window': 60, 'benchmark': 'SX5E', 'min_obs': 30},
        'direction': 1,
        'tags': ['beta', 'regional-equity', 'risk-on'],
    },
    {
        'fid': 'gold_rel_mom_20',
        'name': 'Gold-relative momentum 20d',
        'expr': 'pct_change(close,20) - pct_change(XAU,20)',
        'desc': 'Each asset 20d total return minus the 20d return of gold (XAU). '
                'Assets outperforming the safe-haven benchmark keep trending over '
                '10-20d: positive cross-sectional predictive power (IC +0.032, '
                'ICIR +0.088). Turnover moderate-high (~3.0 rank units / 10d).',
        'deps': ['close', 'XAU close'],
        'params': {'window': 20, 'benchmark': 'XAU'},
        'direction': 1,
        'tags': ['momentum', 'relative', 'cross-asset', 'safe-haven'],
    },
    {
        'fid': 'spx_rel_mom_20',
        'name': 'SPX-relative momentum 20d',
        'expr': 'pct_change(close,20) - pct_change(SPX,20)',
        'desc': 'Each asset 20d total return minus the 20d return of the S&P 500 '
                '(SPX). Assets outperforming the core US equity benchmark keep '
                'trending over 10-20d: positive cross-sectional predictive power '
                '(IC +0.031, ICIR +0.087).',
        'deps': ['close', 'SPX close'],
        'params': {'window': 20, 'benchmark': 'SPX'},
        'direction': 1,
        'tags': ['momentum', 'relative', 'cross-asset'],
    },
    {
        'fid': 'amihud_z_20_60',
        'name': 'Amihud illiquidity z-score 20/60',
        'expr': 'MEAN(ZSCORE(MEAN(|pct_change(close,1)|/volume,60), 60), 20)',
        'desc': 'Per-asset z-score of the 60d Amihud illiquidity ratio '
                '(|daily return| / volume), smoothed by a 20d mean. Elevated '
                'relative illiquidity (liquidity stress) precedes 10d outperformance '
                'in this worldline: positive IC +0.038, ICIR +0.088. Lower coverage '
                '(~43% asset-days) because several index/yield series carry sparse '
                'volume; low turnover (~1.4).',
        'deps': ['close', 'volume'],
        'params': {'ami_window': 60, 'z_window': 60, 'smooth_window': 20, 'min_obs': 30},
        'direction': 1,
        'tags': ['liquidity', 'amihud', 'stress', 'volume'],
    },
]

ok_all = True
for f in FACTORS:
    fid = f['fid']
    fn = FUNCS[fid]
    t1 = time.time()
    panel = factor_to_panel(fn, prices)
    m = validate_factor(fid, panel, prices)
    if m is None:
        print(f'{fid}: insufficient data -> None', flush=True)
        ok_all = False
        continue
    rho, rho_id = lib_max_corr_fast(panel)
    m['max_abs_library_correlation'] = rho
    m['max_corr_library_id'] = rho_id
    ic_ok = abs(m['ic']) >= 0.007
    icir_ok = abs(m['icir']) >= 0.084
    corr_ok = rho < 0.5
    ok = ic_ok and icir_ok and corr_ok
    print(f'{fid}: panel={panel.shape} [{time.time()-t1:.1f}s] '
          f'IC={m["ic"]:.4f} ICIR={m["icir"]:.4f} rho_lib={rho:.3f}({rho_id}) pass={ok}', flush=True)
    if not ok:
        ok_all = False
        continue

    metrics = {k: m[k] for k in ['ic', 'icir', 'ic_hit_ratio', 'n_ic_dates',
                                 'coverage_asset_days', 'coverage_dates_ge8',
                                 'turnover_10d_rank', 'decay_ic_by_horizon',
                                 'max_abs_library_correlation', 'max_corr_library_id']}
    path, arr = persist_factor(
        factor_id=fid,
        factor_name=f['name'],
        expression=f['expr'],
        description=f['desc'],
        dependencies=f['deps'],
        parameters=f['params'],
        expected_direction=f['direction'],
        panel=panel,
        metrics=metrics,
        tags=f['tags'],
        grid=grid,
        prices=prices,
        version='1.0.0',
        status='EFFECTIVE',
        regime_notes=REGIME,
    )
    # stamp recency on the persisted record
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    payload['validation']['last_validated'] = TODAY
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'PERSISTED {fid} -> {path}', flush=True)

    # ---- read back and verify ----
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    arr2 = np.load(Path('factors') / payload['signal_artifact'], allow_pickle=False)
    checks = {
        'json_valid': True,
        'factor_id_ok': payload['factor_id'] == fid,
        'status_ok': payload['validation']['status'] == 'EFFECTIVE',
        'last_validated_ok': payload['validation']['last_validated'] == TODAY,
        'ic_ok': abs(payload['validation']['metrics']['ic']) >= 0.007,
        'icir_ok': abs(payload['validation']['metrics']['icir']) >= 0.084,
        'artifact_shape_ok': arr2.shape == (len(grid), len(WATCHLIST)),
        'artifact_matches': np.allclose(arr2, arr, equal_nan=True),
        'corr_reported': 'max_abs_library_correlation' in payload['validation']['metrics'],
    }
    print(f'VERIFY {fid}: {json.dumps(checks)}', flush=True)
    assert all(checks.values()), f'verification failed for {fid}'
    print(f'OK {fid} persisted+reloadable IC={metrics["ic"]:.4f} ICIR={metrics["icir"]:.4f} '
          f'rho_lib={rho:.3f} decay10={metrics["decay_ic_by_horizon"]["10"]:.4f}', flush=True)

print(f'elapsed={time.time()-t0:.1f}s all_ok={ok_all}', flush=True)
