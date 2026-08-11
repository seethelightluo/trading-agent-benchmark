"""miner_2 2026-11-05: persist round25 passing factors (3 candidates).

Round25 screen (miner_2_20261105_screen_round25_novel.py) found 3 candidates
passing the benchmark admission gates on the warm-up window 2020-01-01..2026-07-15:
  - vol_regime_switch_20x60  IC +0.0375 ICIR +0.1313 rho_lib 0.080
  - range_skew_20            IC -0.0270 ICIR -0.0955 rho_lib 0.116 (reverse direction)
  - volume_entropy_20        IC +0.0411 ICIR +0.0899 rho_lib 0.231

This script re-executes validation (no fabricated metrics), computes library
correlation against the full EFFECTIVE library signal artifacts (vectorized),
persists JSON + .npy artifact via factor_common.persist_factor, then reads
back and verifies id/status/thresholds/artifact before finishing.
"""
import sys, time, json, math, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
from pathlib import Path
from factor_common import (load_prices, factor_to_panel, validate_factor,
                           canonical_grid, signal_matrix, WATCHLIST, persist_factor)

TODAY = '2026-11-05'
t0 = time.time()
prices = load_prices(days=2000)
grid = canonical_grid(prices)
print(f'prices={len(prices)} grid={len(grid)} {grid.min().date()}..{grid.max().date()}', flush=True)

# ---- library: effective factors with signal artifacts ----
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
print(f'library factors with artifacts: {len(lib)}', flush=True)


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


# ---- candidate factor functions (mirror round25 screen) ----
def f_vol_regime_switch_20x60(df, s):
    v = df['close'].pct_change().rolling(20, min_periods=10).std()
    med = v.rolling(60, min_periods=30).median()
    ind = (v > med).astype(float)
    def fn(x):
        if len(x) < 8 or np.all(np.isnan(x)):
            return np.nan
        d = np.diff(np.nan_to_num(x))
        return np.mean(d != 0)
    return ind.rolling(60, min_periods=30).apply(fn, raw=True)


def f_range_skew_20(df, s):
    rr = (df['high'] - df['low']) / df['close'].replace(0, np.nan)
    return rr.rolling(20, min_periods=12).skew()


def f_volume_entropy_20(df, s):
    v = df['volume'].replace(0, np.nan)
    def fn(x):
        x = x[~np.isnan(x)]
        if len(x) < 10 or x.sum() <= 0:
            return np.nan
        p = x / x.sum()
        return -np.sum(p * np.log(p)) / np.log(len(p))
    return v.rolling(20, min_periods=10).apply(fn, raw=True)


FUNCS = {
    'vol_regime_switch_20x60': f_vol_regime_switch_20x60,
    'range_skew_20': f_range_skew_20,
    'volume_entropy_20': f_volume_entropy_20,
}

REGIME = ('Validated 2020-01-01..2026-07-15 on the 15-asset cross-asset universe '
          'across COVID crash 2020, 2020-21 recovery bull, 2022 tightening bear, '
          '2023-24 AI-led equity rally, 2024-26 crypto/commodity cycles. Re-validated '
          f'{TODAY} on the frozen warm-up window; admission gates |IC|>=0.007 |ICIR|>=0.084, '
          'library correlation <0.5.')

FACTORS = [
    {
        'fid': 'vol_regime_switch_20x60',
        'name': 'Vol-regime switching frequency 20x60',
        'expr': 'MEAN(DIFF((RVOL20 > MEDIAN(RVOL20,60)) != 0), 60), RVOL20 = STD(pct_change(close,1),20)',
        'desc': 'Frequency with which an asset flips between high- and low-vol states over the '
                'trailing 60d, where the state is (20d realized vol above its 60d rolling median). '
                'Frequent vol-regime switching (unstable vol states) is followed by 10d '
                'outperformance in this worldline: positive cross-sectional IC +0.037, ICIR +0.131, '
                'high coverage (~73% asset-days), low turnover (~2.1 rank units / 10d), '
                'decay stable out to 20d (IC +0.039).',
        'deps': ['close'],
        'params': {'vol_window': 20, 'median_window': 60, 'switch_window': 60, 'min_obs': 10},
        'direction': 1,
        'tags': ['volatility', 'regime', 'cross-asset'],
    },
    {
        'fid': 'range_skew_20',
        'name': 'Intraday range skew 20d (reverse)',
        'expr': 'SKEW((high-low)/close, 20)',
        'desc': 'Skewness of the daily (high-low)/close range ratio over 20d. Positive range-skew '
                '(occasional very wide intraday ranges / right-tailed range distribution) is '
                'associated with lower forward 10d returns: negative cross-sectional IC -0.027, '
                'ICIR -0.096. Trade in reverse (prefer assets with low/negative range skew). '
                'Decay strengthens toward -0.026 at 10d, -0.023 at 20d.',
        'deps': ['close', 'high', 'low'],
        'params': {'window': 20, 'min_obs': 12},
        'direction': -1,
        'tags': ['volatility', 'skew', 'microstructure', 'cross-asset'],
    },
    {
        'fid': 'volume_entropy_20',
        'name': 'Volume concentration entropy 20d',
        'expr': 'ENTROPY(volume_share_20d) / LOG(20), volume_share = volume / SUM(volume,20)',
        'desc': 'Normalized entropy of the 20d volume share distribution: high entropy means '
                'volume is spread evenly across days (smooth, non-bursty activity), low entropy '
                'means activity concentrated in few days. Smooth activity predicts higher 10d '
                'forward returns: positive cross-sectional IC +0.041, ICIR +0.090, low turnover '
                '(~1.3). Coverage is moderate (~44% asset-days, ~53% dates with >=8 valid) because '
                'several index/yield series carry sparse volume.',
        'deps': ['volume'],
        'params': {'window': 20, 'min_obs': 10},
        'direction': 1,
        'tags': ['volume', 'entropy', 'liquidity', 'cross-asset'],
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
        'artifact_identical': bool(np.allclose(arr, arr2, equal_nan=True)),
        'rho_reported': payload['validation']['metrics']['max_abs_library_correlation'],
    }
    print(f'VERIFY {fid}: {json.dumps(checks, default=str)}', flush=True)
    if not all(checks[k] for k in ['json_valid', 'factor_id_ok', 'status_ok',
                                   'last_validated_ok', 'ic_ok', 'icir_ok',
                                   'artifact_shape_ok', 'artifact_identical']):
        ok_all = False
        print(f'VERIFY FAILED for {fid}', flush=True)

print(f'elapsed={time.time()-t0:.1f}s all_ok={ok_all}', flush=True)
