"""Round-11 screen: novel cross-asset factor batch.

Candidates (all interpretable, OHLCV-based, distinct from the 13-factor library):
 1. vol_trend_20_60      - volume participation trend (20d/60d volume ratio - 1)
 2. clv_20               - close location value, 20d average of (C-L)/(H-L)
 3. trend_r_20           - rolling corr(close, time) over 20d (trend strength+direction)
 4. gain_loss_ratio_40   - profit factor: sum(pos rets)/abs(sum(neg rets)) over 40d
 5. ret_skew_20          - skewness of close-to-close daily returns over 20d
 6. mom_accel_60_120     - momentum acceleration: 60d mom minus 120d mom (skip 5)
 7. range_pos_20         - 20d range position: (C - min(L,20))/(max(H,20)-min(L,20))
 8. down_avg_60          - mean negative daily return over 60d (pain severity)

Validation: shared factor_common battery (daily Spearman IC vs 10d fwd return,
2020-01-01..2026-07-15), max-abs pairwise rho vs ALL effective library signal
artifacts. Admission: |IC|>=0.007, |ICIR|>=0.084, rho<0.5.
"""
import sys, json, glob
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.insert(0, 'scripts')
from factor_common import (WATCHLIST, load_prices, canonical_grid,
                           signal_matrix, factor_to_panel, validate_factor)

np.seterr(all='ignore')
prices = load_prices(days=2500)
grid = canonical_grid(prices)
print(f"prices {len(prices)} assets; canonical grid {len(grid)} dates "
      f"({grid.min().date()}..{grid.max().date()})", flush=True)

# ---------- library artifacts (all EFFECTIVE json + .npy) ----------
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


# ---------- candidate definitions ----------
def vol_trend_20_60(df, s):
    v = df['volume'].astype(float).replace(0, np.nan)
    m20 = v.rolling(20).mean()
    m60 = v.rolling(60).mean()
    return m20 / m60 - 1.0


def clv_20(df, s):
    rng = (df['high'] - df['low']).replace(0, np.nan)
    clv = (df['close'] - df['low']) / rng
    return clv.rolling(20).mean()


def trend_r_20(df, s):
    c = df['close']
    t = np.arange(len(c))
    return c.rolling(20).apply(lambda x: np.corrcoef(x, t[:len(x)])[0, 1], raw=True)


def gain_loss_ratio_40(df, s):
    r = df['close'].pct_change()
    pos = r.clip(lower=0).rolling(40).sum()
    neg = (-r.clip(upper=0)).rolling(40).sum()
    return pos / neg.replace(0, np.nan)


def ret_skew_20(df, s):
    return df['close'].pct_change().rolling(20).skew()


def mom_accel_60_120(df, s):
    c = df['close']
    mom60 = c.shift(5) / c.shift(65) - 1.0
    mom120 = c.shift(5) / c.shift(125) - 1.0
    return mom60 - mom120


def range_pos_20(df, s):
    hi = df['high'].rolling(20).max()
    lo = df['low'].rolling(20).min()
    return (df['close'] - lo) / (hi - lo).replace(0, np.nan)


def down_avg_60(df, s):
    r = df['close'].pct_change()
    return r.clip(upper=0).rolling(60).mean()


candidates = {
    'vol_trend_20_60': dict(fn=vol_trend_20_60, name='Volume participation trend (20d/60d)',
                            expr='mean(volume,20)/mean(volume,60) - 1',
                            deps=['volume'], direction=1),
    'clv_20': dict(fn=clv_20, name='Close location value (20d avg)',
                   expr='mean((close-low)/(high-low),20)', deps=['close', 'high', 'low'], direction=1),
    'trend_r_20': dict(fn=trend_r_20, name='Trend strength (corr close vs time, 20d)',
                       expr='rolling corr(close, time, 20)', deps=['close'], direction=1),
    'gain_loss_ratio_40': dict(fn=gain_loss_ratio_40, name='Profit factor 40d',
                               expr='sum(max(ret,0),40)/abs(sum(min(ret,0),40))',
                               deps=['close'], direction=1),
    'ret_skew_20': dict(fn=ret_skew_20, name='Skew of daily returns (20d)',
                        expr='skew_20d(pct_change(close))', deps=['close'], direction=1),
    'mom_accel_60_120': dict(fn=mom_accel_60_120, name='Momentum acceleration 60-120d',
                             expr='mom60(skip5) - mom120(skip5)', deps=['close'], direction=1),
    'range_pos_20': dict(fn=range_pos_20, name='20d range position',
                         expr='(close-min(low,20))/(max(high,20)-min(low,20))',
                         deps=['close', 'high', 'low'], direction=1),
    'down_avg_60': dict(fn=down_avg_60, name='Mean negative daily return 60d (pain)',
                        expr='mean(min(ret,0),60)', deps=['close'], direction=-1),
}

results = {}
panels = {}
for fid, cfg in candidates.items():
    panel = factor_to_panel(cfg['fn'], prices)
    m = validate_factor(fid, panel, prices)
    if m is None:
        print(f"{fid}: INSUFFICIENT -> skip", flush=True)
        results[fid] = {'ok': False, 'metrics': {'error': 'insufficient'}}
        continue
    mat = signal_matrix(panel, grid)
    rho, lib_id = max_lib_corr(mat)
    m['max_abs_library_correlation'] = rho
    m['max_corr_library_id'] = lib_id
    ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084 and rho < 0.5
    results[fid] = {'ok': ok, 'metrics': m}
    panels[fid] = panel
    print(f"{fid}: panel {panel.shape} | IC={m['ic']:+.4f} ICIR={m['icir']:+.4f} "
          f"hit={m['ic_hit_ratio']:.3f} n={m['n_ic_dates']} cov={m['coverage_asset_days']:.3f} "
          f"ge8={m['coverage_dates_ge8']:.3f} turn={m['turnover_10d_rank']:.2f} "
          f"rho={rho:.3f}({lib_id}) -> {'PASS' if ok else 'FAIL'}", flush=True)
    print("   decay:", {k: round(v, 4) for k, v in m['decay_ic_by_horizon'].items()}, flush=True)

json.dump(results, open('scripts/miner_3_20260730_results_round11.json', 'w'), indent=1, default=str)

# ---------- candidate-candidate pairwise rho (novelty audit) ----------
print("\n=== candidate-candidate mean daily cross-sectional Spearman rho ===")
ids = list(panels)
M = np.full((len(ids), len(ids)), np.nan)
for i, fi in enumerate(ids):
    Ri = rank_rows(signal_matrix(panels[fi], grid))
    for j, fj in enumerate(ids):
        if i == j:
            continue
        Rj = rank_rows(signal_matrix(panels[fj], grid))
        r = row_spearman(Ri, Rj)
        M[i, j] = float(np.nanmean(r)) if np.isfinite(r).any() else np.nan
hdr = "        " + " ".join(f"{i[:9]:>9}" for i in ids)
print(hdr)
for i, idi in enumerate(ids):
    row = " ".join(f"{M[i, j]:9.2f}" if np.isfinite(M[i, j]) else f"{'-':>9}" for j in range(len(ids)))
    print(f"{idi[:9]:>8} {row}")
print("\nsaved scripts/miner_3_20260730_results_round11.json")
