"""Round 10 (fast, vectorized): novel factor candidates vs 12-factor library.

All row-wise Spearman correlations are computed with numpy (rank rows once,
then vectorized Pearson), avoiding the pandas-per-date bottleneck that made the
first version time out.

Candidates:
 [carryover from round-9 batch] vwap_dev_20, macd_hist_12_26, overnight_skew_20,
   max_gap_20, win_rate_40, intraday_ret_skew_20, amihud_trend_20_60, us10y_cond_mom_20
 [round-10 new] cross_vol_ratio_20_60, vol_asym_60, rel_mom_basket_10,
   amihud_level_60, beta_diff_60_120, vol_mom_ratio_10_60, intraday_ret_20,
   range_slope_20, gap_dir_20, rsi_3
"""
import sys, json, glob, time
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.insert(0, 'scripts')
from factor_common import (WATCHLIST, load_prices, load_index, canonical_grid,
                           signal_matrix, factor_to_panel)

np.seterr(all='ignore')
t0 = time.time()
prices = load_prices(days=2500)
grid = canonical_grid(prices)
print(f"prices loaded: {len(prices)} assets; canonical grid {len(grid)} dates "
      f"({grid.min().date()}..{grid.max().date()})", flush=True)

# ---------- library artifacts (real signal matrices) ----------
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
print(f"library artifacts loaded: {len(lib)} -> {sorted(lib)}", flush=True)

MIN_V = 8


def rank_rows(M):
    """Row-wise ascending ranks with NaN preserved; returns float array."""
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
    """Spearman per row between two pre-ranked matrices (same NaN pattern ok)."""
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


# ---------- fast validation (10d admission horizon, decay grid) ----------
def fwd_ret_matrix(h):
    """(T,15) forward h-day simple return matrix aligned to canonical grid."""
    cols = []
    for s in WATCHLIST:
        df = prices[s]
        f = df['close'].shift(-h) / df['close'] - 1.0
        cols.append(f.reindex(grid).values.astype(float))
    return np.column_stack(cols)


def fast_validate(panel):
    """Return metrics dict (same keys as factor_common.validate_factor)."""
    mat = signal_matrix(panel, grid)
    Rf = rank_rows(mat)
    out = {'n_ic_dates': 0}
    horizons = (1, 2, 3, 5, 10, 20)
    ic_series = {}
    for h in horizons:
        F = fwd_ret_matrix(h)
        RF = rank_rows(F)
        ic_series[h] = row_spearman(Rf, RF)
    ic10 = ic_series[10]
    valid = ic10[np.isfinite(ic10)]
    if len(valid) < 100:
        return None
    ic_mean = float(valid.mean())
    ic_std = float(valid.std(ddof=1)) if len(valid) > 1 else 0.0
    icir = ic_mean / ic_std if ic_std > 0 else 0.0
    hit = float((valid > 0).mean()) if ic_mean >= 0 else float((valid < 0).mean())
    total_cells = mat.shape[0] * mat.shape[1]
    valid_cells = int(np.isfinite(mat).sum())
    coverage = valid_cells / total_cells if total_cells else 0.0
    ge8 = float((np.isfinite(mat).sum(axis=1) >= MIN_V).mean())
    rmat = rank_rows(mat)
    if rmat.shape[0] > 10:
        turn = float(np.nanmean(np.abs(np.diff(rmat, axis=0, n=10))))
    else:
        turn = float('nan')
    return {
        'ic': ic_mean, 'icir': icir, 'ic_hit_ratio': hit,
        'n_ic_dates': int(len(valid)), 'coverage_asset_days': coverage,
        'coverage_dates_ge8': ge8, 'turnover_10d_rank': turn,
        'decay_ic_by_horizon': {str(h): (float(np.nanmean(ic_series[h])) if np.isfinite(ic_series[h]).any() else float('nan')) for h in horizons},
    }


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


# ---------- candidate definitions (vectorized where possible) ----------
def vwap_dev_20(df, s):
    vol = df['volume'].replace(0, np.nan)
    pv = (df['close'] * vol).rolling(20).sum()
    vv = vol.rolling(20).sum()
    vwap = pv / vv.replace(0, np.nan)
    return df['close'] / vwap - 1.0


def macd_hist_12_26(df, s):
    c = df['close']
    e12 = c.ewm(span=12, adjust=False).mean()
    e26 = c.ewm(span=26, adjust=False).mean()
    macd = e12 - e26
    return macd - macd.ewm(span=9, adjust=False).mean()


def overnight_skew_20(df, s):
    on = df['open'] / df['close'].shift(1) - 1.0
    return on.rolling(20).skew()


def max_gap_20(df, s):
    gap = (df['open'] / df['close'].shift(1) - 1.0).abs()
    return gap.rolling(20).max()


def win_rate_40(df, s):
    pos = (df['close'].pct_change() > 0).astype(float)
    return pos.rolling(40).mean()


def intraday_ret_skew_20(df, s):
    intr = df['close'] / df['open'] - 1.0
    return intr.rolling(20).skew()


def amihud_trend_20_60(df, s):
    ret = df['close'].pct_change()
    vol = df['volume'].replace(0, np.nan)
    illiq = (ret.abs() / vol)
    return illiq.rolling(20).mean() / illiq.rolling(60).mean().replace(0, np.nan)


def us10y_cond_mom_20(df, s):
    u = prices['US10Y']['close'].reindex(df.index)
    mom = df['close'].shift(5) / df['close'].shift(25) - 1.0
    um = u.shift(5) / u.shift(25) - 1.0
    g = np.sign(um).reindex(df.index).fillna(0.0)
    return mom * g


def cross_vol_ratio_20_60(df, s):
    r = df['close'].pct_change()
    return r.rolling(20).std() / r.rolling(60).std().replace(0, np.nan)


def vol_asym_60(df, s):
    r = df['close'].pct_change()
    up = r.clip(lower=0).rolling(60).std()
    dn = (-r).clip(lower=0).rolling(60).std()
    return dn / up.replace(0, np.nan)


# precompute cross-sectional basket momentum (10d, skip5) once
_basket10 = None


def rel_mom_basket_10(df, s):
    global _basket10
    if _basket10 is None:
        cols = []
        for x in WATCHLIST:
            px = prices[x]['close']
            cols.append((px.shift(5) / px.shift(15) - 1.0).reindex(grid))
        _basket10 = pd.concat(cols, axis=1).mean(axis=1)
    mom = df['close'].shift(5) / df['close'].shift(15) - 1.0
    return mom - _basket10.reindex(df.index)


def amihud_level_60(df, s):
    ret = df['close'].pct_change()
    vol = df['volume'].replace(0, np.nan)
    illiq = (ret.abs() / vol)
    return illiq.rolling(60).mean()


def beta_diff_60_120(df, s):
    spx = prices['SPX']['close'].reindex(df.index)
    r = df['close'].pct_change()
    sr = spx.pct_change()
    b60 = r.rolling(60).cov(sr) / sr.rolling(60).var().replace(0, np.nan)
    b120 = r.rolling(120).cov(sr) / sr.rolling(120).var().replace(0, np.nan)
    return (b60 - b120).reindex(df.index)


def vol_mom_ratio_10_60(df, s):
    vol = df['volume'].replace(0, np.nan)
    return vol.rolling(10).mean() / vol.rolling(60).mean().replace(0, np.nan)


def intraday_ret_20(df, s):
    return (df['close'] / df['open'] - 1.0).rolling(20).sum()


def range_slope_20(df, s):
    y = ((df['high'] - df['low']) / df['close']).values
    w = 20
    i = np.arange(w) - (w - 1) / 2.0
    kern = i / (i @ i)
    s2 = np.full(len(y), np.nan)
    if len(y) >= w:
        s2[w - 1:] = np.convolve(y, kern[::-1], mode='valid')
    return pd.Series(s2, index=df.index)


def gap_dir_20(df, s):
    gap = df['open'] / df['close'].shift(1) - 1.0
    atr = pd.concat([(df['high'] - df['low']),
                     (df['high'] - df['close'].shift(1)).abs(),
                     (df['low'] - df['close'].shift(1)).abs()], axis=1).max(axis=1)
    atr = atr.rolling(14).mean().replace(0, np.nan)
    return (gap / atr).rolling(20).sum()


def rsi_3(df, s):
    c = df['close'].diff()
    up = c.clip(lower=0).ewm(alpha=1 / 3, adjust=False).mean()
    dn = (-c).clip(lower=0).ewm(alpha=1 / 3, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


candidates = {
    'vwap_dev_20': vwap_dev_20,
    'macd_hist_12_26': macd_hist_12_26,
    'overnight_skew_20': overnight_skew_20,
    'max_gap_20': max_gap_20,
    'win_rate_40': win_rate_40,
    'intraday_ret_skew_20': intraday_ret_skew_20,
    'amihud_trend_20_60': amihud_trend_20_60,
    'us10y_cond_mom_20': us10y_cond_mom_20,
    'cross_vol_ratio_20_60': cross_vol_ratio_20_60,
    'vol_asym_60': vol_asym_60,
    'rel_mom_basket_10': rel_mom_basket_10,
    'amihud_level_60': amihud_level_60,
    'beta_diff_60_120': beta_diff_60_120,
    'vol_mom_ratio_10_60': vol_mom_ratio_10_60,
    'intraday_ret_20': intraday_ret_20,
    'range_slope_20': range_slope_20,
    'gap_dir_20': gap_dir_20,
    'rsi_3': rsi_3,
}

results = {}
mats = {}
for fid, fn in candidates.items():
    try:
        panel = factor_to_panel(fn, prices)
        m = fast_validate(panel)
    except Exception as e:
        print(f"{fid}: ERROR {e}; treating as FAIL", flush=True)
        results[fid] = dict(ok=False, metrics={'ic': float('nan'), 'icir': float('nan'), 'error': str(e)})
        continue
    if m is None:
        print(f"{fid}: INSUFFICIENT DATA (panel {panel.shape})", flush=True)
        results[fid] = dict(ok=False, metrics={'ic': float('nan'), 'icir': float('nan'), 'note': 'insufficient'})
        continue
    mat = signal_matrix(panel, grid)
    mats[fid] = mat
    rho, fid_lib = max_lib_corr(mat)
    m['max_abs_library_correlation'] = rho
    m['max_corr_library_id'] = fid_lib
    ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084 and rho < 0.5
    results[fid] = dict(ok=ok, metrics=m)
    print(f"\n=== {fid} === panel {panel.shape}", flush=True)
    print(f"IC={m['ic']:.4f} ICIR={m['icir']:.4f} hit={m['ic_hit_ratio']:.3f} "
          f"n={m['n_ic_dates']} cov={m['coverage_asset_days']:.3f} "
          f"ge8={m['coverage_dates_ge8']:.3f} turn={m['turnover_10d_rank']:.2f} "
          f"maxlibrho={rho:.3f}({fid_lib})", flush=True)
    print("decay:", {k: round(v, 4) for k, v in m['decay_ic_by_horizon'].items()}, flush=True)
    print(f"ADMISSION: |IC|={abs(m['ic']):.4f} {'PASS' if abs(m['ic'])>=0.007 else 'FAIL'} | "
          f"|ICIR|={abs(m['icir']):.4f} {'PASS' if abs(m['icir'])>=0.084 else 'FAIL'} | "
          f"rho={rho:.3f} {'PASS' if rho<0.5 else 'FAIL'} -> {'PASS' if ok else 'FAIL'}", flush=True)

print("\n=== candidate-candidate mean daily cross-sectional Spearman rho ===", flush=True)
ids = list(mats.keys())
M = np.full((len(ids), len(ids)), np.nan)
ranked = {k: rank_rows(v) for k, v in mats.items()}
for i in range(len(ids)):
    for j in range(i + 1, len(ids)):
        rho = row_spearman(ranked[ids[i]], ranked[ids[j]])
        if np.isfinite(rho).any():
            M[i, j] = M[j, i] = float(np.nanmean(rho))
print("        " + " ".join(f"{i[:10]:>10}" for i in ids))
for i, idi in enumerate(ids):
    row = " ".join(f"{M[i, j]:10.2f}" if np.isfinite(M[i, j]) else f"{'-':>10}" for j in range(len(ids)))
    print(f"{idi[:10]:>8} {row}")

json.dump({k: v for k, v in results.items()},
          open('scripts/miner_3_20260730_results_round10.json', 'w'), indent=1, default=str)
print(f"\nsaved scripts/miner_3_20260730_results_round10.json; elapsed {time.time()-t0:.0f}s", flush=True)
