"""miner_2 round-11 novel factor screen.

Tests 6 novel candidates against the full current 16-factor library:
  - r2_60_signed      : R^2 of 60d log-price linear trend, signed by 20d momentum
  - ac_5_60           : 5-day-lag return autocorrelation over 60d
  - obv_z_60          : z-score of On-Balance-Volume over 60d (money-flow level)
  - obv_slope_60      : normalized slope of OBV over 60d (money-flow trend)
  - price_accel_20_60 : momentum acceleration mom20 - mom60
  - body_ratio_20     : mean |close-open|/(high-low) over 20d (directional conviction)
  - tom_5             : turn-of-month return over last 5 trading days of month

Admission gate: |IC10| >= 0.007 and |ICIR10| >= 0.084, max |library rho| < 0.5.
"""
import sys, time
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
from pathlib import Path
from factor_common import (load_prices, factor_to_panel, validate_factor,
                           canonical_grid, signal_matrix, WATCHLIST)

t0 = time.time()
prices = load_prices(days=2000)
grid = canonical_grid(prices)
print(f'prices={len(prices)} grid={len(grid)} {grid.min().date()}..{grid.max().date()}', flush=True)

# ---- load full library signal artifacts (16 effective factors) ----
lib = {}
for p in sorted(Path('factors').glob('*_signal.npy')):
    fid = p.name.replace('_signal.npy', '')
    lib[fid] = np.load(p, allow_pickle=False)
print(f'library factors: {len(lib)}', flush=True)

def lib_max_corr(panel):
    arr = signal_matrix(panel, grid)
    best, best_id = 0.0, None
    for fid, larr in lib.items():
        corrs = []
        for i in range(arr.shape[0]):
            x, y = arr[i], larr[i]
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() >= 8:
                xr = pd.Series(x[m]).rank().values
                yr = pd.Series(y[m]).rank().values
                c = np.corrcoef(xr, yr)[0, 1]
                if np.isfinite(c):
                    corrs.append(c)
        if corrs:
            r = float(np.mean(corrs))
            if abs(r) > best:
                best, best_id = abs(r), fid
    return best, best_id

# ---------------- candidates ----------------
def f_r2_60_signed(df, s):
    logc = np.log(df['close'])
    t = np.arange(60)
    def _r2(y):
        if np.std(y) < 1e-12:
            return np.nan
        c = np.corrcoef(t, y)[0, 1]
        return c * c
    r2v = logc.rolling(60).apply(_r2, raw=True)
    mom = df['close'] / df['close'].shift(20) - 1.0
    return np.sign(mom) * r2v

def f_ac_5_60(df, s):
    r = df['close'].pct_change()
    return r.rolling(60).corr(r.shift(5))

def f_obv_z_60(df, s):
    ret = df['close'].pct_change()
    vol = df['volume']
    obv = (np.sign(ret) * vol).fillna(0.0).cumsum()
    mu = obv.rolling(60).mean()
    sd = obv.rolling(60).std()
    return (obv - mu) / sd

def f_obv_slope_60(df, s):
    ret = df['close'].pct_change()
    vol = df['volume']
    obv = (np.sign(ret) * vol).fillna(0.0).cumsum()
    w = 60
    t = np.arange(w)
    Et, Et2 = t.mean(), (t ** 2).mean()
    denom = Et2 - Et * Et
    if denom <= 0:
        return None
    Ey = obv.rolling(w).mean()
    Ety = obv.rolling(w).apply(lambda a: float((a * t).mean()), raw=True)
    slope = (Ety - Et * Ey) / denom
    scale = vol.rolling(w).mean() * ret.abs().rolling(w).mean() + 1e-12
    return slope / scale

def f_price_accel_20_60(df, s):
    mom20 = df['close'] / df['close'].shift(20) - 1.0
    mom60 = df['close'] / df['close'].shift(60) - 1.0
    return mom20 - mom60

def f_body_ratio_20(df, s):
    body = (df['close'] - df['open']).abs()
    rng = (df['high'] - df['low']).replace(0, np.nan)
    return (body / rng).rolling(20).mean()

def f_tom_5(df, s):
    idx = df.index
    r = df['close'].pct_change()
    s5 = idx.to_series().groupby([idx.year, idx.month]).tail(5).index
    out = pd.Series(np.nan, index=idx)
    for d in s5:
        pos = idx.get_loc(d)
        if pos >= 4:
            w0 = idx[pos - 4]
            out.loc[d] = df['close'].loc[d] / df['close'].loc[w0] - 1.0
    return out

candidates = {
    'r2_60_signed': f_r2_60_signed,
    'ac_5_60': f_ac_5_60,
    'obv_z_60': f_obv_z_60,
    'obv_slope_60': f_obv_slope_60,
    'price_accel_20_60': f_price_accel_20_60,
    'body_ratio_20': f_body_ratio_20,
    'tom_5': f_tom_5,
}

results = {}
for fid, fn in candidates.items():
    t1 = time.time()
    try:
        panel = factor_to_panel(fn, prices)
        if panel.shape[0] < 100:
            print(f'{fid}: insufficient panel {panel.shape} -> skip', flush=True)
            continue
        m = validate_factor(fid, panel, prices)
        if m is None:
            print(f'{fid}: insufficient data -> None', flush=True)
            continue
        rho, rid = lib_max_corr(panel)
        m['max_abs_library_correlation'] = rho
        m['max_corr_library_id'] = rid
        ok_ic = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084
        ok_corr = rho < 0.5
        print(f"{fid}: ic={m['ic']:+.4f} icir={m['icir']:+.4f} hit={m['ic_hit_ratio']:.3f} "
              f"n={m['n_ic_dates']} cov={m['coverage_asset_days']:.3f} ge8={m['coverage_dates_ge8']:.3f} "
              f"turn={m['turnover_10d_rank']:.2f} rho={rho:.3f}({rid}) "
              f"d1={m['decay_ic_by_horizon']['1']:+.4f} d5={m['decay_ic_by_horizon']['5']:+.4f} "
              f"d10={m['decay_ic_by_horizon']['10']:+.4f} d20={m['decay_ic_by_horizon']['20']:+.4f} "
              f"-> {'PASS' if (ok_ic and ok_corr) else 'skip'} [{time.time()-t1:.1f}s]", flush=True)
        results[fid] = (m, panel)
    except Exception as e:
        print(f'{fid}: ERROR {type(e).__name__}: {e}', flush=True)

print(f'\nTOTAL {time.time()-t0:.1f}s')
print('SUMMARY:')
for fid, (m, _) in sorted(results.items()):
    ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084 and m['max_abs_library_correlation'] < 0.5
    print(f"  {fid:22s} ic={m['ic']:+.4f} icir={m['icir']:+.4f} rho={m['max_abs_library_correlation']:.3f} -> {'ADMIT' if ok else 'skip'}")
