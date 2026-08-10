"""miner_2 exploration (2026-08-13): screen novel factor families on the 15-asset
cross-asset benchmark. Candidates computed per-asset own calendar, reindexed to master
grid. IC = daily cross-sectional Spearman vs fwd 10d return. Gates: |IC|>=0.007, |ICIR|>=0.084.
Also reports robust max |library correlation| (mean over overlapping dates, rank-aligned)."""
import sys, json, glob
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
import miner_3_20260813_lib as lib

ASSETS = lib.ASSETS
HORIZON = 10
MIN_ASSETS = lib.MIN_ASSETS
GRID = lib.GRID
N_GRID = lib.N_GRID

def load_full(sym):
    df = lib.load_asset(sym)
    return df
series_dict = {}
for s in ASSETS:
    df = lib.load_asset(s)
    if df is None or len(df) < 100:
        continue
    df = df.copy()
    df['fwd10'] = df['close'].shift(-HORIZON) / df['close'] - 1.0
    series_dict[s] = df
fwd = {s: df['fwd10'] for s, df in series_dict.items()}
FWD = lib.to_grid(fwd)

def mat_from(per_asset_series):
    return lib.to_grid(per_asset_series)

# ---------------- candidate factor constructions (own calendar) ----------------
def build(name):
    out = {}
    for s, df in series_dict.items():
        o, c, h, l, v = df['open'], df['close'], df['high'], df['low'], None
        if 'volume' in df.columns:
            v = df['volume']
        rng = (h - l).replace(0, np.nan)
        if name == 'body_ratio_20':
            out[s] = ((c - o).abs() / rng).rolling(20, min_periods=10).mean()
        elif name == 'body_signed_20':
            out[s] = ((c - o) / rng).rolling(20, min_periods=10).mean()
        elif name == 'lower_shadow_20':
            out[s] = ((o.min(c) - l) / rng).rolling(20, min_periods=10).mean()
        elif name == 'upper_shadow_20':
            out[s] = ((h - o.max(c)) / rng).rolling(20, min_periods=10).mean()
        elif name == 'gap_mean_20':
            gap = o / c.shift(1) - 1.0
            out[s] = gap.rolling(20, min_periods=10).mean()
        elif name == 'range_comp_20x120':
            rr = (h - l) / c
            out[s] = rr.rolling(20, min_periods=10).mean() / rr.rolling(120, min_periods=60).mean()
        elif name == 'kurt_60':
            r = c.pct_change()
            out[s] = r.rolling(60, min_periods=30).kurt()
        elif name == 'downside_ratio_60':
            r = c.pct_change()
            dd = np.where(r < 0, r * r, np.nan)
            dds = pd.Series(dd, index=r.index)
            down = dds.rolling(60, min_periods=30).mean().apply(lambda x: np.sqrt(x))
            tot = r.rolling(60, min_periods=30).std()
            out[s] = down / tot
        elif name == 'var_ratio_60':
            r = c.pct_change()
            s5 = r.rolling(5).sum()
            vr = (s5.rolling(60, min_periods=30).var() / (5 * r.rolling(60, min_periods=30).var())) - 1.0
            out[s] = vr
        elif name == 'sma_dist_60vol':
            sma = c.rolling(60, min_periods=30).mean()
            vol = c.pct_change().rolling(60, min_periods=30).std()
            out[s] = (c / sma - 1.0) / vol
        elif name == 'amihud_60':
            if v is None:
                out[s] = pd.Series(np.nan, index=df.index); continue
            illiq = (c.pct_change().abs() / v) * 1e9
            out[s] = illiq.rolling(60, min_periods=30).mean()
        elif name == 'obv_slope_20':
            if v is None:
                out[s] = pd.Series(np.nan, index=df.index); continue
            obv = (np.sign(c.diff()) * v).fillna(0.0).cumsum()
            obv = pd.Series(obv, index=df.index)
            slope = obv.diff(20)
            out[s] = slope / (obv.diff().rolling(20, min_periods=10).std() + 1e-9)
        elif name == 'vol_trend_20x120':
            if v is None:
                out[s] = pd.Series(np.nan, index=df.index); continue
            out[s] = v.rolling(20, min_periods=10).mean() / v.rolling(120, min_periods=60).mean()
        elif name == 'mfi_14':
            if v is None:
                out[s] = pd.Series(np.nan, index=df.index); continue
            tp = (h + l + c) / 3.0
            mf = tp * v
            pos = mf.where(tp > tp.shift(1), 0.0)
            neg = mf.where(tp < tp.shift(1), 0.0)
            psum = pos.rolling(14, min_periods=7).sum()
            nsum = neg.rolling(14, min_periods=7).sum()
            out[s] = 100.0 - 100.0 / (1.0 + psum / (nsum + 1e-9))
        else:
            raise ValueError(name)
    return out

CANDIDATES = ['body_ratio_20', 'body_signed_20', 'lower_shadow_20', 'upper_shadow_20',
              'gap_mean_20', 'range_comp_20x120', 'kurt_60', 'downside_ratio_60',
              'var_ratio_60', 'sma_dist_60vol', 'amihud_60', 'obv_slope_20',
              'vol_trend_20x120', 'mfi_14']

# library signals
lib_signals = {}
for f in sorted(glob.glob('factors/*.signal.npy')):
    try:
        arr = np.load(f, allow_pickle=True)
        if arr.ndim == 2 and arr.shape[1] == len(ASSETS):
            lib_signals[os.path.basename(f).replace('.signal.npy', '')] = arr
    except Exception:
        pass
import os

def robust_max_lib_corr(fmat):
    ours = lib.cross_sectional_rank(fmat)
    max_abs = 0.0; worst = None; pairs = {}
    for nm, arr in lib_signals.items():
        rows = min(arr.shape[0], ours.shape[0])
        rhos = []
        for t in range(rows):
            x = ours[t]; y = arr[t]
            ok = ~(np.isnan(x) | np.isnan(y))
            if ok.sum() >= MIN_ASSETS:
                xs = pd.Series(x[ok]).rank(); ys = pd.Series(y[ok]).rank()
                c = xs.corr(ys)
                if np.isfinite(c):
                    rhos.append(c)
        if rhos:
            m = float(np.mean(rhos))
            pairs[nm] = round(m, 4)
            if abs(m) > max_abs:
                max_abs = abs(m); worst = nm
    return pairs, worst, max_abs

fwd_by_h = lib.fwd_by_horizon_dict(series_dict, horizons=(1, 2, 3, 5, 10, 20))
results = {}
for name in CANDIDATES:
    try:
        fmat = mat_from(build(name))
    except Exception as e:
        print(name, 'BUILD ERROR', e); continue
    ics = lib.spearman_ic_matrix(fmat, FWD)
    if not ics:
        print(name, 'NO IC DATES'); continue
    summ = lib.summarize(ics, np.array(GRID), name, HORIZON)
    cov_ad, cov_d8 = lib.coverage_stats(fmat)
    rmat = lib.cross_sectional_rank(fmat)
    turn = lib.turnover_10d_rank(rmat)
    dec = lib.decay_curve(fmat, fwd_by_h)
    pairs, worst, maxc = robust_max_lib_corr(fmat)
    results[name] = dict(ic=summ['ic'], icir=summ['icir'], n=summ['n_ic_dates'],
                         hit=summ['hit'], cov_ad=cov_ad, cov_d8=cov_d8, turn=turn,
                         decay=dec, max_lib_corr=maxc, worst=worst)
    flag = 'PASS' if (abs(summ['ic']) >= 0.007 and abs(summ['icir']) >= 0.084) else '   '
    print(f"[{flag}] {name:22s} IC={summ['ic']:+.4f} ICIR={summ['icir']:+.3f} n={summ['n_ic_dates']:4d} "
          f"hit={summ['hit']:.3f} cov_ad={cov_ad:.3f} cov_d8={cov_d8:.3f} turn={turn:.3f} "
          f"maxLibCorr={maxc:.3f}({worst})")
    print('       decay:', {k: round(v, 4) for k, v in dec.items()})
    print('       regime:', {k: (v['ic'], v['icir'], v['n']) for k, v in summ['regime'].items()})

json.dump({k: {kk: vv for kk, vv in v.items() if kk != 'decay'} | {'decay': v['decay']}
           for k, v in results.items()}, open('scripts/miner_2_20260813_explore_results.json', 'w'), indent=1)
print('\nsaved explore results.')
