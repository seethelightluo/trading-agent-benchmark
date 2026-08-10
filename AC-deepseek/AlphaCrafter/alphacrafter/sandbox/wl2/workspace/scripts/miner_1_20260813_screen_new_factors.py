"""miner_1 (2026-08-13) initial screen of NEW factor families.
Per-asset own-calendar computation reindexed to master grid (matches library convention).
IC = daily cross-sectional Spearman vs fwd 10d (own-calendar) return.
Gates: |IC| >= 0.0070, |ICIR| >= 0.0840.

Candidates (all distinct from existing library):
 1. overnight_gap_20   : mean(open/prev_close - 1, 20)  (gap persistence; complements intraday_drift_20)
 2. drawdown_120       : close/rolling_max(close,120) - 1 (drawdown depth; complements range_pos_252)
 3. ret_autocorr_10    : 10d autocorr of daily returns (sign persistence / momentum vs reversal)
 4. vol_ratio_20_60    : vol20/vol60 (vol regime expansion vs contraction)
 5. up_vol_ratio_20    : volume share on up days over 20d (volume asymmetry)
 6. xau_beta_60        : 60d beta of asset rets on XAU rets (gold-sensitivity axis)
 7. ma_accel_20_60     : (MA20-MA60)/MA60 trend level minus its 20d-ago value (trend acceleration)
 8. bollinger_pos_20   : (close - MA20) / (2*std20) (z-score position)
 9. range_pos_10       : (close-min10)/(max10-min10) short-horizon range position
10. ret_skew_10        : 10d return skewness (short-horizon asymmetry; skew_60 failed, try short window)
"""
import json, glob, os
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, 'scripts')
from miner_3_20260813_lib import (ASSETS, GRID, GIDX, HORIZON, MIN_ASSETS, load_asset,
                                  asset_series, to_grid, cross_sectional_rank,
                                  spearman_ic_matrix, summarize, decay_curve,
                                  fwd_by_horizon_dict, turnover_10d_rank,
                                  library_pairwise_corr, coverage_stats, safe_div)

series = asset_series()

# ---- candidate factor builders (per-asset own calendar) ----
def f_overnight_gap(df, w=20, minp=10):
    ret_o = df['open'] / df['close'].shift(1) - 1.0
    return ret_o.rolling(w, min_periods=minp).mean()

def f_drawdown(df, w=120, minp=30):
    return df['close'] / df['close'].rolling(w, min_periods=minp).max() - 1.0

def f_autocorr(df, w=10, minp=5):
    r = df['ret']
    out = pd.Series(np.nan, index=df.index)
    for t in range(w, len(df)):
        seg = r.iloc[t - w:t]
        a, b = seg.iloc[:-1], seg.iloc[1:]
        if a.notna().sum() < minp:
            continue
        c = a.corr(b)
        out.iloc[t] = c
    return out

def f_vol_ratio(df, ws=20, wl=60, minp=10):
    vs = df['ret'].rolling(ws, min_periods=minp).std()
    vl = df['ret'].rolling(wl, min_periods=minp).std()
    return vs / vl

def f_up_vol_ratio(df, w=20, minp=10):
    r = df['ret']
    v = df['volume'].astype(float)
    up_v = (r > 0) * v
    tot = v.rolling(w, min_periods=minp).sum()
    ups = up_v.rolling(w, min_periods=minp).sum()
    return ups / tot

def f_xau_beta(df, xau_r, w=60, minp=15):
    r = df['ret']
    out = pd.Series(np.nan, index=df.index)
    xr = xau_r.reindex(df.index)
    for t in range(w, len(df)):
        a = r.iloc[t - w:t]
        b = xr.iloc[t - w:t]
        ok = a.notna() & b.notna()
        if ok.sum() < minp:
            continue
        aa, bb = a[ok], b[ok]
        vv = bb.var()
        if vv < 1e-12:
            continue
        out.iloc[t] = aa.cov(bb) / vv
    return out

def f_ma_accel(df, fs=20, fl=60, minp=10, accel=20):
    ma_f = df['close'].rolling(fs, min_periods=minp).mean()
    ma_l = df['close'].rolling(fl, min_periods=minp).mean()
    level = ma_f / ma_l - 1.0
    return level - level.shift(accel)

def f_bollinger(df, w=20, minp=10, k=2.0):
    ma = df['close'].rolling(w, min_periods=minp).mean()
    sd = df['close'].rolling(w, min_periods=minp).std()
    return (df['close'] - ma) / (k * sd)

def f_range_pos(df, w=10, minp=5):
    hi = df['close'].rolling(w, min_periods=minp).max()
    lo = df['close'].rolling(w, min_periods=minp).min()
    return (df['close'] - lo) / (hi - lo)

def f_ret_skew(df, w=10, minp=5):
    return df['ret'].rolling(w, min_periods=minp).skew()

builders = {
    'overnight_gap_20': (f_overnight_gap, {}),
    'drawdown_120': (f_drawdown, {}),
    'ret_autocorr_10': (f_autocorr, {}),
    'vol_ratio_20_60': (f_vol_ratio, {}),
    'up_vol_ratio_20': (f_up_vol_ratio, {}),
    'xau_beta_60': (f_xau_beta, {}),
    'ma_accel_20_60': (f_ma_accel, {}),
    'bollinger_pos_20': (f_bollinger, {}),
    'range_pos_10': (f_range_pos, {}),
    'ret_skew_10': (f_ret_skew, {}),
}

xau_r = series['XAU']['ret'] if 'XAU' in series else None
fwd_by_h = fwd_by_horizon_dict(series)

results = {}
for name, (fn, kw) in builders.items():
    d = {}
    for s, df in series.items():
        try:
            if name == 'xau_beta_60':
                d[s] = fn(df, xau_r, **kw)
            else:
                d[s] = fn(df, **kw)
        except Exception as e:
            d[s] = pd.Series(np.nan, index=df.index)
    mat = to_grid(d)
    ics = spearman_ic_matrix(mat, fwd_by_h[HORIZON])
    if not ics:
        print(name, 'NO IC DATES')
        continue
    summ = summarize(ics, np.array(GRID), name, HORIZON)
    cov_ad, cov_d8 = coverage_stats(mat)
    rank = cross_sectional_rank(mat)
    turn = turnover_10d_rank(rank)
    corrs, mx_name, mx_abs = library_pairwise_corr(mat)
    dec = decay_curve(mat, fwd_by_h)
    results[name] = {
        'ic': round(summ['ic'], 4), 'icir': round(summ['icir'], 3),
        'hit': round(summ['hit'], 3), 'n': summ['n_ic_dates'],
        'cov_ad': round(cov_ad, 3), 'cov_d8': round(cov_d8, 3),
        'turnover': round(turn, 3),
        'max_lib_corr': round(mx_abs, 3) if mx_name else None,
        'max_lib_corr_name': mx_name,
        'decay': dec,
        'regime': {k: v for k, v in summ['regime'].items()},
        'pass': abs(summ['ic']) >= 0.0070 and abs(summ['icir']) >= 0.0840,
    }
    print(f"{name:22s} IC {summ['ic']:+.4f} ICIR {summ['icir']:+.3f} hit {summ['hit']:.3f} n {summ['n_ic_dates']:5d} "
          f"cov {cov_ad:.2f}/{cov_d8:.2f} turn {turn:.3f} maxLibCorr {mx_abs if mx_name else None} PASS={results[name]['pass']}")

with open('scripts/miner_1_20260813_screen_results.json', 'w') as f:
    json.dump(results, f, indent=1)
print('\nSaved screen results.')
