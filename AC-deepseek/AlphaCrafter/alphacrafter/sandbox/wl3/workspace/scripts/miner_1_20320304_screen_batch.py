"""miner_1 2032-03-04: exploration screen - batch of NEW candidate factor ideas.

Resume-mode: skips candidates already present in the results JSON. Vectorized
autocorr (no rolling apply). Incremental save after each candidate.
"""
import sys, time, json, os
import numpy as np
import pandas as pd

sys.path.insert(0, 'scripts')
from factor_common import (WATCHLIST, load_prices, factor_to_panel, validate_factor,
                           evaluate_candidate, build_library_panels)

np.seterr(all='ignore')
RESULT_PATH = 'scripts/miner_1_20320304_screen_batch.json'

t0 = time.time()
prices = load_prices(days=3300)
print(f"prices: {len(prices)} assets, last {max(d.index.max() for d in prices.values()).date()} ({time.time()-t0:.1f}s)", flush=True)

xau_r = prices['XAU']['close'].pct_change()
us10y_d = prices['US10Y']['close'].diff()

def rb(r, m, w, min_obs=0.5):
    z = pd.concat([r.rename('r'), m.rename('m')], axis=1, sort=True).dropna()
    if len(z) < 30:
        return pd.Series(np.nan, index=r.index)
    b = z['r'].rolling(w, min_periods=int(w*min_obs)).cov(z['m']) / \
        z['m'].rolling(w, min_periods=int(w*min_obs)).var().replace(0, np.nan)
    return b.reindex(r.index)

def f_kurt_60(df, s):
    return df['close'].pct_change().rolling(60, min_periods=40).kurt()

def f_vol_ratio_10_60(df, s):
    r = df['close'].pct_change()
    v10 = r.rolling(10).std(); v60 = r.rolling(60).std()
    return (v10 / v60).replace([np.inf, -np.inf], np.nan)

def f_us10y_beta_60(df, s):
    return rb(df['close'].pct_change(), us10y_d, 60)

def f_xau_beta_60(df, s):
    return rb(df['close'].pct_change(), xau_r, 60)

def f_overnight_ret_20(df, s):
    gap = (df['open'] / df['close'].shift(1) - 1.0)
    return gap.rolling(20, min_periods=10).sum()

def f_intraday_ret_20(df, s):
    idr = (df['close'] / df['open'] - 1.0)
    return idr.rolling(20, min_periods=10).sum()

def f_downside_vol_ratio_20(df, s):
    r = df['close'].pct_change()
    neg = r.clip(upper=0)
    ddev = np.sqrt((neg ** 2).rolling(20).mean())
    tvol = r.rolling(20).std()
    return (ddev / tvol).replace([np.inf, -np.inf], np.nan)

def f_max_dd_depth_120(df, s):
    c = df['close']
    run_max = c.rolling(120, min_periods=60).max()
    return (c / run_max - 1.0)

def f_autocorr_20(df, s):
    # vectorized lag-1 autocorr: cov(r_t, r_{t-1}) / var(r)
    r = df['close'].pct_change()
    w = 20
    rl = r.shift(1)
    m = r.rolling(w, min_periods=12).mean()
    ml = rl.rolling(w, min_periods=12).mean()
    cov = (r * rl).rolling(w, min_periods=12).mean() - m * ml
    var = r.rolling(w, min_periods=12).var()
    return (cov / var).replace([np.inf, -np.inf], np.nan)

def f_on_intra_vol_ratio_20(df, s):
    gap = df['open'] / df['close'].shift(1) - 1.0
    idr = df['close'] / df['open'] - 1.0
    vg = gap.rolling(20).std(); vi = idr.rolling(20).std()
    return (vg / vi).replace([np.inf, -np.inf], np.nan)

CANDIDATES = {
    'ret_kurtosis_60': f_kurt_60,
    'vol_ratio_10_60': f_vol_ratio_10_60,
    'us10y_beta_60': f_us10y_beta_60,
    'xau_beta_60': f_xau_beta_60,
    'overnight_ret_20': f_overnight_ret_20,
    'intraday_ret_20': f_intraday_ret_20,
    'downside_vol_ratio_20': f_downside_vol_ratio_20,
    'max_dd_depth_120': f_max_dd_depth_120,
    'autocorr_20': f_autocorr_20,
    'on_intra_vol_ratio_20': f_on_intra_vol_ratio_20,
}

summary = {}
if os.path.exists(RESULT_PATH):
    summary = json.load(open(RESULT_PATH))
    print(f"resume: {len(summary)} candidates already done", flush=True)

library_panels = build_library_panels(prices)
print("library panels built", flush=True)

for fid, fn in CANDIDATES.items():
    if fid in summary:
        print(f"{fid}: skip (done)", flush=True)
        continue
    t1 = time.time()
    m, panel = evaluate_candidate(fid, fn, prices, library_panels=library_panels, print_out=False)
    if m is None:
        print(f"{fid}: insufficient data", flush=True)
        summary[fid] = {'error': 'insufficient data'}
        json.dump(summary, open(RESULT_PATH, 'w'), indent=1, default=str)
        continue
    summary[fid] = m
    ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084
    print(f"{fid:22s} ic={m['ic']:+.4f} icir={m['icir']:+.4f} hit={m['ic_hit_ratio']:.3f} "
          f"cov={m['coverage_asset_days']:.3f} ge8={m['coverage_dates_ge8']:.3f} "
          f"turn={m['turnover_10d_rank']:.3f} maxrho={m['max_abs_library_correlation']:.3f}({m['max_corr_library_id']}) "
          f"-> {'PASS' if ok else 'FAIL'} ({time.time()-t1:.1f}s)", flush=True)
    json.dump(summary, open(RESULT_PATH, 'w'), indent=1, default=str)

print("saved", RESULT_PATH, flush=True)
