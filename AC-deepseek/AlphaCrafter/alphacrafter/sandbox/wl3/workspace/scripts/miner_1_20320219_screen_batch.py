"""miner_1 2032-02-19: exploration screen - batch of NEW candidate factor ideas.

Each candidate is a distinct idea not currently in the effective library and not
in the historical .bak archive. Screen on warm-up window (2020-01-01..2026-07-15)
h=10 IC/ICIR + library correlation to prioritize deep validation.

Candidates:
  C1 ret_kurtosis_60     : kurtosis of daily returns (tail-fatness)
  C2 vol_ratio_10_60     : short vol / long vol term-structure ratio
  C3 us10y_beta_60       : rolling beta of asset return to US10Y yield change
  C4 xau_beta_60         : rolling beta of asset return to XAU return
  C5 overnight_ret_20    : cumulative overnight (open/prev close) return
  C6 intraday_ret_20     : cumulative intraday (close/open) return
  C7 downside_vol_ratio_20: downside deviation / total vol
  C8 max_dd_depth_120    : (close - rolling_max)/rolling_max drawdown depth
  C9 autocorr_20         : lag-1 autocorrelation of daily returns
  C10 btcret_beta_60     : rolling beta of asset return to BTC return
"""
import sys, time, json
import numpy as np
import pandas as pd

sys.path.insert(0, 'scripts')
from factor_common import (WATCHLIST, load_prices, factor_to_panel, validate_factor,
                           evaluate_candidate, build_library_panels)

np.seterr(all='ignore')
t0 = time.time()
prices = load_prices(days=3300)
print(f"prices: {len(prices)} assets, last {max(d.index.max() for d in prices.values()).date()} ({time.time()-t0:.1f}s)", flush=True)

# market references
spx_r = prices['SPX']['close'].pct_change()
xau_r = prices['XAU']['close'].pct_change()
btc_r = prices['BTC']['close'].pct_change()
us10y_d = prices['US10Y']['close'].diff()

def rb(r, m, w, cond=None, min_obs=0.5):
    z = pd.concat([r.rename('r'), m.rename('m')], axis=1).dropna()
    if cond is not None:
        z = z[cond.reindex(z.index).astype(bool)]
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
    r = df['close'].pct_change()
    return r.rolling(20, min_periods=12).apply(lambda x: pd.Series(x).autocorr(1) if len(x) > 2 else np.nan, raw=False)

def f_btc_beta_60(df, s):
    return rb(df['close'].pct_change(), btc_r, 60)

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
    'btc_beta_60': f_btc_beta_60,
}

library_panels = build_library_panels(prices)
print("library panels built", flush=True)

summary = {}
for fid, fn in CANDIDATES.items():
    t1 = time.time()
    m, panel = evaluate_candidate(fid, fn, prices, library_panels=library_panels, print_out=False)
    if m is None:
        print(f"{fid}: insufficient data", flush=True)
        continue
    summary[fid] = m
    ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084
    print(f"{fid:22s} ic={m['ic']:+.4f} icir={m['icir']:+.4f} hit={m['ic_hit_ratio']:.3f} "
          f"cov={m['coverage_asset_days']:.3f} ge8={m['coverage_dates_ge8']:.3f} "
          f"turn={m['turnover_10d_rank']:.3f} maxrho={m['max_abs_library_correlation']:.3f}({m['max_corr_library_id']}) "
          f"-> {'PASS' if ok else 'FAIL'} ({time.time()-t1:.1f}s)", flush=True)

with open('scripts/miner_1_20320219_screen_batch.json', 'w') as f:
    json.dump(summary, f, indent=1, default=str)
print("saved scripts/miner_1_20320219_screen_batch.json", flush=True)
