"""miner_2 exploration cycle 2026-07-30 (v3): NEW candidate families distinct from
previously tried (momentum, macro-beta cond, vol-of-vol, skew, breadth, vol-ratio).

Candidates this round:
  1. EQ_BETA_60      : rolling 60d beta of each asset to SPX returns (global equity beta)
  2. RATE_BETA_60    : rolling 60d beta of each asset to US10Y daily returns (duration beta)
  3. US10Y_BETA_COND : rate-beta conditioned on US10Y 20d trend (directional rate regime)
  4. TREND_RSQ_60    : R^2 of 60d linear trend fit (trend quality / persistence)
  5. AUTOCORR_10     : 10d lag-1 autocorrelation of daily returns (trend persistence)
  6. PARK_VS_RLZ_20  : Parkinson range vol vs realized vol ratio (gap/overnight dominance)
  7. MAXDD_60        : trailing 60d max drawdown (recent pain / risk factor)
  8. DIST_LOW_60     : distance above 60d low (reversal-from-lows)
  9. OVN_INTR_20     : intraday vs overnight return composition over 20d
 10. ACCEL_20x10     : trend acceleration = mom20 change over past 10d

All signals computed on the VISIBLE window only (<= 2026-07-29).
"""
import sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, "scripts")
from factor_validate import (closes_panel, macro_closes, forward_returns,
                             ic_series, summary_metrics, library_ic_series_map,
                             max_abs_library_corr, regime_split, load_panel)

VIS = "2026-07-29"
close = closes_panel(VIS)
macro = macro_closes(VIS)
frames = load_panel(visible_through=VIS)
ret = close.pct_change()
high = pd.DataFrame({s: df.set_index("date")["high"].astype(float) for s, df in frames.items()}).sort_index().reindex(close.index)
low = pd.DataFrame({s: df.set_index("date")["low"].astype(float) for s, df in frames.items()}).sort_index().reindex(close.index)
opn = pd.DataFrame({s: df.set_index("date")["open"].astype(float) for s, df in frames.items()}).sort_index().reindex(close.index)

def rolling_beta(asset_ret, mkt_ret, win):
    out = {}
    for a in asset_ret.columns:
        pair = pd.concat([asset_ret[a].rename("a"), mkt_ret.rename("m")], axis=1).dropna()
        b = pair["a"].rolling(win).cov(pair["m"]) / pair["m"].rolling(win).var()
        out[a] = b
    return pd.DataFrame(out).reindex(asset_ret.index)

def eval_factor(name, factor, verbose=True):
    fr = forward_returns(close, 10)
    ics = ic_series(factor, fr, min_valid=8)
    if len(ics) < 30:
        print(f"=== {name} === INSUFFICIENT IC dates: {len(ics)}")
        return None
    m = summary_metrics(ics, factor, fr, close, h=10)
    if m is None:
        print(f"=== {name} === summary INSUFFICIENT")
        return None
    lib_ics = library_ic_series_map(close)
    rho = max_abs_library_corr(ics, lib_ics)
    m["max_abs_library_correlation"] = rho
    rs = regime_split(ics)
    passed = abs(m["ic"]) >= 0.007 and abs(m["icir"] or 0) >= 0.084
    print(f"=== {name} === n_ic={len(ics)}")
    print(f"  ic={m['ic']} icir={m['icir']} hit={m['ic_hit_ratio']} cov_ad={m['coverage_asset_days']} "
          f"cov_d8={m['coverage_dates_ge8']} turn={m['turnover_10d_rank']}")
    print(f"  regimes={ {k:(v['ic'],v['icir'],v['n']) for k,v in rs.items()} }")
    print(f"  decay={m['decay_ic_by_horizon']}")
    print(f"  max_abs_lib_corr={rho}  PASS={passed}")
    return m

cands = {}

# 1. Equity beta 60d to SPX
cands["EQ_BETA_60"] = rolling_beta(ret, ret["SPX"], 60)

# 2. Rate beta 60d to US10Y returns
cands["RATE_BETA_60"] = rolling_beta(ret, ret["US10Y"], 60)

# 3. US10Y beta conditioned on 20d rate trend
rate_mom = close["US10Y"] / close["US10Y"].shift(20) - 1.0
cands["US10Y_BETA_COND_60x20"] = rolling_beta(ret, ret["US10Y"], 60).mul(rate_mom, axis=0)

# 4. Trend R^2 60d on log close
logc = np.log(close)
t = pd.Series(np.arange(len(close)), index=close.index)
r2 = logc.rolling(60).corr(t) ** 2
cands["TREND_RSQ_60"] = r2

# 5. 10d lag-1 return autocorrelation
auto = ret.rolling(11).apply(lambda x: np.corrcoef(x[:-1], x[1:])[0, 1] if len(x) == 11 and np.std(x[:-1]) > 0 and np.std(x[1:]) > 0 else np.nan, raw=True)
cands["AUTOCORR_10"] = auto

# 6. Parkinson vs realized vol ratio 20d
park = np.sqrt((np.log(high / low) ** 2).rolling(20).mean() / (4 * np.log(2)))
rv20 = ret.rolling(20).std()
cands["PARK_VS_RLZ_20"] = park / rv20

# 7. Trailing 60d max drawdown
dd = close / close.rolling(60).max() - 1.0
cands["MAXDD_60"] = dd.rolling(60).min()

# 8. Distance from 60d low
cands["DIST_LOW_60"] = close / close.rolling(60).min() - 1.0

# 9. Intraday vs overnight composition 20d (difference of means / daily vol)
ovn = opn / close.shift(1) - 1.0
intr = close / opn - 1.0
ovn_mean = ovn.rolling(20).mean()
intr_mean = intr.rolling(20).mean()
cands["OVN_INTR_20"] = (intr_mean - ovn_mean) / ret.rolling(20).std()

# 10. Trend acceleration: 20d momentum change over 10d
mom20 = close / close.shift(20) - 1.0
cands["ACCEL_20x10"] = mom20 - mom20.shift(10)

print(f"Universe: {close.shape[1]} tradable assets, {close.shape[0]} visible dates")
for name, f in cands.items():
    try:
        eval_factor(name, f.reindex(close.index))
    except Exception as e:
        print(f"=== {name} === ERROR {e}")
