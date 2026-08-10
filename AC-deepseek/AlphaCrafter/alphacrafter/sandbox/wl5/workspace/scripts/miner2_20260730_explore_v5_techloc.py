"""miner_2 exploration cycle 2026-07-30 (v5): technical price-location / oscillator family.

New candidates distinct from all previously tried families
(momentum-ratio, beta-to-single-index, vol-ratio, skew, drawdown, range-pos, acorr):

  A. RSI_14            : classic 14d relative strength index (momentum oscillator)
  B. CLOSE_STRENGTH_20 : avg daily (close-low)/(high-low) over 20d (close location in day range)
  C. BOLLINGER_POS_20  : (close - MA20) / (2*std20) price z-score vs own band
  D. DIST_HIGH_250     : close / rolling_max(close,250) - 1 (distance below yearly high)
  E. MAX_RET_20        : max daily return over 20d (lottery / tail-preference)
  F. TREND_CONSIST_20  : fraction of positive 5d-block returns over last 12 blocks (win-rate trend)

All validated on the 15-asset cross-asset universe, visible window <= 2026-07-29,
horizon h=10, gates |IC|>=0.007, |ICIR|>=0.084.
"""
import sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, "scripts")
from factor_validate import (closes_panel, load_panel, forward_returns,
                             ic_series, summary_metrics, regime_split,
                             max_abs_library_corr)

VIS = "2026-07-29"
H = 10
close = closes_panel(VIS)
frames = load_panel(visible_through=VIS)
ret = close.pct_change()

high = pd.DataFrame({s: df.set_index("date")["high"].astype(float) for s, df in frames.items()}).sort_index().reindex(close.index)
low = pd.DataFrame({s: df.set_index("date")["low"].astype(float) for s, df in frames.items()}).sort_index().reindex(close.index)
opn = pd.DataFrame({s: df.set_index("date")["open"].astype(float) for s, df in frames.items()}).sort_index().reindex(close.index)

cands = {}

# A. RSI_14 (Wilder-style approximated with simple rolling means of gains/losses)
chg = close.diff()
gain = chg.clip(lower=0.0)
loss = (-chg).clip(lower=0.0)
avg_gain = gain.rolling(14, min_periods=10).mean()
avg_loss = loss.rolling(14, min_periods=10).mean()
rs = avg_gain / avg_loss.replace(0, np.nan)
cands["RSI_14"] = 100.0 - 100.0 / (1.0 + rs)

# B. CLOSE_STRENGTH_20: average intraday close location within daily range
day_range = (high - low).replace(0, np.nan)
close_loc = (close - low) / day_range
cands["CLOSE_STRENGTH_20"] = close_loc.rolling(20, min_periods=12).mean()

# C. BOLLINGER_POS_20: price z-score vs its own 20d mean/std band
ma20 = close.rolling(20, min_periods=12).mean()
sd20 = close.rolling(20, min_periods=12).std()
cands["BOLLINGER_POS_20"] = (close - ma20) / (2.0 * sd20.replace(0, np.nan))

# D. DIST_HIGH_250: distance below 1-year rolling high (negative = below high)
cands["DIST_HIGH_250"] = close / close.rolling(250, min_periods=200).max() - 1.0

# E. MAX_RET_20: max daily return over trailing 20d (lottery preference; expect negative IC)
cands["MAX_RET_20"] = ret.rolling(20, min_periods=12).max()

# F. TREND_CONSIST_20: fraction of positive 5d-block returns over last 12 blocks (12*5=60d span)
b5 = close.shift(0) / close.shift(5) - 1.0  # 5d forward-looking block returns realized at t
pos5 = (b5 > 0).astype(float)
cands["TREND_CONSIST_20"] = pos5.rolling(12, min_periods=8).mean()

fr = forward_returns(close, H)
print(f"Universe: {close.shape[1]} tradable assets | {close.shape[0]} visible dates (<= {VIS})")
results = {}
for name, f in cands.items():
    f = f.reindex(close.index)
    ics = ic_series(f, fr, min_valid=8)
    if len(ics) < 30:
        print(f"=== {name} === INSUFFICIENT IC dates: {len(ics)}")
        results[name] = {"gate_pass": False, "reason": "insufficient IC dates"}
        continue
    m = summary_metrics(ics, f, fr, close, h=H)
    if m is None:
        print(f"=== {name} === summary INSUFFICIENT")
        results[name] = {"gate_pass": False, "reason": "summary insufficient"}
        continue
    m["regime"] = regime_split(ics)
    passed = abs(m["ic"]) >= 0.007 and abs(m["icir"] or 0) >= 0.084
    results[name] = m
    print(f"=== {name} === n_ic={len(ics)}")
    print(f"  ic={m['ic']} icir={m['icir']} hit={m['ic_hit_ratio']} cov_ad={m['coverage_asset_days']} "
          f"cov_d8={m['coverage_dates_ge8']} turn={m['turnover_10d_rank']}")
    print(f"  regimes={ {k: (v['ic'], v['icir'], v['n']) for k, v in m['regime'].items()} }")
    print(f"  decay={m['decay_ic_by_horizon']}")
    print(f"  GATE={'PASS' if passed else 'fail'}")

with open("scripts/miner2_20260730_explore_v5_results.json", "w") as fh:
    json.dump({k: {kk: vv for kk, vv in v.items() if kk != "regime"} for k, v in results.items()},
              fh, indent=1, default=str)
print("saved scripts/miner2_20260730_explore_v5_results.json")
