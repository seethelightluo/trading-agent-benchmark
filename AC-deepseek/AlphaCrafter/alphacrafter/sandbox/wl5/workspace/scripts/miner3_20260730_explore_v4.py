"""miner_3 2026-07-30: Factor exploration v4.

New candidate ideas (avoid duplicating prior work by miner_1/miner_2):
  - NDX_BETA_60      : beta to NDX returns (tech leadership exposure)
  - XAU_BETA_60      : beta to XAU returns (safe-haven sensitivity)
  - WTI_BETA_60      : beta to WTI returns (energy/inflation sensitivity)
  - USDJPY_BETA_60   : beta to USDJPY returns (carry/risk-appetite sensitivity)
  - ETH_BETA_60      : beta to ETH returns (crypto risk sensitivity)
  - MOM_REL_EQ_20    : 20d momentum minus equal-weight universe momentum (relative strength)
  - REV_5D           : 5-day short-term reversal (negative of 5d return)
  - VOL_TERM_5x60    : realized vol 5d / 60d - 1 (vol term structure slope)
  - RANGE_POS_20     : (close - low20)/(high20 - low20) position in 20d range
  - DD_60            : 60d max drawdown depth
  - KURT_20          : 20d return excess kurtosis (tail risk)
  - MOM_60_SKIP5     : 60d momentum skipping last 5 days

Validation horizon h=10, gates |IC|>=0.007, |ICIR|>=0.084 on the 15-asset
cross-asset universe. All data restricted to visible window <= 2026-07-29.
"""
import sys
import json
import numpy as np
import pandas as pd
sys.path.insert(0, "scripts")
from factor_validate import (closes_panel, macro_closes, forward_returns,
                             ic_series, summary_metrics, regime_split,
                             load_panel, WATCH)

VIS = "2026-07-29"
H = 10
close = closes_panel(VIS)
macro = macro_closes(VIS)
frames = load_panel(visible_through=VIS)
ret = close.pct_change()
high = pd.DataFrame({s: df.set_index("date")["high"].astype(float) for s, df in frames.items()}).sort_index().reindex(close.index)
low = pd.DataFrame({s: df.set_index("date")["low"].astype(float) for s, df in frames.items()}).sort_index().reindex(close.index)


def rolling_beta(asset_ret, mkt_ret, win, minp=40):
    out = {}
    for a in asset_ret.columns:
        pair = pd.concat([asset_ret[a].rename("a"), mkt_ret.rename("m")], axis=1).dropna()
        b = pair["a"].rolling(win, min_periods=minp).cov(pair["m"]) / pair["m"].rolling(win, min_periods=minp).var()
        out[a] = b
    return pd.DataFrame(out).reindex(asset_ret.index)


cands = {}

# 1. Tech-leadership beta
cands["NDX_BETA_60"] = rolling_beta(ret, ret["NDX"], 60)
# 2. Safe-haven beta
cands["XAU_BETA_60"] = rolling_beta(ret, ret["XAU"], 60)
# 3. Energy beta
cands["WTI_BETA_60"] = rolling_beta(ret, ret["WTI"], 60)
# 4. Carry beta (USDJPY)
jp = macro["USDJPY"].pct_change().reindex(close.index)
cands["USDJPY_BETA_60"] = rolling_beta(ret, jp, 60)
# 5. Crypto risk beta
cands["ETH_BETA_60"] = rolling_beta(ret, ret["ETH"], 60)

# 6. Relative strength vs equal-weight universe (20d)
mom20 = close / close.shift(20) - 1.0
eq_mom20 = mom20.mean(axis=1)
cands["MOM_REL_EQ_20"] = mom20.sub(eq_mom20, axis=0)

# 7. Short-term reversal 5d
cands["REV_5D"] = 1.0 - close / close.shift(5)

# 8. Vol term structure 5x60
rv5 = ret.rolling(5, min_periods=3).std()
rv60 = ret.rolling(60, min_periods=40).std()
cands["VOL_TERM_5x60"] = rv5 / rv60.clip(lower=1e-9) - 1.0

# 9. Position in 20d range
hi20 = close.rolling(20, min_periods=12).max()
lo20 = close.rolling(20, min_periods=12).min()
cands["RANGE_POS_20"] = (close - lo20) / (hi20 - lo20).clip(lower=1e-9)

# 10. 60d max drawdown depth
cands["DD_60"] = (close / close.rolling(60, min_periods=40).max() - 1.0).rolling(60, min_periods=40).min()

# 11. 20d excess kurtosis
cands["KURT_20"] = ret.rolling(20, min_periods=15).kurt()

# 12. 60d momentum skip 5
cands["MOM_60_SKIP5"] = close.shift(5) / close.shift(65) - 1.0

fr = forward_returns(close, H)
print(f"Universe: {close.shape[1]} tradable assets | {close.shape[0]} visible dates (<= {VIS})")
results = {}
for name, f in cands.items():
    try:
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
    except Exception as e:
        print(f"=== {name} === ERROR {e}")
        results[name] = {"gate_pass": False, "reason": str(e)}

with open("scripts/miner3_20260730_explore_v4_results.json", "w") as fh:
    json.dump({k: {kk: vv for kk, vv in v.items() if kk != "regime"} for k, v in results.items()},
              fh, indent=1, default=str)
print("saved scripts/miner3_20260730_explore_v4_results.json")
