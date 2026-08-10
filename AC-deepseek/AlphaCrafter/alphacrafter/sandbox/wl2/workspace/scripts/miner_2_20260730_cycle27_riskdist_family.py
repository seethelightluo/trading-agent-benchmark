"""miner_2 cycle 27: return-distribution / downside-risk factor family.

Motivation: the effective library holds trend (mom20_volproxy60), macro-beta
(dxy_beta_cond_60x20) and volume (vol_surge_20) signals, all built on close/volume.
Daily return DISTRIBUTION shape (skewness, downside asymmetry, drawdown depth) is
untouched and captures tail-risk regimes orthogonal to those signals.

Candidates (all per-asset, computed on the asset's own calendar):
  - skew_60:        rolling 60d skewness of daily returns (return asymmetry)
  - downside_ratio_60: 60d downside deviation (below-mean) / total std (asymmetry)
  - maxdd_60:       rolling 60d drawdown depth: close/rolling_max(close,60)-1
Sign left raw; ensemble assigns direction from IC sign.
"""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner2_lib import (load_close_panel, per_asset, validate_factor,
                        load_library_signals, report, forward_returns,
                        compute_ic, regime_breakdown)

panel = load_close_panel()
lib = load_library_signals(panel)
fwd_cache = {str(h): forward_returns(panel, h) for h in (1, 2, 3, 5, 10, 20)}
ret = panel.pct_change()

def _downside_ratio(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 15:
        return np.nan
    mu = x.mean()
    dd = np.sqrt(np.mean((x[x < mu] - mu) ** 2)) if (x < mu).any() else 0.0
    sd = x.std()
    return dd / sd if sd > 0 else np.nan

def _maxdd_60(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 15 or x[-1] != x[-1]:
        return np.nan
    peak = np.max(x)
    return (x[-1] / peak - 1.0) if peak > 0 else np.nan

# --- skew_60 ---
f1 = ret.rolling(60, min_periods=30).skew()
m1 = validate_factor(f1, panel, library=lib, fwd_cache=fwd_cache)
p1 = report("skew_60", m1)

# --- downside_ratio_60 ---
f2 = ret.rolling(60, min_periods=30).apply(_downside_ratio, raw=True)
m2 = validate_factor(f2, panel, library=lib, fwd_cache=fwd_cache)
p2 = report("downside_ratio_60", m2)

# --- maxdd_60 ---
f3 = per_asset(panel, lambda s: s.rolling(60, min_periods=30).apply(_maxdd_60, raw=True))
m3 = validate_factor(f3, panel, library=lib, fwd_cache=fwd_cache)
p3 = report("maxdd_60", m3)

print("\n=== REGIME BREAKDOWNS (10d admission IC) ===")
for name, f in [("skew_60", f1), ("downside_ratio_60", f2), ("maxdd_60", f3)]:
    ic_ser = compute_ic(f, fwd_cache["10"]).dropna()
    print(name, regime_breakdown(ic_ser))

print("\n=== SUMMARIES ===")
for name, m, p in [("skew_60", m1, p1), ("downside_ratio_60", m2, p2), ("maxdd_60", m3, p3)]:
    print(name, "PASS" if p else "FAIL", "| ic", m["ic"], "icir", m["icir"],
          "| maxlibcorr", m.get("max_abs_library_correlation"),
          "| decay", m["decay_ic_by_horizon"])
