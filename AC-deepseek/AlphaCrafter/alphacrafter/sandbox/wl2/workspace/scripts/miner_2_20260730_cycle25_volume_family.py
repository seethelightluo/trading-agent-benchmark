"""miner_2 candidate 1 (2026-07-30): volume/liquidity factor family.

Motivation: all library factors are close-price based. Volume is present in the
daily bars but unused so far. Volume dynamics capture participation/liquidity
regimes orthogonal to price trends. Candidates:
  - vol_surge_20:  volume / SMA20(volume) - 1          (volume surge)
  - vol_trend_20_60: volume.shift(5)/volume.shift(25)-1 (20d volume momentum)
  - amihud_60:     mean(|pct_change| / volume, 60d)     (Amihud illiquidity)
Sign is left raw; the ensemble assigns direction from IC sign.
"""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner2_lib import (load_close_panel, load_volume_panel, per_asset,
                        validate_factor, load_library_signals, report,
                        forward_returns)

panel = load_close_panel()
vol = load_volume_panel()
lib = load_library_signals(panel)
fwd_cache = {str(h): forward_returns(panel, h) for h in (1, 2, 3, 5, 10, 20)}

# --- vol_surge_20 ---
f1 = per_asset(vol, lambda s: (s / s.rolling(20, min_periods=10).mean() - 1.0))
m1 = validate_factor(f1, panel, library=lib, fwd_cache=fwd_cache)
p1 = report("vol_surge_20", m1)
print("  regime:", validate_regime if False else "")

# --- vol_trend_20_60 ---
f2 = per_asset(vol, lambda s: s.shift(5) / s.shift(25) - 1.0)
m2 = validate_factor(f2, panel, library=lib, fwd_cache=fwd_cache)
p2 = report("vol_trend_20_60", m2)

# --- amihud_60 ---
f3 = per_asset(panel, lambda s: (s.pct_change().abs() / vol[s.name].reindex(s.index)).rolling(60, min_periods=30).mean())
m3 = validate_factor(f3, panel, library=lib, fwd_cache=fwd_cache)
p3 = report("amihud_60", m3)

print("\n=== SUMMARIES ===")
for name, m, p in [("vol_surge_20", m1, p1), ("vol_trend_20_60", m2, p2), ("amihud_60", m3, p3)]:
    print(name, "PASS" if p else "FAIL", "| ic", m["ic"], "icir", m["icir"],
          "| maxlibcorr", m.get("max_abs_library_correlation"),
          "| decay", m["decay_ic_by_horizon"])
