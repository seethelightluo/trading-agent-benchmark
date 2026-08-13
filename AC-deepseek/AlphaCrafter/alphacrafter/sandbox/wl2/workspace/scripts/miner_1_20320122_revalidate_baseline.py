"""miner_1 2032-01-22: revalidate the full active library with fresh cutoff 2032-01-21.

Admission gates: |IC| >= 0.007, |ICIR| >= 0.084 at h=10 on the 15-instrument cross-section.
Reports regime-split IC (time buckets) and library pairwise correlation max for each factor.
"""
import sys
sys.path.insert(0, "scripts")
from miner_1_20320122_lib import (
    TRADABLES, load_panel, macro_series, per_asset, forward_returns, compute_ic,
    validate_factor, regime_split_ic, report, build_active_library, panel_rank_corr,
)
import numpy as np
import pandas as pd
import json

panel = load_panel()
print(f"panel dates: {panel.index[0].date()} -> {panel.index[-1].date()}, n={len(panel)}")

library = build_active_library(panel)
print(f"library factors: {len(library)}")

fwd_cache = {}
results = {}
print("\n=== library revalidation (h=10 admission, fresh cutoff 2032-01-21) ===")
for name, fp in library.items():
    m = validate_factor(fp, panel, library=library, fwd_cache=fwd_cache)
    results[name] = m
    passed = report(name, m)
    if passed:
        reg = regime_split_ic(fp, forward_returns(panel, 10))
        print("   regime:", reg)

print("\n=== summary table ===")
for name, m in results.items():
    print(f"{name}: ic={m['ic']:+.4f} icir={m['icir']:+.4f} hit={m['ic_hit_ratio']:.3f} "
          f"cov_asset={m['coverage_asset_days']:.3f} cov_dates={m['coverage_dates_ge8']:.3f} "
          f"turn={m['turnover_10d_rank']} maxlibcorr={m['max_abs_library_correlation']}")

with open("scripts/miner_1_20320122_revalidate_baseline.json", "w") as f:
    json.dump(results, f, indent=1, default=str)
print("\nsaved scripts/miner_1_20320122_revalidate_baseline.json")
