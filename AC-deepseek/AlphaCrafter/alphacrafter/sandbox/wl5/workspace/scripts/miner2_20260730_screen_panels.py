"""miner_2: Screen previously-computed signal panels (scripts/_panels) against admission gates.

For each panel: cross-sectional rank IC at h=10, ICIR, hit ratio, coverage,
turnover, regime split, and max-abs IC correlation vs the persisted library.
Only the visible window (<= 2026-07-29) is used.
"""
import sys, os, json
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from factor_validate import (closes_panel, forward_returns, ic_series,
                             summary_metrics, regime_split,
                             library_ic_series_map, max_abs_library_corr,
                             WATCH)

VIS = "2026-07-29"
H = 10
PANEL_DIR = "scripts/_panels"

close = closes_panel(VIS)
fr = forward_returns(close, H)
lib = library_ic_series_map(close, h=H)

panels = sorted(fn for fn in os.listdir(PANEL_DIR) if fn.endswith(".csv"))
rows = []
for fn in panels:
    sig = pd.read_csv(os.path.join(PANEL_DIR, fn), index_col=0, parse_dates=True)
    sig = sig.reindex(columns=close.columns).reindex(close.index)
    ic = ic_series(sig, fr, min_valid=8)
    m = summary_metrics(ic, sig, fr, close, h=H)
    if m is None:
        print(f"{fn:32s} INSUFFICIENT IC DATES ({len(ic.dropna())})")
        continue
    m["max_abs_library_correlation"] = max_abs_library_corr(ic, lib)
    m["regime"] = regime_split(ic)
    rows.append((fn, m))
    print(f"=== {fn}")
    print(json.dumps({k: m[k] for k in
                      ["ic", "icir", "ic_hit_ratio", "n_ic_dates",
                       "coverage_asset_days", "coverage_dates_ge8",
                       "turnover_10d_rank", "max_abs_library_correlation"]},
                     indent=1))
    print("  regime:", json.dumps(m["regime"]))

print("\n--- GATE CHECK (|ic|>=0.007, |icir|>=0.084) ---")
for fn, m in rows:
    gate = abs(m["ic"]) >= 0.007 and abs(m["icir"] or 0) >= 0.084
    print(f"{fn:32s} ic={m['ic']:.4f} icir={m['icir']:.4f} rho={m['max_abs_library_correlation']:.3f} -> {'PASS' if gate else 'FAIL'}")
