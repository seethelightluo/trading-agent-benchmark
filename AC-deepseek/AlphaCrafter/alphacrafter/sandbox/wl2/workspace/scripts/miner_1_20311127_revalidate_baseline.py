"""miner_1 2031-11-27: revalidate the 5 currently-effective factors through visible 2031-11-26."""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner_1_20311127_lib import (load_panel, forward_returns, compute_ic, validate_factor,
                                  build_active_library, regime_split_ic, report, VISIBLE_THROUGH)

print(f"Visible through: {VISIBLE_THROUGH}")
panel = load_panel()
print(f"Panel: {panel.shape[0]} dates x {panel.shape[1]} assets")

fwd_cache = {}
ret10 = forward_returns(panel, 10)
fwd_cache["10"] = ret10
for h in (1, 2, 3, 5, 20):
    fwd_cache[str(h)] = forward_returns(panel, h)

lib = build_active_library(panel)
print("Library signals:", list(lib.keys()))

for fid, sig in lib.items():
    m = validate_factor(sig, panel, library=lib, fwd_cache=fwd_cache)
    print(f"--- {fid} ---")
    report(fid, m)
    print("  regime:", regime_split_ic(sig, ret10))
    print("  decay:", m["decay_ic_by_horizon"])
    print("  library_pairwise:", m.get("library_pairwise_corr"))
