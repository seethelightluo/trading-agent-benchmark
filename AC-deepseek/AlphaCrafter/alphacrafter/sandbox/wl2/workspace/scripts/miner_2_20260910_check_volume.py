"""miner_2 (2026-09-10): check volume data availability across the 15-asset universe."""
import sys, os
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner_2_20260827_lib import ASSETS, load_asset

print("assets:", ASSETS)
for s in ASSETS:
    raw = load_asset(s)
    if raw is None:
        print(f"{s:12s} NO DATA")
        continue
    vol = pd.to_numeric(raw["volume"], errors="coerce")
    n = len(raw)
    nv = int(vol.notna().sum())
    nz = int((vol > 0).sum())
    vmin = float(vol.min()) if nv else float("nan")
    vmax = float(vol.max()) if nv else float("nan")
    vmean = float(vol.mean()) if nv else float("nan")
    print(f"{s:12s} n={n:5d} vol_valid={nv:6d} vol_pos={nz:6d} vmin={vmin:.3g} vmean={vmean:.3g} vmax={vmax:.3g} last5={vol.tail(5).round(3).tolist()}")
