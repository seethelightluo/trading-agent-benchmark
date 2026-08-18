# -*- coding: utf-8 -*-
"""miner_1 2032-01-22: data sanity check."""
import sys
sys.path.insert(0, "scripts")
import miner_1_20320122_common as C

close = C.price_panel()
vol = C.vol_panel()
macro = C.macro_panel()
print("close shape:", close.shape, "date range:", close.index.min(), "->", close.index.max())
print("macro shape:", macro.shape, macro.columns.tolist())
print("volume columns present:", [c for c in close.columns if c in vol.columns])

# forward returns at multiple horizons
for h in [1, 5, 10, 20]:
    fwd = close.shift(-h) / close - 1.0
    fwd = fwd.shift(1)  # avoid lookahead: factor at t predicts return t+1..t+h
    print(f"fwd{h} valid obs:", fwd.notna().sum().sum())
