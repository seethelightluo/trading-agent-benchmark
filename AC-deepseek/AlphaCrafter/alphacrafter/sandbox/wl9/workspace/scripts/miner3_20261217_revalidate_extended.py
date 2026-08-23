"""miner_3 : 2026-12-17 cycle - Revalidate effective factors over extended window.
Only single-series (close/ohlc-based) expressions are re-evaluated here.
Macro/conditional factors (beta, vixreg) require extra series; those are listed
but re-evaluated with explicit formulas in a companion sweep script.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

import miner3_20260730_harness as H

H.VISIBLE_END = "2026-12-16"
H.VALID_END = "2026-12-16"

closes = H.load_closes()
print("assets loaded:", len(closes), "macro loaded:", len(H.load_macro()))
print("validated through:", H.VALID_END)

# factors whose calculation is a pure close-series expression with built-in funcs
EXPR_FACTORS = {
    "ac1_120d": "-abs(1 - s/s.rolling(120).mean())",
    "kaufman_eff_20d": "(s - (s.rolling(20).apply(lambda x: (x.iloc[-1]-x.min())/(x.max()-x.min()) if (x.max()-x.min())!=0 else np.nan))) / (0.5*(s.rolling(20).diff()).abs().mean())",
    "mom_10d_skip5": "s.shift(5)/s.shift(15) - 1.0",
    "mom_120d_skip5": "s.shift(5)/s.shift(125) - 1.0",
    "rng_pos_20d": "(s - s.rolling(20).min())/(s.rolling(20).max()-s.rolling(20).min())",
    "vol_z_20d": "(s.pct_change().rolling(20).std() - s.pct_change().rolling(120).std().rolling(20).mean())/s.pct_change().rolling(120).std().rolling(20).std()",
}

print("\n=== REVALIDATION (extended) ===\n")
summary = {}
for name, expr in FIELD_FACTORS = getattr(__import__("builtins"), "dict")().update({}) or FIELD_FACTORS0.items():
    pass