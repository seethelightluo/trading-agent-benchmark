"""Factor 1: Vol-adjusted momentum (vol_adj_mom_20d_skip5)
Idea: scale 20d-skip5 momentum by inverse 20d volatility -> risk-adjusted trend
signal (signal-to-noise ratio). Novel vs library (mom10 uses raw momentum).
Admission horizon 10d. Uses data up to 2032-02-18.
"""
import numpy as np
import pandas as pd
import sys, json, base64, zlib, io

sys.path.insert(0, '.')
from scripts.factor_validation_lib import (
    ASSETS, DATA_DIR, INDEX_DIR, IC_GATE, ICIR_GATE, MIN_ASSETS_PER_DATE,
)

CURRENT_DATE = pd.Timestamp("2032-02-18")

# Monkey-patch module CURRENT_DATE
import scripts.factor_validation_lib as fvl
fvl.CURRENT_DATE = CURRENT_DATE

# Reload functions with date
def load_closes(end_date):
    closes, vols, opens, highs, lows = {}, {}, {}, {}, {}
    for a in ASSETS:
        df = pd.read_csv(f"{DATA_DIR}/{a}.csv", parse_dates=["date"])
        df = df[df["date"] <= end_date].set_index("date").sort_index()
        closes[a] = df["close"].astype(float); vols[a] = df["volume"].astype(float)
        opens[a] = df["open"].astype(float); highs[a] = df["high"].astype(float)
        lows[a] = df["low"].astype(float)
    return pd.DataFrame(closes), pd.DataFrame(vols), pd.DataFrame(opens), pd.DataFrame(highs), pd.DataFrame(lows)

def load_index(name):
    df = pd.read_csv(f"{INDEX_DIR}/{name}.csv", parse_dates=["date"])
    df = df[df["date"] <= CURRENT_DATE].set_index("date").sort_index()
    return df["close"].astype(float)

close, vol, open_, high, low = load_closes(CURRENT_DATE)
dxy = load_index("DXY"); vix = load_index("VIX"); usdcny = load_index("USDCNY")
macro = {"DXY": dxy, "VIX": vix, "USDCNY": usdcny}

dense = fvl.dense_per_asset(close, vol, open_, high, low)

# Factor definition: (c.shift(5)/c.shift(15)-1) / rolling20 std(daily ret)
def factor_fn(c, v, o, h, l, macro):
    r = c.shift(5) / c.shift(15) - 1.0
    vol20 = c.pct_change().rolling(20).std().clip(lower=1e-8)
    return r / vol20

panel = fvl.factor_panel(factor_fn, close, vol, open_, high, low, macro)

# Validate
result = fvl.validate_factor(factor_fn, close, vol, open_, high, low, macro,
                             horizons=(1, 2, 3, 5, 10, 20), admission_horizon=10)
result["ic"] = round(result["ic"], 6)
result["icir"] = round(result["icir"], 6)

# library correlation
lib = fvl.load_library_panels()
max_corr = fvl.max_library_corr(panel, lib)
result["max_abs_library_correlation"] = round(max_corr, 4)

fvl.print_result("vol_adjusted_momentum_20d_skip5", result)

ok = abs(result["ic"]) >= IC_GATE and abs(result["icir"]) >= ICIR_GATE
print(f"\nPASS GATE: {ok}")
if ok:
    # Persist
    payload = {
        "factor_id": "vol_adj_mom_20d_skip5",
        "factor_name": "Volatility-Adjusted Momentum 20d (skip 5d)",
        "version": "1.0.0",
        "calculation": {
            "expression": "(close.shift(5)/close.shift(15) - 1) / rolling_std(pct_change(close), 20)",
            "description": "20-day momentum with 5-day skip, scaled by inverse 20-day volatility. Captures the signal-to-noise ratio of a price trend so that stable trends in low-vol assets are favored, distinct from raw momentum which ignores risk."
        },
        "dependencies": ["close"],
        "parameters": {"lookback": 10, "skip": 5, "vol_window": 20},
        "expected_direction": 1,
        "validation": {
            "status": "EFFECTIVE",
            "period": "2020-01-01..2032-02-18",
            "last_validated": "2032-02-19",
            "admission_horizon": 10,
            "regime_notes": "Validated across full warm-up + online window through 2032-02-18: COVID crash, 2021-22 tightening, 2023-25 AI/crypto/commodity cycles, 2026-32 high-vol divergent regime. Rank IC on 15-asset tradable universe.",
            "metrics": result,
            "signal_artifact": {
                "format": "base64:zlib:csv",
                "description": "Factor signal panel rows=dates cols=assets",
                "shape": list(panel.shape),
                "n_valid_values": int(panel.notna().sum().sum()),
                "data": fvl.artifact_b64(panel)
            }
        },
        "tags": ["momentum", "volatility", "risk-adjusted"],
        "last_validated": "2032-02-19"
    }
    with open("factors/vol_adj_mom_20d_skip5.json", "w") as f:
        json.dump(payload, f, indent=1)
    print("PERSISTED factors/vol_adj_mom_20d_skip5.json")
