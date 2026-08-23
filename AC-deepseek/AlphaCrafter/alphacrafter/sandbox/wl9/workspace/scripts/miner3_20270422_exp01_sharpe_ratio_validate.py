"""
miner3_20270422_exp01_sharpe_ratio_validate.py
Full historical validation of Rolling Sharpe ratio (risk-adjusted momentum) factor
using the shared validation harness with persistent data (2020-2026).

Factor: sharpe_ratio_63d = rolling 63d mean return / rolling 63d std return
Motivation: Momentum normalized by volatility should provide more consistent
cross-asset signals across equities, commodities, crypto, and rates.
"""
import json, base64, io, zlib
import numpy as np
import pandas as pd
from pathlib import Path

# Import harness utilities
import sys
sys.path.insert(0, str(Path(".").resolve()))
from scripts.miner3_20260730_harness import (
    load_closes, load_macro, to_frame, forward_returns,
    rank_ic, turnover_rank10, decay_profile, library_correlation, evaluate
)

LOOKBACK = 63
HORIZON = 10
LABEL = "sharpe_ratio_63d"

def compute_sharpe_ratio(closes, lookback=63):
    """Compute rolling Sharpe ratio = mean(ret) / std(ret) for each asset."""
    vals = {}
    for a, s in closes.items():
        ret = s.pct_change()
        min_p = max(lookback // 2, 30)
        mean_ret = ret.rolling(lookback, min_periods=min_p).mean()
        std_ret = ret.rolling(lookback, min_periods=min_p).std()
        sharpe = mean_ret / std_ret.replace(0, np.nan)
        # Shift by 1 to avoid lookahead bias
        vals[a] = sharpe.shift(1)
    return vals

def main():
    print(f"=== {LABEL} Full Historical Validation ===\n")
    closes = load_closes()
    print(f"Loaded {len(closes)} assets from persistent data\n")

    # Compute factor values
    factor_vals = compute_sharpe_ratio(closes, LOOKBACK)
    print(f"Factor computed: rolling {LOOKBACK}d Sharpe ratio (shifted 1d)\n")

    # Run full evaluation pipeline
    result = evaluate(closes, factor_vals, LABEL, horizon=HORIZON, verbose=True)
    
    # Additional decay analysis at multiple horizons
    print("\n--- Decay Profile ---")
    decay = decay_profile(closes, result["frame"])
    for h_str, ic_val in decay.items():
        print(f"  Horizon {h_str:>3s}d: IC = {ic_val:.4f}")

    # Factor stability: IC over time chunks
    ic_series = result["ic_series"]
    if len(ic_series) > 0:
        # Split into early/late halves
        mid = len(ic_series) // 2
        early_ic = ic_series.iloc[:mid]
        late_ic = ic_series.iloc[mid:]
        early_mean = early_ic.mean()
        late_mean = late_ic.mean()
        early_std = early_ic.std(ddof=1)
        late_std = late_ic.std(ddof=1)
        print(f"\n--- Temporal Stability ---")
        print(f"  Early half: IC={early_mean:.4f} ICIR={early_mean/early_std:.4f} n={len(early_ic)}")
        print(f"  Late half:  IC={late_mean:.4f} ICIR={late_mean/late_std:.4f} n={len(late_ic)}")
        print(f"  Stability (early-late diff): {abs(early_mean - late_mean):.4f}")

    # Coverage summary
    frame = result["frame"]
    coverage_by_asset = frame.notna().mean().sort_values()
    print(f"\n--- Coverage by Asset ---")
    for a, c in coverage_by_asset.items():
        print(f"  {a:15s}: {c:.3f}")
    print(f"  Overall: {result['coverage_asset_days']:.3f}")

    # Check existing library factors for overlap
    print(f"\n--- Library Correlation Check ---")
    max_r, lib_detail = library_correlation(frame)
    print(f"  Max abs library correlation: {max_r:.4f}")
    for k, v in sorted(lib_detail.items(), key=lambda x: abs(x[1]), reverse=True)[:5]:
        print(f"    vs {k}: r={v:.4f}")

    # Final admission check
    ic_val = result["ic"]
    icir_val = result["icir"]
    passes_ic = abs(ic_val) >= 0.0070
    passes_icir = abs(icir_val) >= 0.0840
    print(f"\n=== ADMISSION GATE ===")
    print(f"IC (h={HORIZON}) = {ic_val:.6f}  threshold: |IC| >= 0.0070")
    print(f"ICIR (h={HORIZON}) = {icir_val:.6f}  threshold: |ICIR| >= 0.0840")
    print(f"PASS: {passes_ic and passes_icir}")

if __name__ == "__main__":
    main()