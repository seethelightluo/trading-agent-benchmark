"""
miner3_20270422_exp03_sharpe_library_corr.py
Detailed library correlation analysis for passing Sharpe ratio variants.
Checks which existing factors the sharpe candidates correlate with, and whether
the factor is redundant with the library.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import sys, json, base64, io, zlib
sys.path.insert(0, str(Path(".").resolve()))
from scripts.miner3_20260730_harness import (
    load_closes, to_frame, library_correlation, evaluate
)

def compute_sharpe_ratio(closes, lookback):
    vals = {}
    for a, s in closes.items():
        ret = s.pct_change()
        min_p = max(lookback // 2, 10)
        if lookback >= 60:
            min_p = max(lookback // 2, 30)
        mean_ret = ret.rolling(lookback, min_periods=min_p).mean()
        std_ret = ret.rolling(lookback, min_periods=min_p).std()
        sharpe = mean_ret / std_ret.replace(0, np.nan)
        vals[a] = sharpe.shift(1)
    return vals

def main():
    print("=== Sharpe Ratio Library Correlation Detail ===\n")
    closes = load_closes()
    
    for lb, label in [(21, "sharpe_21d"), (42, "sharpe_42d"), (126, "sharpe_126d")]:
        vals = compute_sharpe_ratio(closes, lb)
        frame = to_frame(closes, vals)
        
        print(f"\n--- {label} (lookback={lb}) ---")
        # Check library correlation in detail
        FACTOR_DIR = Path("factors")
        for f in sorted(FACTOR_DIR.glob("*.json")):
            if f.name == "factor_ensemble.json":
                continue
            try:
                d = json.load(open(f))
                art = d.get("validation", {}).get("signal_artifact")
                if not art or "data" not in art:
                    continue
                csv = zlib.decompress(base64.b64decode(art["data"])).decode()
                lib = pd.read_csv(io.StringIO(csv), index_col=0, parse_dates=True)
            except Exception as e:
                continue
            common_dates = frame.index.intersection(lib.index)
            if len(common_dates) < 120:
                continue
            a = frame.loc[common_dates]
            b = lib.loc[common_dates, frame.columns]
            pair = pd.concat([a.stack().rename("x"), b.stack().rename("y")], axis=1).dropna()
            if len(pair) < 500:
                continue
            r = float(pair["x"].corr(pair["y"]))
            if abs(r) > 0.50:
                print(f"  HIGH CORR with {f.name}: r={r:.4f}")
            elif abs(r) > 0.30:
                print(f"  MODERATE CORR with {f.name}: r={r:.4f}")
        
        max_r, detail = library_correlation(frame)
        print(f"  Max abs lib corr: {max_r:.4f}")
    
    print("\nDone.")

if __name__ == "__main__":
    main()