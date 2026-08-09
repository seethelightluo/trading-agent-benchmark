"""miner_3 batch 1: basic price/volatility factors on the 15-asset cross-section."""
import sys
sys.path.insert(0, "scripts")
import pandas as pd
import numpy as np
from factor_harness import get_panels, evaluate, WATCH

closes, rets, ohlc, macro = get_panels()

factors = {}
# Momentum
factors["mom_10d"] = closes.pct_change(10)
factors["mom_20d"] = closes.pct_change(20)
factors["mom_60d"] = closes.pct_change(60)
factors["mom_120d"] = closes.pct_change(120)
# Momentum skip-1 (1..20, skip recent)
factors["mom_20d_skip1"] = closes.pct_change(21) - closes.pct_change(1)
# Short-term reversal (negative: high recent ret -> low factor)
factors["rev_5d"] = -closes.pct_change(5)
# Volatility (inverse)
rv20 = rets.rolling(20).std()
rv60 = rets.rolling(60).std()
factors["invvol_20d"] = -rv20
factors["invvol_60d"] = -rv60
# downside semideviation (lower is safer)
def semidev(r, n=20):
    return r.rolling(n).apply(lambda x: np.sqrt(np.mean(np.clip(x, -np.inf, 0) ** 2)), raw=True)
factors["inv_semidev_20d"] = -semidev(rets, 20)
# Distance from high / moving average
factors["dist_60d_high"] = closes / closes.rolling(60).max() - 1.0
factors["dist_120d_high"] = closes / closes.rolling(120).max() - 1.0
factors["dist_sma20"] = closes / closes.rolling(20).mean() - 1.0
# Vol-of-vol: change in vol
factors["vol_chg_20_60"] = rv20 / rv60 - 1.0
# Intraday position (uses SPX OHLC only as demo - skip)
# Parkinson vol (high-low based)
def parkinson(d):
    hl = (np.log(d["high"]) - np.log(d["low"])) ** 2
    return np.sqrt(hl.rolling(20).mean() / (4 * np.log(2)))
pv = pd.concat({a: parkinson(ohlc[a]) for a in WATCH}, axis=1).reindex(closes.index)
factors["inv_parkinson_20d"] = -pv

print("=== BATCH 1 @ h=10 ===")
results = []
for name, f in factors.items():
    f = f.reindex(closes.index)
    results.append(evaluate(f, rets, h=10, name=name, verbose=True))

print("\n=== SAME FACTORS @ h=5 ===")
for name, f in factors.items():
    f = f.reindex(closes.index)
    evaluate(f, rets, h=5, name=name, verbose=True)

print("\n=== PASS GATE (|IC|>=0.007 & |ICIR|>=0.084 @ h=10) ===")
for r in results:
    if abs(r["mean_ic"]) >= 0.007 and abs(r["icir"]) >= 0.084:
        print(f"PASS {r['name']}: IC={r['mean_ic']:+.4f} ICIR={r['icir']:+.3f} t={r['tstat']:+.1f}")
