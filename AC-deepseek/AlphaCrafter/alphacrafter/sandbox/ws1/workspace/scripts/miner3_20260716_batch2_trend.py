"""miner_3 batch 2: risk-adjusted / longer-horizon trend factors + composites."""
import sys
sys.path.insert(0, "scripts")
import pandas as pd
import numpy as np
from factor_harness import get_panels, evaluate, WATCH

closes, rets, ohlc, macro = get_panels()

factors = {}
rv20 = rets.rolling(20).std()
rv60 = rets.rolling(60).std()

# Longer momentum
factors["mom_180d"] = closes.pct_change(180)
factors["mom_250d"] = closes.pct_change(250)
# Risk-adjusted momentum (trend Sharpe)
factors["ts_60d"] = (closes.pct_change(60)) / rv60
factors["ts_120d"] = (closes.pct_change(120)) / rv60
factors["ts_180d"] = (closes.pct_change(180)) / rv60
# SMA ratios
factors["sma_20_60"] = closes.rolling(20).mean() / closes.rolling(60).mean() - 1.0
factors["sma_50_200"] = closes.rolling(50).mean() / closes.rolling(200).mean() - 1.0
# Kaufman efficiency ratio (20d)
def eff_ratio(px, n=20):
    move = (px - px.shift(n)).abs()
    path = px.diff().abs().rolling(n).sum()
    return move / path
factors["eff_ratio_20d"] = eff_ratio(closes, 20)
factors["eff_ratio_60d"] = eff_ratio(closes, 60)
# RSI-ish: up-day fraction
def up_frac(px, n=20):
    d = px.diff()
    return (d > 0).rolling(n).mean()
factors["up_frac_20d"] = up_frac(closes, 20)

print("=== BATCH 2 @ h=10 ===")
results = []
for name, f in factors.items():
    f = f.reindex(closes.index)
    results.append(evaluate(f, rets, h=10, name=name, verbose=True))

print("\n=== BATCH 2 @ h=5 ===")
for name, f in factors.items():
    f = f.reindex(closes.index)
    evaluate(f, rets, h=5, name=name, verbose=True)

print("\n=== PASS GATE ===")
for r in results:
    if abs(r["mean_ic"]) >= 0.007 and abs(r["icir"]) >= 0.084:
        print(f"PASS {r['name']}: IC={r['mean_ic']:+.4f} ICIR={r['icir']:+.3f} t={r['tstat']:+.1f}")
