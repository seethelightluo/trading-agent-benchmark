"""miner_1 2029-03-22: data check + re-validation of effective library factors."""
import json
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, "scripts")
from miner_1_20290322_common import (
    ohlcv_panels, macro_panel, rank_ic_series, summarize_ic, decay_analysis,
    turnover_10d, coverage_stats, regime_split, library_factors,
    IC_THRESHOLD, ICIR_THRESHOLD, VISIBLE_THROUGH, CURRENT_DATE, TRADABLE, MACRO,
)

print("=" * 90)
print(f"DATA CHECK through {VISIBLE_THROUGH} (current {CURRENT_DATE})")
for s in TRADABLE:
    df = pd.read_csv(f"../persistent/stock_data/{s}.csv")
    df["date"] = pd.to_datetime(df["date"])
    sub = df[df["date"] <= VISIBLE_THROUGH]
    print(f"  {s:10s} rows={len(sub):5d} first={sub['date'].iloc[0].date()} last={sub['date'].iloc[-1].date()} "
          f"vol_nz={(sub['volume'] > 0).mean():.2f}")
for s in MACRO:
    df = pd.read_csv(f"../persistent/index_data/{s}.csv")
    df["date"] = pd.to_datetime(df["date"])
    sub = df[df["date"] <= VISIBLE_THROUGH]
    print(f"  {s:10s} (macro) rows={len(sub):5d} last={sub['date'].iloc[-1].date()}")

P = ohlcv_panels()
close = P["close"]
macro = {s: macro_panel(s) for s in MACRO}
print(f"\nClose panel: {close.shape[0]} dates x {close.shape[1]} assets "
      f"({close.index.min().date()}..{close.index.max().date()})")

fwd10 = close.shift(-10) / close - 1.0
libs = library_factors(close, macro)

print("=" * 90)
print(f"LIBRARY RE-VALIDATION through {VISIBLE_THROUGH}")
print(f"Gate: |IC|>={IC_THRESHOLD} |ICIR|>={ICIR_THRESHOLD} at 10d horizon\n")

results = {}
for name, panel in libs.items():
    ic_s = rank_ic_series(panel, fwd10)
    m = summarize_ic(ic_s, f"{name:24s}")
    reg = regime_split(ic_s)
    print("   regimes:", {k: f"IC={v['ic']:.3f}/ICIR={v['icir']:.2f}/n={v['n']}" for k, v in reg.items()})
    dec = decay_analysis(panel, close)
    print("   decay:", {k: round(v, 4) for k, v in dec.items()})
    cov = coverage_stats(panel)
    to = turnover_10d(panel)
    print(f"   coverage_asset_days={cov['coverage_asset_days']:.3f} ge8={cov['coverage_dates_ge8']:.3f} turnover10d={to:.3f}")
    passed = abs(m["ic"]) >= IC_THRESHOLD and abs(m["icir"]) >= ICIR_THRESHOLD
    print(f"   >>> {'PASS' if passed else 'FAIL'} gate\n")
    results[name] = {"metrics": m, "regimes": reg, "decay": dec,
                     "coverage": cov, "turnover": to, "passed": passed}

with open("scripts/miner_1_20290322_revalidate_results.json", "w") as f:
    json.dump(results, f, indent=1, default=str)
print("Saved scripts/miner_1_20290322_revalidate_results.json")
