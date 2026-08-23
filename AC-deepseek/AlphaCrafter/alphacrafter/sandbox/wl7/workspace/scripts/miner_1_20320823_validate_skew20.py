"""miner_1 validation of candidate skew_20d_skip5 at 2032-08-23 (visible through 2032-08-20).

Candidate: skewness of daily returns over 20d window, 5d skip.
Motivation: return-distribution tail asymmetry (positive skew = right-tail upside days)
as a cross-asset quality/risk factor distinct from kurtosis (which measures tail weight
regardless of direction) already in the library.

Admission gates (benchmark-wide, 15-instrument universe):
  abs(daily paper IC)  >= 0.0070
  abs(daily paper ICIR) >= 0.0840
Validation horizon h=10.
"""
import sys, json
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner_shared import (ASSETS, load_close, load_macro, forward_ret, daily_ic, ic_stats,
                          rank_turnover, coverage_stats, library_panel, max_lib_corr,
                          IC_GATE, ICIR_GATE)

END = "2032-08-20"
close = load_close(END)
macro = load_macro(END)
ret = close.pct_change()

# Candidate factor: skewness of daily returns, 20d window, 5d skip
factor = ret.shift(5).rolling(20, min_periods=12).skew()
factor = factor.replace([np.inf, -np.inf], np.nan)

print(f"END={END}  n_dates={len(close)}  n_assets={close.shape[1]}")

# Full-window validation h=10
fwd10 = forward_ret(close, 10)
ic10 = daily_ic(factor, fwd10)
st = ic_stats(ic10, 10)
cov = coverage_stats(factor, fwd10)
turn = rank_turnover(factor, 10)
print(f"\nFULL-WINDOW h=10: IC={st['ic']:+.4f} ICIR={st['icir']:+.3f} hit={st['hit']:.3f} n={st['n']}")
print(f"coverage_asset_days={cov['coverage_asset_days']:.3f} coverage_dates_ge8={cov['coverage_dates_ge8']:.3f} turnover_10d_rank={turn:.2f}")

# Decay across horizons
print("\nDecay (IC by horizon):")
decay = {}
for h in (1, 2, 3, 5, 10, 20):
    fwd = forward_ret(close, h)
    ic = daily_ic(factor, fwd)
    s = ic_stats(ic, h)
    decay[str(h)] = round(s["ic"], 4)
    print(f"  h={h:2d}  IC={s['ic']:+.4f}  ICIR={s['icir']:+.3f}  n={s['n']}")

# Per-year stability
print("\nPer-year h=10 IC (direction-adjusted raw):")
per_year = {}
ic_ser = ic10.dropna()
for yr in range(2020, 2033):
    sub = ic_ser[ic_ser.index.year == yr]
    if len(sub) == 0:
        continue
    s = ic_stats(sub, 10)
    per_year[str(yr)] = dict(ic=round(s["ic"], 4), icir=round(s["icir"], 3), n=int(s["n"]))
    print(f"  {yr}: IC={s['ic']:+.4f} ICIR={s['icir']:+.3f} hit={s['hit']:.3f} n={s['n']}")

# Recent stability (1y / 2y)
recent = ic_ser[ic_ser.index >= (ic_ser.index.max() - np.timedelta64(365, "D"))]
two_y = ic_ser[ic_ser.index >= (ic_ser.index.max() - np.timedelta64(730, "D"))]
print(f"\nRECENT 1y: IC={recent.mean():+.4f} ICIR={recent.mean()/recent.std(ddof=1) if len(recent) > 2 else np.nan:+.3f} n={len(recent)}")
print(f"RECENT 2y: IC={two_y.mean():+.4f} ICIR={two_y.mean()/two_y.std(ddof=1) if len(two_y) > 2 else np.nan:+.3f} n={len(two_y)}")

# Library correlation provenance (real signal artifacts, flattened pairwise)
lib_panels = library_panel(close, macro)
best, pairs = max_lib_corr(factor, lib_panels)
print(f"\nmax_abs_library_correlation={best:.4f}")
print("library_pairwise_corr:", json.dumps(pairs, indent=1))

gate_pass = abs(st["ic"]) >= IC_GATE and abs(st["icir"]) >= ICIR_GATE
print(f"\nGATE (abs IC>=%.4f, abs ICIR>=%.4f): %s" % (IC_GATE, ICIR_GATE, "PASS" if gate_pass else "FAIL"))

# Save validation record
rec = dict(
    factor_id="skew_20d_skip5",
    end=END,
    horizon=10,
    ic=st["ic"], icir=st["icir"], hit=st["hit"], n_ic_dates=st["n"],
    coverage_asset_days=cov["coverage_asset_days"],
    coverage_dates_ge8=cov["coverage_dates_ge8"],
    turnover_10d_rank=turn,
    decay_ic_by_horizon=decay,
    per_year=per_year,
    recent_1y=dict(ic=float(recent.mean()), n=int(len(recent))),
    recent_2y=dict(ic=float(two_y.mean()), n=int(len(two_y))),
    max_abs_library_correlation=best,
    library_pairwise_corr=pairs,
    gate=dict(ic_threshold=IC_GATE, icir_threshold=ICIR_GATE, passed=gate_pass),
)
with open("scripts/miner1_20320823_skew20_validation.json", "w") as fo:
    json.dump(rec, fo, indent=1, default=str)
print("\nsaved scripts/miner1_20320823_skew20_validation.json")
