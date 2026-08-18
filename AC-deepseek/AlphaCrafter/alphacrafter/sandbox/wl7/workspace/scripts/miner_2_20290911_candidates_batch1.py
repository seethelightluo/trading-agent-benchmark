"""miner_2 2029-09-11: validate candidate factors (batch 1).

Candidates (all price-based, distinct from active library / evicted set):
  A. skew_20d          - rolling skewness of 20d daily returns (tail asymmetry)
  B. drawdown_60d      - close / rolling_max(close,60) - 1 (distance from high)
  C. updown_ratio_20d  - mean(gain)/|mean(loss)| over 20d (asymmetry)
  D. range_ratio_20d   - mean((high-low)/close) over 20d (intraday range vol)

Admission gates: abs(IC) >= 0.0070, abs(ICIR) >= 0.0840 (15-instrument universe,
10d primary horizon, daily paper IC).
"""
import sys
sys.path.insert(0, "scripts")
import json
import numpy as np
import pandas as pd
from miner_shared import (ASSETS, load_close, load_macro, forward_ret, daily_ic,
                          ic_stats, coverage_stats, library_panel, max_lib_corr,
                          IC_GATE, ICIR_GATE)

END = "2029-09-10"
close = load_close(END)
macro = load_macro(END)
ret = close.pct_change()
print(f"close panel: {close.shape}, dates {close.index[0].date()}..{close.index[-1].date()}")

# ---- candidate factor panels ----
panels = {}

# A. skew_20d
skew = ret.rolling(20, min_periods=14).skew()
panels["skew_20d"] = skew

# B. drawdown_60d
dd = close / close.rolling(60, min_periods=30).max() - 1.0
panels["drawdown_60d"] = dd

# C. updown_ratio_20d
g = ret.where(ret > 0, 0.0)
l = ret.where(ret < 0, 0.0)
up = g.rolling(20, min_periods=14).mean()
dn = l.rolling(20, min_periods=14).mean().abs()
panels["updown_ratio_20d"] = up / dn

# D. range_ratio_20d: needs high/low on master calendar
def load_hl(name):
    df = pd.read_csv(f"../persistent/stock_data/{name}.csv")
    df["date"] = pd.to_datetime(df["date"])
    s = df.set_index("date").reindex(close.index)
    return s

rr_parts = []
for a in ASSETS:
    s = load_hl(a)
    rng = (s["high"] - s["low"]) / s["close"]
    rr_parts.append(rng.rename(a))
rng_panel = pd.concat(rr_parts, axis=1)
panels["range_ratio_20d"] = rng_panel.rolling(20, min_periods=14).mean()

lib = library_panel(close, macro)

print("\n=== CANDIDATE VALIDATION (primary horizon 10) ===")
results = {}
for name, f in panels.items():
    fwd = forward_ret(close, 10)
    ic = daily_ic(f, fwd)
    st = ic_stats(ic, 10)
    cov = coverage_stats(f, fwd)
    rho, pairs = max_lib_corr(f, lib)
    results[name] = dict(st=st, cov=cov, rho=rho, pairs=pairs)
    ok = (abs(st["ic"]) >= IC_GATE) and (abs(st["icir"]) >= ICIR_GATE)
    print(f"\n[{name}] status={'PASS' if ok else 'fail'}")
    print(f"  IC={st['ic']:.4f} ICIR={st['icir']:.4f} hit={st['hit']:.3f} n_dates={st['n']}")
    print(f"  coverage_asset_days={cov['coverage_asset_days']:.3f} dates_ge8={cov['coverage_dates_ge8']:.3f}")
    print(f"  max_abs_lib_corr={rho:.3f} pairs={pairs}")

# decay across horizons for each
print("\n=== DECAY (IC by horizon) ===")
for name, f in panels.items():
    row = []
    for h in (1, 2, 3, 5, 10, 20):
        ic = daily_ic(f, forward_ret(close, h))
        st = ic_stats(ic, h)
        row.append(f"h{h}:{st['ic']:.4f}/{st['icir']:.3f}")
    print(f"{name:18s} " + "  ".join(row))
