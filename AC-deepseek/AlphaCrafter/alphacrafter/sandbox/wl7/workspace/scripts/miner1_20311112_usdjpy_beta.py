"""miner_1 exploration: USDJPY conditional beta factor (2031-11-12, thru 2031-11-11).
Candidate analog of dxy_beta_cond/eurusd_beta_cond: asset's 60d beta to USDJPY
returns scaled by USDJPY 20d momentum. Insight: yen as global risk-appetite funding
currency; when JPY weakens (USDJPY up) high-beta-to-Yen assets tend to outperform.
Validate against the benchmark-wide gate (abs IC>=0.0070, abs ICIR>=0.0840, h=10)."""
import sys, json
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner_shared import (load_close, load_macro, forward_ret, daily_ic, ic_stats,
                          rank_turnover, coverage_stats, library_panel, summarize,
                          max_lib_corr, ACTIVE_LIB)

END = "2031-11-11"
close = load_close(END)
macro = load_macro(END)
fwd10 = forward_ret(close, 10)

def usdjpy_beta_cond(close, usdjpy, beta_win=60, cond_win=20, min_periods=30):
    ret = close.pct_change()
    fx_r = usdjpy.pct_change()
    cov = ret.rolling(beta_win, min_periods=min_periods).cov(fx_r)
    var = fx_r.rolling(beta_win, min_periods=min_periods).var()
    beta = cov.divide(var, axis=0)
    fx_mom = usdjpy / usdjpy.shift(cond_win) - 1.0
    return beta.multiply(fx_mom, axis=0)

f = usdjpy_beta_cond(close, macro["USDJPY"])
print(f"END={END} n_dates={len(close)} n_assets={close.shape[1]}")

st = ic_stats(daily_ic(f, fwd10), 10)
st_n = ic_stats(daily_ic(-f, fwd10), 10)
cov = coverage_stats(f, fwd10)
turn = rank_turnover(f, 10)
print(f"\n[USDJPY_beta_cond 60x20]  signature given above (+beta*mom)")
print(f"  +sign: IC={st['ic']:+.4f} ICIR={st['icir']:+.3f} hit={st['hit']:.2f} n={st['n']}")
print(f"  -sign: IC={st_n['ic']:+.4f} ICIR={st_n['icir']:+.3f} hit={st_n['hit']:.2f} n={st_n['n']}")
print(f"  coverage_asset_days={cov['coverage_asset_days']:.3f} dates_ge8={cov['coverage_dates_ge8']:.3f}")
print(f"  rank_turnover_10d={turn:.3f}")

# gate on best sign
best_st, best_dir = (st, "+") if abs(st["ic"]) > abs(st_n["ic"]) else (st_n, "-")
gate = abs(best_st["ic"]) >= 0.0070 and abs(best_st["icir"]) >= 0.0840
print(f"\n  GATE (best dir {best_dir}): IC={best_st['ic']:+.4f} ICIR={best_st['icir']:+.3f} -> {'PASS' if gate else 'FAIL'}")

# decay by horizon (use chosen direction)
f_use = f if best_dir == "+" else -f
print("\n  Decay (IC by horizon):")
for h, d in summarize(f_use, close, horizons=(1,2,3,5,10,20)).items():
    print(f"    h={h:>2d}: IC={d['ic']:+.4f} ICIR={d['icir']:+.3f} n={d['n']}")

# per-year
ic = daily_ic(f_use, fwd10)
print("\n  Per-year h10 IC:")
for yr in range(2029, 2033):
    sub = ic.loc[ic.index.year == yr]
    if len(sub) == 0:
        continue
    d = ic_stats(sub, 10)
    print(f"    {yr}: IC={d['ic']:+.3f} ICIR={d['icir']:+.2f} n={d['n']}")

# max abs library correlation (audit provenance)
lib_panels = library_panel(close, macro)
best_rho, pairs = max_lib_corr(f_use, lib_panels)
print(f"\n  max_abs_library_correlation={best_rho:.4f}")
for k, v in sorted(pairs.items(), key=lambda x: -abs(x[1])):
    print(f"    {k}: {v:+.4f}")