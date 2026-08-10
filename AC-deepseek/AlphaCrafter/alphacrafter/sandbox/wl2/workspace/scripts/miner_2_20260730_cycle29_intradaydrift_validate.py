"""miner_2 cycle 29: full pre-persistence validation of intraday_drift_20.

Steps:
  1. Recompute factor exactly as in cycle28b (mean(close/open-1) over 20d).
  2. Full gate metrics at admission horizon 10d (IC, ICIR, hit, coverage, decay).
  3. Library correlation: pairwise rank corr vs ALL persisted signal artifacts
     AND specifically vs the two currently effective factors
     (mom20_volproxy60, dxy_beta_cond_60x20) - correlation gate is 0.5.
  4. Turnover debug: rank turnover at 10d steps + per-asset rank autocorr(1d/5d).
  5. Regime breakdown 2020-2021 / 2022 / 2023-2024 / 2025-2026.
"""
import sys
sys.path.insert(0, "scripts")
import json
import numpy as np
import pandas as pd
from pathlib import Path
from miner2_lib import (load_close_panel, load_ohlc_panels, per_asset,
                        validate_factor, load_library_signals, report,
                        forward_returns, compute_ic, regime_breakdown,
                        panel_rank_corr, coverage_stats)

panel = load_close_panel()
ohlc = load_ohlc_panels()
lib = load_library_signals(panel)
fwd_cache = {str(h): forward_returns(panel, h) for h in (1, 2, 3, 5, 10, 20)}

open_p, close_p = ohlc["open"], ohlc["close"]
intraday = close_p / open_p - 1.0
factor = intraday.rolling(20, min_periods=10).mean()

print("panel dates:", len(panel), "assets:", len(panel.columns))
print("factor dates:", int(factor.notna().sum(axis=1).ge(8).sum()))

m = validate_factor(factor, panel, library=lib, fwd_cache=fwd_cache)
passed = report("intraday_drift_20", m)

print("\n=== LIBRARY CORRELATION (all artifacts) ===")
pair = m["library_pairwise_corr"]
for k, v in sorted(pair.items(), key=lambda kv: -abs(kv[1])):
    print(f"  {k:24s} rho={v:+.4f}")
print("  max_abs_library_correlation =", m["max_abs_library_correlation"])

print("\n=== CORRELATION vs CURRENTLY EFFECTIVE (gate=0.5) ===")
eff = ["mom20_volproxy60", "dxy_beta_cond_60x20"]
for fid in eff:
    if fid in lib:
        rho = panel_rank_corr(factor, lib[fid])
        print(f"  {fid:24s} rho={rho:+.4f} {'OK' if abs(rho) < 0.5 else 'CONFLICT'}")

print("\n=== REGIME BREAKDOWN (10d IC series) ===")
ic_ser = compute_ic(factor, fwd_cache["10"]).dropna()
rb = regime_breakdown(ic_ser)
for k, v in rb.items():
    print(f"  {k}: ic={v['ic']:+.4f} icir={v['icir']:+.4f} n={v['n_dates']}")

print("\n=== TURNOVER DEBUG ===")
ranked = factor.rank(axis=1, pct=True)
valid = ranked.notna().sum(axis=1) >= 8
ridx = ranked.index[valid]
# rank turnover at 10d steps (matching admission horizon)
vals10 = []
for i in range(10, len(ridx), 10):
    a, b = ranked.loc[ridx[i-10]], ranked.loc[ridx[i]]
    mm = a.notna() & b.notna()
    if mm.sum() >= 8:
        vals10.append(float((b[mm] - a[mm]).abs().mean()))
print(f"  rank turnover 10d-step: mean={np.mean(vals10):.4f} n={len(vals10)}")
# per-asset autocorrelation of raw factor (1d, 5d, 10d)
ac = {}
for h in (1, 5, 10):
    acs = []
    for a in factor.columns:
        s = factor[a].dropna()
        if len(s) > 60:
            acs.append(s.autocorr(h))
    ac[h] = float(np.mean(acs))
print("  mean per-asset raw autocorr:", {f"{h}d": round(v, 3) for h, v in ac.items()})
# half-life: decay IC by horizon
print("  decay_ic_by_horizon:", m["decay_ic_by_horizon"])

print("\n=== FULL METRICS ===")
print(json.dumps({k: v for k, v in m.items() if k != "library_pairwise_corr"}, indent=2))
print("PASS_GATE=", passed)
