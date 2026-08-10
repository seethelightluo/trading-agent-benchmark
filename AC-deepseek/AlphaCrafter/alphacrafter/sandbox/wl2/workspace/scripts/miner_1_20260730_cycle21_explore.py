"""miner_1 2026-07-30 cycle 21: explore & validate novel cross-asset factors.

Context: library root is currently empty (all prior factors moved to quarantine/
rejected because their `signal_artifact` was either missing or an embedded dict,
which the post-miner gate cannot load as a matrix).  Correct persistence format:
  - factors/<fid>.json            (signal_artifact = "<fid>.signal.npy" STRING)
  - factors/<fid>.signal.npy      (float matrix, shape (n_dates_union, 15),
                                   dates ascending, columns in TRADABLES order)

This cycle explores NEW constructions not previously screened:
  1. momentum curve at untested horizons/skips (shift-based, gate-safe)
  2. damped / accelerated / long-short momentum combos (shift-based)
  3. short-horizon reversal family (1/3/5/10d)
  4. risk-scaled momentum with shift-based vol proxies
  5. cross-asset relative momentum vs XAU / NDX / WTI / US10Y
  6. trend-location and vol-regime factors (rolling, per-asset calendars;
     artifact-recoverable via .npy even if not union-panel expressible)

Validation: per-asset own-calendar factor computation (no lookahead),
reindexed to union panel; daily Spearman cross-sectional IC vs h-day forward
returns; admission h=10; gate |IC|>=0.007, |ICIR|>=0.084; coverage >=8 assets/
date; turnover; decay; regime splits; pairwise |rho| among passers (>=0.5 =>
redundant).  Reports n_dates and n_instruments explicitly.
"""
import sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner_1_lib import (TRADABLES, load_panel, per_asset, forward_returns,
                         compute_ic, validate_factor, panel_rank_corr,
                         load_library_signals, report)

panel = load_panel()
close = panel
print(f"panel shape: {panel.shape}  dates: {panel.index.min().date()}..{panel.index.max().date()}  "
      f"instruments: {len(panel.columns)}")
print(f"assets: {list(panel.columns)}")

HORIZONS = (1, 2, 3, 5, 10, 20)
ADM_H = 10
fwd_cache = {str(h): forward_returns(panel, h) for h in HORIZONS}

# ---------------------------------------------------------------------------
# Candidate construction (per-asset own calendars via per_asset -> reindex)
# ---------------------------------------------------------------------------
signals = {}

def add_shift_expr(fid, expr):
    """expr uses s = asset close series (own calendar)."""
    signals[fid] = per_asset(close, eval("lambda s: " + expr, {"abs": abs}))

def add_panel_expr(fid, expr):
    """expr uses close = union panel; evaluated directly."""
    try:
        sig = eval(expr, {"__builtins__": {}, "close": close, "pd": pd, "np": np})
        if isinstance(sig, pd.Series):
            sig = sig.to_frame()
        sig = sig.reindex(index=panel.index, columns=panel.columns)
        signals[fid] = sig
    except Exception as e:
        print(f"  {fid:28s} PANEL EVAL FAIL: {e}")

# --- 1. momentum curve, untested horizons/skips (shift-based) ---
add_shift_expr("mom15d_skip5",  "s.shift(5) / s.shift(20) - 1.0")
add_shift_expr("mom40d_skip5",  "s.shift(5) / s.shift(45) - 1.0")
add_shift_expr("mom75d_skip5",  "s.shift(5) / s.shift(80) - 1.0")
add_shift_expr("mom30d_skip10", "s.shift(10) / s.shift(40) - 1.0")
add_shift_expr("mom45d_skip10", "s.shift(10) / s.shift(55) - 1.0")
add_shift_expr("mom90d_skip10", "s.shift(10) / s.shift(100) - 1.0")

# --- 2. damped / accelerated / long-short momentum (shift-based) ---
add_shift_expr("mom20d_damp_rev10", "(s.shift(10)/s.shift(30)-1.0) - 0.5*(s/s.shift(10)-1.0)")
add_shift_expr("mom30d_damp_rev10", "(s.shift(10)/s.shift(40)-1.0) - 0.5*(s/s.shift(10)-1.0)")
add_shift_expr("mom_accel_30_10",   "(s.shift(5)/s.shift(35)-1.0) - (s.shift(5)/s.shift(15)-1.0)")
add_shift_expr("mom20_vs_120",      "(s.shift(5)/s.shift(25)-1.0) - (s.shift(5)/s.shift(125)-1.0)")
add_shift_expr("mom20_vs_60",       "(s.shift(5)/s.shift(25)-1.0) - (s.shift(5)/s.shift(65)-1.0)")
add_shift_expr("mom60_vs_180",      "(s.shift(5)/s.shift(65)-1.0) - (s.shift(5)/s.shift(185)-1.0)")

# --- 3. reversal family (shift-based) ---
add_shift_expr("rev_1d",  "-1.0 * (s / s.shift(1) - 1.0)")
add_shift_expr("rev_3d",  "-1.0 * (s / s.shift(3) - 1.0)")
add_shift_expr("rev_5d",  "-1.0 * (s / s.shift(5) - 1.0)")
add_shift_expr("rev_10d", "-1.0 * (s / s.shift(10) - 1.0)")

# --- 4. risk-scaled momentum with shift-based vol proxies ---
add_shift_expr("mom30_risk_scaled", "(s.shift(5)/s.shift(35)-1.0) / (1.0 + abs(s.shift(5)/s.shift(35)-1.0))")
add_shift_expr("mom20_volproxy60",  "(s.shift(5)/s.shift(25)-1.0) / (1.0 + abs(s.shift(5)/s.shift(65)-1.0))")
add_shift_expr("mom30_mom60_ratio", "((s.shift(5)/s.shift(35)-1.0)+1.0) / ((s.shift(5)/s.shift(65)-1.0)+1.0) - 1.0")

# --- 5. cross-asset relative momentum (shift-based) ---
def rel_mom(anchor):
    return per_asset(close, lambda s: (s.shift(5) / s.shift(25) - 1.0)
                     - (close[anchor].shift(5) / close[anchor].shift(25) - 1.0))
for a in ["XAU", "NDX", "WTI", "US10Y"]:
    signals[f"rel_mom20_{a.lower()}"] = rel_mom(a)

# --- 6. trend-location / vol-regime (rolling, per-asset calendars) ---
signals["dist_high_120d"] = per_asset(close, lambda s: s / s.rolling(120, min_periods=30).max() - 1.0)
signals["range_pos_120d"] = per_asset(close, lambda s: (s - s.rolling(120, min_periods=30).min())
                                      / (s.rolling(120, min_periods=30).max() - s.rolling(120, min_periods=30).min() + 1e-9))
signals["vol_trend_60x20"] = per_asset(close, lambda s: s.pct_change().rolling(60, min_periods=30).std()
                                       / s.pct_change().rolling(20, min_periods=10).std() - 1.0)
signals["zscore_60_rev"] = per_asset(close, lambda s: -1.0 * (s - s.rolling(60, min_periods=30).mean())
                                     / s.rolling(60, min_periods=30).std())
signals["eff_ratio_60"] = per_asset(close, lambda s: (s - s.shift(60)).abs()
                                    / s.diff().abs().rolling(60, min_periods=30).sum())

for fid in list(signals.keys()):
    sig = signals[fid].reindex(index=panel.index, columns=panel.columns)
    signals[fid] = sig
    print(f"  {fid:28s} shape={sig.shape} nan={int(sig.isna().sum().sum())} "
          f"dates_ge8={int((sig.notna().sum(axis=1)>=8).sum())}")

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
print("\n=== validation (admission h=10, per-asset calendars) ===")
library = load_library_signals(panel)   # audit provenance vs prior library family
results = {}
for fid, sig in signals.items():
    m = validate_factor(sig, panel, horizons=HORIZONS, admission_horizon=ADM_H,
                        library=library, fwd_cache=fwd_cache)
    results[fid] = m
    report(fid, m)

passers = [fid for fid, m in results.items()
           if abs(m["ic"]) >= 0.007 and abs(m["icir"]) >= 0.084]
print(f"\nPASSERS (raw gate): {passers} ({len(passers)}/{len(results)})")

# ---------------------------------------------------------------------------
# Pairwise |rho| among passers on the exact artifact matrices (union grid)
# ---------------------------------------------------------------------------
print("\n=== pairwise |rho| among PASSERS (union-grid artifact view, >=0.5 redundant) ===")
names = passers
rho = pd.DataFrame(index=names, columns=names, dtype=float)
for i, a in enumerate(names):
    for j, b in enumerate(names):
        if j <= i:
            continue
        both = pd.concat([signals[a].stack().rename("x"), signals[b].stack().rename("y")], axis=1).dropna()
        r = abs(float(both["x"].corr(both["y"]))) if len(both) > 100 else np.nan
        rho.loc[a, b] = r
        rho.loc[b, a] = r
if names:
    print("        " + "".join(f"{b[:9]:>11s}" for b in names))
    for i, a in enumerate(names):
        print(f"  {a:24s}" + "".join(f"{rho.loc[a,b]:>11.3f}" if pd.notna(rho.loc[a, b]) else f"{'-':>11s}" for b in names))

# greedy decorrelated selection: highest |IC|*|ICIR| first, admit if max|rho|<0.5
qual = {fid: abs(results[fid]["ic"]) * abs(results[fid]["icir"]) for fid in passers}
order = sorted(passers, key=lambda f: -qual[f])
selected = []
for f in order:
    if not selected:
        selected.append(f)
        continue
    mx = max((abs(rho.loc[f, s]) for s in selected if pd.notna(rho.loc[f, s])), default=0.0)
    if mx < 0.5:
        selected.append(f)
    else:
        print(f"  drop {f}: max|rho| vs selected = {mx:.3f}")
print(f"\n=== selected decorrelated set: {selected} ===")

# ---------------------------------------------------------------------------
# Regime breakdown for selected passers
# ---------------------------------------------------------------------------
print("\n=== regime breakdown for selected (IC10 / ICIR10 / n) ===")
regimes = [("2020-01-01", "2021-12-31"), ("2022-01-01", "2022-12-31"),
           ("2023-01-01", "2024-12-31"), ("2025-01-01", "2026-07-29")]
regime_out = {}
for fid in selected:
    sig = signals[fid]
    line = [fid]
    rd = {}
    for r0, r1 in regimes:
        sub_mask = (panel.index >= r0) & (panel.index <= r1)
        ic_ser = compute_ic(sig.loc[sub_mask], fwd_cache[str(ADM_H)].loc[sub_mask]).dropna()
        if len(ic_ser) >= 30:
            sd = ic_ser.std()
            icir = ic_ser.mean() / sd if sd > 0 else 0.0
            line.append(f"{r0[:4]}-{r1[:4]}: {ic_ser.mean():+.4f}/{icir:+.3f}/n={len(ic_ser)}")
            rd[r0[:4]] = {"ic": round(float(ic_ser.mean()), 4),
                          "icir": round(float(icir), 4), "n": int(len(ic_ser))}
    regime_out[fid] = rd
    print("  " + " | ".join(line))

for fid in selected:
    others = [s for s in selected if s != fid]
    mx = max((abs(rho.loc[fid, s]) for s in others if pd.notna(rho.loc[fid, s])), default=0.0)
    results[fid]["max_abs_library_correlation"] = round(mx, 4)
    results[fid]["library_pairwise_corr"] = {s: round(float(rho.loc[fid, s]), 4)
                                             for s in others if pd.notna(rho.loc[fid, s])}
    results[fid]["regime"] = regime_out[fid]

out = {"panel_shape": list(panel.shape), "visible_through": "2026-07-29",
       "n_dates": int(len(panel)), "n_instruments": int(len(panel.columns)),
       "results": {k: v for k, v in results.items()},
       "passers": passers, "selected": selected, "quality": qual,
       "pairwise_rho_passers": {a: {b: round(float(rho.loc[a, b]), 4) for b in names
                                    if pd.notna(rho.loc[a, b])} for a in names},
       "regime": regime_out}
json.dump(out, open("scripts/_miner1_cycle21_results.json", "w"), indent=1, default=float)
print("\nsaved scripts/_miner1_cycle21_results.json")
print("SELECTED_FACTORS=" + json.dumps(selected))
