"""miner_1 2026-07-30 cycle 17: validate momentum-family candidates EXACTLY as the
post-Miner gate recovers them (restricted namespace: close + pd + np on the UNION
panel). Rolling-window stats collapse on the union calendar (crypto trades weekends,
others do not -> NaN gaps), so only shift-based expressions are gate-recoverable.

Admission gates (shared benchmark contract): |IC|>=0.0070 (h=10), |ICIR|>=0.0840,
pairwise |rho|<0.5 vs other library members, coverage with >=8 valid assets per date.

Candidates cover the momentum curve (skip 5d), damped momentum (momentum minus
short-term reversal), acceleration (20d-10d), and risk-scaled momentum.
"""
import sys, json
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner3_lib import (build_panel, forward_returns, spearman_ic,
                        mean_rank_turnover, ADMISSION_HORIZON, HORIZONS, MIN_ASSETS)

prices = build_panel()
panel = pd.DataFrame(prices)

EXPRS = {
    # --- momentum curve (skip 5d to avoid short-term reversal) ---
    "mom_10d_skip5":   "close.shift(5) / close.shift(15) - 1.0",
    "mom_20d_skip5":   "close.shift(5) / close.shift(25) - 1.0",
    "mom_30d_skip5":   "close.shift(5) / close.shift(35) - 1.0",
    "mom_60d_skip5":   "close.shift(5) / close.shift(65) - 1.0",
    "mom_90d_skip5":   "close.shift(5) / close.shift(95) - 1.0",
    "mom_120d_skip5":  "close.shift(5) / close.shift(125) - 1.0",
    "mom_180d_skip5":  "close.shift(5) / close.shift(185) - 1.0",
    "mom_250d_skip5":  "close.shift(5) / close.shift(255) - 1.0",
    # --- damped momentum: momentum minus short-term reversal ---
    "mom20d_damp_rev5": "(close.shift(5)/close.shift(25)-1.0) - 0.5*(close/close.shift(5)-1.0)",
    "mom30d_damp_rev5": "(close.shift(5)/close.shift(35)-1.0) - 0.5*(close/close.shift(5)-1.0)",
    "mom60d_damp_rev5": "(close.shift(5)/close.shift(65)-1.0) - 0.5*(close/close.shift(5)-1.0)",
    # --- momentum acceleration (20d minus 10d, both skip 5) ---
    "mom_accel_20_10": "(close.shift(5)/close.shift(25)-1.0) - (close.shift(5)/close.shift(15)-1.0)",
    # --- risk-scaled momentum (20d momentum scaled by 120d vol proxy) ---
    "mom20_risk_scaled": "(close.shift(5)/close.shift(25)-1.0) / (1.0 + abs(close.shift(5)/close.shift(125)-1.0))",
}

env = {"pd": pd, "np": np, "close": panel}
signals = {}
print("=== restricted-namespace eval (gate view: close+pd+np only) ===")
for fid, exp in EXPRS.items():
    try:
        sig = eval(exp, {"__builtins__": {}}, env)
        ok = isinstance(sig, pd.DataFrame) and sig.shape == panel.shape and sig.notna().sum().sum() > 100
        if ok:
            signals[fid] = sig
        print(f"  {fid:20s} eval={'OK' if ok else 'BAD'}")
    except Exception as e:
        print(f"  {fid:20s} eval=FAIL {type(e).__name__}: {str(e)[:60]}")

fwd10 = forward_returns(prices, ADMISSION_HORIZON)
fwd_cache = {str(h): forward_returns(prices, h) for h in HORIZONS}
print(f"\n=== validation (admission h={ADMISSION_HORIZON}, gate-view signal) ===")
rows = {}
for fid, sig in signals.items():
    ic_series = spearman_ic(sig, fwd10)
    if len(ic_series) == 0:
        print(f"  {fid:20s} NO IC DATES")
        rows[fid] = dict(ic=np.nan, icir=np.nan, hit=np.nan, n=0, cov=0.0,
                         dates_ge8=0.0, turn=np.nan, decay={}, gate=False)
        continue
    ic = float(ic_series.mean())
    icir = float(ic_series.mean() / ic_series.std()) if ic_series.std() > 0 else 0.0
    hit = float((ic_series > 0).mean()) if ic >= 0 else float((ic_series < 0).mean())
    decay = {str(h): round(float(spearman_ic(sig, fwd_cache[str(h)]).mean()), 4)
             for h in HORIZONS}
    valid = float(sig.notna().sum().sum())
    cov = valid / sig.size
    n_ge8 = sum(1 for d in sig.index if sig.loc[d].notna().sum() >= MIN_ASSETS)
    turn = mean_rank_turnover(sig)
    gate = abs(ic) >= 0.007 and abs(icir) >= 0.084
    rows[fid] = dict(ic=ic, icir=icir, hit=hit, n=len(ic_series), cov=cov,
                     dates_ge8=n_ge8 / len(sig), turn=turn, decay=decay, gate=gate)
    print(f"  {fid:20s} n={len(ic_series):5d} ic={ic:+.4f} icir={icir:+.4f} "
          f"hit={hit:.3f} cov={cov:.3f} dates_ge8={n_ge8/len(sig):.3f} turn={turn:.3f} "
          f"gate={'PASS' if gate else 'no '}")

passers = [fid for fid, r in rows.items() if r["gate"]]
print(f"\n=== passers: {passers} ===")

# pairwise pooled Pearson |rho| among passers (real signal artifacts on union panel)
rho = pd.DataFrame(index=passers, columns=passers, dtype=float)
for a in passers:
    for b in passers:
        if a == b:
            continue
        both = pd.concat([signals[a].stack().rename("x"), signals[b].stack().rename("y")], axis=1).dropna()
        rho.loc[a, b] = float(both["x"].corr(both["y"])) if len(both) > 100 else np.nan
print("=== pairwise pooled rho among passers ===")
for a in passers:
    row = "".join(f"{rho.loc[a,b]:>8.3f}" if pd.notna(rho.loc[a, b]) else "      na " for b in passers)
    print(f"  {a:20s}{row}")

# greedy decorrelated selection: highest |ICIR| first, admit if max|rho| vs selected < 0.5
order = sorted(passers, key=lambda f: abs(rows[f]["icir"]), reverse=True)
selected = []
for f in order:
    if not selected:
        selected.append(f)
        continue
    mx = max(abs(rho.loc[f, s]) for s in selected if pd.notna(rho.loc[f, s]))
    if mx < 0.5:
        selected.append(f)
    else:
        print(f"  drop {f}: max|rho| vs selected = {mx:.3f}")
print(f"\n=== selected decorrelated set: {selected} ===")

# for each selected, max_abs_library_correlation vs the other selected members
for f in selected:
    others = [s for s in selected if s != f]
    mx = max((abs(rho.loc[f, s]) for s in others if pd.notna(rho.loc[f, s])), default=0.0)
    rows[f]["max_abs_library_correlation"] = round(mx, 4)
    rows[f]["library_pairwise_corr"] = {s: round(float(rho.loc[f, s]), 4) for s in others if pd.notna(rho.loc[f, s])}

json.dump({k: v for k, v in rows.items()},
          open("scripts/_miner1_cycle17_results.json", "w"), indent=1, default=float)
print("\nsaved scripts/_miner1_cycle17_results.json")
print("SELECTED_FACTORS=" + json.dumps(selected))
