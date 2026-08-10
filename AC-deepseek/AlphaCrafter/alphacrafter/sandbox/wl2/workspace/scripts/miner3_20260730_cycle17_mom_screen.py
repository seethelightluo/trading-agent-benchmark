"""miner_3 2026-07-30 cycle 17: momentum/trend/reversal factor screen.

Key lesson from cycle 16: the post-Miner gate recovers signals by evaluating the
persisted `calculation.expression` in a restricted namespace on the UNION panel
(only `close` + `pd` + `np`). Rolling-window statistics collapse on the union
calendar (crypto trades weekends, others do not -> NaN gaps), so only
shift-based expressions are gate-recoverable with adequate coverage.

This script therefore validates every candidate EXACTLY as the gate would
recover it: eval(expression, {'__builtins__':{}}, {'pd':pd,'np':np,'close':panel}).
Admission gates (shared benchmark contract): |IC|>=0.0070 (h=10),
|ICIR|>=0.0840, pairwise |rho|<0.5 vs other passers, coverage with >=8 valid
assets per date across a majority of dates.
"""
import sys, json
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner3_lib import (build_panel, forward_returns, spearman_ic,
                        mean_rank_turnover, ADMISSION_HORIZON, HORIZONS,
                        MIN_ASSETS)

prices = build_panel()
panel = pd.DataFrame(prices)

EXPRS = {
    # --- library re-validation (self-contained corrected forms) ---
    "mom_10d_skip5":   "close.shift(5) / close.shift(15) - 1.0",
    "mom_120d_skip5":  "close.shift(5) / close.shift(125) - 1.0",
    # --- momentum curve (skip 5d to avoid short-term reversal) ---
    "mom_20d_skip5":   "close.shift(5) / close.shift(25) - 1.0",
    "mom_30d_skip5":   "close.shift(5) / close.shift(35) - 1.0",
    "mom_60d_skip5":   "close.shift(5) / close.shift(65) - 1.0",
    "mom_90d_skip5":   "close.shift(5) / close.shift(95) - 1.0",
    "mom_180d_skip5":  "close.shift(5) / close.shift(185) - 1.0",
    "mom_250d_skip5":  "close.shift(5) / close.shift(255) - 1.0",
    # --- 12-1 momentum (skip last 20d) ---
    "mom_12_1":        "close.shift(20) / close.shift(250) - 1.0",
    # --- short-term reversal ---
    "rev_5d":          "-(close / close.shift(5) - 1.0)",
    "rev_10d":         "-(close / close.shift(10) - 1.0)",
    # --- momentum acceleration (60d vs 20d) ---
    "mom_accel_60_20": "(close.shift(5)/close.shift(65)-1.0) - (close.shift(5)/close.shift(25)-1.0)",
    # --- momentum dampened by short-term reversal (20d minus 0.5*5d) ---
    "mom20d_damp_rev5": "(close.shift(5)/close.shift(25)-1.0) - 0.5*(close/close.shift(5)-1.0)",
    # --- 30d trend slope via ratio of two shifted prices (robust to gaps) ---
    "trend_slope_30x5": "(close.shift(5)/close.shift(10)-1.0) - (close.shift(25)/close.shift(30)-1.0)",
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
        print(f"  {fid:18s} eval={'OK' if ok else 'BAD'}")
    except Exception as e:
        print(f"  {fid:18s} eval=FAIL {type(e).__name__}: {str(e)[:60]}")

fwd10 = forward_returns(prices, ADMISSION_HORIZON)
print(f"\n=== validation (admission h={ADMISSION_HORIZON}, gate view signal) ===")
rows = {}
for fid, sig in signals.items():
    ic_series = spearman_ic(sig, fwd10)
    if len(ic_series) == 0:
        print(f"  {fid:18s} NO IC DATES")
        rows[fid] = dict(ic=np.nan, icir=np.nan, hit=np.nan, n=0, cov=0.0,
                         dates_ge8=0.0, turn=np.nan, decay={}, gate=False)
        continue
    ic = float(ic_series.mean())
    icir = float(ic_series.mean() / ic_series.std()) if ic_series.std() > 0 else 0.0
    hit = float((ic_series > 0).mean()) if ic >= 0 else float((ic_series < 0).mean())
    decay = {str(h): round(float(spearman_ic(sig, forward_returns(prices, h)).mean()), 4)
             for h in HORIZONS}
    valid = float(sig.notna().sum().sum())
    cov = valid / sig.size
    n_ge8 = sum(1 for d in sig.index if sig.loc[d].notna().sum() >= MIN_ASSETS)
    turn = mean_rank_turnover(sig)
    gate = abs(ic) >= 0.007 and abs(icir) >= 0.084
    rows[fid] = dict(ic=ic, icir=icir, hit=hit, n=len(ic_series), cov=cov,
                     dates_ge8=n_ge8 / len(sig), turn=turn, decay=decay, gate=gate)
    print(f"  {fid:18s} n={len(ic_series):5d} ic={ic:+.4f} icir={icir:+.4f} "
          f"hit={hit:.3f} cov={cov:.3f} dates_ge8={n_ge8/len(sig):.3f} turn={turn:.3f} "
          f"gate={'PASS' if gate else 'no '} decay10={decay['10']:+.4f}")

# pairwise pooled rho among all eval-able candidates (gate view)
names = list(signals.keys())
rho = pd.DataFrame(index=names, columns=names, dtype=float)
for i, a in enumerate(names):
    for j, b in enumerate(names):
        if j <= i:
            continue
        both = pd.concat([signals[a].stack().rename("x"), signals[b].stack().rename("y")], axis=1).dropna()
        rho.loc[a, b] = float(both["x"].corr(both["y"])) if len(both) > 100 else np.nan
        rho.loc[b, a] = rho.loc[a, b]

print("\n=== pairwise |rho| among candidates (gate view, pooled) ===")
for i, a in enumerate(names):
    row = "".join(f"{abs(rho.loc[a,b]):>7.3f}" if j < i else "       " for j, b in enumerate(names))
    print(f"  {a:18s}{row}")

print("\n=== passers & max |rho| vs other passers ===")
passers = [fid for fid, r in rows.items() if r["gate"]]
for a in passers:
    mx = max((abs(rho.loc[a, b]) for b in passers if b != a and pd.notna(rho.loc[a, b])), default=0.0)
    print(f"  {a:18s} max_rho_vs_passers={mx:.3f}  ic={rows[a]['ic']:+.4f} icir={rows[a]['icir']:+.4f}")

out = {k: {kk: (vv if kk != "decay" else vv) for kk, vv in v.items()} for k, v in rows.items()}
json.dump(out, open("scripts/_cycle17_results.json", "w"), indent=1)
print("\nsaved scripts/_cycle17_results.json")
