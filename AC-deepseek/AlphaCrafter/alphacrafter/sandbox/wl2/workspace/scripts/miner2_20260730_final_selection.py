"""miner_2 2026-07-30: combined correlation gate & final admission selection.

Combine cycle-17 momentum passers and cycle-18 vol/trend/corr passers, compute
pairwise pooled |rho| on the gate-view signals (eval in restricted namespace),
then greedily admit factors in descending |icir| (stability) subject to
max pairwise |rho| < 0.5 vs already-admitted factors.

Also recompute, for every admitted factor, max_abs_library_correlation vs the
other admitted factors (the gate's provenance/audit field) from real signal
artifacts.
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
env = {"pd": pd, "np": np, "close": panel}

CANDIDATES = {
    # --- cycle-17 momentum passers ---
    "mom_10d_skip5":    "close.shift(5) / close.shift(15) - 1.0",
    "mom_20d_skip5":    "close.shift(5) / close.shift(25) - 1.0",
    "mom_30d_skip5":    "close.shift(5) / close.shift(35) - 1.0",
    "mom_180d_skip5":   "close.shift(5) / close.shift(185) - 1.0",
    "mom_250d_skip5":   "close.shift(5) / close.shift(255) - 1.0",
    "mom20d_damp_rev5": "(close.shift(5)/close.shift(25)-1.0) - 0.5*(close/close.shift(5)-1.0)",
    # --- cycle-18 passers ---
    "vol60":            "close.pct_change().rolling(60, min_periods=15).std()",
    "vol_of_vol20x60":  "close.pct_change().rolling(20, min_periods=5).std().rolling(60, min_periods=15).std()",
    "range_pos_252":    "(close - close.rolling(252, min_periods=30).min()) / (close.rolling(252, min_periods=30).max() - close.rolling(252, min_periods=30).min())",
    "spx_corr60":       "close.pct_change().rolling(60, min_periods=15).corr(close['SPX'].pct_change())",
}

fwd10 = forward_returns(prices, ADMISSION_HORIZON)
signals = {}
stats = {}
for fid, exp in CANDIDATES.items():
    sig = eval(exp, {"__builtins__": {}}, env)
    signals[fid] = sig
    ic_series = spearman_ic(sig, fwd10)
    ic = float(ic_series.mean())
    icir = float(ic_series.mean() / ic_series.std())
    hit = float((ic_series > 0).mean()) if ic >= 0 else float((ic_series < 0).mean())
    decay = {str(h): round(float(spearman_ic(sig, forward_returns(prices, h)).mean()), 4)
             for h in HORIZONS}
    cov = float(sig.notna().sum().sum()) / sig.size
    n_ge8 = sum(1 for d in sig.index if sig.loc[d].notna().sum() >= MIN_ASSETS)
    turn = mean_rank_turnover(sig)
    stats[fid] = dict(ic=ic, icir=icir, hit=hit, n=len(ic_series), cov=cov,
                      dates_ge8=n_ge8 / len(sig), turn=turn, decay=decay,
                      quality=abs(ic) * abs(icir))

names = list(signals.keys())
rho = pd.DataFrame(index=names, columns=names, dtype=float)
for i, a in enumerate(names):
    for j, b in enumerate(names):
        if j <= i:
            continue
        both = pd.concat([signals[a].stack().rename("x"), signals[b].stack().rename("y")], axis=1).dropna()
        r = float(both["x"].corr(both["y"])) if len(both) > 100 else np.nan
        rho.loc[a, b] = r
        rho.loc[b, a] = r

print("=== candidate stats (gate view) ===")
for fid in names:
    s = stats[fid]
    print(f"  {fid:18s} ic={s['ic']:+.4f} icir={s['icir']:+.4f} "
          f"cov={s['cov']:.3f} dates_ge8={s['dates_ge8']:.3f} turn={s['turn']:.3f} "
          f"quality={s['quality']:.5f}")

print("\n=== pairwise |rho| (pooled, gate view) ===")
for i, a in enumerate(names):
    row = "".join(f"{abs(rho.loc[a,b]):>7.3f}" if j < i else "       " for j, b in enumerate(names))
    print(f"  {a:18s}{row}")

print("\n=== greedy admission (desc quality, max rho < 0.5 vs admitted) ===")
order = sorted(names, key=lambda f: -stats[f]["quality"])
admitted = []
for a in order:
    mx = max((abs(rho.loc[a, b]) for b in admitted), default=0.0)
    ok = mx < 0.5
    print(f"  {a:18s} quality={stats[a]['quality']:.5f} max_rho_vs_admitted={mx:.3f} -> {'ADMIT' if ok else 'reject'}")
    if ok:
        admitted.append(a)

print("\n=== final admission set ===")
out = {}
for a in admitted:
    others = [b for b in admitted if b != a]
    mx = max((abs(rho.loc[a, b]) for b in others), default=0.0)
    out[a] = dict(
        expression=CANDIDATES[a],
        ic=round(stats[a]["ic"], 4),
        icir=round(stats[a]["icir"], 4),
        hit=round(stats[a]["hit"], 4),
        n=stats[a]["n"],
        cov=round(stats[a]["cov"], 4),
        dates_ge8=round(stats[a]["dates_ge8"], 4),
        turn=round(stats[a]["turn"], 4),
        decay=stats[a]["decay"],
        max_abs_library_correlation=round(mx, 4),
    )
    print(f"  {a:18s} ic={out[a]['ic']:+.4f} icir={out[a]['icir']:+.4f} "
          f"max_rho_vs_admitted={mx:.3f}")

json.dump(out, open("scripts/_final_admission.json", "w"), indent=1)
print("\nsaved scripts/_final_admission.json")
