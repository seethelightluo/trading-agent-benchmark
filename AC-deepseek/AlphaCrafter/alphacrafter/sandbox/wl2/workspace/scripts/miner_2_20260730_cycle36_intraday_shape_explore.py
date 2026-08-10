"""miner_2 cycle36: intraday OHLC-shape factor family.

Rationale: the active library is built almost entirely from CLOSE-only series
(momentum, vol, streaks, days-since-high, calmness) plus macro-beta conditionals.
Untested so far: the *intraday shape* of the candle -- where the close sits
inside the daily range (C-position), the upper/lower shadow proportions, their
asymmetry, and the dispersion of overnight gaps. These use OHLC data, so they
are structurally decorrelated from close-path factors.

Candidates (one family: intraday shape & gap structure):
  - upper_shadow_20 : mean upper-shadow fraction (high - max(o,c)) / (h-l)
  - lower_shadow_20 : mean lower-shadow fraction (min(o,c) - low) / (h-l)
  - close_pos_20    : mean (close - low) / (high - low)  [C-position]
  - shadow_asym_20  : (upper - lower) / (upper + lower)  [shadow asymmetry]
  - gap_std_20      : std of overnight gap = open/prev_close - 1 over 20d

Admission gates (10d horizon): abs(IC) >= 0.0070, abs(ICIR) >= 0.0840,
max_abs_library_correlation < 0.5.
"""
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, "scripts")
from miner2_lib import (TRADABLES, load_ohlc_panels, compute_ic,
                        forward_returns, validate_factor, regime_breakdown,
                        report, per_asset)

P = load_ohlc_panels()
op, hi, lo, cl = P["open"], P["high"], P["low"], P["close"]
print(f"panels {cl.shape}, last date {cl.index[-1].date()}")

# ---- intraday shape series per asset ----
shapes = {}
for a in TRADABLES:
    o, h, l, c = op[a], hi[a], lo[a], cl[a]
    d = pd.concat([o.rename("o"), h.rename("h"), l.rename("l"), c.rename("c")], axis=1)
    d = d.dropna()
    rng = (d["h"] - d["l"]).replace(0, np.nan)
    upper = (d["h"] - d[["o", "c"]].max(axis=1)) / rng
    lower = (d[["o", "c"]].min(axis=1) - d["l"]) / rng
    cpos = (d["c"] - d["l"]) / rng
    asym = (upper - lower) / (upper + lower).replace(0, np.nan)
    gap = d["o"] / d["c"].shift(1) - 1.0
    shapes[a] = pd.DataFrame({"upper": upper, "lower": lower, "cpos": cpos,
                              "asym": asym, "gap": gap}, index=d.index)

W, MP = 20, 10


def roll_mean(col, w=W, mp=MP):
    return col.rolling(w, min_periods=mp).mean()


def roll_std(col, w=W, mp=MP):
    return col.rolling(w, min_periods=mp).std()


def build(name, fn):
    out = {}
    for a in TRADABLES:
        out[a] = fn(shapes[a][name]).reindex(cl.index)
    return pd.DataFrame(out, index=cl.index)


cands = {
    "upper_shadow_20": build("upper", roll_mean),
    "lower_shadow_20": build("lower", roll_mean),
    "close_pos_20": build("cpos", roll_mean),
    "shadow_asym_20": build("asym", roll_mean),
    "gap_std_20": build("gap", roll_std),
}

# ---- library: all real .signal.npy artifacts with matching shape ----
idx = cl.index
lib = {}
for f in sorted(Path("factors").glob("*.signal.npy")):
    arr = np.load(f)
    if arr.shape == cl.shape:
        fid = f.name.replace(".signal.npy", "")
        if fid != "downside_dev_60":  # deprecated, keep out of active library
            lib[fid] = pd.DataFrame(arr, index=idx, columns=cl.columns)
print(f"[lib] loaded {len(lib)} artifacts: {sorted(lib.keys())}")

fwd = {str(h): forward_returns(cl, h) for h in (1, 2, 3, 5, 10, 20)}

print("\n=== VALIDATION (admission horizon=10d) ===")
results = {}
for name, f in cands.items():
    m = validate_factor(f, cl, library=lib, fwd_cache=fwd)
    p = report(name, m)
    print("    decay:", m["decay_ic_by_horizon"])
    print("    pairwise:", m.get("library_pairwise_corr"))
    print()
    results[name] = {"metrics": m, "pass": p}

print("=== REGIME BREAKDOWN (10d IC) ===")
for name, f in cands.items():
    ic_ser = compute_ic(f, fwd["10"]).dropna()
    reg = regime_breakdown(ic_ser)
    print(f"  {name:18s} | " + " | ".join(
        f"{k}: ic={v['ic']:+.4f} icir={v['icir']:+.3f} n={v['n_dates']}"
        for k, v in reg.items()))

json.dump({k: {"metrics": v["metrics"], "pass": v["pass"]} for k, v in results.items()},
          open("scripts/miner_2_20260730_cycle36_results.json", "w"), indent=1, default=str)
print("\nwrote scripts/miner_2_20260730_cycle36_results.json")
