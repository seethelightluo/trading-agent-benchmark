"""miner_1 persistence run for rev5x_er_soft:
5d reversal scaled by soft trend-efficiency weight (1 - ER20)+.

Rebuilds the signal artifact aligned to the library calendar (2388 rows,
BTC calendar 2020-01-01..2026-07-15), recomputes admission metrics on the
validation window 2021-01-01..2026-07-15, and reports library correlation.
"""
import os, sys, json, glob
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from miner1_common import SYMBOLS, load_close, ic_analysis, coverage, turnover, decay_analysis

VAL_START = pd.Timestamp("2021-01-01")
CUT = pd.Timestamp("2026-07-15")
closes = load_close()

def er_series(df, win=20):
    net = (df["close"] / df["close"].shift(win) - 1.0).abs()
    path = np.log(df["close"] / df["close"].shift(1)).abs().rolling(win).sum()
    return net / (path + 1e-12)

def factor_values(df):
    rev5 = -(df["close"] / df["close"].shift(5) - 1.0)          # 5d reversal
    w = (1.0 - er_series(df, 20)).clip(lower=0)                  # soft efficiency weight
    return rev5 * w

def build_panel(fn):
    cols = {}
    for s in SYMBOLS:
        try:
            fv = fn(closes[s])
            if fv is not None and len(fv):
                cols[s] = fv
        except Exception as e:
            print(f"  [warn] {s}: {e}")
    return pd.DataFrame(cols)

# ---- full-history panel aligned to library calendar (BTC calendar: 2388 rows) ----
cal = closes["BTC"].index  # 2388 rows, 2020-01-01..2026-07-15
panel_full = build_panel(factor_values).reindex(cal)
arr = np.full((len(cal), len(SYMBOLS)), np.nan, dtype=np.float32)
for j, s in enumerate(SYMBOLS):
    arr[:, j] = panel_full[s].values
np.save("factors/miner1_20260716_rev5x_er_soft.npy", arr)
print("artifact saved: factors/miner1_20260716_rev5x_er_soft.npy",
      arr.shape, "finite:", int(np.isfinite(arr).sum()), "/", arr.size)

# ---- validation-window metrics ----
panel = panel_full[panel_full.index >= VAL_START]
print("val panel:", panel.shape, panel.index.min().date(), "..", panel.index.max().date())
print("coverage   :", round(coverage(panel, closes), 4))
print("turnover10 :", round(turnover(panel), 4))

ics = {h: ic_analysis(panel, closes, fwd_days=h) for h in (1, 2, 3, 5, 10, 20, 30)}
for h, r in ics.items():
    print(f"h={h:>2}: IC={r['ic']:+.4f} ICIR={r['icir']:+.3f} hit={r['hit']:.3f} n_dates={r['n_dates']} n_obs={r['n_obs']}")

print("\nby-year (IC1):")
for yr in range(2021, 2027):
    sub = panel[(panel.index >= pd.Timestamp(f"{yr}-01-01")) & (panel.index <= pd.Timestamp(f"{yr}-12-31"))]
    r = ic_analysis(sub, closes, fwd_days=1)
    print(f"  {yr}: IC={r['ic']:+.4f} ICIR={r['icir']:+.3f} hit={r['hit']:.3f} n={r['n_dates']}")

# ---- library correlation (spearman, daily cross-section, overlapping rows) ----
from scipy.stats import spearmanr
lib = {}
for f in glob.glob("factors/*.npy"):
    if "rev5x_er_soft" in f:
        continue
    try:
        lib[os.path.basename(f)] = np.load(f)
    except Exception as e:
        print("skip", f, e)
rhos = {}
for k, a in lib.items():
    n = min(arr.shape[0], a.shape[0])
    vals = []
    for i in range(n):
        x, y = arr[i], a[i]
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() >= 4:
            rho = spearmanr(x[m], y[m])[0]
            if np.isfinite(rho):
                vals.append(rho)
    rhos[k] = (float(np.mean(vals)) if vals else np.nan, len(vals))
    print(f"  rho vs {k}: {rhos[k][0]:+.4f} (n={rhos[k][1]})")
max_abs = max((abs(v[0]) for v in rhos.values() if np.isfinite(v[0])), default=np.nan)
print("max_abs_library_correlation:", round(max_abs, 4))

# ---- dump metrics json for persistence ----
r1 = ics[1]
out = {
    "ic": round(float(r1["ic"]), 4),
    "icir": round(float(r1["icir"]), 4),
    "hit1": round(float(r1["hit"]), 3),
    "n_ic_dates": int(r1["n_dates"]),
    "n_obs": int(r1["n_obs"]),
    "coverage": round(float(coverage(panel, closes)), 4),
    "turnover_10d": round(float(turnover(panel)), 4),
    "decay_ic": {str(h): (round(float(r["ic"]), 4) if np.isfinite(r["ic"]) else None) for h, r in ics.items()},
    "by_year_ic1": {
        str(yr): {"ic": round(float(byr["ic"]), 4), "icir": round(float(byr["icir"]), 4), "n": int(byr["n_dates"])}
        for yr in range(2021, 2027) for byr in [ic_analysis(panel[(panel.index >= pd.Timestamp(f"{yr}-01-01")) & (panel.index <= pd.Timestamp(f"{yr}-12-31"))], closes, fwd_days=1)]
    },
    "max_abs_library_correlation": round(max_abs, 4) if np.isfinite(max_abs) else None,
    "library_rhos": {k: round(v[0], 4) if np.isfinite(v[0]) else None for k, v in rhos.items()},
}
with open("scripts/miner1_rev5x_metrics.json", "w") as f:
    json.dump(out, f, indent=1)
print("\nmetrics json dumped")
