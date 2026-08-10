"""miner_1 focused validation of rev5x_er_soft (reversal 5d x (1-ER) soft weight)."""
import os, sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from miner1_common import SYMBOLS, MACRO, CUT, START, load_close, ic_analysis, coverage, turnover

VAL_START = pd.Timestamp("2021-01-01")
closes = load_close()
macros = load_close(MACRO, dir_="../persistent/index_data")

def er_series(df, win=20):
    net = (df["close"] / df["close"].shift(win) - 1.0).abs()
    path = np.log(df["close"] / df["close"].shift(1)).abs().rolling(win).sum()
    return net / (path + 1e-12)

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

panel = build_panel(lambda df: (-(df["close"] / df["close"].shift(5) - 1.0)) * (1.0 - er_series(df, 20)).clip(lower=0))
panel = panel[panel.index >= VAL_START]

print("full panel shape:", panel.shape, "dates:", panel.index.min().date(), "..", panel.index.max().date())
print("coverage:", round(coverage(panel, closes), 3), "turnover10:", round(turnover(panel), 3))

ics = {h: ic_analysis(panel, closes, fwd_days=h) for h in (1, 2, 3, 5, 10, 20)}
for h, r in ics.items():
    print(f"h={h:>2}: IC={r['ic']:+.4f} ICIR={r['icir']:+.3f} hit={r['hit']:.3f} n_dates={r['n_dates']} n_obs={r['n_obs']}")

print("\nby-year (IC1):")
for yr in range(2021, 2027):
    sub = panel[(panel.index >= pd.Timestamp(f"{yr}-01-01")) & (panel.index <= pd.Timestamp(f"{yr}-12-31"))]
    r = ic_analysis(sub, closes, fwd_days=1)
    print(f"  {yr}: IC={r['ic']:+.4f} ICIR={r['icir']:+.3f} hit={r['hit']:.3f} n={r['n_dates']}")

print("\nsub-period IC1:")
for a, b in [("2021-01-01", "2023-06-30"), ("2023-07-01", "2026-07-15"), ("2024-07-01", "2026-07-15")]:
    sub = panel[(panel.index >= pd.Timestamp(a)) & (panel.index <= pd.Timestamp(b))]
    r = ic_analysis(sub, closes, fwd_days=1)
    print(f"  {a}..{b}: IC={r['ic']:+.4f} ICIR={r['icir']:+.3f} hit={r['hit']:.3f} n={r['n_dates']}")

# ---- dump full-history aligned signal artifact (2388 x 15, matching library layout) ----
full = build_panel(lambda df: (-(df["close"] / df["close"].shift(5) - 1.0)) * (1.0 - er_series(df, 20)).clip(lower=0))
# align to the same calendar as existing artifacts: take er20 npy as template
ref = np.load("factors/miner1_20260716_er20.npy")  # (2388, 15)
# er20 panel dates: rebuild from closes (all symbols share dates after cutoff)
cal = closes[SYMBOLS[0]].index
full = full.reindex(cal)
n_cal = ref.shape[0]
if len(cal) >= n_cal:
    arr = np.full((n_cal, len(SYMBOLS)), np.nan, dtype=np.float32)
    sub = full.iloc[-n_cal:]
    for j, s in enumerate(SYMBOLS):
        col = sub[s].values
        arr[:, j] = col[:n_cal]
else:
    arr = full.values[:n_cal].astype(np.float32)
np.save("factors/miner1_20260716_rev5x_er_soft.npy", arr)
print("\nartifact saved: factors/miner1_20260716_rev5x_er_soft.npy", arr.shape, "finite:", np.isfinite(arr).sum())

# sanity: re-load and compare lib corr
lib = {"er20": np.load("factors/miner1_20260716_er20.npy"),
       "mom": np.load("factors/miner2_20260716_mom_10d_skip5.npy"),
       "nclv": np.load("factors/miner2_20260716_nclv_1d.npy")}
from scipy.stats import spearmanr
for k, a in lib.items():
    n = min(arr.shape[0], a.shape[0])
    rhos = []
    for i in range(n):
        x, y = arr[i], a[i]
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() >= 4:
            r = spearmanr(x[m], y[m])[0]
            if np.isfinite(r):
                rhos.append(r)
    print(f"  artifact rho vs {k}: {np.mean(rhos):+.3f} (n={len(rhos)})")
