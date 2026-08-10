"""Miner2 deep validation of short-term mean-reversion factors (PASSING candidates).

Candidates from screen #1: rev_1d, rev_2d, rev_5d, nclv_1d (all PASS gate).
New variants: vol-scaled reversal, cross-sectional (relative) reversal,
composite nclv+rev, gap-aware reversal.

Checks: decay across horizons, year-by-year IC, VIX-regime split, pooled
pairwise correlations between candidates (library conflict / duplication check),
per-symbol IC contribution.
Gates: |IC1| >= 0.0070, |ICIR1| >= 0.0840. Window 2021-01 .. 2026-07-15.
"""
import sys, time
import numpy as np
import pandas as pd
import scipy.stats as st

sys.path.insert(0, "scripts")
from miner1_common import SYMBOLS, load_close
import miner2_fast as F

t0 = time.time()
closes = load_close()
idx = None
for s, df in closes.items():
    idx = df.index if idx is None else idx.intersection(df.index)
idx = idx[(idx >= pd.Timestamp("2021-01-01"))]
CP = pd.DataFrame({s: closes[s]["close"].reindex(idx).astype(float) for s in SYMBOLS})
HP = pd.DataFrame({s: closes[s]["high"].reindex(idx).astype(float) for s in SYMBOLS})
LP = pd.DataFrame({s: closes[s]["low"].reindex(idx).astype(float) for s in SYMBOLS})
OP = pd.DataFrame({s: closes[s]["open"].reindex(idx).astype(float) for s in SYMBOLS})
RET = CP.pct_change()
print(f"loaded {len(idx)} common dates {idx.min().date()}..{idx.max().date()} ({time.time()-t0:.1f}s)")

fwd = {h: F.fwd_returns(closes, h).reindex(idx) for h in (1, 2, 3, 5, 10, 20, 30)}
N_CELLS = len(idx) * len(SYMBOLS)

panels = {}
panels["rev_1d"] = -(CP / CP.shift(1) - 1.0)
panels["rev_2d"] = -(CP / CP.shift(2) - 1.0)
panels["rev_5d"] = -(CP / CP.shift(5) - 1.0)
panels["nclv_1d"] = -((CP - LP) / (HP - LP).replace(0, np.nan))
vol20 = RET.rolling(20).std()
panels["rev_1d_vs"] = panels["rev_1d"] / (vol20 + 1e-12)
panels["nclv_1d_vs"] = panels["nclv_1d"] / (vol20 + 1e-12)
# cross-sectional (relative) reversal: demeaned by cross-sectional mean
xmean = panels["rev_1d"].mean(axis=1)
panels["rev_1d_rel"] = panels["rev_1d"].sub(xmean, axis=0)
# composite: z-scores of rev_1d and nclv_1d summed
zr = panels["rev_1d"].sub(panels["rev_1d"].mean(axis=1), axis=0).div(panels["rev_1d"].std(axis=1) + 1e-12, axis=0)
zn = panels["nclv_1d"].sub(panels["nclv_1d"].mean(axis=1), axis=0).div(panels["nclv_1d"].std(axis=1) + 1e-12, axis=0)
panels["composite_z"] = zr + zn
# gap-aware reversal: sign flip when overnight gap contradicts close (2*close - open - prev_close)/prev_close
gap = OP / CP.shift(1) - 1.0
intra = CP / OP - 1.0
panels["rev_gapadj_1d"] = -(gap + intra) / (vol20 + 1e-12)

print("\n=== full metrics (decay) ===")
summary = {}
for name, p in panels.items():
    r = F.fast_ic_all(p.reindex(idx), closes, horizons=(1, 2, 3, 5, 10, 20, 30))
    cov = float(p.reindex(idx).notna().sum().sum()) / N_CELLS
    to = F.turnover10(p.reindex(idx))
    ic1, icir1 = r[1]["ic"], r[1]["icir"]
    passed = (abs(ic1) >= 0.0070) and (abs(icir1) >= 0.0840)
    summary[name] = {"r": r, "cov": cov, "to": to, "passed": passed}
    dec = {h: v["ic"] for h, v in r.items()}
    print(f"{name:14s} cov={cov:.3f} to={to:.3f} PASS={passed} | "
          + " ".join(f"h{h}:IC{v['ic']:+.4f}/ICIR{v['icir']:+.3f}" for h, v in r.items()))

print("\n=== pooled pairwise correlations (Pearson on stacked panel) ===")
stacked = {n: p.stack() for n, p in panels.items()}
names = list(panels.keys())
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a, b = stacked[names[i]], stacked[names[j]]
        common = a.index.intersection(b.index)
        rho = st.pearsonr(a.loc[common], b.loc[common])[0]
        print(f"  corr({names[i]}, {names[j]}) = {rho:+.3f}  n={len(common)}")

print("\n=== year-by-year IC1 (rev_1d, nclv_1d, rev_1d_rel, composite_z) ===")
fwd1 = fwd[1]
for yr in range(2021, 2027):
    lo = pd.Timestamp(f"{yr}-01-01"); hi = pd.Timestamp(f"{yr}-12-31")
    m = (idx >= lo) & (idx <= hi)
    row = []
    for nm in ("rev_1d", "nclv_1d", "rev_1d_rel", "composite_z"):
        r = F.fast_ic(panels[nm].loc[idx[m]], fwd1.loc[idx[m]])
        row.append(f"{r['ic']:+.4f}/{r['icir']:+.3f}")
    print(f"{yr}  " + "  ".join(f"{nm[:9]}={v}" for nm, v in zip(("rev_1d", "nclv_1d", "rev_1d_rel", "composite_z"), row)))

print("\n=== VIX regime split (IC1) ===")
vix_raw = pd.read_csv("../persistent/index_data/VIX.csv")
vix_raw["date"] = pd.to_datetime(vix_raw["date"])
vix = vix_raw.set_index("date")["close"].reindex(idx)
med = vix.median()
for nm in ("rev_1d", "nclv_1d", "rev_1d_rel", "composite_z"):
    parts = []
    for label, m in [("lowVIX", vix < med), ("highVIX", vix >= med)]:
        r = F.fast_ic(panels[nm].loc[idx[m]], fwd1.loc[idx[m]])
        parts.append(f"{label}:IC={r['ic']:+.4f}/ICIR={r['icir']:+.3f}/n={r['n_dates']}")
    print(f"{nm:12s} " + " | ".join(parts))

print("\n=== per-symbol IC contribution (nclv_1d, rev_1d) ===")
for nm in ("nclv_1d", "rev_1d"):
    cols = []
    for s in SYMBOLS:
        f = panels[nm][s].dropna()
        r = fwd1[s].reindex(f.index).dropna()
        common = f.index.intersection(r.index)
        if len(common) > 50:
            ic = st.spearmanr(f.loc[common], r.loc[common])[0]
            cols.append((s, ic, len(common)))
    cols.sort(key=lambda x: -abs(x[1]))
    print(f"  {nm}: " + ", ".join(f"{s}={ic:+.3f}(n={n})" for s, ic, n in cols))

print(f"\ndeep validation done in {time.time()-t0:.1f}s")
