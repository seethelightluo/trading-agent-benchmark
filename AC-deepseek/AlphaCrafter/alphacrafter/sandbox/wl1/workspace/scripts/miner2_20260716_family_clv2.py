"""Miner2 CLV family: negative close-location-value (close near day's low => bounce).

Full validation: decay, year robustness, VIX regime, per-asset coverage,
and library correlation (library currently empty -> 0.0 for first factor).
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
print(f"common dates: {len(idx)}  ({idx.min().date()} .. {idx.max().date()})")

def nclv_panel(win):
    cols = {}
    for s in SYMBOLS:
        d = closes[s].reindex(idx)
        rng = (d["high"] - d["low"]).replace(0, np.nan)
        clv = (d["close"] - d["low"]) / rng
        cols[s] = -clv.rolling(win).mean()
    return pd.DataFrame(cols)

print("per-symbol valid cells (win=3):")
p = nclv_panel(3)
print(p.notna().sum().to_string())
print("total cells:", p.shape[0]*p.shape[1], "valid:", int(p.notna().sum().sum()))

N_CELLS = len(idx) * len(SYMBOLS)

def screen10(nm):
    panel = nclv_panel(nm).reindex(idx)
    res10 = F.ic_all(panel, closes, horizons=(1, 2, 3, 5, 10, 20, 30))
    cov = float(panel.notna().sum().sum()) / N_CELLS
    to = F.turnover10(panel)
    ic1 = res10[1]
    passed = (abs(ic1["ic"]) >= 0.0070) and (abs(ic1["icir"]) >= 0.0840)
    print(f"\n== nclv_{nm:>2}d cov={cov:.3f} to={to:.3f} PASS={passed} ==")
    for h, v in res10.items():
        print(f"  h{h:>2} IC={v['ic']:+.4f} ICIR={v['icir']:+.3f} hit={v['hit']:.3f} n={v['n_dates']}")
    return panel, cov, to, res10, passed

panels = {}
for nm in (1, 2, 3):
    panel, cov, to, res10, passed = screen10(nm)
    panels[nm] = (panel, cov, to, res10, passed)

print("\n=== year-by-year IC (nclv_1d) ===")
panel = panels[1][0]
fwd1 = F.fwd_returns(closes, 1).reindex(idx)
for yr in range(2021, 2027):
    lo = pd.Timestamp(f"{yr}-01-01"); hi = pd.Timestamp(f"{yr}-12-31")
    m = (idx >= lo) & (idx <= hi)
    r = F.fast_ic(panel.loc[idx[m]], fwd1.loc[idx[m]])
    if r["n_dates"]:
        print(f"{yr}  IC={r['ic']:+.4f} ICIR={r['icir']:+.3f} hit={r['hit']:.3f} n={r['n_dates']}")

vix_raw = pd.read_csv("../persistent/index_data/VIX.csv")
vix_raw["date"] = pd.to_datetime(vix_raw["date"])
vix = vix_raw.set_index("date")["close"].reindex(idx)
med = vix.median()
print("\n=== VIX regime split (nclv_1d) ===")
for label, m in [("lowVIX", vix < med), ("highVIX", vix >= med)]:
    r = F.fast_ic(panel.loc[idx[m]], fwd1.loc[idx[m]])
    print(f"{label:8s} IC={r['ic']:+.4f} ICIR={r['icir']:+.3f} hit={r['hit']:.3f} n={r['n_dates']}")

print("\n=== correlation with rev_1d (library conflict check) ===")
CP = pd.DataFrame({s: closes[s]["close"].reindex(idx).astype(float) for s in SYMBOLS})
rev1 = -np.log(CP / CP.shift(1))
x = panel.stack().dropna()
y = rev1.stack().dropna()
common = x.index.intersection(y.index)
r_pool = st.pearsonr(x.loc[common], y.loc[common])[0]
print(f"pooled pearson(nclv_1d, rev_1d) = {r_pool:+.4f}  n={len(common)}")

print(f"\n[elapsed {time.time()-t0:.1f}s]")
