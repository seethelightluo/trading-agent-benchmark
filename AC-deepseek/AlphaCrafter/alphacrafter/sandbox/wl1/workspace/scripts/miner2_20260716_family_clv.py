"""Miner2 family exploration: Close-Location-Value (CLV) / intraday-range position.

Idea: Where the close sits inside the daily high-low range carries information
about order-flow imbalance. Assets whose closes persistently sit in the upper
part of the range (strong buying pressure) may continue (momentum flavor), or
conversely may mean-revert (reversal flavor). We test both directions via raw
CLV and its negative. Uses only OHLC -> full 15-name coverage.

Gates (15-instrument universe): |IC1| >= 0.0070, |ICIR1| >= 0.0840.
Window: 2021-01 .. 2026-07-15 (warm-up only), >=8 names per date for IC.
"""
import sys, time
import numpy as np
import pandas as pd
sys.path.insert(0, "scripts")
from miner1_common import SYMBOLS, load_close
import miner2_fast as F

t0 = time.time()
closes = load_close()
idx = None
for s, df in closes.items():
    idx = df.index if idx is None else idx.intersection(df.index)
idx = idx[(idx >= pd.Timestamp("2021-01-01"))]
print(f"loaded {len(idx)} common dates {idx.min().date()}..{idx.max().date()} ({time.time()-t0:.1f}s)")

def clv_panel(win):
    cols = {}
    for s in SYMBOLS:
        d = closes[s].reindex(idx)
        rng = (d["high"] - d["low"]).replace(0, np.nan)
        clv = (d["close"] - d["low"]) / rng
        cols[s] = clv.rolling(win).mean()
    return pd.DataFrame(cols)

fwd = {h: F.fwd_returns(closes, h).reindex(idx) for h in (1, 2, 3, 5, 10, 20, 30)}
N_CELLS = len(idx) * len(SYMBOLS)

def run(name, panel):
    panel = panel.reindex(idx)
    cov = float(panel.notna().sum().sum()) / N_CELLS
    to = F.turnover10(panel)
    ic1 = F.fast_ic(panel, fwd[1])
    ic5 = F.fast_ic(panel, fwd[5])
    ic10 = F.fast_ic(panel, fwd[10])
    passed = (abs(ic1["ic"]) >= 0.0070) and (abs(ic1["icir"]) >= 0.0840)
    print(f"{name:14s} cov={cov:.3f} to={to:.3f} | "
          f"IC1={ic1['ic']:+.4f} ICIR1={ic1['icir']:+.3f} hit1={ic1['hit']:.2f} n1={ic1['n_dates']} | "
          f"IC5={ic5['ic']:+.4f} ICIR5={ic5['icir']:+.3f} | IC10={ic10['ic']:+.4f} | {'PASS' if passed else 'fail'}")
    return {"name": name, "cov": cov, "to": to, "ic1": ic1, "ic5": ic5, "ic10": ic10, "passed": passed}

print("=== CLV family (positive = closing near highs) ===")
cands = {}
for win in (1, 2, 3, 5, 10, 20):
    base = clv_panel(win)
    cands[f"clv_{win}d"] = base                       # momentum flavor
    cands[f"nclv_{win}d"] = -base                     # reversal flavor

res = {n: run(n, p) for n, p in cands.items()}

# daily-range signed variant: (2*close - high - low)/(high - low)  in [-1,1]
print("\n=== signed range position (SRP) family ===")
for win in (1, 3, 5, 10):
    cols = {}
    for s in SYMBOLS:
        d = closes[s].reindex(idx)
        rng = (d["high"] - d["low"]).replace(0, np.nan)
        srp = (2.0 * d["close"] - d["high"] - d["low"]) / rng
        cols[s] = srp.rolling(win).mean()
    cands[f"srp_{win}d"] = pd.DataFrame(cols)
    res[f"srp_{win}d"] = run(f"srp_{win}d", cands[f"srp_{win}d"])

# decay for best candidates
print("\n--- decay (clv_3d, nclv_3d, srp_5d) ---")
for nm in ("clv_3d", "nclv_3d", "srp_5d"):
    r = F.fast_ic_all(cands[nm].reindex(idx), closes, horizons=(1, 2, 3, 5, 10, 20, 30))
    print(f"{nm}: " + ", ".join(f"h{h} IC={v['ic']:+.4f} ICIR={v['icir']:+.3f}" for h, v in r.items()))

# year robustness for srp_5d & nclv_3d
print("\nyear      srp_5d_IC1  srp5_ICIR   nclv3_IC1  nclv3_ICIR")
for yr in range(2021, 2027):
    lo = pd.Timestamp(f"{yr}-01-01"); hi = pd.Timestamp(f"{yr}-12-31")
    m = (idx >= lo) & (idx <= hi)
    a = F.fast_ic(cands["srp_5d"].loc[idx[m]], fwd[1].loc[idx[m]])
    b = F.fast_ic(cands["nclv_3d"].loc[idx[m]], fwd[1].loc[idx[m]])
    print(f"{yr}  {a['ic']:+.4f}  {a['icir']:+.3f}  {b['ic']:+.4f}  {b['icir']:+.3f}")

print(f"\nCLV family done in {time.time()-t0:.1f}s | {sum(v['passed'] for v in res.values())} passed gate")
