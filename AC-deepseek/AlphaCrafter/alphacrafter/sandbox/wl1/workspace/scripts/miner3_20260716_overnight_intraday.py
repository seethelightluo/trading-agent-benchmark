"""Miner3 exploration: overnight vs intraday return decomposition.

Hypothesis: the split between the overnight gap (open vs prev close) and the
intraday session move (close vs open) encodes different information in a
cross-asset universe. Assets that persistently gap up overnight but fade
intraday may behave differently going forward. Classic equity anomaly: positive
overnight drift on average, but cross-sectionally the gap component can carry
reversal or momentum information depending on market regime.

Constructs (all signal=rolling mean of the daily component):
  - on_nd: overnight gap (open/prev_close - 1) smoothed over nd
  - id_nd: intraday move (close/open - 1) smoothed over nd
  - on_id_ratio_nd: relative balance (overnight minus intraday) smoothed
Gates: |IC1|>=0.0070 and |ICIR1|>=0.0840 on 15-instrument universe.
Window: 2020-01-01..2026-07-15 warm-up.
"""
import sys, os, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from miner1_common import SYMBOLS, load_close
import miner3_fast as F

t0 = time.time()
closes = load_close()
idx = None
for s, df in closes.items():
    idx = df.index if idx is None else idx.intersection(df.index)
idx = idx[(idx >= pd.Timestamp("2021-01-01"))]  # 1y warmup
OP = pd.DataFrame({s: closes[s]["open"].reindex(idx).astype(float) for s in SYMBOLS})
CP = pd.DataFrame({s: closes[s]["close"].reindex(idx).astype(float) for s in SYMBOLS})
print(f"loaded {len(idx)} common dates {idx.min().date()}..{idx.max().date()} ({time.time()-t0:.1f}s)")

fwd = {h: F.fwd_returns(closes, h).reindex(idx) for h in (1, 2, 3, 5, 10, 20, 30)}
N_CELLS = len(idx) * len(SYMBOLS)


def run(name, panel, verbose=True):
    panel = panel.reindex(idx)
    cov = float(panel.notna().sum().sum()) / N_CELLS
    to = F.turnover10(panel)
    ic1 = F.fast_ic(panel, fwd[1])
    ic5 = F.fast_ic(panel, fwd[5])
    ic10 = F.fast_ic(panel, fwd[10])
    passed = (abs(ic1["ic"]) >= 0.0070) and (abs(ic1["icir"]) >= 0.0840)
    if verbose:
        print(f"{name:20s} cov={cov:.3f} to={to:.3f} | IC1={ic1['ic']:+.4f} ICIR1={ic1['icir']:+.3f} "
              f"hit1={ic1['hit']:.2f} n1={ic1['n_dates']} | IC5={ic5['ic']:+.4f} ICIR5={ic5['icir']:+.3f} "
              f"| IC10={ic10['ic']:+.4f} | {'PASS' if passed else 'fail'}")
    return {"name": name, "cov": cov, "to": to, "ic1": ic1, "ic5": ic5, "ic10": ic10, "passed": passed}


# daily components
on = OP / CP.shift(1) - 1.0   # overnight gap
id_ = CP / OP - 1.0           # intraday session move
tot = CP / CP.shift(1) - 1.0  # total

cands = {}
for nd in (3, 5, 10, 20):
    cands[f"ong_mean_{nd}d"] = on.rolling(nd).mean()
    cands[f"id_mean_{nd}d"] = id_.rolling(nd).mean()
    cands[f"oni_diff_{nd}d"] = (on - id_).rolling(nd).mean()
    cands[f"oni_price_{nd}d"] = (on - tot).rolling(nd).mean()

# overnight share of total magnitude (gap intensity)
for nd in (10, 20):
    cands[f"ong_share_{nd}d"] = (on.abs().rolling(nd).mean()) / (tot.abs().rolling(nd).mean() + 1e-12)

res = {n: run(n, p) for n, p in cands.items()}
print(f"\n{sum(r['passed'] for r in res.values())} passed gate | {time.time()-t0:.1f}s")