"""Miner3 fast vectorized screen: volatility / risk family on 15-asset universe.
Research window capped at 2026-07-15 (warm-up). Uses common-date close panel.
Gates: |IC1|>=0.0070 and |ICIR1|>=0.0840 (15-instrument cross-asset universe).
"""
import sys, os, time
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
from miner1_common import SYMBOLS, load_close
import miner3_fast as F

t0 = time.time()
closes = load_close()
idx = None
for s, df in closes.items():
    idx = df.index if idx is None else idx.intersection(df.index)
idx = idx[(idx >= pd.Timestamp("2021-01-01"))]  # leave 1y warmup for rolling windows
CP = pd.DataFrame({s: closes[s]["close"].reindex(idx).astype(float) for s in SYMBOLS})
HP = pd.DataFrame({s: closes[s]["high"].reindex(idx).astype(float) for s in SYMBOLS})
LP = pd.DataFrame({s: closes[s]["low"].reindex(idx).astype(float) for s in SYMBOLS})
RET = CP.pct_change()
LOG = np.log(CP / CP.shift(1))
print(f"loaded {len(idx)} common dates {idx.min().date()}..{idx.max().date()} ({time.time()-t0:.1f}s)")

fwd1 = F.fwd_returns(closes, 1).reindex(idx)
fwd5 = F.fwd_returns(closes, 5).reindex(idx)


def run(name, panel):
    panel = panel.reindex(idx)
    cov = panel.notna().sum().sum() / (len(idx) * panel.shape[1])
    to = F.turnover10(panel)
    ic1 = F.fast_ic(panel, fwd1)
    ic5 = F.fast_ic(panel, fwd5)
    ic10 = F.fast_ic(panel, F.fwd_returns(closes, 10).reindex(idx))
    passed = (abs(ic1["ic"]) >= 0.007) and (abs(ic1["icir"]) >= 0.084)
    print(f"{name:24s} cov={cov:.3f} to={to:.3f} | IC1={ic1['ic']:+.4f} ICIR1={ic1['icir']:+.3f} "
          f"hit1={ic1['hit']:.2f} n1={ic1['n_dates']} | IC5={ic5['ic']:+.4f} ICIR5={ic5['icir']:+.3f} "
          f"| IC10={ic10['ic']:+.4f} | {'PASS' if passed else 'fail'}")
    return {"name": name, "cov": cov, "to": to, "ic1": ic1, "ic5": ic5, "ic10": ic10, "passed": passed}


cands = {}
# realized vol: raw and inverse
for nd in (10, 20, 60, 120):
    v = RET.rolling(nd).std() * np.sqrt(252)
    cands[f"vol_{nd}d"] = v
    cands[f"inv_vol_{nd}d"] = -v
# vol ratio (regime shift)
cands["vol_ratio_5_60"] = RET.rolling(5).std() / RET.rolling(60).std()
cands["vol_ratio_10_60"] = RET.rolling(10).std() / RET.rolling(60).std()
cands["vol_ratio_20_120"] = RET.rolling(20).std() / RET.rolling(120).std()
# downside vol / upside-downside
for nd in (20, 60):
    dn = RET.clip(upper=0).rolling(nd).std()
    up = RET.clip(lower=0).rolling(nd).std()
    cands[f"downside_vol_{nd}d"] = dn
    cands[f"up_dn_ratio_{nd}d"] = up / (dn + 1e-12)
# parkinson volatility
hl = np.log(HP / LP)
for nd in (20, 60):
    cands[f"parkinson_{nd}d"] = (hl ** 2).rolling(nd).mean() / (4 * np.log(2)) * np.sqrt(252)
# range ratio
for nd in (20, 60):
    cands[f"range_ratio_{nd}d"] = (CP.rolling(nd).max() - CP.rolling(nd).min()) / CP.rolling(nd).mean()
# skewness / kurtosis
for nd in (60, 120):
    cands[f"skew_{nd}d"] = RET.rolling(nd).skew()
    cands[f"kurt_{nd}d"] = RET.rolling(nd).kurt()
# tail risk proxies
cands["neg_freq_60d"] = (RET < 0).rolling(60).mean()
cands["neg_freq_120d"] = (RET < 0).rolling(120).mean()
cands["var95_60d"] = RET.rolling(60).quantile(0.05)
cands["var95_120d"] = RET.rolling(120).quantile(0.05)
cands["max_dd_20d"] = CP / CP.rolling(20).max() - 1.0
cands["max_dd_60d"] = CP / CP.rolling(60).max() - 1.0
cands["max_dd_120d"] = CP / CP.rolling(120).max() - 1.0

res = [run(name, p) for name, p in cands.items()]
print(f"\nscreen {time.time()-t0:.1f}s | {sum(r['passed'] for r in res)} passed gate")