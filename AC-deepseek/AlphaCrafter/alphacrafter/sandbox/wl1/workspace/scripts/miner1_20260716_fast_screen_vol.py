"""Miner1-2 fast family screen #2: volatility / risk / reversal.

Vectorized on common-date panel. Includes vol factors, skewness, drawdown,
short-term reversal, overnight/gap, and liquidity-ish price-based proxies.
"""
import sys, os, time
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from miner1_common import SYMBOLS, load_close
import miner3_fast as F

t0 = time.time()
closes = load_close()
idx = pd.Index(sorted(set(closes["SPX"].index)))
for s, df in closes.items():
    idx = idx.intersection(df.index)
CP = pd.DataFrame({s: closes[s]["close"].reindex(idx).astype(float) for s in SYMBOLS})
HP = pd.DataFrame({s: closes[s]["high"].reindex(idx).astype(float) for s in SYMBOLS})
LP = pd.DataFrame({s: closes[s]["low"].reindex(idx).astype(float) for s in SYMBOLS})
RET = CP.pct_change()
LOG = np.log(CP / CP.shift(1))
print(f"loaded common dates={len(idx)} ({time.time()-t0:.1f}s)")

fwd1 = F.fwd_returns(closes, 1).reindex(idx)
fwd5 = F.fwd_returns(closes, 5).reindex(idx)


def run(name, panel):
    panel = panel.reindex(idx)
    cov = panel.notna().sum().sum() / (len(idx) * panel.shape[1])
    to = F.turnover10(panel)
    ic1 = F.fast_ic(panel, fwd1)
    ic5 = F.fast_ic(panel, fwd5)
    passed = (abs(ic1["ic"]) >= 0.017) and (abs(ic1["icir"]) >= 0.084)
    print(f"{name:24s} cov={cov:.3f} to={to:.3f} | IC1={ic1['ic']:+.4f} ICIR1={ic1['icir']:+.3f} "
          f"hit1={ic1['hit']:.2f} n1={ic1['n_dates']} | IC5={ic5['ic']:+.4f} ICIR5={ic5['icir']:+.3f} "
          f"| {'PASS' if passed else 'fail'}")
    return passed


cands = {}
# realized volatility (negative sign: low vol assets outperform? test both)
for nd in (10, 20, 60, 120):
    v = RET.rolling(nd).std() * np.sqrt(252)
    cands[f"vol_{nd}d"] = v
    cands[f"inv_vol_{nd}d"] = -v
# vol ratio
cands["vol_ratio_5_60"] = RET.rolling(5).std() / RET.rolling(60).std()
cands["vol_ratio_10_60"] = RET.rolling(10).std() / RET.rolling(60).std()
cands["vol_ratio_20_120"] = RET.rolling(20).std() / RET.rolling(120).std()
# downside/upside
for nd in [20, 60]:
    dn = RET.clip(upper=0).rolling(nd).std()
    cands[f"downside_vol_{nd}d"] = dn
    up = RET.clip(lower=0).rolling(nd).std()
    cands[f"up_dn_ratio_{nd}d"] = up / (dn + 1e-12)
# parkinson
hl = np.log(HP / LP)
for nd in [20, 60]:
    cands[f"parkinson_{nd}d"] = (hl ** 2).rolling(nd).mean() / (4 * np.log(2)) * np.sqrt(252)
# range
for nd in [20, 60]:
    cands[f"range_ratio_{nd}d"] = (CP.rolling(nd).max() - CP.rolling(nd).min()) / CP.rolling(nd).mean()
# skewness (negative skew = crash risk)
for nd in [60, 120]:
    cands[f"skew_{nd}d"] = RET.rolling(nd).skew()
    cands[f"neg_skew_{nd}d"] = -RET.rolling(nd).skew()
# kurtosis/crash
cands["kurt_60d"] = RET.rolling(60).kurt()
# max drawdown
for nd in [20, 60, 120]:
    cands[f"max_dd_{nd}d"] = CP.rolling(nd).max() / CP - 1.0
# reversal factors (negative short-term momentum = buy losers)
for nd in [1, 2, 5, 10, 20]:
    cands[f"rev_{nd}d"] = -(CP / CP.shift(nd) - 1.0)
# z of price vs ma (mean reversion)
cands["z_price_ma20"] = (CP - CP.rolling(20).mean()) / CP.rolling(20).std()
cands["z_price_ma60"] = (CP - CP.rolling(60).mean()) / CP.rolling(60).std()
# autocorrelation of returns (trend persistence)
for nd in [10, 20]:
    r = RET
    cands[f"autocorr_{nd}d"] = r.rolling(nd).apply(lambda x: np.corrcoef(x[:-1], x[1:])[0, 1], raw=True)
# neg freq (fraction of down days)
cands["neg_freq_60d"] = (RET < 0).rolling(60).mean()
# var
cands["var95_60d"] = RET.rolling(60).quantile(0.05)
cands["var95_20d"] = RET.rolling(20).quantile(0.05)

npass = 0
for name, panel in cands.items():
    try:
        npass += int(run(name, panel))
    except Exception as e:
        print(f"{name}: ERROR {e}")
print(f"\nscreen2 finished in {time.time()-t0:.1f}s | {npass} passed gate")