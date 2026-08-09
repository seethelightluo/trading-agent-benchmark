"""Miner1 fast vectorized family screen #1: momentum / trend / range / mean-reversion.
Uses common-date aligned close panel and vectorized cross-sectional IC.
"""
import sys, os, time
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from miner1_common import SYMBOLS, load_close
import miner3_fast as F

t0 = time.time()
closes = load_close()
# align all symbols on common dates
idx = None
for s, df in closes.items():
    idx = df.index if idx is None else idx.intersection(df.index)
CP = pd.DataFrame({s: closes[s]["close"].reindex(idx) for s in SYMBOLS}).astype(float)
print(f"loaded {len(SYMBOLS)} symbols, common dates={len(idx)} {idx.min().date()}..{idx.max().date()} ({time.time()-t0:.1f}s)")

RET = CP.pct_change()
fwd1 = F.fwd_returns(closes, 1).reindex(idx)
fwd5 = F.fwd_returns(closes, 5).reindex(idx)
fwd10 = F.fwd_returns(closes, 10).reindex(idx)


def run(name, panel, horizon_ics=True):
    panel = panel.reindex(idx)
    cov = panel.notna().sum().sum() / (len(idx) * panel.shape[1])
    to = F.turnover10(panel)
    ic1 = F.fast_ic(panel, fwd1)
    ic5 = F.fast_ic(panel, fwd5)
    ic10 = F.fast_ic(panel, fwd10)
    passed = (abs(ic1["ic"]) >= 0.007) and (abs(ic1["icir"]) >= 0.084)
    print(f"{name:24s} cov={cov:.3f} to={to:.3f} | IC1={ic1['ic']:+.4f} ICIR1={ic1['icir']:+.3f} "
          f"hit1={ic1['hit']:.2f} n1={ic1['n_dates']} | IC5={ic5['ic']:+.4f} ICIR5={ic5['icir']:+.3f} "
          f"| IC10={ic10['ic']:+.4f} | {'PASS' if passed else 'fail'}")
    return {"name": name, "cov": cov, "to": to, "ic1": ic1, "ic5": ic5, "ic10": ic10, "passed": passed}


def mom(nd, skip=0):
    if skip:
        return CP.shift(skip) / CP.shift(skip + nd) - 1.0
    return CP / CP.shift(nd) - 1.0


def rs_mom(nd):
    m = RET.rolling(nd).mean() * 252
    v = RET.rolling(nd).std() * np.sqrt(252)
    return m / v


def ts_mom(nd):
    pos = RET.clip(lower=0).rolling(nd).sum()
    neg = (-RET.clip(upper=0)).rolling(nd).sum()
    return (pos - neg) / (pos + neg + 1e-12)


def dist_high(nd):
    return CP / CP.rolling(nd).max() - 1.0


def dist_low(nd):
    return CP / CP.rolling(nd).min() - 1.0


def norm_ma_trend(n, m):
    return ((CP.rolling(n).mean() - CP.rolling(m).mean()) / CP.rolling(m).std())


def ma_slope(n, m):
    return CP.rolling(n).mean() / CP.rolling(m).mean() - 1.0


def close_vs_ma(nd):
    return CP / CP.rolling(nd).mean() - 1.0


cands = {
    "mom_20d": mom(20),
    "mom_60d": mom(60),
    "mom_120d": mom(120),
    "mom_250d": mom(250),
    "mom_60d_skip5": mom(60, skip=5),
    "mom_120d_skip20": mom(120, skip=20),
    "sharpe_60d": rs_mom(60),
    "ts_mom_60d": ts_mom(60),
    "ma20_60_slope": ma_slope(20, 60),
    "ma60_120_slope": ma_slope(60, 120),
    "close_vs_ma60": close_vs_ma(60),
    "close_vs_ma120": close_vs_ma(120),
    "dist_52w_high": dist_high(252),
    "dist_20w_high": dist_high(120),
    "dist_52w_low": dist_low(252),
    "norm_ma20_60": norm_ma_trend(20, 60),
    "norm_ma60_120": norm_ma_trend(60, 120),
}

results = []
for name, panel in cands.items():
    results.append(run(name, panel))
print(f"\nscreen finished in {time.time()-t0:.1f}s | {sum(r['passed'] for r in results)} passed gate")