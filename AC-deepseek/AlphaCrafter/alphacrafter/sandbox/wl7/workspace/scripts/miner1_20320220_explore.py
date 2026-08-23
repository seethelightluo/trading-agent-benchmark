"""miner_1 novel factor exploration at 2032-02-20 (visible through 2032-02-19).
Monitors only — does NOT touch live account. Explores risk-on/beta/cross-asset regime families.
"""
import sys, json
sys.path.insert(0, "scripts")
import numpy as np, pandas as pd
from miner_shared import (load_close, load_macro, forward_ret, daily_ic, ic_stats,
                          rank_turnover, coverage_stats, library_panel, IC_GATE,
                          ICIR_GATE, max_lib_corr)

END = "2032-02-19"
close = load_close(END)
macro = load_macro(END)
lib_panels = library_panel(close, macro)
ret = close.pct_change()
fwd = forward_ret(close, 10)

# --- candidate constructions ---
cands = {}

# use only through END
c = close

# 1. return-of-risk (post-2029 risk-on friendly): 20d return / 20d vol (reward-to-vol)
r20 = c / c.shift(20) - 1.0
v20 = ret.rolling(20).std()
cands["roi_risk_20"] = r20 / v20

# 2. trend eff ratio with volatility (close relative to its EWMA position)
def ewma(s, span):
    return s.ewm(span=span, adjust=False).mean()
cands["close_over_ewma50"] = c / ewma(c, 50)

# 3. high-beta risk tilt: beta_ew * 20d momentum (betas rising -> leveraged momentum)
mkt = ret.mean(axis=1)
beta = ret.rolling(60).cov(mkt) / mkt.rolling(60).var()
mom20 = c / c.shift(20) - 1.0
cands["beta_mom_60x20"] = beta.multiply(mom20, axis=0)

# 4. cross-asset breadth momentum: asset vs median of all-asset momentum (demeaned momentum)
cands["breadth_mom_20"] = mom20.subtract(mom20.median(axis=1), axis=0)

# 5. vol-scaled max return (high reward per unit risk)
vol20 = ret.rolling(20).std()
cands["maxret_risk_adj_20"] = ret.rolling(20).max() / (vol20 + 1e-9)

# 6. realized vol percentile rank (risk-off avoidance in risk-on regime)
cands["volile20_ranked"] = -(ret.rolling(20).std())

# 7. US10Y level signal: bond return as defensive/reflation proxy
cands["bond_ret_20"] = c["US10Y"] / c["US10Y"].shift(20) - 1.0

print(f"{'candidate':24s} {'IC10':>7s} {'ICIR10':>7s} {'hit':>5s} {'n':>5s} {'IC_1y':>7s} {'ICIR_1y':>7s} {'IC_2y':>7s} {'turn':>6s} | maxlibrho")
def full(cand):
    ic = daily_ic(cand, fwd)
    st = ic_stats(ic, 10)
    ic_ser = ic.dropna()
    r1 = ic_ser[ic_ser.index >= (ic_ser.index.max() - np.timedelta64(365,'D'))]
    r2 = ic_ser[ic_ser.index >= (ic_ser.index.max() - np.timedelta64(730,'D'))]
    ic1 = r1.mean() if len(r1) else np.nan
    icir1 = (r1.mean()/r1.std(ddof=1)) if len(r1)>2 else np.nan
    ic2 = r2.mean() if len(r2) else np.nan
    turn = rank_turnover(cand, 10)
    rho, pairs = max_lib_corr(cand, lib_panels)
    return st, ic1, icir1, ic2, turn, rho

for name, cand in cands.items():
    st, ic1, icir1, ic2, turn, rho = full(cand)
    gate = abs(st["ic"]) >= IC_GATE and abs(st["icir"]) >= ICIR_GATE
    flag = "PASS" if gate else ""
    print(f"{name:24s} {st['ic']:+.4f} {st['icir']:+.3f} {st['hit']:5.2f} {st['n']:5d} "
          f"{ic1:+.4f} {icir1:+.3f} {ic2:+.4f} {turn:6.2f} | {rho:.3f} {flag}")