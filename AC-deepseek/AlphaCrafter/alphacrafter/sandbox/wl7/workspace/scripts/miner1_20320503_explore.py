"""miner_1 novel factor exploration at 2032-05-03 (visible through 2032-05-01).
Monitors only — does NOT touch live account. Explores spread/mean-reversion families
and relative-volatility signals, which appear under-represented in the active library.
"""
import sys, json
sys.path.insert(0, "scripts")
import numpy as np, pandas as pd
from miner_shared import (load_close, load_macro, forward_ret, daily_ic, ic_stats,
                          rank_turnover, coverage_stats, library_panel, IC_GATE,
                          ICIR_GATE, max_lib_corr)

END = "2032-05-01"
close = load_close(END)
macro = load_macro(END)
lib_panels = library_panel(close, macro)
ret = close.pct_change()
fwd = forward_ret(close, 10)

c = close
mom20 = c / c.shift(20) - 1.0
mom60 = c / c.shift(60) - 1.0
vol20 = ret.rolling(20).std()
vol60 = ret.rolling(60).std()

cands = {}
# 1. Short-lag momentum minus long-lag momentum (reversal/convergence)
cands["spread_mom_20v60"] = mom20 - mom60
# 2. vol_ratio_60v20 (relative realized-vol reversal, direction -)
cands["vol_ratio_60v20"] = -(vol60 / (vol20 + 1e-9))
# 3. low-vol bonus: negative raw vol20 (defensive)
cands["lowvol20"] = -vol20
# 4. close vs 20d mean deviation (mean reversion, direction negative on stretch)
ma20 = c.rolling(20).mean()
cands["dev_ma20"] = -(c / ma20 - 1.0)
# 5. 20d downside semi-vol rank (safe-asset tilt, cf downside ratio but simpler)
neg = ret.where(ret < 0, 0.0)
semivol = (neg ** 2).rolling(20).mean().apply(np.sqrt)
cands["semivol20"] = -semivol
# 6. momentum breadth: momentum relative to median (already in library rel_mom but independent test)
cands["mom_breadth_20"] = mom20.subtract(mom20.median(axis=1), axis=0)
# 7. 5d vs 60d momentum (faster reversal)
cands["spread_mom_5v60"] = (c / c.shift(5) - 1.0) - mom60

print(f"{'candidate':24s} {'IC10':>7s} {'ICIR10':>7s} {'hit':>5s} {'n':>5s} {'IC_1y':>8s} {'ICIR_1y':>8s} {'turn':>6s} | maxlibrho")
def full(cand):
    ic = daily_ic(cand, fwd)
    st = ic_stats(ic, 10)
    ic_ser = ic.dropna()
    r1 = ic_ser[ic_ser.index >= (ic_ser.index.max() - np.timedelta64(365, 'D'))]
    ic1 = r1.mean() if len(r1) else np.nan
    icir1 = (r1.mean()/r1.std(ddof=1)) if len(r1) > 2 else np.nan
    turn = rank_turnover(cand, 10)
    rho, pairs = max_lib_corr(cand, lib_panels)
    return st, ic1, icir1, turn, rho

for name, cand in cands.items():
    st, ic1, icir1, turn, rho = full(cand)
    gate = abs(st["ic"]) >= IC_GATE and abs(st["icir"]) >= ICIR_GATE
    flag = "PASS" if gate else ""
    print(f"{name:24s} {st['ic']:+.4f} {st['icir']:+.3f} {st['hit']:5.2f} {st['n']:5d} "
          f"{ic1:+.4f} {icir1:+.3f} {turn:6.2f} | {rho:.3f} {flag}")