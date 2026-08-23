"""miner_1 novel factor exploration at 2032-05-31 (visible through 2032-05-28).
Monitors only -- does NOT touch live account, date.json, or account.json.
Targets under-represented families in the active library: spread/reversal,
relative-vol, risk-adjusted reward, and cross-asset macro-conditional signals.
"""
import sys, json
sys.path.insert(0, "scripts")
import numpy as np, pandas as pd
from miner_shared import (load_close, load_macro, forward_ret, daily_ic, ic_stats,
                          rank_turnover, coverage_stats, library_panel, IC_GATE,
                          ICIR_GATE, max_lib_corr)

END = "2032-05-28"
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
# 1. short-lag momentum minus long-lag momentum (convergence/reversal)
cands["spread_mom_20v60"] = mom20 - mom60
# 2. relative realized-vol reversal (low recent vol vs long vol)
cands["vol_ratio_60v20"] = -(vol60 / (vol20 + 1e-9))
# 3. lowvol20 (defensive, cf downside but simpler raw)
cands["lowvol20"] = -vol20
# 4. dev_ma20 mean reversion on stretch
ma20 = c.rolling(20).mean()
cands["dev_ma20"] = -(c / ma20 - 1.0)
# 5. risk-adjusted reward 20d
cands["roi_risk_20"] = mom20 / (vol20 + 1e-9)
# 6. beta_ew momentum interaction
mkt = ret.mean(axis=1)
beta = ret.rolling(60).cov(mkt) / mkt.rolling(60).var()
cands["beta_mom_60x20"] = beta.multiply(mom20, axis=0)
# 7. cross-asset breadth momentum
cands["breadth_mom_20"] = mom20.subtract(mom20.median(axis=1), axis=0)
# 8. maxret risk-adjusted
cands["maxret_risk_adj_20"] = ret.rolling(20).max() / (vol20 + 1e-9)
# 9. upside/downside semi-vol (asymmetry) -- under-represented
up = ret.where(ret > 0, 0.0)
down = ret.where(ret < 0, 0.0)
upsv = (up ** 2).rolling(20).mean().apply(np.sqrt) + 1e-9
dnv = (down ** 2).rolling(20).mean().apply(np.sqrt) + 1e-9
cands["asym_upside20"] = upsv / dnv
# 10. intraday/range efficiency: closeness of close to 20d high (trend strength)
cands["close_pos_20"] = c / c.rolling(20).max()
# 11. autcorr reversal: sign persistence of returns (mean-reversion signal)
sign = ret.rolling(20).apply(lambda x: (x[:10].sum() * x[10:].sum()), raw=True)
cands["autocorr_10x10"] = -sign

print(f"{'candidate':24s} {'IC10':>7s} {'ICIR10':>7s} {'hit':>5s} {'n':>5s} {'IC_1y':>7s} {'ICIR_1y':>7s} {'IC_2y':>7s} {'turn':>6s} | maxlibrho")
def full(cand):
    ic = daily_ic(cand, fwd)
    st = ic_stats(ic, 10)
    ic_ser = ic.dropna()
    r1 = ic_ser[ic_ser.index >= (ic_ser.index.max() - np.timedelta64(365, 'D'))]
    r2 = ic_ser[ic_ser.index >= (ic_ser.index.max() - np.timedelta64(730, 'D'))]
    ic1 = r1.mean() if len(r1) else np.nan
    icir1 = (r1.mean()/r1.std(ddof=1)) if len(r1) > 2 else np.nan
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