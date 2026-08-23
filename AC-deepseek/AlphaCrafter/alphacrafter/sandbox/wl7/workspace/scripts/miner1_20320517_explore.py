"""miner_1 novel factor exploration at 2032-05-17 (visible through 2032-05-14).
Monitors only — does NOT touch live account. Explores cross-asset spread, vol-risk-adjust,
and macro-gated factor families that are under-represented in the active library."""
import sys, json
sys.path.insert(0, "scripts")
import numpy as np, pandas as pd
from miner_shared import (load_close, load_macro, forward_ret, daily_ic, ic_stats,
                          rank_turnover, coverage_stats, library_panel, IC_GATE,
                          ICIR_GATE, max_lib_corr)

END = "2032-05-14"
close = load_close(END)
macro = load_macro(END)
lib_panels = library_panel(close, macro)
ret = close.pct_change()
fwd = forward_ret(close, 10)
vol10 = ret.rolling(10).std()
vol20 = ret.rolling(20).std()
vol60 = ret.rolling(60).std()
c = close
mom5 = c/c.shift(5)-1.0
mom20 = c/c.shift(20)-1.0
mom60 = c/c.shift(60)-1.0
ma20 = c.rolling(20).mean()
ma60 = c.rolling(60).mean()
vix = macro["VIX"]; dxy = macro["DXY"]; eurusd = macro["EURUSD"]; usdjpy = macro["USDJPY"]

cands = {}
# 1. vol-adjusted spread momentum (risk-normalized convergence), looking for higher ICIR
cands["spread_mom_20v60_vadj"] = (mom20 - mom60) / vol20
# 2. trend-strength: price-vs-MA60 normalized by vol (risk-adjusted trend momentum)
cands["trend_ma60_vadj"] = ((c/ma60 - 1.0) / vol20)
# 3. cross-asset: asset return correlation to DXY momentum direction (risk carry)
dxy_r = dxy.pct_change()
cov = ret.rolling(60).cov(dxy_r); var = dxy_r.rolling(60).var()
beta_dxy = cov.divide(var, axis=0)
dxy_mom = dxy/dxy.shift(20)-1.0
cands["dxy_carry_60x20"] = -beta_dxy.multiply(dxy_mom, axis=0)  # anti-DXY strength carry
# 4. USDJPY-gated risk on: return beta to USDJPY up-move
jpy_r = usdjpy.pct_change()
covj = ret.rolling(60).cov(jpy_r); varj = jpy_r.rolling(60).var()
beta_jpy = covj.divide(varj, axis=0)
cands["jpy_beta_60"] = beta_jpy  # risk-on (JPY weak) beta
# 5. momentum breadth vs own history (residual momentum), normalized
rel20 = mom20.subtract(mom20.median(axis=1), axis=0)
cands["relmom_vadj"] = rel20 / vol20
# 6. short-term mean reversion on stretched vol (contrarian normalized by realized stretch)
stretch = (c/ma20 - 1.0)
cands["meanrev_ma20_vadj"] = -stretch / vol20
# 7. low-vol tilt with positive-momentum gate (defensive + trend confirm)
cands["lowvol_trmom"] = -(vol20) * (1.0 + mom20) 
# 8. vol ratio with price-trend gate (vol compression confirms trend persist)
mv = (vol20/vol60.replace(0,np.nan))
cands["vol_compress_tr"] = -(mv) * np.sign(mom60) # compress + uptrend
# 9. speed of trend: 10d mom - 60d mom risk adj (medium-horizon momentum quality)
mix = (c/c.shift(10)-1.0) - (mom60*0.5)
cands["trend_accel_10v60"] = mix / vol20
# 10. cross-asset dispersion beta: correlation-to-EW-market (concentration) 
ret_ew = ret.mean(axis=1)
ave_corr = pd.DataFrame(index=ret.index, columns=ret.columns, dtype=float)
for a in ret.columns:
    others = ret.drop(columns=[a])
    ave_corr[a] = ret[a].rolling(60, min_periods=30).corr(others).mean(axis=1)
cands["avg_corr_60"] = ave_corr  # high-conviction diversification score

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