"""miner_3 novel factor exploration at 2032-05-17 (visible through 2032-05-14).
Monitors only — does NOT touch live account or advance dates.
Explores cross-asset spread/momentum, macro-gated (USDJPY/EURUSD), vol-adjusted
trend, contrarian stretch, and low-vol-tilt families under the 15-asset universe.
Reports both full-sample and last-1y stats to gauge timeliness.
"""
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

# helpers for macro-beta
def beta_to(ret, fx, win=60, minp=30):
    cov = ret.rolling(win, min_periods=minp).cov(fx)
    var = fx.rolling(win, min_periods=minp).var()
    return cov.divide(var, axis=0)

cands = {}
# --- cross-asset momentum / spread ---
cands["spread_mom_20v60"] = mom20 - mom60
cands["spread_mom_5v60"]  = mom5 - mom60
cands["spread_mom_10v60"] = (c/c.shift(10)-1.0) - mom60
# vol-adjusted spread momentum (risk-normalized convergence)
cands["spread_mom_vadj"] = (mom20 - mom60) / vol20
# trend strength: price-vs-MA60 normalized by vol
cands["trend_ma60_vadj"] = (c/ma60 - 1.0) / vol20
# --- macro-gated ---
dxy_r = dxy.pct_change(); jpy_r = usdjpy.pct_change(); eur_r = eurusd.pct_change()
b_dxy = beta_to(ret, dxy_r); b_jpy = beta_to(ret, jpy_r); b_eur = beta_to(ret, eur_r)
cands["jpy_beta_60"] = b_jpy            # risk-on (JPY weak) beta, no gate
cands["dxy_beta_plain_60"] = -b_dxy     # anti-DXY beta (plain, un-gated)
# beta to DXY *conditioned* on DXY momentum direction (carry) - variant of lib
dxy_mom20 = dxy/dxy.shift(20)-1.0
cands["dxy_carry_60x20"] = -b_dxy.multiply(dxy_mom20, axis=0)
# --- eurusd-gated risk carry ---
eur_mom20 = eurusd/eurusd.shift(20)-1.0
cands["eur_carry_60x20"] = b_eur.multiply(eur_mom20, axis=0)
# USDJPY-gated carry: JPY weakness as risk-on carry signal
cands["jpy_carry_60x20"] = b_jpy.multiply(jpy_r.rolling(20).sum().replace(0,np.nan), axis=0)
# --- contrarian / stretch ---
cands["meanrev_ma20_vadj"] = -(c/ma20-1.0)/vol20
# --- low-vol tilt with positive momentum gate ---
cands["lowvol_trmom"] = -vol20*(1.0+mom20)
# vol compression + price trend gate
mv = vol20/vol60.replace(0,np.nan)
cands["vol_compress_tr"] = -mv*np.sign(mom60)
# --- relative momentum vol-adjusted (breadth normalized) ---
rel20 = mom20.subtract(mom20.median(axis=1), axis=0)
cands["relmom_vadj"] = rel20/vol20
# --- VIX-gated beta carry ---
vix_r = vix.pct_change(); b_vix = beta_to(ret, vix_r)
vix_mom20 = vix/vix.shift(20)-1.0
cands["vix_carry_60x20"] = -b_vix.multiply(vix_mom20, axis=0)
# --- cross-asset dispersion (avg correlation) alpha ---
ret_ew = ret.mean(axis=1)
ave_corr = pd.DataFrame(index=ret.index, columns=ret.columns, dtype=float)
for a in ret.columns:
    others = ret.drop(columns=[a])
    ave_corr[a] = ret[a].rolling(60, min_periods=30).corr(others).mean(axis=1)
cands["avg_corr_60"] = ave_corr

print(f"{'candidate':26s} {'IC10':>7s} {'ICIR10':>7s} {'hit':>5s} {'n':>5s} {'IC_1y':>8s} {'ICIR_1y':>8s} {'turn':>6s} | rho  flag")
def full(cand):
    ic = daily_ic(cand, fwd)
    st = ic_stats(ic, 10)
    ic_ser = ic.dropna()
    r1 = ic_ser[ic_ser.index >= (ic_ser.index.max() - np.timedelta64(365,'D'))]
    ic1 = r1.mean() if len(r1) else np.nan
    icir1 = (r1.mean()/r1.std(ddof=1)) if len(r1) > 2 else np.nan
    turn = rank_turnover(cand, 10)
    rho, pairs = max_lib_corr(cand, lib_panels)
    gate = abs(st["ic"]) >= IC_GATE and abs(st["icir"]) >= ICIR_GATE
    return st, ic1, icir1, turn, rho, gate

for name, cand in cands.items():
    st, ic1, icir1, turn, rho, gate = full(cand)
    flag = "PASS" if gate else ""
    print(f"{name:26s} {st['ic']:+.4f} {st['icir']:+.3f} {st['hit']:5.2f} {st['n']:5d} "
          f"{ic1:+.4f} {icir1:+.3f} {turn:6.2f} | {rho:.3f} {flag}")