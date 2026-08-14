"""Trader dry-run 2035-08-02: compute target weights WITHOUT submitting orders.

Replicates strategy_hook's weight pipeline (reads factor_ensemble.json
dynamically) using sim data visible through the previous completed day.
Does NOT call rebalance_to_weights / step / backtest.
"""
import json
import math
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, ".")

import strategy as st

# ---- replicate hook inputs ----
assets = list(st.get_account_dict()["watch_list"])
frames = {a: st.get_df(a) for a in assets}
close = {a: st.series(frames[a]) for a in assets}
open_ = {a: st.series(frames[a], "open") for a in assets}

frozen = st.detect_frozen(close)
live = [a for a in assets if a not in frozen]

ret = {a: close[a].pct_change() for a in assets}
panel = pd.concat([ret[a].rename(a) for a in assets], axis=1, join="inner").dropna()

ens = st.load_ensemble()
ens_ids = {fid for fid, _, _ in ens}
print("ensemble:", [(f, round(w, 3), d) for f, w, d in ens])
print("frozen:", sorted(frozen))
print("live:", sorted(live))

r_spx = ret["SPX"]
r_300 = ret["000300.SH"]
d_cn = close["CN10Y"].pct_change()
dxy = st.series(st.get_df("DXY"))
r_dxy = dxy.pct_change() if dxy is not None else None
vix = st.series(st.get_df("VIX"))
r_vix = vix.pct_change() if vix is not None else None
comm_basket = panel[["XAU", "COPPER", "WTI"]].mean(axis=1)

sig = {fid: {} for fid in ens_ids}
for a in assets:
    c, o, r = close[a], open_[a], ret[a]
    if "down_beta_60" in ens_ids:
        sig["down_beta_60"][a] = st.down_beta(r, r_spx)
    if "spx_beta_60" in ens_ids:
        sig["spx_beta_60"][a] = st.beta_last(r, r_spx)
    if "hs300_beta_60" in ens_ids:
        sig["hs300_beta_60"][a] = st.beta_last(r, r_300)
    if "cn10y_beta_60" in ens_ids:
        sig["cn10y_beta_60"][a] = st.beta_last(r, d_cn)
    if "vol_adj_mom_20_60" in ens_ids:
        sig["vol_adj_mom_20_60"][a] = ((c.iloc[-6] / c.iloc[-26] - 1.0) / max(float(r.tail(60).std()), 1e-6)) if len(c) >= 30 else None
    if "dxy_beta_cond_60x20" in ens_ids:
        if r_dxy is not None:
            b = st.beta_last(r, r_dxy)
            sig["dxy_beta_cond_60x20"][a] = b * (dxy.iloc[-1] / dxy.iloc[-21] - 1.0) if b is not None else None
        else:
            sig["dxy_beta_cond_60x20"][a] = None
    if "hilo_vol_ratio_20" in ens_ids:
        if len(c) >= 25:
            rng = (c.rolling(20).max() - c.rolling(20).min()) / c
            rv = r.rolling(20).std()
            q = (rng / rv).dropna()
            sig["hilo_vol_ratio_20"][a] = float(q.iloc[-1]) if len(q) else None
        else:
            sig["hilo_vol_ratio_20"][a] = None
    if "intraday_ret_skew_20" in ens_ids:
        if o is not None:
            ir = (c / o - 1.0).dropna().tail(20)
            sig["intraday_ret_skew_20"][a] = float(ir.skew()) if len(ir) >= 5 else None
        else:
            sig["intraday_ret_skew_20"][a] = None
    if "comm_basket_beta_60" in ens_ids:
        sig["comm_basket_beta_60"][a] = st.beta_last(r, comm_basket)
    if "vol_of_vol20x60" in ens_ids:
        rv20 = r.rolling(20).std()
        sig["vol_of_vol20x60"][a] = float(rv20.tail(60).std()) if len(rv20.dropna()) >= 40 else None

score = {a: 0.0 for a in assets}
for fid, w, d in ens:
    rk = st.cs_rank(sig.get(fid, {}), assets)
    for a in assets:
        score[a] += w * d * rk[a]

lp = panel[live] if live else panel
market = lp.mean(axis=1)
wealth = (1.0 + market).cumprod()
mdd = float((wealth / wealth.rolling(60).max() - 1.0).tail(20).min())
mkt20 = float(market.tail(20).mean())
vol20 = float(lp.tail(20).std().mean())
vol_med = float(lp.tail(120).std().median(axis=0))
risk_off = (mkt20 < 0.0 and mdd < -0.025) or (vol20 > 1.25 * max(vol_med, 1e-6))
risk_on = mkt20 > 0.0 and mdd > -0.015
def_floor = 0.18 if risk_off else (0.11 if risk_on else 0.13)
spread = 2.0 if risk_off else (3.0 if risk_on else 2.0)
print(f"regime: mkt20={mkt20*100:.3f}% mdd20={mdd*100:.2f}% vol20={vol20*100:.2f}% "
      f"risk_off={risk_off} risk_on={risk_on} def_floor={def_floor} spread={spread}")

vix_level = float(vix.iloc[-1]) if vix is not None and len(vix) else None
eq_live = [a for a in st.EQ_ASSETS if a in live]
eq_ret21 = float(np.mean([close[a].iloc[-1] / close[a].iloc[-22] - 1.0 for a in eq_live])) if eq_live else 0.0
stress = risk_off and ((vix_level is not None and vix_level >= st.VIX_STRESS) or eq_ret21 < st.EQ_RET21_STRESS)
print(f"stress: vix={vix_level} eq_ret21={eq_ret21*100:.2f}% stress={stress}")

w = st.build_weights(score, assets, panel, def_floor, spread)
w = st.apply_frozen_override(w, assets, frozen)
w = st.risk_trim(w, assets, live, stress)
w = st.apply_all_caps(w, assets, live, stress=stress)

acct = st.get_account_dict()
mv = {p["symbol"]: float(p.get("market_value", 0.0)) for p in acct.get("positions", [])}
nav = float(acct.get("net_assets", 0.0))
cur = {a: mv.get(a, 0.0) / nav for a in assets}

print("\n%12s %8s %8s %8s" % ("asset", "target", "current", "delta"))
tot = 0.0
for a in sorted(assets, key=lambda x: -w[x]):
    d = w[a] - cur[a]
    tot += w[a]
    print("%12s %8.2f %8.2f %+8.2f" % (a, w[a] * 100, cur[a] * 100, d * 100))
print("sum target:", round(tot, 6))

gross_turn = 0.5 * sum(abs(w[a] - cur[a]) for a in assets)
print(f"one-way gross turnover: {gross_turn*100:.2f}%  (3bp gate needs edge > {gross_turn*0.0003*100:.4f}% of NAV)")

scale = float(lp.tail(252).std(axis=1, ddof=0).median()) if len(lp) >= 30 else 0.01
k = st.FC_K_MULT * scale
forecast = {a: float(k * (w[a] - cur[a])) for a in assets}
edge = sum((w[a] - cur[a]) * forecast[a] for a in assets)
print(f"scale={scale:.5f} k={k:.5f} signed gross edge={edge:.6f} "
      f"gate={gross_turn*0.0003:.6f} -> {'EXECUTE' if edge > gross_turn*0.0003 else 'SKIP'}")
