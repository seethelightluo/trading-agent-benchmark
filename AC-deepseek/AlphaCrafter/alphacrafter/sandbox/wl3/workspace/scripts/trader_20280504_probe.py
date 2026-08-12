"""Read-only probe: replicate strategy_hook computation up to target weights
(no rebalance_to_weights call, no account mutation). 2028-05-04."""
import json
import sys
sys.path.insert(0, ".")
import numpy as np
import pandas as pd

import strategy as st

from alphacrafter.sim.utils import get_account_dict

assets = list(get_account_dict()["watch_list"])
frames = {a: st.get_df(a) for a in assets}
close = {a: st.series(frames[a]) for a in assets}
open_ = {a: st.series(frames[a], "open") for a in assets}

frozen = st.detect_frozen(close)
live = [a for a in assets if a not in frozen]
print("frozen:", sorted(frozen))
print("live  :", sorted(live))

ret = {a: close[a].pct_change() for a in assets}
panel = pd.concat([ret[a].rename(a) for a in assets], axis=1, join="inner").dropna()
print("panel rows:", len(panel))

ens = st.load_ensemble()
ens_ids = {fid for fid, _, _ in ens}
print("ensemble ids:", sorted(ens_ids))

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
        sig["vol_adj_mom_20_60"][a] = ((c.iloc[-6] / c.iloc[-26] - 1.0)
                                       / max(float(r.tail(60).std()), 1e-6)
                                       if len(c) >= 30 else None)
    if "dxy_beta_cond_60x20" in ens_ids:
        if r_dxy is not None:
            b = st.beta_last(r, r_dxy)
            sig["dxy_beta_cond_60x20"][a] = b * (dxy.iloc[-1] / dxy.iloc[-21] - 1.0) if b is not None else None
        else:
            sig["dxy_beta_cond_60x20"][a] = None
    if "vix_beta_cond_60x20" in ens_ids:
        if r_vix is not None and len(vix) >= 25:
            b = st.beta_last(r, r_vix)
            sig["vix_beta_cond_60x20"][a] = b * (vix.iloc[-1] / vix.iloc[-21] - 1.0) if b is not None else None
        else:
            sig["vix_beta_cond_60x20"][a] = None
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
    if "vol_regime_switch_20x60" in ens_ids:
        rv20 = r.rolling(20).std()
        above = (rv20 > rv20.rolling(60).median()).astype(float)
        flips = above.diff().abs().rolling(60).mean().dropna()
        sig["vol_regime_switch_20x60"][a] = float(flips.iloc[-1]) if len(flips) else None
    if "dd_duration_120_resid" in ens_ids:
        sig["dd_duration_120_resid"][a] = st.dd_duration_resid(c, r, r_spx)

# coverage check
for fid in sorted(ens_ids):
    vals = {a: v for a, v in sig[fid].items() if v is not None and np.isfinite(v)}
    print(f"  {fid:28s} coverage={len(vals):2d}/15")

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
vix_level = float(vix.iloc[-1]) if vix is not None and len(vix) else None
eq_live = [a for a in st.EQ_ASSETS if a in live]
eq_ret21 = float(np.mean([close[a].iloc[-1] / close[a].iloc[-22] - 1.0 for a in eq_live]))
stress = risk_off and ((vix_level is not None and vix_level >= st.VIX_STRESS)
                       or eq_ret21 < st.EQ_RET21_STRESS)
print(f"regime: mkt20={mkt20*100:.2f}% mdd20={mdd*100:.2f}% vol20={vol20*100:.2f}% "
      f"risk_off={risk_off} risk_on={risk_on} def_floor={def_floor} spread={spread}")
print(f"stress: vix={vix_level} eq_ret21={eq_ret21*100:.2f}% -> {stress}")

weights = st.build_weights(score, assets, panel, def_floor, spread)
weights = st.apply_frozen_override(weights, assets, frozen)
weights = st.risk_trim(weights, assets, live, stress)

print("\ntarget weights:")
tot = 0.0
for a in sorted(assets, key=lambda x: -weights[x]):
    print(f"  {a:10s} {weights[a]*100:6.2f}%")
    tot += weights[a]
print("sum:", round(tot, 6), "| cash target: 0")

eqw = sum(weights[a] for a in st.EQ_ASSETS)
print(f"live-equity total: {eqw*100:.1f}% | XAU {weights['XAU']*100:.1f}% "
      f"COPPER {weights['COPPER']*100:.1f}% WTI {weights['WTI']*100:.1f}% "
      f"ETH {weights['ETH']*100:.1f}%")

# score ranks print
order = sorted(assets, key=lambda a: -score[a])
print("\nscore order:", [(a, round(score[a], 3)) for a in order])
