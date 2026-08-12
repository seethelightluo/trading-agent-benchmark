"""Dry-run of the full weight pipeline + gate outcome at 2028-04-06."""
import json, math
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_account_dict

import importlib.util
spec = importlib.util.spec_from_file_location("strat", "strategy.py")
strat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(strat)

assets = list(get_account_dict()["watch_list"])
frames = {a: strat.get_df(a) for a in assets}
close = {a: strat.series(frames[a]) for a in assets}
open_ = {a: strat.series(frames[a], "open") for a in assets}
ret = {a: close[a].pct_change() for a in assets}
panel = pd.concat([ret[a].rename(a) for a in assets], axis=1, join="inner").dropna()

frozen = strat.detect_frozen(close)
live = [a for a in assets if a not in frozen]
print("frozen:", sorted(frozen))
print("live:", sorted(live))

# ---- replicate regime posture ----
lp = panel[live]
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

vix = strat.series(strat.get_df("VIX"))
vix_level = float(vix.iloc[-1]) if vix is not None and len(vix) else None
eq_live = [a for a in strat.EQ_ASSETS if a in live]
eq_ret21 = float(np.mean([close[a].iloc[-1] / close[a].iloc[-22] - 1.0 for a in eq_live])) if eq_live else 0.0
stress = risk_off and ((vix_level is not None and vix_level >= strat.VIX_STRESS) or eq_ret21 < strat.EQ_RET21_STRESS)
print(f"risk_off={risk_off} risk_on={risk_on} stress={stress} VIX={vix_level:.1f} eq_ret21={eq_ret21*100:.2f}%")

# ---- factor signals & score ----
ens = strat.load_ensemble()
ens_ids = {fid for fid, _, _ in ens}
r_spx = ret["SPX"]
r_300 = ret["000300.SH"]
d_cn = close["CN10Y"].pct_change()
dxy = strat.series(strat.get_df("DXY"))
r_dxy = dxy.pct_change() if dxy is not None else None
comm_basket = panel[["XAU", "COPPER", "WTI"]].mean(axis=1)

sig = {fid: {} for fid in ens_ids}
for a in assets:
    c, o, r = close[a], open_[a], ret[a]
    if "down_beta_60" in ens_ids: sig["down_beta_60"][a] = strat.down_beta(r, r_spx)
    if "spx_beta_60" in ens_ids: sig["spx_beta_60"][a] = strat.beta_last(r, r_spx)
    if "hs300_beta_60" in ens_ids: sig["hs300_beta_60"][a] = strat.beta_last(r, r_300)
    if "cn10y_beta_60" in ens_ids: sig["cn10y_beta_60"][a] = strat.beta_last(r, d_cn)
    if "vol_adj_mom_20_60" in ens_ids:
        sig["vol_adj_mom_20_60"][a] = (c.iloc[-6] / c.iloc[-26] - 1.0) / max(float(r.tail(60).std()), 1e-6) if len(c) >= 30 else None
    if "dxy_beta_cond_60x20" in ens_ids:
        if r_dxy is not None:
            b = strat.beta_last(r, r_dxy)
            sig["dxy_beta_cond_60x20"][a] = b * (dxy.iloc[-1] / dxy.iloc[-21] - 1.0) if b is not None else None
        else: sig["dxy_beta_cond_60x20"][a] = None
    if "hilo_vol_ratio_20" in ens_ids:
        if len(c) >= 25:
            rng = (c.rolling(20).max() - c.rolling(20).min()) / c
            rv = r.rolling(20).std()
            q = (rng / rv).dropna()
            sig["hilo_vol_ratio_20"][a] = float(q.iloc[-1]) if len(q) else None
        else: sig["hilo_vol_ratio_20"][a] = None
    if "intraday_ret_skew_20" in ens_ids:
        if o is not None:
            ir = (c / o - 1.0).dropna().tail(20)
            sig["intraday_ret_skew_20"][a] = float(ir.skew()) if len(ir) >= 5 else None
        else: sig["intraday_ret_skew_20"][a] = None
    if "comm_basket_beta_60" in ens_ids: sig["comm_basket_beta_60"][a] = strat.beta_last(r, comm_basket)
    if "vol_of_vol20x60" in ens_ids:
        rv20 = r.rolling(20).std()
        sig["vol_of_vol20x60"][a] = float(rv20.tail(60).std()) if len(rv20.dropna()) >= 40 else None

score = {a: 0.0 for a in assets}
for fid, w, d in ens:
    rk = strat.cs_rank(sig.get(fid, {}), assets)
    for a in assets:
        score[a] += w * d * rk[a]

# ---- weights ----
weights = strat.build_weights(score, assets, panel, def_floor, spread)
weights = strat.apply_frozen_override(weights, assets, frozen)
weights = strat.risk_trim(weights, assets, live, stress)

print("\nFINAL TARGET:")
for a in sorted(assets, key=lambda x: -weights[x]):
    print("  %-10s w=% .4f" % (a, weights[a]))
eqw = sum(weights[a] for a in strat.EQ_ASSETS)
print("eq total:", round(eqw, 4), "| sum:", round(sum(weights.values()), 6))

# ---- gate check with two forecast schemes ----
exec_w = json.load(open("../persistent/account.json"))["last_executed_target_weights"]
cur = {a: float(exec_w.get(a, 0.0)) for a in assets}
dw = {a: weights[a] - cur[a] for a in assets}
one_way_turnover = sum(abs(dw[a]) for a in assets) / 2.0
thresh_bps = one_way_turnover * 3.0

# scheme A: current z-score forecast
vals = np.array([score[a] for a in assets], dtype=float)
mu, sd = float(vals.mean()), float(vals.std())
scale = float(lp.tail(252).std(axis=1, ddof=0).median()) if len(lp) >= 30 else 0.01
if not math.isfinite(scale) or scale <= 0:
    scale = 0.01
fA = {a: ((score[a] - mu) / sd) * scale if sd > 1e-12 else 0.0 for a in assets}
for a in frozen:
    fA[a] = 0.0
edgeA = 10000.0 * sum(fA[a] * dw[a] for a in assets)

# scheme B: implied alpha from final target (k = 2*scale)
k = 2.0 * scale
eqw0 = 1.0 / len(assets)
fB = {a: k * (weights[a] - eqw0) for a in assets}
edgeB = 10000.0 * sum(fB[a] * dw[a] for a in assets)

print("\nscale=%.5f k=%.5f one_way_turnover=%.4f thresh_bps=%.3f" % (scale, k, one_way_turnover, thresh_bps))
print("scheme A (z-score): gross_edge_bps=%+.2f -> %s" % (edgeA, "EXECUTE" if edgeA > thresh_bps else "SKIP"))
print("scheme B (implied alpha): gross_edge_bps=%+.2f -> %s" % (edgeB, "EXECUTE" if edgeB > thresh_bps else "SKIP"))
print("\nforecasts B per asset:")
for a in sorted(assets, key=lambda x: -fB[x]):
    print("  %-10s w=% .4f fcast=%+.3f%% dw=%+.4f" % (a, weights[a], fB[a]*100, dw[a]))
