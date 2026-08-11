"""Trader dry-run probe for block start 2026-08-27.
Computes factor signals, composite score, regime posture, and target weights
exactly as strategy.py would, WITHOUT submitting orders."""
import json
import sys
sys.path.insert(0, ".")
import strategy as strat
from alphacrafter.sim.utils import get_account_dict

assets = list(get_account_dict()["watch_list"])
frames = {a: strat.get_df(a) for a in assets}
close = {a: strat.series(frames[a]) for a in assets}
open_ = {a: strat.series(frames[a], "open") for a in assets}
ret = {a: close[a].pct_change() for a in assets}
panel = strat.pd.concat([ret[a].rename(a) for a in assets], axis=1, join="inner").dropna()

r_spx = ret["SPX"]
r_300 = ret["000300.SH"]
d_cn = close["CN10Y"].pct_change()
dxy = strat.series(strat.get_df("DXY"))
vix = strat.series(strat.get_df("VIX"))
r_dxy = dxy.pct_change() if dxy is not None else None
r_vix = vix.pct_change() if vix is not None else None

ens = strat.load_ensemble()
print("ensemble n:", len(ens))
for fid, w, d in ens:
    print("  %-24s w=%7.4f dir=%+d" % (fid, w, d))

sig = {fid: {} for fid, _, _ in ens}
for a in assets:
    c, o, r = close[a], open_[a], ret[a]
    sig["down_beta_60"][a] = strat.down_beta(r, r_spx)
    sig["spx_beta_60"][a] = strat.beta_last(r, r_spx)
    sig["hs300_beta_60"][a] = strat.beta_last(r, r_300)
    sig["cn10y_beta_60"][a] = strat.beta_last(r, d_cn)
    sig["vol_adj_mom_20_60"][a] = (c.iloc[-6]/c.iloc[-26]-1.0)/max(float(r.tail(60).std()),1e-6) if len(c)>=30 else None
    if r_dxy is not None:
        b = strat.beta_last(r, r_dxy)
        sig["dxy_beta_cond_60x20"][a] = b*(dxy.iloc[-1]/dxy.iloc[-21]-1.0) if b is not None else None
    else:
        sig["dxy_beta_cond_60x20"][a] = None
    if r_vix is not None:
        b = strat.beta_last(r, r_vix)
        sig["vix_beta_cond_60x20"][a] = -b*(vix.iloc[-1]/vix.iloc[-21]-1.0) if b is not None else None
    else:
        sig["vix_beta_cond_60x20"][a] = None
    if o is not None:
        ir = (c/o-1.0).dropna().tail(20)
        sig["intraday_ret_skew_20"][a] = float(ir.skew()) if len(ir)>=5 else None
    else:
        sig["intraday_ret_skew_20"][a] = None
    rv20 = r.rolling(20).std()
    sig["vol_of_vol20x60"][a] = float(rv20.tail(60).std()) if len(rv20.dropna())>=40 else None
    c120 = c.tail(126); prev_max = c120.shift(1).rolling(120, min_periods=60).max()
    hit = (c120 > prev_max) & prev_max.notna(); idx_hit = strat.np.where(hit.values)[0]
    days = (len(c120)-1-idx_hit[-1]) if len(idx_hit) else 120
    sig["dd_duration_120_resid"][a] = strat.math.log1p(max(0, days))

mom_vals = {a: (close[a].iloc[-6]/close[a].iloc[-121]-1.0 if len(close[a])>=122 else 0.0) for a in assets}
ys = strat.np.array([sig["dd_duration_120_resid"][a] for a in assets], dtype=float)
xs = strat.np.array([mom_vals[a] for a in assets], dtype=float)
xm, ym = xs.mean(), ys.mean()
vx = float(strat.np.var(xs))
b_ = float(strat.np.cov(xs, ys)[0,1]/vx) if vx > 1e-14 else 0.0
for i, a in enumerate(assets):
    sig["dd_duration_120_resid"][a] = ys[i] - b_*(xs[i]-xm)

score = {a: 0.0 for a in assets}
for fid, w, d in ens:
    rk = strat.cs_rank(sig.get(fid, {}), assets)
    for a in assets:
        score[a] += w*d*rk[a]

market = panel.mean(axis=1)
wealth = (1.0+market).cumprod()
mdd = float((wealth/wealth.rolling(60).max()-1.0).tail(20).min())
mkt20 = float(market.tail(20).mean())
vol20 = float(panel.tail(20).std().mean())
vol_med = float(panel.tail(120).std().median(axis=0))
risk_off = (mkt20 < 0.0 and mdd < -0.03) or (vol20 > 1.3*max(vol_med,1e-6))
risk_on = mkt20 > 0.0 and mdd > -0.02
def_floor = 0.15 if risk_off else (0.10 if risk_on else 0.12)
spread = 2.0 if risk_off else (3.0 if risk_on else 2.5)
print("regime: mkt20=%.4f mdd=%.4f vol20=%.4f vol_med=%.4f risk_off=%s risk_on=%s def_floor=%.2f spread=%.1f" % (mkt20, mdd, vol20, vol_med, risk_off, risk_on, def_floor, spread))

weights = strat.build_weights(score, assets, panel, def_floor, spread)
print("score/weights top->bottom:")
for a in sorted(assets, key=lambda a: -score[a]):
    print("  %-10s score=%6.3f w=%6.3f" % (a, score[a], weights[a]))
print("sum weights:", round(sum(weights.values()), 6))
assert all(w >= 0 for w in weights.values())
assert abs(sum(weights.values()) - 1.0) < 1e-6
print("PROBE OK")
