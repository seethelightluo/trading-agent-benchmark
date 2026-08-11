"""Trader probe 2026-10-22: regime assessment + proposed target (no execution)."""
import json
import sys
sys.path.insert(0, ".")
import strategy as st

from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

acc = get_account_dict()
assets = list(acc["watch_list"])
frames = {a: st.get_df(a) for a in assets}
close = {a: st.series(frames[a]) for a in assets}
ret = {a: close[a].pct_change() for a in assets}
panel = pd = __import__("pandas").concat(
    [ret[a].rename(a) for a in assets], axis=1, join="inner").dropna()

market = panel.mean(axis=1)
wealth = (1.0 + market).cumprod()
mdd = float((wealth / wealth.rolling(60).max() - 1.0).tail(20).min())
mkt20 = float(market.tail(20).mean())
mkt60 = float(market.tail(60).mean())
vol20 = float(panel.tail(20).std().mean())
vol_med = float(panel.tail(120).std().median(axis=0))
risk_off = (mkt20 < 0.0 and mdd < -0.03) or (vol20 > 1.3 * max(vol_med, 1e-6))
risk_on = mkt20 > 0.0 and mdd > -0.02
def_floor = 0.15 if risk_off else (0.10 if risk_on else 0.12)
spread = 2.0 if risk_off else (3.0 if risk_on else 2.5)

print(f"current_date visible_through check: last panel date = {panel.index[-1]}")
print(f"mkt20={mkt20*100:.2f}% mkt60={mkt60*100:.2f}% mdd20={mdd*100:.2f}% vol20={vol20*100:.2f}% vol_med={vol_med*100:.2f}%")
print(f"risk_off={risk_off} risk_on={risk_on} -> def_floor={def_floor} spread={spread}")

# per-asset 20d returns to see dispersion
r20 = {a: float((close[a].iloc[-1] / close[a].iloc[-21] - 1.0) * 100) for a in assets}
for a in sorted(assets, key=lambda x: -r20[x]):
    print(f"  {a}: 20d {r20[a]:+.2f}%")

# factor load
ens = st.load_ensemble()
print("ensemble:", [(f, round(w, 3), d) for f, w, d in ens])

# replicate score -> weights
sig = {fid: {} for fid, _, _ in ens}
r_spx = ret["SPX"]; r_300 = ret["000300.SH"]; d_cn = close["CN10Y"].pct_change()
dxy = st.series(st.get_df("DXY")); vix = st.series(st.get_df("VIX"))
r_dxy = dxy.pct_change(); r_vix = vix.pct_change()
open_ = {a: st.series(frames[a], "open") for a in assets}
for a in assets:
    c, o, r = close[a], open_[a], ret[a]
    sig["down_beta_60"][a] = st.down_beta(r, r_spx)
    sig["spx_beta_60"][a] = st.beta_last(r, r_spx)
    sig["hs300_beta_60"][a] = st.beta_last(r, r_300)
    sig["cn10y_beta_60"][a] = st.beta_last(r, d_cn)
    sig["vol_adj_mom_20_60"][a] = (c.iloc[-6]/c.iloc[-26]-1.0)/max(float(r.tail(60).std()), 1e-6) if len(c) >= 30 else None
    b = st.beta_last(r, r_dxy)
    sig["dxy_beta_cond_60x20"][a] = b * (dxy.iloc[-1]/dxy.iloc[-21]-1.0) if b is not None else None
    b = st.beta_last(r, r_vix)
    sig["vix_beta_cond_60x20"][a] = -b * (vix.iloc[-1]/vix.iloc[-21]-1.0) if b is not None else None
    ir = (c/o - 1.0).dropna().tail(20)
    sig["intraday_ret_skew_20"][a] = float(ir.skew()) if len(ir) >= 5 else None
    rv20 = r.rolling(20).std()
    sig["vol_of_vol20x60"][a] = float(rv20.tail(60).std()) if len(rv20.dropna()) >= 40 else None
    c120 = c.tail(126)
    prev_max = c120.shift(1).rolling(120, min_periods=60).max()
    hit = (c120 > prev_max) & prev_max.notna()
    idx_hit = list(__import__("numpy").where(hit.values)[0])
    days = (len(c120)-1-idx_hit[-1]) if idx_hit else 120
    sig["dd_duration_120_resid"][a] = __import__("math").log1p(max(0, days))

score = {a: 0.0 for a in assets}
for fid, w, d in ens:
    rk = st.cs_rank(sig.get(fid, {}), assets)
    for a in assets:
        score[a] += w * d * rk[a]

w = st.build_weights(score, assets, panel, def_floor, spread)
tot = sum(w.values())
print(f"sum(w)={tot:.6f}")
for a in sorted(assets, key=lambda x: -w[x]):
    print(f"  {a}: {w[a]*100:.2f}%")
