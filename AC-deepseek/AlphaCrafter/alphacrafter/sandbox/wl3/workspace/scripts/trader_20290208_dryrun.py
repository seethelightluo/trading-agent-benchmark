"""Trader dry-run 2029-02-08: replicate strategy decision path, print target."""
import json
import math
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_index_daily_data, get_stock_daily_data

import importlib.util
spec = importlib.util.spec_from_file_location("strat", "strategy.py")
# Avoid importing the module (it imports rebalance_to_weights etc. fine, but
# we only need its pure functions). We'll copy the decision path instead.

OBS = {"DXY", "VIX", "USDCNY", "USDJPY", "EURUSD"}
DEF = {"XAU", "US10Y", "CN10Y"}
EQ_ASSETS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX"]
CAP = 0.18; EQ_CAP = 0.40; ETH_CAP = 0.06

def get_df(sym, days=260):
    try:
        if sym in OBS:
            return get_index_daily_data(sym, days=days)
        return get_stock_daily_data(sym, days=days)
    except Exception:
        return None

def series(df, col="close"):
    if df is None or col not in df or len(df) < 40:
        return None
    s = df[col].astype(float)
    try:
        s.index = pd.to_datetime(df["date"])
    except Exception:
        s.index = pd.RangeIndex(len(s))
    return s

def beta_last(y, x, win=60, min_obs=20):
    q = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna().tail(win)
    if len(q) < min_obs:
        return None
    vx = float(q.x.var())
    if vx <= 1e-14:
        return None
    return float(q.y.cov(q.x) / vx)

def dd_duration_resid(c, r, r_spx):
    try:
        hi = c.rolling(120).max()
        if isinstance(c.index, pd.DatetimeIndex):
            last_high = c.index.to_series().where(c == hi).ffill()
            dur = np.log1p((c.index - last_high).days.fillna(0).astype(float))
        else:
            pos = pd.Series(np.arange(len(c)), index=c.index)
            dur = np.log1p((pos - pos.where(c == hi).ffill()).fillna(0).astype(float))
        mom = c.shift(5) / c.shift(125) - 1.0
        zmom = (mom - mom.rolling(250).mean()) / mom.rolling(250).std()
        b = beta_last(r, r_spx)
        v = float(dur.iloc[-1]) - (b * float(zmom.iloc[-1]) if b is not None else 0.0)
        return v if np.isfinite(v) else None
    except Exception:
        return None

def cs_rank(values, assets):
    valid = sorted((float(v), a) for a, v in values.items() if v is not None and np.isfinite(float(v)))
    out = {a: 0.5 for a in assets}
    n = max(1, len(valid) - 1)
    for i, (_, a) in enumerate(valid):
        out[a] = i / n
    return out

def detect_frozen(close, lookback=120):
    out = set()
    for a, c in close.items():
        if c is None:
            continue
        q = c.dropna().tail(lookback)
        if len(q) >= 20 and q.nunique() <= 2:
            out.add(a)
    return out

acct = get_account_dict()
assets = list(acct["watch_list"])
frames = {a: get_df(a) for a in assets}
close = {a: series(frames[a]) for a in assets}
open_ = {a: series(frames[a], "open") for a in assets}
frozen = detect_frozen(close)
live = [a for a in assets if a not in frozen]
print("frozen:", sorted(frozen))
print("live:", sorted(live))

ret = {a: close[a].pct_change() for a in assets}
panel = pd.concat([ret[a].rename(a) for a in assets], axis=1, join="inner").dropna()

ens = json.load(open("factor_ensemble.json"))["selected_factors"]
ens = [(s["factor_id"], float(s["weight"]), int(s["direction"])) for s in ens]
ens_ids = {fid for fid, _, _ in ens}
print("ensemble:", [(f, w, d) for f, w, d in ens])

r_spx = ret["SPX"]; r_300 = ret["000300.SH"]
d_cn = close["CN10Y"].pct_change()
dxy = series(get_df("DXY")); r_dxy = dxy.pct_change() if dxy is not None else None
vix = series(get_df("VIX")); r_vix = vix.pct_change() if vix is not None else None
comm_basket = panel[["XAU", "COPPER", "WTI"]].mean(axis=1)

sig = {fid: {} for fid in ens_ids}
for a in assets:
    c, o, r = close[a], open_[a], ret[a]
    if "down_beta_60" in ens_ids: sig["down_beta_60"][a] = None
    if "spx_beta_60" in ens_ids: sig["spx_beta_60"][a] = None
    if "hs300_beta_60" in ens_ids: sig["hs300_beta_60"][a] = beta_last(r, r_300)
    if "cn10y_beta_60" in ens_ids: sig["cn10y_beta_60"][a] = beta_last(r, d_cn)
    if "vol_adj_mom_20_60" in ens_ids:
        sig["vol_adj_mom_20_60"][a] = (c.iloc[-6]/c.iloc[-26]-1.0)/max(float(r.tail(60).std()),1e-6) if len(c)>=30 else None
    if "dxy_beta_cond_60x20" in ens_ids: sig["dxy_beta_cond_60x20"][a] = None
    if "vix_beta_cond_60x20" in ens_ids:
        if r_vix is not None and len(vix) >= 25:
            b = beta_last(r, r_vix)
            sig["vix_beta_cond_60x20"][a] = b*(vix.iloc[-1]/vix.iloc[-21]-1.0) if b is not None else None
        else:
            sig["vix_beta_cond_60x20"][a] = None
    if "hilo_vol_ratio_20" in ens_ids:
        if len(c) >= 25:
            rng = (c.rolling(20).max()-c.rolling(20).min())/c
            rv = r.rolling(20).std()
            q = (rng/rv).dropna()
            sig["hilo_vol_ratio_20"][a] = float(q.iloc[-1]) if len(q) else None
        else:
            sig["hilo_vol_ratio_20"][a] = None
    if "intraday_ret_skew_20" in ens_ids:
        if o is not None:
            ir = (c/o-1.0).dropna().tail(20)
            sig["intraday_ret_skew_20"][a] = float(ir.skew()) if len(ir) >= 5 else None
        else:
            sig["intraday_ret_skew_20"][a] = None
    if "comm_basket_beta_60" in ens_ids: sig["comm_basket_beta_60"][a] = beta_last(r, comm_basket)
    if "vol_of_vol20x60" in ens_ids:
        rv20 = r.rolling(20).std()
        sig["vol_of_vol20x60"][a] = float(rv20.tail(60).std()) if len(rv20.dropna()) >= 40 else None
    if "vol_regime_switch_20x60" in ens_ids:
        rv20 = r.rolling(20).std()
        above = (rv20 > rv20.rolling(60).median()).astype(float)
        flips = above.diff().abs().rolling(60).mean().dropna()
        sig["vol_regime_switch_20x60"][a] = float(flips.iloc[-1]) if len(flips) else None
    if "dd_duration_120_resid" in ens_ids:
        sig["dd_duration_120_resid"][a] = dd_duration_resid(c, r, r_spx)

score = {a: 0.0 for a in assets}
for fid, w, d in ens:
    rk = cs_rank(sig.get(fid, {}), assets)
    for a in assets:
        score[a] += w * d * rk[a]

print("\n=== SCORES (desc) ===")
for a in sorted(score, key=lambda x: -score[x]):
    print(f"  {a}: {score[a]:+.4f}")

# regime
lp = panel[live] if live else panel
market = lp.mean(axis=1)
wealth = (1.0+market).cumprod()
mdd = float((wealth/wealth.rolling(60).max()-1.0).tail(20).min())
mkt20 = float(market.tail(20).mean())
vol20 = float(lp.tail(20).std().mean())
vol_med = float(lp.tail(120).std().median(axis=0))
risk_off = (mkt20 < 0.0 and mdd < -0.025) or (vol20 > 1.25*max(vol_med, 1e-6))
risk_on = mkt20 > 0.0 and mdd > -0.015
def_floor = 0.18 if risk_off else (0.11 if risk_on else 0.13)
spread = 2.0 if risk_off else (3.0 if risk_on else 2.0)
vix_level = float(vix.iloc[-1]) if vix is not None and len(vix) else None
eq_live = [a for a in EQ_ASSETS if a in live]
eq_ret21 = float(np.mean([close[a].iloc[-1]/close[a].iloc[-22]-1.0 for a in eq_live])) if eq_live else 0.0
stress = risk_off and ((vix_level is not None and vix_level >= 30.0) or eq_ret21 < -0.05)
print(f"\n=== REGIME ===\nmkt20={mkt20*100:+.3f}% mdd20={mdd*100:.2f}% vol20={vol20*100:.2f}% vol_med={vol_med*100:.2f}%")
print(f"risk_off={risk_off} risk_on={risk_on} def_floor={def_floor} spread={spread}")
print(f"VIX={vix_level:.1f} eq_ret21={eq_ret21*100:+.2f}% -> stress={stress}")

# build weights (copy of strategy logic)
from strategy import build_weights, apply_frozen_override, risk_trim
weights = build_weights(score, assets, panel, def_floor, spread)
weights = apply_frozen_override(weights, assets, frozen)
weights = risk_trim(weights, assets, live, stress)

print("\n=== TARGET WEIGHTS ===")
cur = {}
mv = {p["symbol"]: float(p.get("market_value", 0.0)) for p in acct.get("positions", [])}
nav = float(acct.get("net_assets", 0.0))
if nav > 0 and sum(mv.values()) > 0:
    cur = {a: mv.get(a, 0.0)/nav for a in assets}
for a in assets:
    print(f"  {a}: target={weights[a]*100:6.2f}%  current={cur.get(a,0)*100:6.2f}%  delta={ (weights[a]-cur.get(a,0))*100:+6.2f}pp")
print(f"  SUM: {sum(weights.values())*100:.4f}%")
eqw = sum(weights[a] for a in EQ_ASSETS)
eqc = sum(cur.get(a,0) for a in EQ_ASSETS)
print(f"  live+HSI/SX5E eq complex: target={eqw*100:.2f}% current={eqc*100:.2f}%")
print(f"  XAU={weights['XAU']*100:.2f}% COPPER={weights['COPPER']*100:.2f}% WTI={weights['WTI']*100:.2f}% ETH={weights['ETH']*100:.2f}%")

# gross edge proxy (turnover * k * ... ) -> just turnover and migration edge
turn = sum(abs(weights[a]-cur.get(a,0)) for a in assets)/2
print(f"\n  one-way turnover={turn*100:.2f}%  migrated notional={turn*nav:.0f}")
scale = float(lp.tail(252).std(axis=1, ddof=0).median()) if len(lp) >= 30 else 0.01
k = 2.0*scale
edge = sum((weights[a]-cur.get(a,0))*(k*(weights[a]-cur.get(a,0))) for a in assets)
print(f"  scale={scale*100:.3f}% k={k*100:.4f} gross_edge={edge*10000:.2f}bp 3bp_gate_cutoff={3*turn*10000:.2f}bp")
