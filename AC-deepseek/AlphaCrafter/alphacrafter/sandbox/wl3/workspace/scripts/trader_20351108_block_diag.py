"""Trader diagnostic: reconstruct 2035-11-08 block-start proposal and attribution.

Uses data truncated to 2035-11-07 (visible at the 11-08 decision) for the
target, and full data for per-asset block returns (11-07 close -> 11-21 close).
"""
import json, math
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

OBS_ONLY = {"DXY", "VIX", "USDCNY", "USDJPY", "EURUSD"}

def get_df(symbol, days=400):
    try:
        if symbol in OBS_ONLY:
            return get_index_daily_data(symbol, days=days)
        return get_stock_daily_data(symbol, days=days)
    except Exception:
        return None

def series(df, col="close"):
    if df is None or col not in df or len(df) < 40:
        return None
    s = df[col].astype(float)
    s.index = pd.to_datetime(df["date"])
    return s

def cs_rank(values, assets):
    valid = sorted((float(v), a) for a, v in values.items() if v is not None and np.isfinite(float(v)))
    out = {a: 0.5 for a in assets}
    n = max(1, len(valid) - 1)
    for i, (_, a) in enumerate(valid):
        out[a] = i / n
    return out

def beta_last(y, x, win=60, min_obs=20):
    q = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna().tail(win)
    if len(q) < min_obs:
        return None
    vx = float(q.x.var())
    if vx <= 1e-14:
        return None
    return float(q.y.cov(q.x) / vx)

def down_beta(y, x, win=60):
    q = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    q = q[q.x < 0].tail(win)
    if len(q) < 20:
        return None
    vx = float(q.x.var())
    if vx <= 1e-14:
        return None
    return float(q.y.cov(q.x) / vx)

def detect_frozen(close, lookback=120):
    out = set()
    for a, c in close.items():
        if c is None:
            continue
        q = c.dropna().tail(lookback)
        if len(q) >= 20 and q.nunique() <= 2:
            out.add(a)
    return out

def build_weights(score, assets, panel, def_floor, spread, cap=0.18):
    order = sorted(assets, key=lambda a: (-score[a], a))
    lin = {a: 1.0 - i / max(1, len(order) - 1) for i, a in enumerate(order)}
    vols = {a: max(float(panel[a].tail(20).std()), 0.003) for a in assets}
    vmed = float(np.median([vols[a] for a in assets]))
    pref = {a: (1.0 + spread * lin[a]) * math.sqrt(vmed / vols[a]) for a in assets}
    total = sum(max(0.0, float(x)) for x in pref.values())
    w = {a: max(0.0, float(pref[a])) / total for a in assets}
    for a in ("XAU", "US10Y", "CN10Y"):
        w[a] = max(w[a], def_floor)
    tot = sum(w.values())
    if tot > 0:
        w = {a: x / tot for a, x in w.items()}
    for _ in range(200):
        excess = sum(max(0.0, x - cap) for x in w.values())
        if excess < 1e-12:
            break
        w = {a: min(cap, x) for a, x in w.items()}
        room = [a for a, x in w.items() if x < cap - 1e-9]
        if not room:
            break
        p = {a: max(0.0, pref.get(a, 0.0)) for a in room}
        den = sum(p.values())
        if den <= 0:
            p = {a: 1.0 for a in room}
            den = len(room)
        for a in room:
            w[a] += excess * p[a] / den
    tot = sum(w.values())
    if tot <= 0:
        w = {a: 1.0 / len(assets) for a in assets}
    else:
        w = {a: x / tot for a, x in w.items()}
    w[assets[-1]] += 1.0 - sum(w.values())
    return {a: max(0.0, float(x)) for a, x in w.items()}

def apply_frozen_override(w, assets, frozen, cap=0.18, floor=0.005):
    if not frozen or len(frozen) >= len(assets) - 1:
        return w
    w = dict(w)
    live = [a for a in assets if a not in frozen]
    for a in frozen:
        w[a] = floor
    tot = sum(w.values())
    if tot <= 0:
        return {a: 1.0 / len(assets) for a in assets}
    w = {a: x / tot for a, x in w.items()}
    for _ in range(200):
        excess = sum(max(0.0, w[a] - cap) for a in live)
        if excess < 1e-12:
            break
        for a in live:
            w[a] = min(cap, w[a])
        room = [a for a in live if w[a] < cap - 1e-9]
        if not room:
            break
        p = {a: max(w[a], 1e-9) for a in room}
        den = sum(p.values())
        for a in room:
            w[a] += excess * p[a] / den
    tot = sum(w.values())
    if tot <= 0:
        return {a: 1.0 / len(assets) for a in assets}
    w = {a: x / tot for a, x in w.items()}
    w[assets[-1]] += 1.0 - sum(w.values())
    return {a: max(0.0, float(x)) for a, x in w.items()}

# ---- data as of 11-07 (truncate) ----
assets = list(get_account_dict()["watch_list"])
frames = {a: get_df(a) for a in assets}
close = {}
for a in assets:
    s = series(frames[a])
    s = s[s.index <= "2035-11-07"]
    close[a] = s
open_ = {a: series(frames[a], "open") for a in assets}
open_ = {a: s[s.index <= "2035-11-07"] for a, s in open_.items() if s is not None}

frozen = detect_frozen(close)
live = [a for a in assets if a not in frozen]
print("frozen:", sorted(frozen))
print("live:", sorted(live))

ret = {a: close[a].pct_change() for a in assets}
panel = pd.concat([ret[a].rename(a) for a in assets], axis=1, join="inner").dropna()
print("panel rows:", len(panel))

ens = json.load(open("factor_ensemble.json"))["selected_factors"]
ens = [(s["factor_id"], float(s["weight"]), int(s["direction"])) for s in ens]
ens_ids = {fid for fid, _, _ in ens}
print("ensemble:", [(f, w, d) for f, w, d in ens])

r_spx = ret["SPX"]
r_300 = ret["000300.SH"]
d_cn = close["CN10Y"].pct_change()
dxy = series(get_df("DXY")); dxy = dxy[dxy.index <= "2035-11-07"]
r_dxy = dxy.pct_change() if dxy is not None else None
vix = series(get_df("VIX")); vix = vix[vix.index <= "2035-11-07"]
comm_basket = panel[["XAU", "COPPER", "WTI"]].mean(axis=1)

sig = {fid: {} for fid in ens_ids}
for a in assets:
    c, o, r = close[a], open_.get(a), ret[a]
    if "down_beta_60" in ens_ids:
        sig["down_beta_60"][a] = down_beta(r, r_spx)
    if "spx_beta_60" in ens_ids:
        sig["spx_beta_60"][a] = beta_last(r, r_spx)
    if "hs300_beta_60" in ens_ids:
        sig["hs300_beta_60"][a] = beta_last(r, r_300)
    if "cn10y_beta_60" in ens_ids:
        sig["cn10y_beta_60"][a] = beta_last(r, d_cn)
    if "vol_adj_mom_20_60" in ens_ids:
        sig["vol_adj_mom_20_60"][a] = ((c.iloc[-6] / c.iloc[-26] - 1.0) / max(float(r.tail(60).std()), 1e-6)) if len(c) >= 30 else None
    if "dxy_beta_cond_60x20" in ens_ids:
        if r_dxy is not None:
            b = beta_last(r, r_dxy)
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
        sig["comm_basket_beta_60"][a] = beta_last(r, comm_basket)
    if "vol_of_vol20x60" in ens_ids:
        rv20 = r.rolling(20).std()
        sig["vol_of_vol20x60"][a] = float(rv20.tail(60).std()) if len(rv20.dropna()) >= 40 else None

score = {a: 0.0 for a in assets}
for fid, w, d in ens:
    rk = cs_rank(sig.get(fid, {}), assets)
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
print(f"regime: risk_off={risk_off} risk_on={risk_on} def_floor={def_floor} spread={spread} mkt20={mkt20:.4f} mdd20={mdd:.4f} vol20={vol20:.4f}")

w = build_weights(score, assets, panel, def_floor, spread)
w = apply_frozen_override(w, assets, frozen)
print("--- proposed target weights (11-08 decision) ---")
for a in assets:
    print(f"  {a}: {w[a]*100:.2f}%")
print("sum:", sum(w.values()))

# ---- block attribution: 11-07 close -> 11-21 close ----
print("--- block returns 11-07 -> 11-21 ---")
contrib = 0.0
for a in assets:
    s_full = series(frames[a])
    c0 = s_full[s_full.index <= "2035-11-07"].iloc[-1]
    c1 = s_full[s_full.index <= "2035-11-21"].iloc[-1]
    r = c1 / c0 - 1.0
    contrib += w[a] * r
    print(f"  {a}: w={w[a]*100:.2f}%  r={r*100:+.2f}%  contrib={w[a]*r*100:+.2f}pp")
print(f"approx block attribution: {contrib*100:+.2f}pp")
