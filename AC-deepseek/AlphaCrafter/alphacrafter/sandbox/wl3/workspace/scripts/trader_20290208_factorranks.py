"""Inspect per-factor ranks for WTI (and others) at 2029-02-08 block start."""
import sys, json
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_index_daily_data, get_stock_daily_data

OBS = {"DXY", "VIX", "USDCNY", "USDJPY", "EURUSD"}

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

acct = get_account_dict()
assets = list(acct["watch_list"])
frames = {a: get_df(a) for a in assets}
close = {a: series(frames[a]) for a in assets}
open_ = {a: series(frames[a], "open") for a in assets}
ret = {a: close[a].pct_change() for a in assets}
panel = pd.concat([ret[a].rename(a) for a in assets], axis=1, join="inner").dropna()

ens = json.load(open("factor_ensemble.json"))["selected_factors"]
ens = [(s["factor_id"], float(s["weight"]), int(s["direction"])) for s in ens]
ens_ids = {fid for fid, _, _ in ens}

r_spx = ret["SPX"]; r_300 = ret["000300.SH"]
d_cn = close["CN10Y"].pct_change()
dxy = series(get_df("DXY")); r_dxy = dxy.pct_change() if dxy is not None else None
vix = series(get_df("VIX")); r_vix = vix.pct_change() if vix is not None else None
comm_basket = panel[["XAU", "COPPER", "WTI"]].mean(axis=1)

sig = {fid: {} for fid in ens_ids}
for a in assets:
    c, o, r = close[a], open_[a], ret[a]
    if "hs300_beta_60" in ens_ids: sig["hs300_beta_60"][a] = beta_last(r, r_300)
    if "cn10y_beta_60" in ens_ids: sig["cn10y_beta_60"][a] = beta_last(r, d_cn)
    if "vol_adj_mom_20_60" in ens_ids:
        sig["vol_adj_mom_20_60"][a] = (c.iloc[-6]/c.iloc[-26]-1.0)/max(float(r.tail(60).std()),1e-6) if len(c)>=30 else None
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

print(f"{'asset':10s}", end="")
for fid, _, _ in ens:
    print(f"{fid[:14]:>16s}", end="")
print(f"{'COMPOSITE':>12s}")
for a in assets:
    print(f"{a:10s}", end="")
    comp = 0.0
    for fid, w, d in ens:
        rk = cs_rank(sig.get(fid, {}), assets)
        v = sig[fid].get(a)
        val = rk[a] if v is not None else float('nan')
        comp += w*d*rk[a]
        print(f"{val:16.2f}", end="")
    print(f"{comp:12.4f}")

print("\nraw signal values for WTI:")
for fid, _, _ in ens:
    print(f"  {fid}: {sig[fid].get('WTI')}")
print("\nraw signal values for XAU:")
for fid, _, _ in ens:
    print(f"  {fid}: {sig[fid].get('XAU')}")
