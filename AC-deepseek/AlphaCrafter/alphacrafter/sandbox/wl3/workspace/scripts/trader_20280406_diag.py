import json, math, numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

OBS = {"DXY", "VIX", "USDCNY", "USDJPY", "EURUSD"}

def get_df(sym, days=300):
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
    s.index = pd.to_datetime(df["date"])
    return s

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

def cs_rank(values, assets):
    valid = sorted((float(v), a) for a, v in values.items() if v is not None and np.isfinite(float(v)))
    out = {a: 0.5 for a in assets}
    n = max(1, len(valid) - 1)
    for i, (_, a) in enumerate(valid):
        out[a] = i / n
    return out

assets = list(get_account_dict()["watch_list"])
frames = {a: get_df(a) for a in assets}
close = {a: series(frames[a]) for a in assets}
open_ = {a: series(frames[a], "open") for a in assets}
ret = {a: close[a].pct_change() for a in assets}
panel = pd.concat([ret[a].rename(a) for a in assets], axis=1, join="inner").dropna()

ens = json.load(open("factor_ensemble.json"))["selected_factors"]
ens = [(s["factor_id"], float(s["weight"]), int(s["direction"])) for s in ens]
ens_ids = {f[0] for f in ens}
print("ensemble:", [f[0] for f in ens])

r_spx = ret["SPX"]
r_300 = ret["000300.SH"]
d_cn = close["CN10Y"].pct_change()
dxy = series(get_df("DXY"))
r_dxy = dxy.pct_change() if dxy is not None else None
comm_basket = panel[["XAU", "COPPER", "WTI"]].mean(axis=1)

sig = {fid: {} for fid in ens_ids}
for a in assets:
    c, o, r = close[a], open_[a], ret[a]
    if "down_beta_60" in ens_ids:
        sig["down_beta_60"][a] = down_beta(r, r_spx)
    if "spx_beta_60" in ens_ids:
        sig["spx_beta_60"][a] = beta_last(r, r_spx)
    if "hs300_beta_60" in ens_ids:
        sig["hs300_beta_60"][a] = beta_last(r, r_300)
    if "cn10y_beta_60" in ens_ids:
        sig["cn10y_beta_60"][a] = beta_last(r, d_cn)
    if "vol_adj_mom_20_60" in ens_ids:
        sig["vol_adj_mom_20_60"][a] = (
            (c.iloc[-6] / c.iloc[-26] - 1.0) / max(float(r.tail(60).std()), 1e-6)
            if len(c) >= 30 else None)
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

frozen = set()
for a, c in close.items():
    q = c.dropna().tail(120)
    if len(q) >= 20 and q.nunique() <= 2:
        frozen.add(a)
live = [a for a in assets if a not in frozen]

vals = np.array([score[a] for a in assets], dtype=float)
mu, sd = float(vals.mean()), float(vals.std())
scale = float(panel[live].tail(252).std(axis=1, ddof=0).median()) if len(panel) >= 30 else 0.01
forecast = {a: ((score[a] - mu) / sd) * scale if sd > 1e-12 else 0.0 for a in assets}
for a in frozen:
    forecast[a] = 0.0

print("\nscore / forecast per asset (sorted by score):")
for a in sorted(assets, key=lambda x: -score[x]):
    print("  %-10s score=% .4f forecast=%+.3f%%" % (a, score[a], forecast[a] * 100))

exec_w = json.load(open("../persistent/account.json"))["last_executed_target_weights"]
print("\ncurrent executed weights vs forecast (sorted by weight):")
for a in sorted(assets, key=lambda x: -exec_w.get(x, 0)):
    print("  %-10s w=% .4f  fcast=%+.3f%%  score=% .4f" % (a, exec_w.get(a, 0), forecast[a] * 100, score[a]))

# approximate gross edge from moving toward score-implied target
dw = {a: 0.0 for a in assets}
# proposal will trim eq to 40% and boost XAU/COPPER/WTI; approximate gross edge:
approx_prop = dict(exec_w)
# rough stress-trimmed guess from 03-23 style
print("\nfrozen:", sorted(frozen), "live:", sorted(live))
