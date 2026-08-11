"""Trader probe: verify the 4 ensemble factor computations on live data (2026-08-26)."""
from math import isfinite
import json
import pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

N_DAYS = 300
assets = list(get_account_dict()["watch_list"])

def stock(a, n=N_DAYS):
    try:
        return get_stock_daily_data(a, days=n)
    except Exception:
        return None

frames = {a: stock(a) for a in assets}
closes = {a: (f.close.astype(float) if f is not None and "close" in f and len(f) >= 140 else None)
          for a, f in frames.items()}
usable = [c.rename(a) for a, c in closes.items() if c is not None]
print("usable assets:", len(usable), "of", len(assets))
print("min len:", min(len(c) for c in usable), "max len:", max(len(c) for c in usable))
for a in assets:
    if closes.get(a) is None:
        print("MISSING:", a)

panel = pd.concat(usable, axis=1, join="inner")
print("panel shape:", panel.shape, "last date:", panel.index[-1])

vf = get_index_daily_data("VIX", days=N_DAYS)
vix_close = vf.close.astype(float) if vf is not None and "close" in vf else None
print("VIX rows:", None if vix_close is None else len(vix_close),
      "last:", None if vix_close is None else vix_close.index[-1])

raw = {fid: {} for fid in ["mom_10d_skip5", "mom_120d_skip5", "vix_beta_cond_60x20", "vol_of_vol20x60"]}
for a in assets:
    c = closes.get(a)
    if c is None:
        for fid in raw:
            raw[fid][a] = None
        continue
    ret = c.pct_change()
    s5, s15, s125 = c.shift(5), c.shift(15), c.shift(125)
    mom10 = (s5 / s15 - 1.0).iloc[-1]
    mom120 = (s5 / s125 - 1.0).iloc[-1]
    vov = ret.rolling(20).std().rolling(60).std().iloc[-1]
    vb = None
    if vix_close is not None:
        vix_ret = vix_close.pct_change()
        z = pd.concat([ret.rename("a"), vix_ret.rename("v")], axis=1).dropna().tail(60)
        var = float(z["v"].var())
        beta = float(z["a"].cov(z["v"]) / var) if len(z) >= 30 and var > 1e-14 else None
        vix_move = (vix_close / vix_close.shift(20) - 1.0).iloc[-1] if len(vix_close) > 21 else None
        vb = (-beta * vix_move) if (beta is not None and vix_move is not None and isfinite(vix_move)) else None
    raw["mom_10d_skip5"][a] = float(mom10) if isfinite(mom10) else None
    raw["mom_120d_skip5"][a] = float(mom120) if isfinite(mom120) else None
    raw["vol_of_vol20x60"][a] = float(vov) if isfinite(vov) else None
    raw["vix_beta_cond_60x20"][a] = vb

for fid, vals in raw.items():
    nv = sum(1 for v in vals.values() if v is not None)
    print(f"{fid}: n_valid={nv}  sample:", {a: round(v, 4) for a, v in list(vals.items())[:5] if v is not None})

def ranks(values, assets):
    valid = sorted((float(v), a) for a, v in values.items() if v is not None and isfinite(float(v)))
    out = {a: 0.5 for a in assets}
    for i, (_, a) in enumerate(valid):
        out[a] = i / max(1, len(valid) - 1)
    return out

ens = json.load(open("factor_ensemble.json"))
factors = ens["selected_factors"]
score = {a: 0.0 for a in assets}
for f in factors:
    fid, w, d = f["factor_id"], f.get("weight", 0.0), f.get("direction", 1)
    r = ranks(raw.get(fid, {}), assets)
    for a in assets:
        score[a] += (w * d) * (r[a] - 0.5)

print("\ncomposite score top/bottom:")
for a, s in sorted(score.items(), key=lambda kv: -kv[1]):
    print(f"  {a:10s} {s:+.4f}")
