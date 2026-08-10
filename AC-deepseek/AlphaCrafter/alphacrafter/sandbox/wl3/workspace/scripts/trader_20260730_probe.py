"""Trader probe: compute the 10-factor ensemble signals at current date."""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from alphacrafter.sim.utils import (
    get_account_dict,
    get_stock_daily_data,
    get_index_daily_data,
)

N = 300  # lookback days

account = get_account_dict()
assets = list(account["watch_list"])

# ---- data ----
frames = {}
for a in assets:
    df = get_stock_daily_data(symbol=a, days=N)
    frames[a] = df

closes = {a: df["close"].astype(float) for a, df in frames.items()}
opens = {a: df["open"].astype(float) for a, df in frames.items()}
rets = {a: closes[a].pct_change() for a in assets}
panel = pd.concat([rets[a].rename(a) for a in assets], axis=1).dropna()

def beta_of(y, x, win):
    """rolling beta of y on x over last win observations (aligned)"""
    d = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna().tail(win)
    if len(d) < max(12, win // 2):
        return None
    var = float(d["x"].var())
    if var <= 1e-16:
        return None
    return float(d["y"].cov(d["x"]) / var)

def last_beta(y, x, win):
    d = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    if len(d) < max(12, win // 2):
        return None
    q = d.tail(win)
    var = float(q["x"].var())
    if var <= 1e-16:
        return None
    return float(q["y"].cov(q["x"]) / var)

spx_ret = rets["SPX"]
hs300_ret = rets["000300.SH"]
cn10y_diff = closes["CN10Y"].diff()
vix = get_index_daily_data("VIX", days=N)
dxy = get_index_daily_data("DXY", days=N)
vix_ret = vix["close"].astype(float).pct_change() if vix is not None else None
dxy_ret = dxy["close"].astype(float).pct_change() if dxy is not None else None

# align observation-only series to panel index
if vix_ret is not None:
    vix_ret = vix_ret.reindex(panel.index)
if dxy_ret is not None:
    dxy_ret = dxy_ret.reindex(panel.index)
    dxy_lvl = dxy["close"].astype(float).reindex(panel.index)
if cn10y_diff is not None:
    cn10y_diff = cn10y_diff.reindex(panel.index)

# ---- 1. down_beta_60 : beta on SPX-down days only (60 down days) ----
down_mask = (spx_ret < 0).reindex(panel.index)
down_idx = panel.index[down_mask]
signals = {}
def down_beta(a):
    sub = panel.loc[down_idx, [a, "SPX"]].dropna().tail(60)
    if len(sub) < 15:
        return None
    var = float(sub["SPX"].var())
    if var <= 1e-16:
        return None
    return float(sub[a].cov(sub["SPX"]) / var)
signals["down_beta_60"] = {a: down_beta(a) for a in assets}

# ---- 2. cn10y_beta_60 : beta(ret, diff(CN10Y), 60) ----
signals["cn10y_beta_60"] = {
    a: last_beta(panel[a], cn10y_diff, 60) for a in assets
}

# ---- 3. spx_beta_60 ----
signals["spx_beta_60"] = {
    a: last_beta(panel[a], spx_ret, 60) for a in assets
}

# ---- 4. vol_adj_mom_20_60 : (mom20 skip5)/std60 ----
def vam(a):
    c = closes[a]
    if len(c) < 70:
        return None
    mom = float(c.iloc[-6] / c.iloc[-26] - 1.0)  # close[t-5]/close[t-25]-1
    vol = float(panel[a].tail(60).std())
    if vol <= 1e-12:
        return None
    return mom / vol
signals["vol_adj_mom_20_60"] = {a: vam(a) for a in assets}

# ---- 5. dxy_beta_cond_60x20 : beta(ret,DXY,60)*(DXY/DXY.shift(20)-1) ----
def dxy_beta(a):
    b = last_beta(panel[a], dxy_ret, 60)
    if b is None or dxy_lvl is None:
        return None
    m = float(dxy_lvl.iloc[-1] / dxy_lvl.iloc[-21] - 1.0)
    return b * m
signals["dxy_beta_cond_60x20"] = {a: dxy_beta(a) for a in assets}

# ---- 6. hs300_beta_60 ----
signals["hs300_beta_60"] = {
    a: last_beta(panel[a], hs300_ret, 60) for a in assets
}

# ---- 7. intraday_ret_skew_20 : skew(close/open-1,20) ----
def skew20(a):
    o, c = opens[a], closes[a]
    d = (c / o - 1.0).dropna().tail(20)
    if len(d) < 10 or float(d.std()) <= 1e-16:
        return None
    return float(d.skew())
signals["intraday_ret_skew_20"] = {a: skew20(a) for a in assets}

# ---- 8. vol_of_vol20x60 : std(rolling std(ret,20),60) ----
def vov(a):
    rv = panel[a].rolling(20).std()
    s = rv.tail(60)
    if len(s.dropna()) < 30:
        return None
    return float(s.std())
signals["vol_of_vol20x60"] = {a: vov(a) for a in assets}

# ---- 9. dd_duration_120_resid : log1p(days since 120d high) resid vs mom120 z ----
def dd_duration(a):
    c = closes[a]
    if len(c) < 130:
        return None
    window = c.tail(120)
    roll_max = window.cummax()
    # days since last 120d high
    days = 0
    for i in range(len(window) - 1, -1, -1):
        if window.iloc[i] >= roll_max.iloc[i] - 1e-12:
            days = len(window) - 1 - i
            break
    return math.log1p(days)

def mom120(a):
    c = closes[a]
    if len(c) < 130:
        return None
    return float(c.iloc[-6] / c.iloc[-126] - 1.0)

y = {a: dd_duration(a) for a in assets}
z = {a: mom120(a) for a in assets}
valid = [a for a in assets if y[a] is not None and z[a] is not None]
if valid:
    zvals = np.array([z[a] for a in valid])
    zs = (zvals - zvals.mean()) / (zvals.std() + 1e-12)
    yvals = np.array([y[a] for a in valid])
    beta = float(np.dot(yvals - yvals.mean(), zs)) / float(np.dot(zs, zs) + 1e-12)
    signals["dd_duration_120_resid"] = {
        a: (y[a] - beta * zs[valid.index(a)] if a in valid else None) for a in assets
    }
else:
    signals["dd_duration_120_resid"] = {a: None for a in assets}

# ---- 10. vix_beta_cond_60x20 : -beta(ret,VIX,60)*(VIX/VIX.shift(20)-1) ----
def vix_beta(a):
    b = last_beta(panel[a], vix_ret, 60)
    if b is None or vix_ret is None:
        return None
    vix_lvl = vix["close"].astype(float).reindex(panel.index)
    m = float(vix_lvl.iloc[-1] / vix_lvl.iloc[-21] - 1.0)
    return -b * m
signals["vix_beta_cond_60x20"] = {a: vix_beta(a) for a in assets}

# ---- combine with ensemble ----
ens = json.load(open(Path(__file__).parent.parent / "factors" / "factor_ensemble.json"))
sel = ens["selected_factors"]

def ranks(d):
    valid = sorted((float(v), a) for a, v in d.items() if v is not None and math.isfinite(float(v)))
    out = {a: np.nan for a in assets}
    for i, (_, a) in enumerate(valid):
        out[a] = i / max(1, len(valid) - 1)
    return out

score = {a: 0.0 for a in assets}
print("=" * 100)
for f in sel:
    fid, w, d = f["factor_id"], float(f["weight"]), int(f["direction"])
    r = ranks(signals[fid])
    n_valid = sum(1 for v in r.values() if not np.isnan(v))
    for a in assets:
        if not np.isnan(r[a]):
            score[a] += w * d * r[a]
    print(f"{fid:26s} w={w:.4f} dir={d:+d} valid={n_valid}/15")
    print("   ", {a: round(float(signals[fid][a]), 4) if signals[fid][a] is not None else None for a in assets})

print("=" * 100)
print("COMPOSITE SCORE (higher=preferred):")
for a in sorted(assets, key=lambda x: -score[x]):
    print(f"  {a:10s} {score[a]:+.4f}")
