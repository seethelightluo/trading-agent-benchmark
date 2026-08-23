"""miner_3 (2026-08-20): Sweep Q - cross-sectional & regime-conditional dimensions.

Library: momentum, vol, beta families, range/position, skew/kurt. Try:
  - r2_vspx_20   : rolling R2 of asset return vs SPX (systematic share)
  - cs_mom_rel20 : 20d momentum minus cross-asset median (idiosyncratic relative momentum)
  - mom_10_vixreg: momentum signed by VIX trend regime (risk-off flip)
  - pos_skew_20  : skewness of 20d range position series
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner3_20260730_harness import ASSETS, evaluate, load_closes, load_macro

closes = load_closes()
macro = load_macro()

def rt(a):
    return closes[a].pct_change()

mkt = rt("SPX")

def r2_vs_spx(asset_r, mkt_r, w=20, minp=15):
    df = pd.concat([asset_r.rename("a"), mkt_r.rename("m")], axis=1)
    out = []
    for i in range(len(df)):
        if i < w - 1:
            out.append(np.nan); continue
        sub = df.iloc[i-w+1:i+1]
        m = sub["m"].to_numpy(); a = sub["a"].to_numpy()
        fm = np.isfinite(m) & np.isfinite(a)
        if fm.sum() < minp or np.nanstd(m) == 0 or np.nanstd(a) == 0:
            out.append(np.nan); continue
        out.append(np.corrcoef(m[fm], a[fm])[0, 1] ** 2)
    return pd.Series(out, index=df.index)

cand = {"r2_vspx_20": {a: r2_vs_spx(rt(a), mkt, 20) for a in closes}}

# cross-sectional relative momentum
mom = {a: closes[a] / closes[a].shift(20) - 1.0 for a in closes}
mom_frame = pd.DataFrame(mom)
med = mom_frame.median(axis=1)
mom_cs = {a: (mom_frame[a] - med) for a in closes}
cand["cs_mom_rel20"] = mom_cs

# regime-conditional momentum signed by VIX trend
def vix_regime(n=10):
    v = macro["VIX"].reindex(closes["SPX"].index)
    return v.pct_change(n)

vr = vix_regime()
regime = {}
for a in closes:
    m10 = closes[a] / closes[a].shift(5) - 1.0
    signv = pd.Series(np.where(vr.shift(5).notna(), np.where(vr.shift(5) > 0, -1.0, 1.0), np.nan), index=vr.index)
    regime[a] = m10 * signv
cand["mom_10_vixreg"] = regime

# skewness of range position over 20d
def load_ohlc():
    out = {}
    for a in ASSETS:
        df = pd.read_csv(f"../persistent/stock_data/{a}.csv", parse_dates=["date"]).sort_values("date")
        df = df[df["date"] <= "2026-07-29"]
        out[a] = df.set_index("date")
    return out

ohlc = load_ohlc()
skew_pos = {}
for a in closes:
    hi = ohlc[a]["high"]; lo = ohlc[a]["low"]; cl = closes[a]
    rng = (hi - lo).replace(0, np.nan)
    pos = (cl - lo).div(rng)
    skew_pos[a] = pos.rolling(20, min_periods=15).skew()
cand["pos_skew_20"] = skew_pos

for name, vals in cand.items():
    try:
        evaluate(closes, vals, name, horizon=10)
    except Exception as e:
        print(name, "ERROR:", repr(e))
    print()