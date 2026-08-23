"""miner_3 (2026-08-20): Sweep P - orthogonal residual/serial dimensions.

Library heavily covers: momentum, vol, VIX/USDCNY/DXY beta, range/position.
Target fresh dimensions with likely LOW library correlation:
  - ret_autocorr_5_20 : 20d window, lag-5 daily-return autocorrelation
  - idio_mom_30    : idiosyncratic 5d mean return residual vs SPX beta over 30d
  - up_down_ratio20: upside/downside return capture ratio (asymmetry)
  - price_vol_div  : EWM price momentum minus EWM volume momentum (price/volume divergence)
  - gap_reversion  : overnight gap (open vs prev close)
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

def ret_autocorr(asset_r, n=20, lag=5):
    def ac(x):
        x = x.dropna()
        return x.autocorr(lag=lag) if len(x) > n//2 else np.nan
    return asset_r.rolling(n, min_periods=12).apply(ac, raw=False)

def load_ohlc():
    out = {}
    for a in ASSETS:
        df = pd.read_csv(f"../persistent/stock_data/{a}.csv", parse_dates=["date"]).sort_values("date")
        df = df[df["date"] <= "2026-07-29"]
        out[a] = df.set_index("date")
    return out

ohlc = load_ohlc()

def beta_(r, mkt, w=30):
    df = pd.concat([r.rename("a"), mkt.rename("m")], axis=1)
    out = []
    for i in range(len(df)):
        if i < w-1:
            out.append(np.nan); continue
        sub = df.iloc[i-w+1:i+1]
        m = sub["m"].to_numpy(); a = sub["a"].to_numpy()
        fm = np.isfinite(m) & np.isfinite(a)
        if fm.sum() < 15 or (np.nanstd(m) == 0):
            out.append(np.nan); continue
        out.append(np.polyfit(m[fm], a[fm], 1)[0])
    return pd.Series(out, index=df.index)

mkt = rt("SPX")

def idio_mom(close, mkt_r, w=30, lag=5):
    r = close.pct_change()
    b = beta_(r, mkt_r, w)
    resid = r - b * mkt_r
    return resid.rolling(lag).mean()

def up_down_ratio(close, n=20):
    r = close.pct_change()
    up = r.where(r > 0).rolling(n, min_periods=10).mean()
    dn = (-r).where(r < 0).rolling(n, min_periods=10).mean()
    return up / dn.replace(0, np.nan)

def price_vol_div(close, vol_series, span=15, lag=3):
    r = close.pct_change()
    vr = vol_series.pct_change()
    ep = r.ewm(span=span, adjust=False).mean()
    ev = vr.ewm(span=span, adjust=False).mean()
    return ep - ev

vol = {a: ohlc[a]["volume"].astype(float) for a in ASSETS}

candidates = {
    "ret_autocorr_5_20": {a: ret_autocorr(rt(a), 20, 5) for a in closes},
    "idio_mom_30": {a: idio_mom(closes[a], mkt, 30, 5) for a in closes},
    "up_down_ratio20": {a: up_down_ratio(closes[a], 20) for a in closes},
    "price_vol_div15": {a: price_vol_div(closes[a], vol[a], 15, 3) for a in closes},
}

print("assets:", len(closes), "macro:", len(macro))
for name, vals in candidates.items():
    try:
        evaluate(closes, vals, name, horizon=10)
    except Exception as e:
        print(name, "ERROR:", repr(e))
    print()