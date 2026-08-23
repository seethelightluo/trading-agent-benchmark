""" miner_3 (2026-08-16): Sweep H - volatility-regime/quality/semi-vol trend family.
Current library has mom/skew/rng_pos/vol_z/days_since_high/beta_VIX/vix_beta_cond.
Try to find NEW low-correlation alpha."""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, "scripts")
from miner3_20260730_harness import load_closes, evaluate

closes = load_closes()

def semi_vol_ratio(close, n=60, minp=20):
    r = close.pct_change()
    neg = r.where(r < 0, 0.0)
    down = np.sqrt((neg**2).rolling(n, min_periods=minp).mean())
    tot = r.rolling(n, min_periods=minp).std(ddof=0).replace(0, np.nan)
    return down / tot

def upside_downside_ratio(close, n=60, minp=20):
    r = close.pct_change()
    pos = r.where(r > 0, 0.0)
    up = np.sqrt((pos**2).rolling(n, min_periods=minp).mean())
    dn = np.sqrt((r.where(r<0,0.0)**2).rolling(n, min_periods=minp).mean()).replace(0,np.nan)
    return up/dn

def max_dd_60(close, n=60):
    rollmax = close.rolling(n, min_periods=n).max()
    dd = close/rollmax.replace(0,np.nan)-1.0
    m = dd.rolling(n, min_periods=n).min()
    return m   # negative depth to max

def trend_r2_signed(close, n=60):
    x = np.arange(n)
    out = pd.Series(np.nan, index=close.index)
    c = close
    for i in range(n-1, len(c)):
        y = c.iloc[i-n+1:i+1].values
        if np.any(~np.isfinite(y)) or np.std(y)==0:
            continue
        slope = np.polyfit(x, y, 1)[0]
        yhat = np.polyval(np.polyfit(x,y,1), x)
        r2 = 1 - np.var(y-yhat)/np.var(y)
        out.iloc[i] = r2*np.sign(slope)
    return out

def vol_cluster_mom(close, n_vol=60, n_ret=20, skip=3):
    vol = close.pct_change().rolling(n_vol, min_periods=20).std(ddof=0)
    mean = vol.rolling(252, min_periods=60).mean()
    std = vol.rolling(252,min_periods=60).std(ddof=0).replace(0,np.nan)
    volz = (vol - mean)/std
    lowq = (volz < -0.5).astype(float)
    mom = close/close.shift(n_ret+skip)-1.0
    return mom*lowq  # momentum, active only when in low-vol cluster

candidates = {
    "semi_vol_ratio_60d": {a: semi_vol_ratio(closes[a],60) for a in closes},
    "upside_downside_60d": {a: upside_downside_ratio(closes[a],60) for a in closes},
    "max_dd_60d": {a: max_dd_60(closes[a],60) for a in closes},
    "trend_r2_60_signed": {a: trend_r2_signed(closes[a],60) for a in closes},
    "vol_cluster_mom_20x60": {a: vol_cluster_mom(closes[a],60,20,3) for a in closes},
}
for name, vals in candidates.items():
    try:
        evaluate(closes, vals, name, horizon=10)
    except Exception as e:
        print(name, "ERROR:", repr(e))
    print()