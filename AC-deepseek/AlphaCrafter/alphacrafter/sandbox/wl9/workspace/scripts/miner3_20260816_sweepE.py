"""Exploration sweep E (miner_3, 2026-08-16): liquidity/risk interactions.

vol_z_20d passed and was persisted. Next distinct families:
1) vpc_20d: rolling corr(asset return, volume pct-change) - volume-price concordance
2) vol_pressure_20d: mean(volume on up days) / mean(volume on down days)
3) riskadj_mom_10x20: mom_10d_skip5 / realized vol 20d (Sharpe-like momentum)
4) drawdown_252: 1 - close/rolling_max(close,252) (distance from 52w high)
5) volz_x_skew = vol_z_20d * sign(skew_20d) (conditional volume expansion)
6) btc_beta_60: rolling beta of each asset to BTC returns (risk-on beta)

Gate: abs(IC)>=0.0070 & abs(ICIR)>=0.0840 at h=10 on 15-asset universe;
persistence additionally requires max_abs_library_correlation < 0.5.
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner3_20260730_harness import ASSETS, evaluate, load_closes  # noqa: E402

VOL_ASSETS = ["000300.SH", "000688.SH", "SPX", "HSI", "N225", "SX5E", "NDX", "BTC", "ETH"]
closes = load_closes()


def load_ohlc():
    out = {}
    for a in ASSETS:
        df = pd.read_csv(f"../persistent/stock_data/{a}.csv", parse_dates=["date"]).sort_values("date")
        out[a] = df.set_index("date")
    return out


ohlc = load_ohlc()
vols = {a: ohlc[a]["volume"].astype(float) for a in VOL_ASSETS}


def vpc_20(close, volume, n=20):
    r = close.pct_change()
    dv = volume.pct_change()
    return r.rolling(n).corr(dv)


def vol_pressure_20(close, volume, n=20):
    r = close.pct_change()
    up = volume.where(r > 0).rolling(n).mean()
    dn = volume.where(r < 0).rolling(n).mean()
    return up / dn.replace(0, np.nan)


def riskadj_mom(close, n=10, skip=5, v=20):
    mom = close.shift(skip) / close.shift(skip + n) - 1.0
    rv = close.pct_change().rolling(v).std(ddof=0).replace(0, np.nan)
    return mom / rv


def drawdown_252(close, n=252):
    return 1.0 - close / close.rolling(n).max()


def volz_x_skew(close, volume, n=20):
    mu = volume.rolling(n).mean()
    sd = volume.rolling(n).std(ddof=0).replace(0, np.nan)
    vz = (volume - mu) / sd
    r = close.pct_change()
    m = r.rolling(n).mean()
    s = r.rolling(n).std(ddof=0).replace(0, np.nan)
    skew = ((r - m) ** 3).rolling(n).mean() / (s ** 3)
    return vz * np.sign(skew)


def beta_btc_60(close, btc, n=60):
    ra = close.pct_change()
    rb = btc.pct_change()
    cov = ra.rolling(n).cov(rb)
    var = rb.rolling(n).var(ddof=0).replace(0, np.nan)
    return cov / var


btc = closes["BTC"]

candidates = {
    "vpc_20d": {a: vpc_20(closes[a], vols[a], 20) if a in vols else pd.Series(np.nan, index=closes[a].index) for a in closes},
    "vol_pressure_20d": {a: vol_pressure_20(closes[a], vols[a], 20) if a in vols else pd.Series(np.nan, index=closes[a].index) for a in closes},
    "riskadj_mom_10x20": {a: riskadj_mom(closes[a]) for a in closes},
    "drawdown_252": {a: drawdown_252(closes[a], 252) for a in closes},
    "volz_x_skew_20d": {a: volz_x_skew(closes[a], vols[a], 20) if a in vols else pd.Series(np.nan, index=closes[a].index) for a in closes},
    "btc_beta_60": {a: beta_btc_60(closes[a], btc, 60) for a in closes},
}

for name, vals in candidates.items():
    try:
        evaluate(closes, vals, name, horizon=10)
    except Exception as e:
        print(name, "ERROR:", repr(e))
    print()