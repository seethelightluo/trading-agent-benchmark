"""miner_3 (2026-08-19): Sweep N - fresh orthogonal dimensions.

Library covers: momentum (mom_10/120), vol (bb_width, vol_z, vol_of_vol evicted),
range/position (rng_pos), skew/kurt, VIX-beta family, days_since_high, kaufman_eff,
streak, dxy_corr_change, cny_beta. High-correlation collisions observed for RSI (0.88
with rng_pos) and beta/vol families.

We target dimensions NOT well covered and likely lower-correlated:
  - ewm_mom_20  : EWM-weighted momentum (exponential memory, smoother than linear)
  - mom_40_20   : intermediate 40d momentum skipping 5 (fresh horizon)
  - hi_lo_pos_10: 10d intraday high-low position (fresh window of rng_pos)
  - volume_liquidity_z : 20d vol z-score of volume (volume dimension genuinely new)
  - range_ac_5  : autocorrelation of 5d rolling range (vol persistence, fresh)
  - corr_spx_20 : rolling 20d correlation of asset return with SPX (fresh linkage
                  vs existing 60d VIX/USDCNY betas; uses shorter window + index)
Gate: abs(IC)>=0.0070 & abs(ICIR)>=0.0840 at h=10; persistence needs max lib corr <0.5.
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner3_20260730_harness import ASSETS, evaluate, load_closes, load_macro

closes = load_closes()
macro = load_macro()


def ewm_mom(close, span=20, lag=5):
    """EWM-weighted momentum: ratio of ewm price now vs ewm price 'lag' ago."""
    e = close.ewm(span=span, adjust=False).mean()
    return (e / e.shift(lag)) - 1.0


def hi_lo_pos(close, hi, lo, n=10):
    rng = (hi - lo).replace(0, np.nan)
    return (close - lo) / rng


def vol_zscore(vol, n=20, minp=10):
    m = vol.rolling(n, min_periods=minp).mean()
    s = vol.rolling(n, min_periods=minp).std(ddof=0).replace(0, np.nan)
    return (vol - m) / s


def range_autocorr(close, hi, lo, n=20, lag=1):
    rng = (hi - lo) / close
    def ac(x):
        x = x.dropna()
        return x.autocorr(lag=lag) if len(x) > 12 else np.nan
    return rng.rolling(n, min_periods=12).apply(ac, raw=False)


def load_ohlc():
    out = {}
    for a in ASSETS:
        df = pd.read_csv(f"../persistent/stock_data/{a}.csv", parse_dates=["date"]).sort_values("date")
        df = df[df["date"] <= "2026-07-29"]
        out[a] = df.set_index("date")
    return out


ohlc = load_ohlc()


def mkt(name):
    return macro[name].pct_change()


def rolling_corr(asset_r, mktr, w, minp=12):
    df = pd.concat([asset_r.rename("a"), mktr.rename("m")], axis=1)
    return df["a"].rolling(w, min_periods=minp).corr(df["m"])


def rt(a):
    return closes[a].pct_change()


mkt_spx = rt("SPX")

candidates = {
    "ewm_mom_20": {a: ewm_mom(closes[a], 20, 5) for a in closes},
    "mom_40_20": {a: closes[a] / closes[a].shift(45) - 1.0 for a in closes},
    "hi_lo_pos_10": {a: hi_lo_pos(closes[a], ohlc[a]["high"], ohlc[a]["low"], 10) for a in closes},
    "volume_liquidity_z_20": {a: vol_zscore(ohlc[a]["volume"].astype(float), 20) for a in closes},
    "range_ac_20": {a: range_autocorr(closes[a], ohlc[a]["high"], ohlc[a]["low"], 20) for a in closes},
    "corr_spx_20": {a: rolling_corr(rt(a), mkt_spx, 20) for a in closes},
}

print("assets:", len(closes), "macro:", len(macro))
for name, vals in candidates.items():
    try:
        evaluate(closes, vals, name, horizon=10)
    except Exception as e:
        print(name, "ERROR:", repr(e))
    print()