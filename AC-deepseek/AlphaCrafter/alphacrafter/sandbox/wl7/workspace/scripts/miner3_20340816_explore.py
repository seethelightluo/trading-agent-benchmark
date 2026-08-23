"""miner_3 factor exploration - 2034-08-16 (data thru 2034-08-15).

Candidate cross-asset factors on the 15-instrument tradable universe:
  1) up_ratio_20        : (up semi-vol / total vol) over 20d (upside capture)
  2) skew_20_skip5      : 20d return skewness ending 5d ago
  3) cny_beta_cond_60x20: -beta(asset, USDCNY, 60) * USDCNY 20d move (China FX)
  4) hl_range_20        : (high-low)/close over 20d normalized (range trend)
  5) vol_adj_mom_60x20  : vol-adjusted 60d vs 20d momentum spread
  6) lead_lag_60_20skip5: (60d momentum - 20d momentum) end 5d ago
  7) updown_asym_20     : (up semi - down semi)/total vol over 20d
  8) jpy_beta_cond_60x20:-beta(asset,USDJPY,60)*USDJPY 20d move (carry/JP macro)

Uses the shared 15-instrument cross-sectional gate |IC|>=0.0070, |ICIR|>=0.0840.
Factor signals computed per-asset on own trading calendar then union-reindexed.
"""
import json
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

WATCH = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX',
         'XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
DAYS = 4000


def load_panel():
    closes = {}
    for s in WATCH:
        df = get_stock_daily_data(s, days=DAYS)
        if df is None or len(df) < 300:
            print('WARN', s, 'insufficient')
            continue
        closes[s] = df.set_index('date')['close'].astype(float)
    panel = pd.DataFrame(closes).sort_index()
    panel = panel[~panel.index.duplicated(keep='last')]
    return panel


def load_index(name):
    df = get_index_daily_data(name, days=DAYS)
    if df is None:
        return None
    return df.set_index('date')['close'].astype(float).sort_index()


def ap(func, panel):
    """Apply func on each asset's own calendar then reindex to union grid."""
    out = {}
    for col in panel.columns:
        s = panel[col].dropna()
        if len(s) < 80:
            out[col] = pd.Series(np.nan, index=panel.index)
            continue
        f = func(s)
        if f is None:
            out[col] = pd.Series(np.nan, index=panel.index)
        else:
            out[col] = f.reindex(panel.index)
    return pd.DataFrame(out, index=panel.index)


def fwd_returns(panel, h):
    return panel.shift(-h) / panel - 1.0


def xsec_ic(fdf, rdf, minv=8):
    dates, ics = [], []
    for dt in fdf.index:
        f = fdf.loc[dt]; r = rdf.loc[dt]
        m = f.notna() & r.notna() & np.isfinite(f) & np.isfinite(r)
        if m.sum() >= minv:
            ics.append(f[m].corr(