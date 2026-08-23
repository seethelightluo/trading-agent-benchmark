"""
miner_3 2032-12-23 exploration: probe data availability and screen several
fresh candidate factor ideas on the 15-instrument cross-asset universe.

Candidates (all interpretable, new relative to evicted library):
 A) vol_target_mom_20x60     : momentum/volatility scaling (risk-adjusted momentum)
 B) sma_dist_40              : distance of close from 40d SMA (trend reversion)
 C) hi_mom_20                : close relative to 20d high (breakout vs mean-reversion)
 D) wti_lead_20              : WTI 20d return leading others (commodity cycle proxy)
 E) gold_equity_div_60       : XAU vs SPX relative strength (regime rotation)
 F) cv_raw_20                : raw 20d realized vol cross-section (low-vol tilt)
 G) kauf_eff_60              : Kaufman efficiency ratio 60d (trend smoothness)
 H) drawdown_recovery_40     : 1 - close/max(close,40) (recovery depth)
"""
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

current_date = "2032-12-23"
watchlist = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
             'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']

data = {}
for sym in watchlist:
    df = get_stock_daily_data(symbol=sym, days=600)
    if df is not None and len(df) > 150:
        data[sym] = df.reset_index(drop=True)
    else:
        print(f"WARNING: {sym} insufficient ({len(df) if df is not None else 0})")

print(f"Loaded {len(data)} instruments")
for sym, df in data.items():
    print(f"  {sym}: {len(df)} days, first={df['date'].iloc[0].date()} last={df['date'].iloc[-1].date()}")

# ---- factor calculators (return full arrays, NaN where undefined) ----
def f_vol_target_mom(df, s=20, l=60):
    c = df['close'].values
    mom = np.full(len(c), np.nan); vol = np.full(len(c), np.nan)
    if len(c) > l:
        mom[l:] = c[l:] / c[:-l] - 1
    if len(c) > s:
        r = np.diff(c) / c[:-1]
        vol[s:] = pd.Series(r).rolling(s).std().values
    return mom / vol

def f_sma_dist(df, w=40):
    c = df['close'].values
    sma = pd.Series(c).rolling(w).mean().values
    return c / sma - 1

def f_hi_mom(df, w=20):
    c = df['close'].values
    roll_high = pd.Series(c).rolling(w).max().values
    return c / roll_high - 1

def f_wti_lead(df, w=20):
    if df['symbol'].iloc[0] != 'WTI':
        return np.full(len(df), np.nan)
    c = df['close'].values
    mom = np.full(len(c), np.nan)
    if len(c) > w:
        mom[w:] = c[w:] / c[:-w] - 1
    return mom

def f_gold_equity_div(df, w=60):
    c = df['close'].values
    if df['symbol'].iloc[0] != 'XAU':
        return np.full(len(c), np.nan)
    mom = np.full(len(c), np.nan)
    if len(c) > w:
        mom[w:] = c[w:] / c[:-w] - 1
    return mom

def f_cv_raw(df, s=20):
    c = df['close'].values
    vol = np.full(len(c), np.nan)
    if len(c) > s:
        r = np.diff(c) / c[:-1]
        vol[s:] = pd.Series(r).rolling(s).std().values
    return vol

def f_kauf_eff(df, w=60):
    c = df['close'].values
    eff = np.full(len(c), np.nan)
    if len(c) > w:
        r = np.diff(c) / c[:-1]
        path = np.abs(r)
        vol_sum = pd.Series(path).rolling(w).sum().values
        net = np.abs(c[w:] / c[:-w] - 1)
        eff[w:] = net / vol_sum
    return eff

def f_drawdown_recovery(df, w=40):
    c = df['close'].values
    roll_max = pd.Series(c).rolling(w).max().values
    return 1 - c / roll_max

FACTORS = {
    'A_vol_target_mom_20x60': f_vol_target_mom,
    'B_sma_dist_40': f_sma_dist,
    'C_hi_mom_20': f