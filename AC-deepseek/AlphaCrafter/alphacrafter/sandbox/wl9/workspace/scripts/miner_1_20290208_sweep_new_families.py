"""miner_1 2029-02-08 exploration sweep: new cross-asset factor families.
Universe: 15 tradable instruments. IC = cross-sectional rank corr of factor vs
forward 10d mean return, dates with >=8 valid assets. Split into warm-up
(<=2026-07-15) and live (2026-07-16..2029-02-08) windows.
"""
import pandas as pd
import numpy as np
from pathlib import Path

VISIBLE_END = '2029-02-08'
SPLIT = '2026-07-15'
STOCK_DIR = Path('../persistent/stock_data')
INDEX_DIR = Path('../persistent/index_data')
ASSETS = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

closes = {}
for a in ASSETS:
    df = pd.read_csv(STOCK_DIR / f'{a}.csv', parse_dates=['date']).sort_values('date')
    df = df[df['date'] <= VISIBLE_END].set_index('date')['close'].astype(float)
    closes[a] = df

rets = pd.DataFrame({a: closes[a].pct_change() for a in ASSETS}).dropna()
rets = rets[rets.index >= '2020-03-01']
print(f"Panel: {rets.shape[0]} dates x {rets.shape[1]} assets, {rets.index[0]:%Y-%m-%d} .. {rets.index[-1]:%Y-%m-%d}")

def load_macro(name):
    df = pd.read_csv(INDEX_DIR / f'{name}.csv', parse_dates=['date'])
    return df[df['date'] <= VISIBLE_END].set_index('date')['close'].astype(float)

vix = load_macro('VIX'); dxy = load_macro('DXY'); usdcny = load_macro('USDCNY')
usd = load_macro('USDJPY'); eur = load_macro('EURUSD')
r_vix = vix.pct_change(); r_dxy = dxy.pct_change(); r_usdcny = usdcny.pct_change()
r_usd = usd.pct_change(); r_eur = eur.pct_change()
r_us10y = closes['US10Y'].pct_change()

def compute_ic(factor_vals, forward_rets, label):
    common =