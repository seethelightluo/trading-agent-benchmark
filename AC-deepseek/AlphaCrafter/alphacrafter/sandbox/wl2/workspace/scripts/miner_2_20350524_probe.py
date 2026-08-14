"""miner_2 probe 2035-05-24: quick data sanity + volume availability + baseline IC on the 15-asset benchmark.
Data through visible_through=2035-05-23 only (no future leakage).
"""
import pandas as pd, numpy as np
from scipy.stats import spearmanr

SYMS = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END = pd.Timestamp('2035-05-23')

def load_col(colname):
    cols = {}
    for s in SYMS:
        df = pd.read_csv(f'../persistent/stock_data/{s}.csv')
        df['date'] = pd.to_datetime(df['date'])
        cols[s] = df.set_index('date')[colname]
    out = pd.DataFrame(cols).sort_index()
    return out[out.index <= END]

P = load_col('close')
V = load_col('volume')
R = P.pct_change()

print("=== price shape ===", P.shape)
print("=== volume NaN fraction per symbol ===")
print(V.isna().mean().round(3).to_dict())
print("=== volume zero fraction per symbol ===")
print((V.fillna(0) == 0).mean().round(3).to_dict())
print("=== last date ===", P.index[-1])
print("=== last 10d returns (2035-05-23) ===")
print((P.iloc[-1] / P.iloc[-11] - 1).round(4).to_string())
print("=== VIX check ===")
vix = pd.read_csv('../persistent/index_data/VIX.csv')
vix['date'] = pd.to_datetime(vix['date']); vix = vix.set_index('date')['close']; vix = vix[vix.index <= END]
print("VIX last:", vix.iloc[-1], "mean60:", vix.tail(60).mean().round(2), "min/max60:", vix.tail(60).min().round(2), vix.tail(60).max().round(2))
