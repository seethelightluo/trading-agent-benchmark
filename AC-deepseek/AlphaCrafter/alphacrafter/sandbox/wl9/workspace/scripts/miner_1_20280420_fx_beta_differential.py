"""
Factor candidate: fx_beta_diff_20d
Idea: FX-beta differential factor. For each of the 15 tradable cross-asset
instruments, estimate rolling 20d beta of daily returns vs USDCNY (CNY carry /
EM dollar-debt proxy) and vs DXY (global dollar index). The differential
beta_dxy - beta_cny captures an asset's net sensitivity to dollar strength
relative to CNY weakness.

Motivation: In a cross-asset dollar-driven synthetic worldline, assets that
react more to DXY than to USDCNY (or vice versa) have different carry/risk
profiles. The differential is a clean, interpretable cross-sectional factor
that is NOT a simple momentum/vol factor and should be low-correlated with the
existing library (beta_VIX_60 is beta on VIX returns, not FX returns).

Validation: cross-sectional Pearson rank IC on 10d forward returns over
2020-01-01..2028-04-20 (online date). Note: data files contain data beyond the
current date; we TRUNCATE at 2028-04-20 to avoid lookahead.
"""
import numpy as np
import pandas as pd

CUTOFF = pd.Timestamp('2028-04-20')
HORIZON = 10
WINDOW = 20
MIN_ASSETS = 8

TRADABLE = ['000300.SH', '000688.SH', 'SPX', 'HSI', 'N225', 'SX5E'.replace("'SX5E'", "'SX5E'"), 'SOX', 'NDX', 'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']

# load asset closes
closes = {}
rets = {}
for s in TRADABLE:
    df = pd.read_csv(f'../persistent/stock_data/{s}.csv', parse_dates=['date'])
    df = df[df['date'] <= CUTOFF].set_index('date')['close']
    closes[s] = df
    rets[s] = df.pct_change()

#print('dates:', len(closes['SPX']), closes['SPX'].index.min(), closes['SPX'].index.max())
print('Asset close series loaded; dates per asset:')
for s in TRADABLE:
    print(s, len(closes[s])