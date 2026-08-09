import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-12-16')
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cut').set_index('date').close for s in U}
R=pd.DataFrame({s:p.pct_change() for s,p in P.items()}); med=R.median(axis=1)
F=-(R.sub(med,axis=0)).rolling(2,min_periods=2).sum()
F.to_csv('scripts/miner_3_20261217_idio_reversal_2d_signal.csv',index_label='date')
