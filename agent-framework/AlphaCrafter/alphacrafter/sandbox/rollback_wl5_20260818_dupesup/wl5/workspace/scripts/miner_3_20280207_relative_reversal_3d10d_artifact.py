import pandas as pd,numpy as np
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2028-02-06'); b=Path('../persistent/stock_data')
px={s:pd.read_csv(b/f'{s}.csv',parse_dates=['date']).set_index('date')['close'].sort_index() for s in U}; P=pd.DataFrame(px).sort_index().loc[:end].ffill(); r=P.pct_change(3); f=-r.sub(r.median(axis=1),axis=0); f.to_csv('scripts/miner_3_20280207_relative_reversal_3d10d_signal.csv'); print('saved',f.shape,'coverage',float(f.notna().mean().mean()))
