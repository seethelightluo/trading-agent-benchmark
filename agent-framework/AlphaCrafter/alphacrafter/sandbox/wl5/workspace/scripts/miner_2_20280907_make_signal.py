import pandas as pd,numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.concat({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U},axis=1).sort_index().loc[:'2028-09-06']; r=p.pct_change(); disp=r.std(axis=1).rolling(5).mean(); shock=disp>disp.rolling(60,min_periods=30).quantile(.70); f=-r.rolling(5).sum().where(shock,0)
f.loc[:'2020-03-01']=np.nan
f.stack().rename('signal').to_csv('scripts/miner_2_20280907_shock_reversal10_signal.csv',header=True)
print('rows',f.stack().shape[0],'dates',f.dropna(how='all').shape[0])
