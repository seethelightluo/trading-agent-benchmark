import pandas as pd, numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2026-12-17')
rows=[]
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=END]; r=d.close.pct_change(); f=d.close.pct_change(20)/(r.abs().rolling(20).sum()+1e-12); rows.append(pd.DataFrame({'date':d.date,'symbol':s,'signal':f}))
pd.concat(rows,ignore_index=True).dropna().to_csv('scripts/miner_3_20261217_range_efficiency_signal.csv',index=False)
print('wrote signal artifact')
