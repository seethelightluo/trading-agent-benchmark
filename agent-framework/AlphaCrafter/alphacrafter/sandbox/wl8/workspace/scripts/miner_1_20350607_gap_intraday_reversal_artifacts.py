import os,pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; rows=[]
for s in U:
 p='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(p):
  d=pd.read_csv(p); d.date=pd.to_datetime(d.date); d=d.sort_values('date').set_index('date'); intr=d.close/d.open-1; vol=d.close.pct_change().rolling(30,min_periods=20).std(); f=-(intr.rolling(5,min_periods=5).sum()).shift(1)/(vol.shift(1)+1e-12); y=d.close.shift(-10)/d.close-1; rows.append(pd.DataFrame({'date':d.index,'symbol':s,'signal':f,'forward_10d_return':y}).dropna())
p=pd.concat(rows,ignore_index=True); p.to_csv('scripts/miner_1_20350607_gap_intraday_reversal_signal.csv',index=False)
a=[]
for dt,g in p.groupby('date'):
 if len(g)>=8 and g.signal.nunique()>1 and g.forward_10d_return.nunique()>1:a.append({'date':dt,'ic':spearmanr(g.signal,g.forward_10d_return).statistic,'n':len(g)})
pd.DataFrame(a).to_csv('scripts/miner_1_20350607_gap_intraday_reversal_ic.csv',index=False); print(len(p),len(a))
