import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index().ffill(); r=p.pct_change()
# Interpretable medium-term acceleration: recent 20d return minus long 60d return/3, lagged one day.
f=(r.rolling(20,min_periods=15).sum()-r.rolling(60,min_periods=45).sum()/3).shift(1)
y=p.pct_change(10).shift(-10); D=[];A=[];ns=[];turn=[]
for d in f.index:
 z=pd.concat([f.loc[d],y.loc[d]],axis=1).dropna()
 if len(z)>=8:
  D.append(d); A.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z));turn.append(np.mean(f.loc[d].rank()!=f.shift(1).loc[d].rank()))
A=np.array(A);D=pd.DatetimeIndex(D)
print('all dates',len(A),'avgN',np.mean(ns),'coverage',f.notna().sum(1).mean()/15,'turnover',np.mean(turn),'IC',A.mean(),'ICIR',A.mean()/A.std(ddof=1),'hit',np.mean(A>0))
for name,lo,hi in [('2020_22','2020','2022-12-31'),('2023_25','2023','2025-12-31'),('2026_28','2026','2028-12-31')]:
 q=A[(D>=lo)&(D<=hi)];print(name,'dates',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',np.mean(q>0))
f.to_csv('scripts/miner_2_20290924_acceleration_signal.csv',index_label='date')
