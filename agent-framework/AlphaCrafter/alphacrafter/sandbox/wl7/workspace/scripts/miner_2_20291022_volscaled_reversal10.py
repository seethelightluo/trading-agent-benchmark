import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index().ffill(); r=p.pct_change()
# Volatility-scaled 10-day reversal, lagged one day.
vol=r.rolling(20,min_periods=15).std(); f=(-r.rolling(10,min_periods=8).sum()/vol.rolling(10,min_periods=8).mean()).shift(1)
y=p.pct_change(10).shift(-10); A=[];D=[];ns=[];turn=[]
for d in f.index:
 z=pd.concat([f.loc[d],y.loc[d]],axis=1).dropna()
 if len(z)>=8:
  D.append(d); ns.append(len(z)); A.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); turn.append(np.mean(f.loc[d].rank()!=f.shift(1).loc[d].rank()))
A=np.array(A);D=pd.DatetimeIndex(D)
print('factor=volscaled_reversal10 horizon=10');print('dates',len(A),'avgN',round(np.mean(ns),2),'coverage',round(f.notna().sum(1).mean()/15,4),'turnover',round(np.mean(turn),4),'IC',round(A.mean(),6),'ICIR',round(A.mean()/A.std(ddof=1),6),'hit',round(np.mean(A>0),4))
for name,lo,hi in [('2020_22','2020','2022-12-31'),('2023_25','2023','2025-12-31'),('2026_28','2026','2028-12-31'),('2029','2029','2029-10-22')]:
 q=A[(D>=lo)&(D<=hi)];print(name,'dates',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round(np.mean(q>0),4))
f.to_csv('scripts/miner_2_20291022_volscaled_reversal10_signal.csv',index_label='date')
