import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame()
for s in U:
 d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv')); d['date']=pd.to_datetime(d['date'])
 P[s]=d.set_index('date')['close']
P=P.sort_index(); R=P.pct_change()
d=pd.read_csv('../persistent/index_data/VIX.csv');d['date']=pd.to_datetime(d['date']);v=d.set_index('date')['close'].reindex(P.index).ffill()
vm=(v/v.rolling(120,min_periods=60).median()).clip(.5,2.0).shift(1)
vol=R.rolling(20,min_periods=20).std()*np.sqrt(5)
f=R.rolling(5,min_periods=5).sum().mul(-1).div(vol).mul(vm,axis=0).shift(1)
ics={}
for h in [1,5,10,20]:
 vals=[]
 for i in range(len(P)-h):
  z=pd.concat([f.iloc[i],P.pct_change(h).iloc[i+h]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 a=np.array(vals);ics[h]=a
 print(h,'dates',len(a),'N',15,'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1)*np.sqrt(len(a)),6),'hit',round((a>0).mean(),4))
print('coverage',round(f.notna().mean().mean(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),6))
for n in [250,500]:
 a=ics[10][-n:];print('recent',n,round(a.mean(),6),round(a.mean()/a.std(ddof=1)*np.sqrt(len(a)),6))
