import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; fs={}
for s in U:
 d=None
 try:d=get_index_daily_data(s,2600)
 except:pass
 if d is None or len(d)<150:
  try:d=get_stock_daily_data(s,2600)
  except:pass
 if d is not None:fs[s]=d.set_index(pd.to_datetime(d.date)).close.astype(float)
p=pd.DataFrame(fs).sort_index().ffill()
# 60d relative trend, 10d skip, lagged one day; tests intermediate persistence
raw=p.shift(1)/p.shift(61)-1
f=raw.sub(raw.median(axis=1),axis=0)
ics=[]; rows=[]
for i in range(62,len(p)-10):
 z=pd.concat([f.iloc[i],p.iloc[i+10]/p.iloc[i]-1],axis=1).dropna()
 if len(z)>=8:
  q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman');ics.append(q);rows.append([p.index[i],len(z),q])
a=np.array(ics);dt=pd.to_datetime([x[0] for x in rows]);
print('candidate relative 60d momentum skip10');print('assets',len(fs),'dates',len(a),'avg_n',round(np.mean([x[1] for x in rows]),3),'coverage',round(np.mean([x[1] for x in rows])/15,4));print('IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4),'turnover',round(np.nanmean(np.abs(f.diff()).sum(1)/f.notna().sum(1)),4))
for label,m in [('2020-24',(dt<'2025-01-01')),('2025-26',(dt>='2025-01-01')&(dt<'2027-01-01')),('2027-28',(dt>='2027-01-01')&(dt<'2029-01-01')),('recent',(dt>='2028-09-01'))]:
 q=a[m];print(label,len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6))
pd.DataFrame(rows,columns=['date','n','ic']).to_csv('scripts/miner_2_20290604_relative60_signal.csv',index=False)
