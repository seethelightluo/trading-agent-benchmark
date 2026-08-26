import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv'); d.date=pd.to_datetime(d.date); p[s]=d.set_index('date').close
p=pd.DataFrame(p).sort_index(); v=pd.read_csv('../persistent/index_data/VIX.csv'); v.date=pd.to_datetime(v.date); v=v.set_index('date').close.reindex(p.index).ffill()
mom=p.pct_change(20); vr=v.pct_change(10); calm=(vr<=0).astype(float)
f=mom.mul((2*calm-1),axis=0).shift(1)
for h in [1,5,10,20]:
 vals=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],p.pct_change(h).shift(-h).loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
 a=np.array(vals); print(h,'dates',len(a),'avgN',np.mean(ns),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0))
for label,lo,hi in [('2020-22','2020','2022-12-31'),('2023-25','2023','2025-12-31'),('2026-28','2026','2028-12-31'),('2029','2029','2029-08-13')]:
 a=[]
 for dt in f.loc[lo:hi].index:
  z=pd.concat([f.loc[dt],p.pct_change(10).shift(-10).loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 a=np.array(a); print(label,len(a),a.mean(),a.mean()/a.std(ddof=1))
f.to_csv('scripts/miner_2_20290813_vix_conditioned_momentum_signal.csv',index_label='date')
