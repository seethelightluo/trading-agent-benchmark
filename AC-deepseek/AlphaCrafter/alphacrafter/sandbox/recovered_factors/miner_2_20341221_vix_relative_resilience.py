import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; E=pd.Timestamp('2034-12-20')
p={}
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()
 p[a]=d.loc[d.index<=E,'close'].astype(float)
p=pd.DataFrame(p).sort_index(); r=p.pct_change(fill_method=None)
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).drop_duplicates('date').set_index('date')['close'].astype(float).reindex(p.index).ffill()
v=vix.pct_change(); vz=(vix-vix.rolling(120,min_periods=60).mean())/(vix.rolling(120,min_periods=60).std()+1e-9)
# During elevated VIX, favor assets with resilient 10d relative performance, scaled by risk;
# use only completed observations at t, with a smooth activation and cross-sectional benchmark.
rel=r.rolling(10,min_periods=8).sum().sub(r.mean(axis=1).rolling(10,min_periods=8).sum(),axis=0)
vol=r.rolling(20,min_periods=15).std(); activation=np.tanh(np.maximum(vz,0)/2)
f=rel.div(vol+1e-9).mul(activation,axis=0)
print('candidate vix-conditioned relative resilience; cutoff',E.date(),'rows',len(p),'assets',len(A))
for h in [1,5,10,20]:
 vals=[]; ns=[]; fr=p.shift(-h)/p-1
 for t in f.index:
  q=pd.concat([f.loc[t],fr.loc[t]],axis=1).dropna()
  if len(q)>=8:
   z=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(z): vals.append(z); ns.append(len(q))
 q=np.asarray(vals); print('H',h,'dates',len(q),'meanN %.2f IC %.6f ICIR %.6f hit %.4f'%(np.mean(ns),np.mean(q),np.mean(q)/(np.std(q,ddof=1)+1e-12),np.mean(q>0)))
print('coverage %.6f turnover %.6f'%(f.notna().mean().mean(),f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()))
fr=p.shift(-10)/p-1; vals=[]; ds=[]
for t in f.index:
 q=pd.concat([f.loc[t],fr.loc[t]],axis=1).dropna()
 if len(q)>=8:
  z=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
  if np.isfinite(z): vals.append(z); ds.append(t)
v=np.asarray(vals); ds=pd.Series(ds)
for lo,hi in [('2020','2024-12-31'),('2025','2029-12-31'),('2030','2034-12-20'),('2033','2034-12-20')]:
 q=(ds>=lo)&(ds<=hi); y=v[q]; print('regime',lo,hi,'dates',len(y),'IC %.6f ICIR %.6f hit %.4f'%(np.mean(y),np.mean(y)/(np.std(y,ddof=1)+1e-12),np.mean(y>0)) if len(y)>1 else 'nan')
print('library_audit FAILED: admitted factor panels are not persisted in reconstructable form; max_abs_library_correlation unavailable, so candidate is not admissible')
