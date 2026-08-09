import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
D={os.path.basename(f)[:-4]:pd.read_csv(f,parse_dates=['date']).set_index('date') for f in glob.glob('../persistent/stock_data/*.csv')}
p=pd.DataFrame({k:v.close for k,v in D.items()}).sort_index().astype(float)
v=pd.DataFrame({k:v.volume for k,v in D.items()}).reindex(p.index).astype(float)
# Cross-sectional volume surprise: recent completed 5-session volume relative to lagged 60-session baseline.
# Rank transform makes the signal comparable across heterogeneous asset classes.
vr=(v.shift(1).rolling(5,min_periods=3).mean()/v.shift(1).rolling(60,min_periods=30).mean()).clip(0.25,4.0)
sig=vr.rank(axis=1,pct=True)
print('candidate=cross_sectional_volume_surprise_rank_5_60')
print('dates',len(p),'instruments',len(p.columns),'coverage',round(sig.notna().sum().sum()/sig.size,6),'meanN',round(sig.notna().sum(axis=1).mean(),2))
for h in [1,5,10,20]:
 a=[];ns=[]; f=p.shift(-h)/p-1
 for d in p.index:
  z=pd.concat([sig.loc[d],f.loc[d]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q):a.append(q);ns.append(len(z))
 a=np.asarray(a);print('horizon',h,'dates',len(a),'meanN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
print('turnover_proxy',round(sig.diff().abs().mean(axis=1).dropna().mean(),6))
for h in [1,5,10,20]:
 f=p.shift(-h)/p-1
 for label,mask in [('2020-23',p.index<'2024-01-01'),('2024-27',(p.index>='2024-01-01')&(p.index<'2028-01-01')),('2028+',p.index>='2028-01-01'),('latest120',p.index>=p.index[-120])]:
  a=[]
  for d in p.index[mask]:
   z=pd.concat([sig.loc[d],f.loc[d]],axis=1).dropna()
   if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
    q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
    if np.isfinite(q):a.append(q)
  a=np.asarray(a);print('regime',h,label,'dates',len(a),'IC',round(a.mean(),6) if len(a) else None,'ICIR',round(a.mean()/a.std(ddof=1),6) if len(a)>1 else None)
for lag in [1,5,10,20]:
 z=pd.concat([sig.stack().rename('a'),sig.shift(lag).stack().rename('b')],axis=1).dropna();print('decay',lag,round(spearmanr(z.a,z.b).statistic,6),len(z))
