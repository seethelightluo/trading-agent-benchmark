import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2027-02-24')
def load(path):
 d=pd.read_csv(path,parse_dates=['date']); d['date']=d.date.dt.normalize(); return d.drop_duplicates('date').set_index('date').sort_index().close
P={s:load('../persistent/stock_data/'+s+'.csv').loc[:cut] for s in U}; p=pd.concat(P,axis=1).sort_index().ffill();r=p.pct_change()
base=p.pct_change(60)/(r.rolling(40,min_periods=30).std()*np.sqrt(40)+1e-12); base*=.75+.25*np.tanh(p.pct_change(10)*8)
v=load('../persistent/index_data/VIX.csv').reindex(p.index).ffill(); z=(v-v.rolling(120,min_periods=60).mean())/(v.rolling(120,min_periods=60).std()+1e-12); f=base/(1+z.clip(lower=0))
print('universe',len(U),'dates',len(p),'vix coverage',v.notna().mean())
for h in [10,20]:
 for name,g in [('base',base),('macro',f)]:
  a=[];ns=[];ds=[]
  for i in range(len(p)-h):
   q=pd.concat([g.iloc[i],p.iloc[i+h]/p.iloc[i]-1],axis=1).dropna()
   if len(q)>=8 and q.iloc[:,0].nunique()>1:
    x=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
    if np.isfinite(x):a.append(x);ns.append(len(q));ds.append(p.index[i])
  x=np.array(a); print(name,h,'dates',len(x),'avgN',np.mean(ns),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round(np.mean(x>0),4),'recent',round(x[np.array(ds)>=pd.Timestamp('2026-07-16')].mean(),6))
print('coverage',f.notna().mean().mean(),'turn',f.rank(pct=True).diff().abs().mean(axis=1).mean())
