import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=get_stock_daily_data(s,days=3000)
 if x is not None and len(x): D[s]=x.assign(date=pd.to_datetime(x.date)).set_index('date').close.astype(float)
P=pd.DataFrame(D).sort_index().ffill(); r=np.log(P).diff()
vix=pd.read_csv('../persistent/index_data/VIX.csv'); vix['date']=pd.to_datetime(vix['date'])
v=vix.set_index('date')['close'].astype(float).reindex(P.index).ffill()
vpct=v.rolling(252,min_periods=120).rank(pct=True).clip(0,1)
disp=r.rolling(5).std().mean(axis=1).rolling(20).mean()
dscore=(disp/disp.rolling(252,min_periods=120).median()).clip(0,3)/3
res=np.log(P/P.shift(10)).sub(np.log(P/P.shift(10)).median(axis=1),axis=0)
f=(-res/(r.rolling(30).std()+1e-8)).mul(vpct*dscore,axis=0).shift(1)
for h in [5,10,20]:
 q=[]; ns=[]
 for i in range(31,len(P)-h):
  z=pd.concat([f.iloc[i],np.log(P.iloc[i+h]/P.iloc[i])],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:
   q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z))
 q=pd.Series(q).dropna(); print(h,'dates',len(q),'avgN',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4),'coverage',round(P.notna().mean().mean(),4))
 for w in [365,730,1095]:
  x=q.tail(w); print(' recent',w,'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'n',len(x))
print('universe',len(D),'dates',len(P),'mean stress',round(vpct.mean(),4),'mean dispersion score',round(dscore.mean(),4))
