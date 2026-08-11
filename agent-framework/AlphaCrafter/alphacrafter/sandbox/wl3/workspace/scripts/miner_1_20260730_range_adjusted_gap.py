import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; E=pd.Timestamp('2026-07-15'); D={}
for s in U:
 d=get_stock_daily_data(s,days=3000)
 if d is not None:
  d=d[pd.to_datetime(d.date)<=E].copy(); d.date=pd.to_datetime(d.date); D[s]=d.set_index('date')
pd.set_option('mode.use_inf_as_na',True); C=pd.concat({s:d.close for s,d in D.items()},axis=1).sort_index().ffill(); O=pd.concat({s:d.open for s,d in D.items()},axis=1).sort_index().reindex(C.index).ffill(); H=pd.concat({s:d.high for s,d in D.items()},axis=1).sort_index().reindex(C.index).ffill(); L=pd.concat({s:d.low for s,d in D.items()},axis=1).sort_index().reindex(C.index).ffill(); R=C.pct_change(); V=R.rolling(20,min_periods=10).std(); gap=O/C.shift(1)-1; rng=(H-L)/C.shift(1); f=(-gap/(V*(1+rng))).replace([np.inf,-np.inf],np.nan)
x=[]; cs=[]; tr=[]; d5=[]; d10=[]
for i in range(len(C)-10):
 q=pd.concat([f.iloc[i].rename('f'),R.iloc[i+1].rename('y')],axis=1).dropna()
 if len(q)>=8 and q.f.nunique()>1:
  x.append(spearmanr(q.f,q.y).statistic);cs.append(len(q)/15)
  for h,a in [(5,d5),(10,d10)]:
   z=pd.concat([f.iloc[i],C.pct_change(h).iloc[i+h]],axis=1).dropna()
   if len(z)>=8 and z.iloc[:,0].nunique()>1:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
  if i:
   z=pd.concat([f.iloc[i],f.iloc[i-1]],axis=1).dropna()
   if len(z)>=8:tr.append((z.iloc[:,0].rank()!=z.iloc[:,1].rank()).mean())
x=np.array(x);print('dates',len(x),'avg_names',np.mean(cs)*15,'coverage',np.mean(cs),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'hit',(x>0).mean(),'turn',np.mean(tr),'d5',np.mean(d5),'d10',np.mean(d10));
for n,z in [('early',x[:len(x)//2]),('late',x[len(x)//2:]),('recent250',x[-250:])]:print(n,len(z),z.mean(),z.mean()/z.std(ddof=1),(z>0).mean())
