import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 try:x=get_index_daily_data(s,days=3200)
 except Exception:x=get_stock_daily_data(s,days=3200)
 if x is not None and len(x)>100:D[s]=x.sort_values('date').set_index('date')['close'].astype(float)
p=pd.DataFrame(D).sort_index().ffill();r=p.pct_change();v=r.rolling(40,min_periods=25).std()*np.sqrt(40);f=-(p.pct_change(40)/v)
print('data_dates',len(p),'assets',len(D),'cutoff',p.index.max())
for h in [1,5,10,20]:
 a=[];ns=[];rec=[]
 for i in range(1,len(p)-h):
  q=pd.concat([f.iloc[i-1].rename('f'),(p.iloc[i+h]/p.iloc[i]-1).rename('r')],axis=1).dropna()
  if len(q)>=8:
   c=q.f.corr(q.r)
   if np.isfinite(c):a.append(c);ns.append(len(q)); rec += [c] if i>=len(p)-250 else []
 a=np.array(a);rec=np.array(rec)
 print({'h':h,'dates':len(a),'avg_n':round(np.mean(ns),2),'IC':round(a.mean(),6),'ICIR':round(a.mean()/a.std(ddof=1),6),'hit':round(np.mean(a>0),4),'coverage':round(np.mean(ns)/15,4),'recent250_IC':round(rec.mean(),6),'recent250_ICIR':round(rec.mean()/rec.std(ddof=1),6)})
print('turnover',round(float(f.rank(axis=1,pct=True).diff().abs().stack().mean()),6))
