import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 try:x=get_index_daily_data(s,days=3200)
 except Exception:x=get_stock_daily_data(s,days=3200)
 if x is not None and len(x)>120:D[s]=x.sort_values('date').set_index('date')['close'].astype(float)
p=pd.DataFrame(D).sort_index().ffill();r=p.pct_change(); dn=r.where(r<0,0).rolling(60,min_periods=30).std(); fac=p.pct_change(40)/(dn*np.sqrt(60))
for h in [1,5,10,20]:
 a=[];ns=[];cov=[];turn=[];rec=[]
 for i in range(101,len(p)-h):
  z=pd.concat([fac.iloc[i-1].rename('f'),(p.iloc[i+h]/p.iloc[i]-1).rename('r')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8:
   q=z.f.corr(z.r)
   if np.isfinite(q):a.append(q);ns.append(len(z));cov.append(len(z)/15);rec += [q] if i>=len(p)-250 else []
  if i>101:turn.append((fac.iloc[i-1].rank(pct=True)-fac.iloc[i-2].rank(pct=True)).abs().mean())
 a=np.array(a);rr=np.array(rec);print({'h':h,'dates':len(a),'avg_n':round(np.mean(ns),2),'IC':round(a.mean(),6),'ICIR':round(a.mean()/a.std(ddof=1),6),'hit':round(np.mean(a>0),3),'coverage':round(np.mean(cov),3),'turnover':round(np.mean(turn),5),'recent250_IC':round(rr.mean(),6),'recent250_ICIR':round(rr.mean()/rr.std(ddof=1),6)})
