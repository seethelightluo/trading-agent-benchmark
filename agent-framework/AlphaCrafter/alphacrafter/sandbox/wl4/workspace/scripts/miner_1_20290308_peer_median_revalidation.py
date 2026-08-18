import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 try:x=get_index_daily_data(s,days=3200)
 except FileNotFoundError:x=get_stock_daily_data(s,days=3200)
 if x is not None and len(x)>30:D[s]=x.sort_values('date').set_index('date')['close'].astype(float)
pd_=pd.DataFrame(D).sort_index().ffill(); r=pd_.pct_change(5)
fac=r.sub(r.median(axis=1),axis=0) * 0 # placeholder
# leave-one-out peer median
fac=pd.DataFrame(index=r.index,columns=r.columns,dtype=float)
for c in r: fac[c]=r.drop(columns=c).median(axis=1)
for h in [1,5,10,20]:
 ic=[]; cov=[]; turn=[]
 for i in range(6,len(pd_)-h):
  z=pd.concat([fac.iloc[i-1].rename('f'),(pd_.iloc[i+h]/pd_.iloc[i]-1).rename('r')],axis=1).dropna()
  if len(z)>=8:
   q=z.f.corr(z.r)
   if np.isfinite(q):ic.append(q);cov.append(len(z)/15)
   if i>6:turn.append((fac.iloc[i-1].rank(pct=True)-fac.iloc[i-2].rank(pct=True)).abs().mean())
 a=np.array(ic);print({'h':h,'dates':len(a),'avg_n':round(np.mean(cov)*15,2),'IC':round(a.mean(),6),'ICIR':round(a.mean()/a.std(ddof=1),6),'coverage':round(np.mean(cov),3),'turnover':round(np.mean(turn),5),'recent250_IC':round(a[-250:].mean(),6)})
