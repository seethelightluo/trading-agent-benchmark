import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for a in A:
 d=get_stock_daily_data(a,days=1800)
 if d is not None and len(d)>100: px[a]=d.set_index('date').close.astype(float)
p=pd.concat(px,axis=1).sort_index().ffill(); n=len(px)
for look in [3,5,10,20]:
 x=p.pct_change(look)
 F=pd.DataFrame({a:(x.gt(0).sum(axis=1)-x[a].gt(0).astype(int))/(n-1) for a in px})
 vals={h:[] for h in [1,5,10]}; turns=[]; cov=[]
 for i in range(len(p)-10):
  for h in vals:
   q=pd.concat([F.iloc[i].rename('f'),p.pct_change(h).iloc[i+h].rename('y')],axis=1).dropna()
   if len(q)>=8: vals[h].append(q.f.corr(q.y))
  if i:
   z=pd.concat([F.iloc[i],F.iloc[i-1]],axis=1).dropna(); turns.append((z.iloc[:,0].rank()!=z.iloc[:,1].rank()).mean())
  cov.append(F.iloc[i].notna().mean())
 print('look',look,'dates',len(vals[1]),'n',n,'IC',*[round(np.nanmean(vals[h]),5) for h in vals],'ICIR',round(np.nanmean(vals[1])/np.nanstd(vals[1],ddof=1),5),'hit',round(np.mean(np.array(vals[1])>0),5),'cov',round(np.mean(cov),5),'turn',round(np.mean(turns),5),'decay',*[round(np.nanmean(vals[h]),5) for h in [5,10]])
print('done')
