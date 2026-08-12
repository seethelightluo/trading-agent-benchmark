import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)<300: d=get_index_daily_data(s,4000)
 if d is not None: D[s]=d.set_index('date')['close'].astype(float)
p=pd.DataFrame(D).sort_index().ffill()
r=np.log(p).diff()
# One interpretable candidate: high-dispersion residual reversal.
# residual = asset 5d return minus equal-weight universe 5d return; activate only
# when trailing 20d cross-sectional dispersion is above its expanding median.
r5=r.rolling(5).sum(); resid=r5.sub(r5.mean(axis=1),axis=0)
disp=r.rolling(20).std().mean(axis=1)
med=disp.rolling(252,min_periods=126).median()
f=(-resid).where(disp>med)
# lag signal one day, forward 5/10/20 returns
rows=[]
for t in f.index:
 sig=f.loc[t]
 if t not in r.index: continue
 for h in [5,10,20]:
  j=r.index.searchsorted(t,side='right')
  k=j+h-1
  if j>=len(r) or k>=len(r): continue
  fw=r.iloc[j:k+1].sum()
  z=pd.concat([sig,fw],axis=1).dropna()
  if len(z)>=8:
   rows.append((t,h,len(z),z.iloc[:,0].corr(z.iloc[:,1])))
x=pd.DataFrame(rows,columns=['date','h','n','ic'])
print('dates',len(x.date.unique()),'instruments',len(U),'avg_n',x.n.mean(),'min_n',x.n.min())
for h in [5,10,20]:
 a=x[x.h==h].ic.dropna(); print(h,'obs',len(a),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit', (a>0).mean())
 for lo,hi in [('2020','2022'),('2023','2025'),('2026','2027'),('2028','2030')]:
  q=x[(x.h==h)&(x.date.dt.year.astype(str)>=lo)&(x.date.dt.year.astype(str)<=hi)].ic
  print(lo,round(q.mean(),6),round(q.mean()/q.std(ddof=1),4),len(q))
print('coverage',f.notna().stack().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
f.to_csv('scripts/miner_3_20300919_dispersion_resid_signal.csv')
