import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
raw={}
for s in U:
 d=get_stock_daily_data(s,days=4000)
 if d is not None and len(d)>100:
  x=d.copy(); x.date=pd.to_datetime(x.date); raw[s]=x.set_index('date').close.astype(float)
p=pd.DataFrame(raw).sort_index().ffill(limit=3)
r=np.log(p).diff()
# residual short shock: remove common market return beta over rolling 60d, then reverse 3d residual move
m=r.mean(axis=1)
res=pd.DataFrame(index=r.index,columns=r.columns,dtype=float)
for s in U:
 cov=r[s].rolling(60,min_periods=30).cov(m); var=m.rolling(60,min_periods=30).var()
 res[s]=r[s]-cov/var*m
f=(-res.rolling(3,min_periods=3).sum()/(res.rolling(30,min_periods=15).std()*np.sqrt(3))).shift(1)
f=f.sub(f.mean(axis=1),axis=0)
fr=np.log(p.shift(-1)/p)
vals=[]; ds=[]; ns=[]
for dt in f.index:
 ok=f.loc[dt].notna()&fr.loc[dt].notna()
 if ok.sum()>=8:
  vals.append(f.loc[dt,ok].corr(fr.loc[dt,ok],method='spearman')); ds.append(dt); ns.append(ok.sum())
z=pd.Series(vals,index=ds).dropna()
print('dates',len(z),'avg_n',round(np.mean(ns),2),'IC',round(z.mean(),7),'ICIR',round(z.mean()/z.std(ddof=1),7),'hit',round((z>0).mean(),4))
for h in [3,5,10]:
 frh=np.log(p.shift(-h)/p); q=[]; dd=[]
 for dt in f.index:
  ok=f.loc[dt].notna()&frh.loc[dt].notna()
  if ok.sum()>=8:q.append(f.loc[dt,ok].corr(frh.loc[dt,ok],method='spearman'));dd.append(dt)
 q=pd.Series(q,index=dd).dropna(); print('H',h,len(q),round(q.mean(),7),round(q.mean()/q.std(ddof=1),7))
print('regimes')
for a,b in [('2020','2022-12-31'),('2023','2025-12-31'),('2026','2027-12-31'),('2028','2029-12-31')]:
 q=z.loc[a:b]; print(a,len(q),round(q.mean(),7),round(q.mean()/q.std(ddof=1),7))
print('coverage',round(f.notna().sum().sum()/f.size,4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
f.to_csv('scripts/miner_1_20290823_residual_shock_signal.csv')
