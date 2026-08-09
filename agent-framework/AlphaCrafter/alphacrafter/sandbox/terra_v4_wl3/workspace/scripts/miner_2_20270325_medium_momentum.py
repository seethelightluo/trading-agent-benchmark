import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in U:
 d=get_stock_daily_data(s,days=2600)
 if d is not None and len(d): d=d.assign(date=pd.to_datetime(d.date)).set_index('date');D[s]=d.close.astype(float)
p=pd.DataFrame(D).sort_index();r=p.pct_change();fwd=p.shift(-1)/p-1
# medium-term momentum with short-term reversal filter, volatility normalized
fac=(p.pct_change(60)-p.pct_change(5))/(r.rolling(20).std()*np.sqrt(20)+1e-8)
rows=[]; turns=[];prev=None
for dt in fac.index:
 z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
  rr=fac.loc[dt].rank(pct=True)
  if prev is not None:turns.append((rr-prev).abs().dropna().mean())
  prev=rr
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
for label,a,b in [('all',q.index.min(),q.index.max()),('2020-22','2020-01-01','2022-12-31'),('2023-24','2023-01-01','2024-12-31'),('2025-26','2025-01-01','2026-12-31'),('2027','2027-01-01','2027-12-31')]:
 x=q.loc[a:b]
 if len(x):print(label,len(x),x.ic.mean(),x.ic.mean()/(x.ic.std(ddof=1)+1e-12),np.mean(x.ic>0),x.n.mean())
print('coverage',len(q)/len(fac),'turnover',np.mean(turns),'dates',len(q),'instruments',len(U))
for h in [1,5,10]:
 yy=p.shift(-h)/p-1;v=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8:v.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('horizon',h,np.nanmean(v),len(v))
fac.reset_index().to_csv('scripts/miner_2_20270325_medium_momentum_signal.csv',index=False)
