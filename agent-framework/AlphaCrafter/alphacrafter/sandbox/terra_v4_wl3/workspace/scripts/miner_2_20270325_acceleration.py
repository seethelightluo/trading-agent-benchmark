import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 d=get_stock_daily_data(s,days=2600)
 if d is not None and len(d): d=d.assign(date=pd.to_datetime(d.date)).set_index('date'); D[s]=d.close.astype(float)
p=pd.DataFrame(D).sort_index(); r=p.pct_change(); fwd=p.shift(-1)/p-1
# acceleration: recent 5d return minus expected quarter of 20d return, scaled by 20d volatility
fac=(p.pct_change(5)-p.pct_change(20)/4)/(r.rolling(20).std()+1e-8)
rows=[]; turns=[]; prev=None
for dt in fac.index:
 z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
  rr=fac.loc[dt].rank(pct=True)
  if prev is not None: turns.append((rr-prev).abs().dropna().mean())
  prev=rr
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
for label,a,b in [('all',q.index.min(),q.index.max()),('2020-22','2020-01-01','2022-12-31'),('2023-24','2023-01-01','2024-12-31'),('2025-26','2025-01-01','2026-12-31'),('2027','2027-01-01','2027-12-31')]:
 x=q.loc[a:b]
 if len(x): print(label,'dates',len(x),'IC',x.ic.mean(),'ICIR',x.ic.mean()/(x.ic.std(ddof=1)+1e-12),'hit',np.mean(x.ic>0),'N',x.n.mean())
print('coverage',len(q)/len(fac),'turnover',np.mean(turns),'dates',len(q),'instruments',len(U))
for h in [1,5,10]:
 yy=p.shift(-h)/p-1; vals=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('horizon',h,'IC',np.nanmean(vals),'dates',len(vals))
fac.reset_index().to_csv('scripts/miner_2_20270325_acceleration_signal.csv',index=False)
