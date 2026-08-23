import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d): px[s]=d.assign(date=pd.to_datetime(d.date)).set_index('date').close
P=pd.DataFrame(px).sort_index()
rng=P.rolling(20,min_periods=15); hi=rng.max().shift(1); lo=rng.min().shift(1)
rev=-(P-hi)/(hi-lo+1e-12)
mom=P/P.shift(60)-1; mom[P.shift(60).isna()]=np.nan
def cs_z(x):
 return x.sub(x.mean(axis=1),axis=0).div(x.std(axis=1).replace(0,np.nan),axis=0)
f=cs_z(rev)+0.35*cs_z(mom)
for h in [5,10,20]:
 fr=P.shift(-h)/P-1; rows=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
 q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); a=q.ic
 print('H',h,'dates',len(q),'avgN',round(q.n.mean(),3),'coverage',round(q.n.mean()/15,6),'IC',round(a.mean(),8),'ICIR',round(a.mean()/a.std(ddof=1)*np.sqrt(len(a)),8),'hit',round((a>0).mean(),6))
 print('years',q.groupby(q.index.year).ic.mean().round(4).to_dict())
print('turnover_proxy',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),8))
