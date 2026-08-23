import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)<100:
  d=get_index_daily_data(s,5000)
 return None if d is None else d.set_index(pd.to_datetime(d.date)).close.astype(float)
P=pd.concat({s:load(s) for s in U if load(s) is not None},axis=1).sort_index(); r=P.pct_change()
mom=P.pct_change(20); vol=r.rolling(20).std()*np.sqrt(252); slow=P.pct_change(60)
f=(mom/vol).where(slow>0).shift(1)
for h in [5,10,20,40]:
 fr=P.pct_change(h).shift(-h); vals=[]; counts=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(c): vals.append(c); counts.append(len(z))
 q=np.array(vals); print(f'{h}D dates={len(q)} avg_n={np.mean(counts):.3f} coverage={np.mean(counts)/15:.4f} IC={q.mean():.8f} ICIR={q.mean()/q.std(ddof=1)*np.sqrt(len(q)):.8f} hit={np.mean(q>0):.4f}')
rank=f.rank(axis=1,pct=True); print('overall_dates',len(f),'avg_valid',f.notna().sum(axis=1).mean(),'coverage',f.notna().sum().sum()/(len(f)*15),'turnover',rank.diff().abs().mean(axis=1).mean())
fr=P.pct_change(20).shift(-20); q=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8:q.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman')))
q=pd.DataFrame(q,columns=['date','ic']).set_index('date')
for a,b in [('2020','2027'),('2027','2032'),('2032','2035-01-17')]:
 x=q.loc[a:b,'ic']; print('regime',a,b,'dates',len(x),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1)*np.sqrt(len(x)))
