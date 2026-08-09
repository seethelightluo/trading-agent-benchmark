import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=get_stock_daily_data(s,days=3000)
 if x is not None and len(x):
  x=x.copy();x.date=pd.to_datetime(x.date).dt.normalize();D[s]=x.set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change()
# downside asymmetry: prefer assets with low downside deviation relative to total risk
sd=r.rolling(30).std(); down=(-r.clip(upper=0)).rolling(30).apply(lambda x: np.sqrt(np.mean(x*x)), raw=True); f=-(down/sd)
rows=[]
for d in f.index:
 v=f.loc[d]; rec={'date':d,'n':v.notna().sum()}
 for h in [1,5,10]:
  y=p.pct_change(h).shift(-h).loc[d];z=pd.concat([v,y],axis=1).dropna()
  rec['ic'+str(h)]=z.iloc[:,0].rank().corr(z.iloc[:,1].rank()) if len(z)>=8 else np.nan
 rows.append(rec)
a=pd.DataFrame(rows).set_index('date').dropna(subset=['ic1'])
print('candidate downside-asymmetry; dates',len(a),'avg_names',a.n.mean(),'coverage',f.notna().mean().mean(),'turnover',f.rank(pct=True).diff().abs().mean().mean())
for h in [1,5,10]:
 q=a['ic'+str(h)].dropna();print(h,'IC %.5f ICIR %.5f hit %.3f n=%d'%(q.mean(),q.mean()/q.std(ddof=1),(q>0).mean(),len(q)))
for y,g in a.groupby(a.index.year):
 q=g.ic1;print('year',y,'IC %.5f ICIR %.5f'%(q.mean(),q.mean()/q.std(ddof=1)))
for label,g in [('early',a[a.index<'2023-01-01']),('late',a[a.index>='2023-01-01']),('recent',a[a.index>='2025-01-01'])]:
 q=g.ic1;print(label,'IC %.5f ICIR %.5f hit %.3f n=%d'%(q.mean(),q.mean()/q.std(ddof=1),(q>0).mean(),len(q)))
