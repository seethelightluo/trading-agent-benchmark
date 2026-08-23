import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)<100: d=get_index_daily_data(s,4000)
 return None if d is None else d.sort_values('date').copy()
px={s:load(s) for s in U}
for s,d in px.items():
 if d is not None:
  r=d.close.pct_change(); d['r10']=d.close.pct_change(10); d['breadth']=((r>0).rolling(10).mean()*2-1); d['downvol']=r.where(r<0).rolling(20).std()*np.sqrt(20)
rel=pd.concat([d.set_index('date').r10.rename(s) for s,d in px.items() if d is not None],axis=1); med=rel.median(axis=1)
rows=[]
for s,d0 in px.items():
 if d0 is None: continue
 d=d0.set_index('date'); common=d.index.intersection(med.index)
 for dt in common:
  i=d.index.get_loc(dt); den=d.loc[dt,'downvol'];
  if i+10<len(d) and pd.notna(d.loc[dt,'r10']) and pd.notna(den) and den>1e-8:
   f=(d.loc[dt,'r10']-med.loc[dt])*d.loc[dt,'breadth']/den; fw=d.close.iloc[i+10]/d.close.iloc[i]-1
   if np.isfinite(f) and np.isfinite(fw): rows.append((dt,s,f,fw))
x=pd.DataFrame(rows,columns=['date','symbol','factor','fwd'])
def stats(z):
 q=[]
 for _,g in z.groupby('date'):
  if len(g)>=8: q.append(g.factor.corr(g.fwd,method='spearman'))
 q=pd.Series(q).dropna(); return len(q),round(z.groupby('date').size().mean(),2),q.mean(),q.mean()/q.std(ddof=1),(q>0).mean(),z.symbol.nunique()
print('range',x.date.min(),x.date.max(),'rows',len(x),'assets',x.symbol.nunique()); print('overall',stats(x))
for a,b in [('2020','2022'),('2023','2024'),('2025','2026'),('2027','2029'),('2030','2033')]: print('regime',a,b,stats(x[(x.date>=a)&(x.date<=b)]))
for h in [5,10,20,40]:
 q=[]
 for dt,g in x.groupby('date'):
  vals=[]
  for _,r in g.iterrows():
   d=px[r.symbol].set_index('date'); i=d.index.get_loc(r.date)
   if i+h<len(d): vals.append((r.factor,d.close.iloc[i+h]/d.close.iloc[i]-1))
  if len(vals)>=8: q.append(pd.DataFrame(vals,columns=['a','b']).a.corr(pd.DataFrame(vals,columns=['a','b']).b,method='spearman'))
 q=pd.Series(q).dropna(); print('decay',h,len(q),q.mean(),q.mean()/q.std(ddof=1))
ranks=x.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True); print('turnover',ranks.diff().abs().mean(axis=1).mean(),'coverage',x.symbol.nunique()/15,'avg_n',x.groupby('date').size().mean())
