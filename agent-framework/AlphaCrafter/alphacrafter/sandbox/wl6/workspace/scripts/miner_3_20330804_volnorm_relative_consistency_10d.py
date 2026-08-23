import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s,n=4000):
 d=get_stock_daily_data(s,n)
 if d is None or len(d)<100: d=get_index_daily_data(s,n)
 return d.sort_values('date').copy() if d is not None else None
px={s:get(s) for s in U}
for s,d in px.items():
 if d is not None:
  d['r']=d.close.pct_change(); d['ret20']=d.close.pct_change(20)
  d['breadth']=(d.r>0).rolling(20).mean()*2-1
  d['vol']=d.r.rolling(20).std()*np.sqrt(20)
rel=pd.concat([d.set_index('date')['ret20'].rename(s) for s,d in px.items() if d is not None],axis=1)
median=rel.median(axis=1); rows=[]
for s,d0 in px.items():
 if d0 is None: continue
 d=d0.set_index('date'); common=d.index.intersection(median.index)
 for dt in common:
  loc=d.index.get_loc(dt)
  if loc+10<len(d): rows.append((dt,s,(d.loc[dt,'ret20']-median.loc[dt])*d.loc[dt,'breadth']/max(d.loc[dt,'vol'],1e-8),d.close.iloc[loc+10]/d.close.iloc[loc]-1))
x=pd.DataFrame(rows,columns=['date','symbol','factor','fwd']).replace([np.inf,-np.inf],np.nan).dropna()
def stats(z):
 q=[]
 for _,g in z.groupby('date'):
  if len(g)>=8: q.append(g.factor.corr(g.fwd,method='spearman'))
 q=pd.Series(q).dropna(); return len(q),round(z.groupby('date').size().mean(),2),q.mean(),q.mean()/q.std(ddof=1),(q>0).mean(),z.symbol.nunique()
print('dates',x.date.min(),x.date.max(),'rows',len(x)); print('overall',stats(x))
for a,b in [('2020','2022'),('2023','2024'),('2025','2026'),('2027','2029'),('2030','2033')]: print('regime',a,b,stats(x[(x.date>=a)&(x.date<=b)]))
for h in [5,10,20,40]:
 q=[]
 for _,g in x.groupby('date'):
  vals=[]
  for _,r in g.iterrows():
   d=px[r.symbol].set_index('date'); loc=d.index.get_loc(r.date)
   if loc+h<len(d): vals.append((r.factor,d.close.iloc[loc+h]/d.close.iloc[loc]-1))
  if len(vals)>=8:
   v=pd.DataFrame(vals,columns=['factor','fwd']); q.append(v.factor.corr(v.fwd,method='spearman'))
 q=pd.Series(q).dropna(); print('decay',h,len(q),q.mean(),q.mean()/q.std(ddof=1))
rank=x.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True)
print('turnover',rank.diff().abs().mean(axis=1).mean(),'coverage',x.symbol.nunique()/15,'avg_n',x.groupby('date').size().mean())
