import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 d=get_stock_daily_data(s,2300)
 if d is None or len(d)<100:d=get_index_daily_data(s,2300)
 return None if d is None else d.sort_values('date')
px={s:load(s) for s in U}; rows=[]
for s,d in px.items():
 if d is None:continue
 r=d.close.pct_change(); f=d.close.pct_change(10)*(2*r.gt(0).rolling(10).mean()-1)
 for i in range(20,len(d)-10):rows.append((d.date.iloc[i],s,f.iloc[i],d.close.iloc[i+10]/d.close.iloc[i]-1))
x=pd.DataFrame(rows,columns=['date','symbol','factor','fwd']).dropna()
def st(z):
 q=[]
 for _,g in z.groupby('date'):
  if len(g)>=8:q.append(g.factor.corr(g.fwd,method='spearman'))
 q=pd.Series(q).dropna();return len(q),z.groupby('date').size().mean(),q.mean(),q.mean()/q.std(ddof=1),(q>0).mean(),z.symbol.nunique()
print('overall',st(x))
for a,b in [('2025','2026'),('2027','2029'),('2030','2033')]:print('regime',a,b,st(x[(x.date>=a)&(x.date<=b)]))
for h in [5,10,20,40]:
 z=[]
 for s,d in px.items():
  if d is None:continue
  r=d.close.pct_change();f=d.close.pct_change(10)*(2*r.gt(0).rolling(10).mean()-1)
  for i in range(20,len(d)-h):z.append((d.date.iloc[i],s,f.iloc[i],d.close.iloc[i+h]/d.close.iloc[i]-1))
 z=pd.DataFrame(z,columns=['date','symbol','factor','fwd']).dropna();print('decay',h,st(z)[:4])
p=x.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True);print('turnover',p.diff().abs().mean(axis=1).mean(),'coverage',len(x)/(len(px)*x.date.nunique()),'dates',x.date.nunique(),'avg_n',x.groupby('date').size().mean())
