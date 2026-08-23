import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 for fn in (get_stock_daily_data,get_index_daily_data):
  try:
   d=fn(s,2000)
   if d is not None and len(d)>=100:return d.sort_values('date').copy()
  except Exception: pass
 return None
px={s:load(s) for s in U}; print('available',[(s,len(d) if d is not None else 0) for s,d in px.items()])
for s,d in px.items():
 if d is not None:
  r=d.close.pct_change(); d['f']=d.close.pct_change(5)/(r.rolling(20).std()*np.sqrt(20)); d['f']*=((r>0).rolling(10).mean()*2-1)
rows=[]
for s,d in px.items():
 if d is None:continue
 for i in range(20,len(d)-10):
  f=d.f.iloc[i]; fw=d.close.iloc[i+10]/d.close.iloc[i]-1
  if np.isfinite(f) and np.isfinite(fw):rows.append((d.date.iloc[i],s,f,fw))
x=pd.DataFrame(rows,columns=['date','symbol','f','fw'])
q=[]
for _,g in x.groupby('date'):
 if len(g)>=8:q.append(g.f.corr(g.fw,method='spearman'))
q=pd.Series(q).dropna();print('dates',len(q),'assets',x.symbol.nunique(),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'avgN',x.groupby('date').size().mean())
for a,b in [('2020','2022'),('2023','2025'),('2026','2029'),('2030','2033')]:
 z=x[(x.date>=a)&(x.date<=b)]; qq=[]
 for _,g in z.groupby('date'):
  if len(g)>=8:qq.append(g.f.corr(g.fw,method='spearman'))
 qq=pd.Series(qq).dropna();print(a,b,len(qq),qq.mean(),qq.mean()/qq.std(ddof=1) if len(qq)>1 else np.nan)
r=x.pivot(index='date',columns='symbol',values='f').rank(axis=1,pct=True);print('coverage',x.symbol.nunique()/15,'turnover',r.diff().abs().mean(axis=1).mean())
