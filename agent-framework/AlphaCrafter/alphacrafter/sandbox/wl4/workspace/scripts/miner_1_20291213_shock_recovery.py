import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
try: macro=get_index_daily_data('VIX',days=4000)
except: macro=None
D={}
for s in U:
 try: d=get_index_daily_data(s,days=4000)
 except: d=None
 if d is None:
  try: d=get_stock_daily_data(s,days=4000)
  except: d=None
 if d is not None:
  d=d[['date','close']].copy(); d['r']=d.close.pct_change(); D[s]=d.set_index('date')
rows=[]
for s,d in D.items():
 for i in range(30,len(d)-10):
  dt=d.index[i]; r=d.r.iloc[:i]; w=r.iloc[-20:]; worst=w.idxmin(); pos=d.index.get_loc(worst)
  if pos+3>=i: continue
  rebound=d.close.iloc[i-1]/d.close.iloc[pos]-1; vol=w.std(); fac=rebound/(vol*np.sqrt(3)) if vol>0 else np.nan
  fwd=d.close.iloc[i+10]/d.close.iloc[i]-1; rows.append((dt,s,fac,fwd))
x=pd.DataFrame(rows,columns=['date','symbol','factor','fwd']).dropna(); ics=[]
for dt,g in x.groupby('date'):
 if len(g)>=8 and g.factor.nunique()>2: ics.append(g.factor.corr(g.fwd,method='spearman'))
ic=pd.Series(ics).dropna(); turns=[]
for s,g in x.groupby('symbol'):
 z=g.sort_values('date').factor.rank(pct=True); turns+=list(z.diff().abs().dropna())
print('assets',len(D),'dates',len(ic),'avgN',x.groupby('date').size().mean(),'obs',len(x),'IC',ic.mean(),'ICIR',ic.mean()/ic.std(),'hit',(ic>0).mean(),'turnover',np.mean(turns))
for label,sub in [('2020-2025',x[x.date<'2026-01-01']),('2026-2027',x[(x.date>='2026-01-01')&(x.date<'2028-01-01')]),('2028-2029',x[x.date>='2028-01-01'])]:
 q=[]
 for _,g in sub.groupby('date'):
  if len(g)>=8 and g.factor.nunique()>2:q.append(g.factor.corr(g.fwd,method='spearman'))
 q=pd.Series(q).dropna(); print(label,len(q),q.mean(),q.mean()/q.std() if len(q)>1 else np.nan)
print('recent250',ic.tail(250).mean(),ic.tail(250).mean()/ic.tail(250).std())
