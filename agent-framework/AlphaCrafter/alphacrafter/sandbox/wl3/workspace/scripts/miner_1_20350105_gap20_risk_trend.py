import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 d=get_stock_daily_data(s,4200)
 if d is None or len(d)<150: d=get_index_daily_data(s,4200)
 return d
rows=[]
for s in U:
 df=get(s)
 if df is None or len(df)<150: continue
 x=df[['date','close']].drop_duplicates('date').set_index('date').sort_index()
 f=(x.close.pct_change(20).shift(5)/(x.close.pct_change().rolling(20).std().shift(5)*np.sqrt(20))).rename('f')
 fr=x.close.pct_change(5).shift(-5).rename('fr')
 z=pd.concat([f,fr],axis=1).dropna().reset_index(); z['asset']=s; rows.append(z)
d=pd.concat(rows,ignore_index=True)
def calc(z):
 out=[]
 for dt,g in z.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1: out.append(g.f.corr(g.fr,method='spearman'))
 return pd.Series(out).dropna()
ic=calc(d)
print('dates',len(ic),'rows',len(d),'avg_instruments',d.groupby('date').size().mean(),'coverage',len(d)/(d.date.nunique()*15))
print('IC',ic.mean(),'ICIR',ic.mean()/ic.std(ddof=1),'hit',(ic>0).mean())
for n in [120,252,504]:
 q=ic.tail(n); print('recent',n,'n',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1))
p=d.pivot(index='date',columns='asset',values='f').rank(axis=1,pct=True); print('turnover',p.diff().abs().mean(axis=1).dropna().mean())
for h in [1,3,5,10]:
 rr=[]
 for s in U:
  df=get(s)
  if df is None: continue
  x=df[['date','close']].drop_duplicates('date').set_index('date').sort_index()
  f=x.close.pct_change(20).shift(5)/(x.close.pct_change().rolling(20).std().shift(5)*np.sqrt(20)); fr=x.close.pct_change(h).shift(-h)
  rr.append(pd.concat([f.rename('f'),fr.rename('fr')],axis=1).dropna().reset_index())
 ii=calc(pd.concat(rr,ignore_index=True)); print('horizon',h,'IC',ii.mean(),'ICIR',ii.mean()/ii.std(ddof=1),'dates',len(ii))
