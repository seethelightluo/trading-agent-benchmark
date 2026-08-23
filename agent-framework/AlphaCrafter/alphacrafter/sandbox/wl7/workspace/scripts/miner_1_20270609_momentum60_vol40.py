import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2027-06-08')
def fetch(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,3000)
   if d is not None and len(d):
    d=d.copy();d.date=pd.to_datetime(d.date).dt.normalize();return d.drop_duplicates('date').set_index('date').sort_index().loc[:CUT]
  except Exception: pass
D={s:fetch(s) for s in U};D={s:d for s,d in D.items() if d is not None}; rows=[]
for s,d in D.items():
 c=d.close.astype(float); r=c.pct_change(); vol=r.rolling(40).std()*np.sqrt(40)
 f=(c.pct_change(60)/(vol+1e-8)).shift(1); fr=c.shift(-10)/c-1
 rows.append(pd.DataFrame({'date':c.index,'asset':s,'f':f,'fr':fr}))
q=pd.concat(rows,ignore_index=True).replace([np.inf,-np.inf],np.nan).dropna()
def stats(x):
 v=[]; nn=[]
 for _,g in x.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1:v.append(g.f.corr(g.fr,method='spearman'));nn.append(len(g))
 z=pd.Series(v);return len(z),np.mean(nn),z.mean(),z.mean()/z.std(ddof=1)*np.sqrt(252),(z>0).mean()
print('assets',len(D),'dates',q.date.nunique(),'avg_n',q.groupby('date').size().mean(),'coverage',len(q)/(q.date.nunique()*15));print('10d',stats(q))
for h in [1,5,10,20]:
 rr=[]
 for s,d in D.items():
  c=d.close.astype(float);f=(c.pct_change(60)/(c.pct_change().rolling(40).std()*np.sqrt(40)+1e-8)).shift(1);rr.append(pd.DataFrame({'date':c.index,'asset':s,'f':f,'fr':c.shift(-h)/c-1}))
 print('horizon',h,stats(pd.concat(rr,ignore_index=True).dropna()))
for a,b in [(2020,2022),(2023,2024),(2025,2027)]: print('regime',a,b,stats(q[(q.date.dt.year>=a)&(q.date.dt.year<=b)]))
r=q.pivot(index='date',columns='asset',values='f').rank(axis=1,pct=True);print('turnover',r.diff().abs().mean().mean());q.to_csv('scripts/miner_1_20270609_momentum60_vol40_signal.csv',index=False)
