import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2027-04-21')
def fetch(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,3000)
   if d is not None and len(d):
    d=d.copy(); d.date=pd.to_datetime(d.date).dt.normalize(); return d.drop_duplicates('date').set_index('date').sort_index().loc[:CUT]
  except Exception: pass
D={s:fetch(s) for s in U}; D={s:d for s,d in D.items() if d is not None and len(d)>100}
# Cross-sectional residual momentum: asset 20d return relative to same-day universe median, scaled by own 30d vol, lagged.
rows=[]
for s,d in D.items():
 c=d.close.astype(float); ret20=c.pct_change(20); vol=c.pct_change().rolling(30,min_periods=20).std()
 rows.append(pd.DataFrame({'date':c.index,'asset':s,'r20':ret20.values,'vol':vol.values}))
x=pd.concat(rows,ignore_index=True); med=x.pivot(index='date',columns='asset',values='r20').median(axis=1)
x['f']=((x.r20-x.date.map(med))/(x.vol+1e-12)).shift(0)
# lag signal by one calendar panel date per asset (strictly prior observation)
x['f']=x.groupby('asset').f.shift(1)
for h in [1,5,10,20]:
 rows=[]
 for s,d in D.items():
  c=d.close.astype(float); rows.append(pd.DataFrame({'date':c.index,'asset':s,'fr':(c.shift(-h)/c-1).values}))
 y=pd.concat(rows,ignore_index=True); q=x.merge(y,on=['date','asset']).replace([np.inf,-np.inf],np.nan).dropna()
 vals=[]; ns=[]
 for dt,g in q.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1: vals.append(g.f.corr(g.fr,method='spearman')); ns.append(len(g))
 z=pd.Series(vals); ic=z.mean(); icir=ic/z.std(ddof=1)*np.sqrt(252)
 print('horizon',h,'dates',len(z),'avg_assets',np.mean(ns),'IC',round(ic,6),'ICIR',round(icir,4),'hit',round((z>0).mean(),4))
 if h==1:
  rank=q.pivot(index='date',columns='asset',values='f').rank(axis=1,pct=True); print('coverage',round(len(q)/(q.date.nunique()*15),4),'turnover',round(rank.diff().abs().mean().mean(),4)); q.to_csv('scripts/miner_1_20270422_residual_momentum_signal.csv',index=False)
for a,b in [(2020,2022),(2023,2024),(2025,2027)]:
 g=q[(q.date.dt.year>=a)&(q.date.dt.year<=b)]; vals=[]
 for _,v in g.groupby('date'):
  if len(v)>=8 and v.f.nunique()>1 and v.fr.nunique()>1: vals.append(v.f.corr(v.fr,method='spearman'))
 print('regime',a,b,'dates',len(vals),'IC',round(np.mean(vals),6) if vals else None)
print('assets',len(D),'dates',q.date.nunique())
