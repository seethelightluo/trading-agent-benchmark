import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2027-02-23')
def fetch(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,3000)
   if d is not None and len(d):
    d=d.copy(); d.date=pd.to_datetime(d.date).dt.normalize(); return d.drop_duplicates('date').set_index('date').sort_index().loc[:CUT]
  except Exception: pass
 return None
D={s:fetch(s) for s in U}; D={s:d for s,d in D.items() if d is not None}; rows=[]
for s,d in D.items():
 c=d.close.astype(float); r=c.pct_change(); vol=r.rolling(20,min_periods=15).std()
 r10=r.rolling(10,min_periods=8).sum(); r30=r.rolling(30,min_periods=20).sum(); r60=r.rolling(60,min_periods=40).sum()
 # trend agreement: volatility-scaled multi-horizon return, rewarded when horizons agree
 base=(.30*r10+.45*r30+.25*r60)/(vol+1e-12)
 agree=(np.sign(r10)+np.sign(r30)+np.sign(r60))/3
 f=(base*(0.5+0.5*np.abs(agree))*np.sign(agree)).shift(1)
 rows.append(pd.DataFrame({'date':c.index,'asset':s,'f':f,'fr':c.shift(-1)/c-1}))
q=pd.concat(rows,ignore_index=True).replace([np.inf,-np.inf],np.nan).dropna()
def st(x):
 z=[]; ns=[]
 for _,g in x.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1:z.append(g.f.corr(g.fr,method='spearman'));ns.append(len(g))
 z=pd.Series(z); return len(z),round(np.mean(ns),2),round(z.mean(),5),round(z.mean()/z.std(ddof=1)*np.sqrt(252),4),round((z>0).mean(),4)
print('assets',len(D),'rows',len(q),'dates',q.date.nunique(),'avg_n',round(q.groupby('date').size().mean(),2),'coverage',round(len(q)/(q.date.nunique()*len(D)),4))
print('daily',st(q))
for a,b in [(2020,2022),(2023,2024),(2025,2027)]: print('regime',a,b,st(q[(q.date.dt.year>=a)&(q.date.dt.year<=b)]))
r=q.pivot(index='date',columns='asset',values='f').rank(axis=1,pct=True); print('turnover',round(r.diff().abs().mean().mean(),5))
for h in [1,5,10]:
 z=[]
 for _,g in q.assign(fr=q.groupby('asset').fr.shift(-(h-1))).dropna().groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1:z.append(g.f.corr(g.fr,method='spearman'))
 print('decay',h,round(np.mean(z),5),len(z))
q.to_csv('scripts/miner_3_20270224_trend_agreement_signal.csv',index=False)
