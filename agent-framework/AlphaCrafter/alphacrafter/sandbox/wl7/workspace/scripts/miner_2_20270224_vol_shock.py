import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2027-02-24')
def fetch(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,3000)
   if d is not None and len(d):
    d=d.copy();d.date=pd.to_datetime(d.date).dt.normalize();return d.drop_duplicates('date').set_index('date').sort_index().loc[:CUT]
  except Exception: pass
D={s:fetch(s) for s in U};D={s:d for s,d in D.items() if d is not None};rows=[]
for s,d in D.items():
 c=d.close.astype(float);r=c.pct_change();v10=r.rolling(10,min_periods=7).std();v60=r.rolling(60,min_periods=30).std();shock=(v10/(v60+1e-12)).shift(1);f=(-(c.pct_change(3))/(v10+1e-12)*shock).shift(1)
 for h in [1,5,10]:
  pass
 for i in range(len(c)): rows.append({'date':c.index[i],'asset':s,'f':f.iloc[i],**{f'fr{h}':(c.shift(-h)/c-1).iloc[i] for h in [1,5,10]}})
q=pd.DataFrame(rows).replace([np.inf,-np.inf],np.nan)
def stats(x,col):
 z=[];ns=[]
 for _,g in x[['date','asset','f',col]].dropna().groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g[col].nunique()>1:z.append(g.f.corr(g[col],method='spearman'));ns.append(len(g))
 z=pd.Series(z);return len(z),round(np.mean(ns),2),round(z.mean(),5),round(z.mean()/z.std(ddof=1)*np.sqrt(252),4),round((z>0).mean(),4)
print('assets',len(D),'rows',len(q),'dates',q.date.nunique(),'coverage',round(q.f.notna().mean(),4))
for h in [1,5,10]:print('horizon',h,stats(q,f'fr{h}'))
for a,b in [(2020,2022),(2023,2024),(2025,2027)]:print('regime',a,b,stats(q[(q.date.dt.year>=a)&(q.date.dt.year<=b)],'fr1'))
r=q.pivot(index='date',columns='asset',values='f').rank(axis=1,pct=True);print('turnover',round(float(r.diff().abs().mean().mean()),5));q.to_csv('scripts/miner_2_20270224_vol_shock_signal.csv',index=False)
