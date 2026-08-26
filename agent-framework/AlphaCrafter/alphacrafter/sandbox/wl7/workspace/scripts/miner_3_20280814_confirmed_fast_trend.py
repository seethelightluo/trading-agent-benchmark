import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2028-08-13')
def load(s):
    for fn in (get_index_daily_data,get_stock_daily_data):
        try:
            d=fn(s,3000)
            if d is not None and len(d):
                d=d.copy(); d.date=pd.to_datetime(d.date).dt.normalize(); return d.drop_duplicates('date').set_index('date').sort_index().loc[:CUT]
        except Exception: pass
D={s:load(s) for s in U}; D={s:d for s,d in D.items() if d is not None}
rows=[]
for s,d in D.items():
 c=pd.to_numeric(d.close,errors='coerce'); r10=c.pct_change(10); r60=c.pct_change(60); vol=c.pct_change().rolling(20,min_periods=15).std()*np.sqrt(252)
 # fast trend signal confirmed only when medium trend agrees; lag all inputs one day
 f=(r10/(vol+0.03))*np.sign(r60)
 rows.append(pd.DataFrame({'date':c.index,'asset':s,'f':f.shift(1),'fr1':c.shift(-1)/c-1,'fr5':c.shift(-5)/c-1,'fr10':c.shift(-10)/c-1,'fr20':c.shift(-20)/c-1}))
x=pd.concat(rows,ignore_index=True).replace([np.inf,-np.inf],np.nan)
def stat(z,col):
 vals=[]; ns=[]
 for _,g in z.groupby('date'):
  g=g.dropna(subset=['f',col])
  if len(g)>=8 and g.f.nunique()>1 and g[col].nunique()>1: vals.append(g.f.corr(g[col],method='spearman')); ns.append(len(g))
 v=pd.Series(vals).dropna(); return len(v),float(np.mean(ns)),float(v.mean()),float(v.mean()/v.std(ddof=1)*np.sqrt(252)),float((v>0).mean())
q=x.dropna(subset=['f','fr1']); print('assets',len(D),'dates',q.date.nunique(),'avg_n',q.groupby('date').size().mean(),'coverage',len(q)/(q.date.nunique()*len(D)))
for h in ['fr1','fr5','fr10','fr20']: print(h,stat(q,h))
for a,b in [(2020,2022),(2023,2024),(2025,2026),(2027,2028)]: print('regime',a,b,stat(q[(q.date.dt.year>=a)&(q.date.dt.year<=b)],'fr10'))
r=q.pivot(index='date',columns='asset',values='f').rank(axis=1,pct=True); print('turnover',float(r.diff().abs().mean().mean()))
q.to_csv('scripts/miner_3_20280814_confirmed_fast_trend_signal.csv',index=False)
