import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2027-03-18')
def fetch(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,3000)
   if d is not None and len(d):
    d=d.copy(); d.date=pd.to_datetime(d.date).dt.normalize(); return d.drop_duplicates('date').set_index('date').sort_index().loc[:CUT]
  except Exception: pass
 return None
D={s:fetch(s) for s in U}; D={s:d for s,d in D.items() if d is not None}
# Efficiency-weighted trend: medium-term return rewarded when daily path is directional;
# lag one day. This is interpretable and reduces noisy/choppy momentum.
def build(h):
 rows=[]
 for s,d in D.items():
  c=d.close; r=c.pct_change(); net=c.pct_change(20).abs(); path=r.abs().rolling(20,min_periods=15).sum(); trend=c.pct_change(20)
  f=(trend*(net/(path+1e-12))).shift(1); fr=c.shift(-h)/c-1
  rows.append(pd.DataFrame({'date':c.index,'asset':s,'signal':f,'forward_return':fr}))
 return pd.concat(rows,ignore_index=True).replace([np.inf,-np.inf],np.nan).dropna()
def calc(q):
 a=[]; ns=[]
 for dt,g in q.groupby('date'):
  if len(g)>=8 and g.signal.nunique()>1 and g.forward_return.nunique()>1:
   a.append(g.signal.corr(g.forward_return,method='spearman')); ns.append(len(g))
 x=pd.Series(a); return len(x),float(np.mean(ns)),float(x.mean()),float(x.mean()/x.std(ddof=1)*np.sqrt(252)),float((x>0).mean())
q=build(1); print('assets',len(D),'dates',q.date.nunique(),'coverage',len(q)/(q.date.nunique()*len(D)),'avg_n',q.groupby('date').size().mean())
for h in [1,5,10,20]: print('horizon',h,calc(build(h)))
for lo,hi in [(2020,2022),(2023,2024),(2025,2027)]: print('regime',lo,hi,calc(q[(q.date.dt.year>=lo)&(q.date.dt.year<=hi)]))
p=q.pivot(index='date',columns='asset',values='signal').rank(axis=1,pct=True); print('turnover',p.diff().abs().mean().mean())
q.to_csv('scripts/miner_3_20270319_efficiency_trend_signal.csv',index=False)
