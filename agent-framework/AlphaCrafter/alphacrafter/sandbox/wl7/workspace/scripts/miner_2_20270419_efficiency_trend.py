import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2027-04-18')
def fetch(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,3000)
   if d is not None and len(d):
    d=d.copy(); d.date=pd.to_datetime(d.date).dt.normalize(); return d.drop_duplicates('date').set_index('date').sort_index().loc[:CUT]
  except Exception: pass
D={s:fetch(s) for s in U}; D={s:d for s,d in D.items() if d is not None}
p=pd.concat({s:d.close.astype(float) for s,d in D.items()},axis=1).sort_index(); r=p.pct_change(); rows=[]
# Efficient directional trend: net 20d return / total absolute daily movement, with volatility penalty.
for s in D:
 rr=r[s]; net=rr.rolling(20).sum(); path=rr.abs().rolling(20).sum(); vol=rr.rolling(40).std()*np.sqrt(20)
 f=(net/(path+1e-12)/(vol+1e-12)).shift(1)
 rows.append(pd.DataFrame({'date':p.index,'asset':s,'f':f}))
base=pd.concat(rows,ignore_index=True)
def evalh(h):
 xs=[]
 for s in D:
  fr=p[s].shift(-h)/p[s]-1
  z=base[base.asset.eq(s)].copy(); z['fr']=fr.reindex(z.date).to_numpy(); xs.append(z)
 x=pd.concat(xs).replace([np.inf,-np.inf],np.nan).dropna(); vals=[]; ns=[]
 for dt,g in x.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1: vals.append(g.f.corr(g.fr,method='spearman')); ns.append(len(g))
 z=pd.Series(vals); return len(z),np.mean(ns),z.mean(),z.mean()/z.std(ddof=1)*np.sqrt(252),(z>0).mean()
q=base.dropna(); print('assets',len(D),'dates',q.date.nunique(),'rows',len(q),'coverage',len(q)/(q.date.nunique()*15))
for h in [1,5,10,20]: print('horizon',h,evalh(h))
for lo,hi in [(2020,2022),(2023,2024),(2025,2027)]:
 x=q[(q.date.dt.year>=lo)&(q.date.dt.year<=hi)]; xs=[]
 for s in D:
  z=x[x.asset.eq(s)].copy(); z['fr']=p[s].shift(-1).reindex(z.date).to_numpy()/p[s].reindex(z.date).to_numpy()-1; xs.append(z)
 y=pd.concat(xs).dropna(); vals=[]
 for dt,g in y.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1: vals.append(g.f.corr(g.fr,method='spearman'))
 print('regime',lo,hi,len(vals),np.mean(vals) if vals else np.nan)
rank=q.pivot(index='date',columns='asset',values='f').rank(axis=1,pct=True); print('turnover',rank.diff().abs().mean().mean())
q.to_csv('scripts/miner_2_20270419_efficiency_trend_signal.csv',index=False)
print('signal_artifact','scripts/miner_2_20270419_efficiency_trend_signal.csv')
