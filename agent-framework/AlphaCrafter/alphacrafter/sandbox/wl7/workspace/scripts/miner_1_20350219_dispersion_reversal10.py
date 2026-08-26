import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2035-02-18')
def load(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,days=4000)
   if d is not None and len(d):
    d=d.copy(); d.date=pd.to_datetime(d.date).dt.normalize(); return d.drop_duplicates('date').set_index('date').sort_index().loc[:END]
  except Exception: pass
D={s:load(s) for s in U}; D={s:d for s,d in D.items() if d is not None and len(d)>100}
rows=[]
for s,d in D.items():
 c=d.close.astype(float); r=c.pct_change()
 rows.append(pd.DataFrame({'date':c.index,'asset':s,'ret10':c.pct_change(10),'r':r,'fr1':c.shift(-1)/c-1,'fr5':c.shift(-5)/c-1,'fr10':c.shift(-10)/c-1}))
x=pd.concat(rows,ignore_index=True)
piv=x.pivot(index='date',columns='asset',values='ret10'); daily=x.pivot(index='date',columns='asset',values='r')
# Relative reversal, damped when cross-sectional dispersion is unusually high.
res=piv.sub(piv.median(axis=1),axis=0); disp=daily.sub(daily.median(axis=1),axis=0).rolling(20,min_periods=15).std().median(axis=1)
damp=(disp.rolling(120,min_periods=60).median()/(disp+1e-12)).clip(0.5,2.0)
sig=res.mul(-damp,axis=0).shift(1)
x['f']=x.set_index(['date','asset']).index.map(lambda z: sig.loc[z[0],z[1]] if z[0] in sig.index and z[1] in sig.columns else np.nan)

def stats(q,col):
 vals=[]; ns=[]
 for _,g in q.groupby('date'):
  g=g.dropna(subset=['f',col])
  if len(g)>=8 and g.f.nunique()>1 and g[col].nunique()>1:
   v=g.f.corr(g[col],method='spearman')
   if np.isfinite(v): vals.append(v);ns.append(len(g))
 v=np.array(vals); return len(v),np.mean(ns),v.mean(),v.mean()/v.std(ddof=1)*np.sqrt(252),np.mean(v>0)
q=x.dropna(subset=['f','fr1'])
print('assets',len(D),'dates',q.date.nunique(),'avg_n',q.groupby('date').size().mean(),'coverage',len(q)/(q.date.nunique()*len(U)))
for col in ['fr1','fr5','fr10']: print(col,stats(q,col))
for a,b in [(2020,2023),(2024,2027),(2028,2031),(2032,2035)]:
 z=q[(q.date.dt.year>=a)&(q.date.dt.year<=b)]; print('regime',a,b,stats(z,'fr10'))
r=sig.rank(axis=1,pct=True); print('turnover',r.diff().abs().mean().mean())
out=sig.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_1_20350219_dispersion_reversal10_signal.csv',index=False)
