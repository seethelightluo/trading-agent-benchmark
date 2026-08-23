import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2027-05-18')
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
 c=d.close.astype(float); h=d.high.astype(float); l=d.low.astype(float); o=d.open.astype(float)
 # Close-location weighted reversal: recent losses are stronger when they closed near lows,
 # with all inputs lagged before the forecast date.
 clv=((2*c-h-l)/(h-l+1e-12)).clip(-1,1)
 ret3=c.pct_change(3); vol20=c.pct_change().rolling(20,min_periods=10).std()
 f=(-(ret3/(vol20*np.sqrt(3)+1e-12))*((1-clv)/2)).shift(1)
 rows.append(pd.DataFrame({'date':c.index,'asset':s,'f':f}))
P=pd.concat(rows,ignore_index=True)
def evaluate(h):
 rr=[]
 for s,d in D.items():
  c=d.close.astype(float); fr=c.shift(-h)/c-1
  z=P[P.asset==s].set_index('date').f.reindex(c.index)
  rr.append(pd.DataFrame({'date':c.index,'asset':s,'f':z.values,'fr':fr.values}))
 x=pd.concat(rr,ignore_index=True).replace([np.inf,-np.inf],np.nan).dropna(); vals=[]; ns=[]
 for dt,g in x.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1: vals.append(g.f.corr(g.fr,method='spearman')); ns.append(len(g))
 z=pd.Series(vals); return {'dates':len(z),'avg_n':float(np.mean(ns)),'ic':float(z.mean()),'icir':float(z.mean()/z.std(ddof=1)*np.sqrt(252)),'hit':float((z>0).mean()),'coverage':len(x)/(x.date.nunique()*15)}
print('assets',len(D),'raw_dates',P.date.nunique())
for h in [1,5,10,20]: print('horizon',h,evaluate(h))
# same-horizon regime diagnostics
for a,b in [(2020,2022),(2023,2024),(2025,2027)]:
 x=P[(P.date.dt.year>=a)&(P.date.dt.year<=b)].copy(); rr=[]
 for s,d in D.items():
  c=d.close.astype(float); fr=c.shift(-1)/c-1; z=x[x.asset==s].set_index('date').f.reindex(c.index)
  rr.append(pd.DataFrame({'date':c.index,'f':z.values,'fr':fr.values}))
 y=pd.concat(rr).dropna(); v=[]
 for _,g in y.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1:v.append(g.f.corr(g.fr,method='spearman'))
 print('regime',a,b,'dates',len(v),'ic',float(np.mean(v)) if v else None)
r=P.pivot(index='date',columns='asset',values='f').rank(axis=1,pct=True); print('turnover',float(r.diff().abs().mean().mean()))
P.to_csv('scripts/miner_2_20270518_clv_reversal_signal.csv',index=False)
