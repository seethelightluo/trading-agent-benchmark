import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];CUT=pd.Timestamp('2027-08-12')
def load(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,3000)
   if d is not None and len(d):
    d=d.copy();d.date=pd.to_datetime(d.date).dt.normalize();return d.drop_duplicates('date').set_index('date').sort_index().loc[:CUT]
  except Exception:pass
D={s:load(s) for s in U};D={s:d for s,d in D.items() if d is not None}; rr=[]
for s,d in D.items():
 c=pd.to_numeric(d.close,errors='coerce');rr.append(pd.DataFrame({'date':c.index,'asset':s,'r20':c.pct_change(20),'r5':c.pct_change(5),'fr1':c.shift(-1)/c-1,'fr5':c.shift(-5)/c-1,'fr10':c.shift(-10)/c-1}))
x=pd.concat(rr,ignore_index=True); x['f']=x['r20']-x.groupby('date')['r20'].transform('median');x['f']=x.f.shift(0) # returns are already known at date; decision uses prior close, then lag below
x['f']=x.groupby('asset').f.shift(1);x=x.replace([np.inf,-np.inf],np.nan).dropna(subset=['f','fr1'])
def st(col,z):
 a=[];ns=[]
 for _,g in z.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g[col].nunique()>1:a.append(g.f.corr(g[col],method='spearman'));ns.append(len(g))
 a=pd.Series(a).dropna();return len(a),np.mean(ns),a.mean(),a.mean()/a.std(ddof=1)*np.sqrt(252),(a>0).mean()
print('assets',len(D),'dates',x.date.nunique(),'avg_n',x.groupby('date').size().mean(),'coverage',len(x)/(x.date.nunique()*15))
for c in ['fr1','fr5','fr10']:print(c,st(c,x))
for a,b in [(2020,2022),(2023,2024),(2025,2027)]:print('regime',a,b,st('fr5',x[(x.date.dt.year>=a)&(x.date.dt.year<=b)]))
p=x.pivot(index='date',columns='asset',values='f').rank(axis=1,pct=True);print('turnover',float(p.diff().abs().mean().mean()))
x.to_csv('scripts/miner_2_20270813_relative_momentum_signal.csv',index=False)
