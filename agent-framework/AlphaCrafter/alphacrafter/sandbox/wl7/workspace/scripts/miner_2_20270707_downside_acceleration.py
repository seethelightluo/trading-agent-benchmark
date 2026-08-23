import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,3000)
   if d is not None and len(d):
    d=d.copy(); d.date=pd.to_datetime(d.date).dt.normalize(); return d.drop_duplicates('date').set_index('date').sort_index()
  except Exception: pass
D={s:get(s) for s in U}; D={s:d for s,d in D.items() if d is not None}; rows=[]
for s,d in D.items():
 c=d.close.astype(float); r=c.pct_change(); down=r.clip(upper=0).pow(2).rolling(20,min_periods=15).mean().pow(.5)
 f=((c.pct_change(5)-c.pct_change(20)/4)/(down+0.002)).shift(1)
 q=pd.DataFrame({'date':c.index,'asset':s,'f':f.values,'fr':(c.shift(-1)/c-1).values}); rows.append(q)
x=pd.concat(rows,ignore_index=True).replace([np.inf,-np.inf],np.nan).dropna(); vals=[]; ns=[]; dates=[]
for dt,g in x.groupby('date'):
 if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1: vals.append(g.f.corr(g.fr,method='spearman')); ns.append(len(g)); dates.append(dt)
z=pd.Series(vals,index=dates); print('assets',len(D),'dates',len(z),'avg_n',round(np.mean(ns),2),'ic',round(z.mean(),6),'icir_ann',round(z.mean()/z.std(ddof=1)*np.sqrt(252),6),'daily_icir',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),4),'coverage',round(len(x)/(x.date.nunique()*len(U)),4))
for h in [5,10,20]:
 vals=[]
 for s,d in D.items():
  c=d.close.astype(float); r=c.pct_change(); down=r.clip(upper=0).pow(2).rolling(20,min_periods=15).mean().pow(.5); f=((c.pct_change(5)-c.pct_change(20)/4)/(down+0.002)).shift(1); vals.append(pd.DataFrame({'date':c.index,'f':f.values,'fr':(c.shift(-h)/c-1).values}))
 y=pd.concat(vals,ignore_index=True).dropna(); vv=[]
 for _,g in y.groupby('date'):
  if len(g)>=8:vv.append(g.f.corr(g.fr,method='spearman'))
 print('decay',h,'dates',len(vv),'ic',round(np.nanmean(vv),6))
for a,b in [(2020,2022),(2023,2024),(2025,2027)]:
 v=[]
 for _,g in x[x.date.dt.year.between(a,b)].groupby('date'):
  if len(g)>=8:v.append(g.f.corr(g.fr,method='spearman'))
 print('regime',a,b,'dates',len(v),'ic',round(np.nanmean(v),6),'daily_icir',round(np.nanmean(v)/np.nanstd(v,ddof=1),6))
r=x.pivot(index='date',columns='asset',values='f').rank(axis=1,pct=True); print('turnover',round(r.diff().abs().mean().mean(),6)); x[['date','asset','f']].to_csv('scripts/miner_2_20270707_downside_acceleration_signal.csv',index=False)
