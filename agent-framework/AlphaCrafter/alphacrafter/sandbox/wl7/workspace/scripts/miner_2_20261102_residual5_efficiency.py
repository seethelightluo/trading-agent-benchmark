import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; rows=[]
for s in U:
 d=get_stock_daily_data(s,3000)
 if d is None or len(d)<50: continue
 d=d.sort_values('date'); c=pd.to_numeric(d.close,errors='coerce'); r=c.pct_change(); v=r.rolling(20).std()
 # residual reversal: 5d shock, conditioned on range efficiency (avoid persistent trends)
 r5=c.pct_change(5); r20=c.pct_change(20); eff=(c.diff().abs().rolling(20).sum()+1e-9)/(c.rolling(20).max()-c.rolling(20).min()+1e-9)
 f=(-r5/(v*np.sqrt(5))*(1-0.4*np.tanh((eff-2)/1.5))).shift(1)
 z=pd.DataFrame({'date':pd.to_datetime(d.date),'s':s,'f':f})
 for h in [1,5,10,20]:z['fr'+str(h)]=c.shift(-h)/c-1
 rows.append(z)
x=pd.concat(rows); out=[]
for h in [1,5,10,20]:
 a=[]; ns=[]
 for dt,g in x.groupby('date'):
  g=g.dropna(subset=['f','fr'+str(h)])
  if len(g)>=8:a.append(g.f.corr(g['fr'+str(h)],method='spearman'));ns.append(len(g))
 a=np.array(a); print('h',h,'dates',len(a),'avg_names',np.mean(ns),'IC',np.nanmean(a),'ICIR',np.nanmean(a)/np.nanstd(a,ddof=1),'hit',np.mean(a>0))
a=[]
for dt,g in x.groupby('date'):
 g=g.dropna(subset=['f','fr1'])
 if len(g)>=8:a.append(g.f.corr(g.fr1,method='spearman'))
r=x.pivot(index='date',columns='s',values='f').rank(axis=1,pct=True); print('turnover',r.diff().abs().mean(axis=1).dropna().mean())
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31')]:
 q=[]
 for dt,g in x[(x.date>=lo)&(x.date<=hi)].groupby('date'):
  g=g.dropna(subset=['f','fr1'])
  if len(g)>=8:q.append(g.f.corr(g.fr1,method='spearman'))
 q=np.array(q);print(lo,len(q),np.nanmean(q),np.nanmean(q)/np.nanstd(q,ddof=1))
