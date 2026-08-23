import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-11-02')
def fetch(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   x=fn(s,3000)
   if x is not None and len(x):
    x=x.copy(); x.date=pd.to_datetime(x.date).dt.normalize()
    return x.drop_duplicates('date').set_index('date').sort_index()
  except Exception: pass
D={s:fetch(s) for s in U}; D={s:x.loc[:cut] for s,x in D.items() if x is not None and (x.index<=cut).any()}
px=pd.DataFrame({s:x.close for s,x in D.items()}).sort_index().ffill(); r=px.pct_change()
# Medium-term acceleration: recent 20d trend relative to preceding 40d trend,
# volatility-normalized and lagged one completed day at evaluation.
m20=px.pct_change(20); m60=px.pct_change(60)
vol=r.rolling(40,min_periods=20).std()*np.sqrt(40)
f=((m20-(m60-m20)/2)/(vol+1e-12)).shift(1)
print('assets',len(D),'dates',len(px),'range',px.index.min(),px.index.max(),'cutoff',cut.date())
def calc(h):
 fr=px.shift(-h)/px-1; vals=[]; ns=[]
 for dt in f.index:
  z=pd.DataFrame({'f':f.loc[dt],'fr':fr.loc[dt]}).dropna()
  if len(z)>=8 and z.f.nunique()>1 and z.fr.nunique()>1:
   vals.append(z.f.corr(z.fr,method='spearman')); ns.append(len(z))
 a=pd.Series(vals)
 return len(a),np.mean(ns),a.mean(),a.mean()/a.std(ddof=1),np.mean(a>0)
for h in [1,5,10,20]: print('horizon',h,'dates avg_names IC ICIR hit',calc(h))
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-11-02')]:
 fr=px.shift(-1)/px-1; a=[]
 for dt in f.loc[lo:hi].index:
  z=pd.DataFrame({'f':f.loc[dt],'fr':fr.loc[dt]}).dropna()
  if len(z)>=8 and z.f.nunique()>1 and z.fr.nunique()>1:a.append(z.f.corr(z.fr,method='spearman'))
 a=pd.Series(a); print('regime',lo,'dates',len(a),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0))
print('coverage',f.notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
