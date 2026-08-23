import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,3000)
   if d is not None and len(d):
    d=d.copy(); d.date=pd.to_datetime(d.date).dt.normalize(); return d.drop_duplicates('date').set_index('date').sort_index()
  except Exception: pass
D={s:fetch(s) for s in U}; D={s:d for s,d in D.items() if d is not None}
px=pd.DataFrame({s:d.close for s,d in D.items()}).sort_index().ffill(); r=px.pct_change()
r5=r.rolling(5,min_periods=5).sum(); assetvol=r.rolling(20,min_periods=10).std()
# High-dispersion conditional reversal: cross-sectional dispersion is lagged and
# normalized by its 60-day median; the multiplier is clipped for stability.
disp=r.std(axis=1); disprel=(disp/(disp.rolling(60,min_periods=20).median()+1e-12)).clip(.5,2.0)
f=(-r5/(assetvol+1e-12)).mul(disprel,axis=0)
print('assets',len(D),'dates',len(px),'range',px.index.min(),px.index.max())
def calc(h):
 vals=[]; ns=[]
 fr=px.shift(-h)/px-1
 for dt in f.index:
  z=pd.DataFrame({'f':f.loc[dt],'fr':fr.loc[dt]}).dropna()
  if len(z)>=8 and z.f.nunique()>1 and z.fr.nunique()>1:
   vals.append(z.f.corr(z.fr,method='spearman')); ns.append(len(z))
 a=pd.Series(vals)
 return len(a),float(np.mean(ns)),float(a.mean()),float(a.mean()/a.std(ddof=1)),float((a>0).mean())
for h in [1,5,10,20]: print('horizon',h,'dates avg_names IC ICIR hit',calc(h))
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31')]:
 fr=px.shift(-1)/px-1; a=[]
 for dt in f.loc[lo:hi].index:
  z=pd.DataFrame({'f':f.loc[dt],'fr':fr.loc[dt]}).dropna()
  if len(z)>=8 and z.f.nunique()>1 and z.fr.nunique()>1:a.append(z.f.corr(z.fr,method='spearman'))
 a=pd.Series(a); print('regime',lo,'dates',len(a),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1))
print('coverage',f.notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean(),'valid_assets',len(D))
