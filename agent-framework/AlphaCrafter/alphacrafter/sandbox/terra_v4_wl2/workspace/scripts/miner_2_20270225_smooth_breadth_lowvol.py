import pandas as pd,numpy as np
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,days=5000)
   if d is not None and len(d): return d
  except Exception: pass
px=pd.DataFrame({s:get(s).set_index('date')['close'] for s in U}).sort_index(); r=px.pct_change()
# Smooth breadth-conditioned defensive factor: low volatility weighted by continuous market breadth.
breadth=(r.shift(1).rolling(5).mean()>0).sum(axis=1)/r.notna().sum(axis=1)
vol=r.shift(1).rolling(20).std(); sig=(-vol).mul((0.25+0.75*breadth),axis=0)
# cross-sectional rank stabilizes scale while retaining direction
sig=sig.rank(axis=1,pct=True)
for h in [1,5,10]:
 vals=[]; ns=[]
 f=px.shift(-h)/px-1
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
 x=np.asarray(vals); print('H',h,'dates',len(x),'avgN',round(np.mean(ns),2),'IC',round(np.mean(x),6),'ICIR',round(np.mean(x)/np.std(x,ddof=1),6),'hit',round(np.mean(x>0),4))
print('coverage',round(sig.notna().sum().sum()/(len(U)*len(sig)),4),'matrix_dates',len(sig))
for lab,st,en in [('2020-22','2020','2022-12-31'),('2023-24','2023','2024-12-31'),('2025-26','2025','2026-12-31'),('2027','2027','2027-12-31')]:
 x=[]
 for dt in sig.index:
  if str(dt)[:10]>=st and str(dt)[:10]<=en:
   z=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
   if len(z)>=8 and z.iloc[:,0].nunique()>1:x.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 x=np.asarray(x);print(lab,len(x),round(np.mean(x),6) if len(x) else np.nan,round(np.mean(x)/np.std(x,ddof=1),6) if len(x)>1 else np.nan)
# save exact artifact for provenance
out=sig.copy();out.index.name='date';out.to_csv('../persistent/factor_signals_miner_2_20270225_smooth_breadth_lowvol.csv')
