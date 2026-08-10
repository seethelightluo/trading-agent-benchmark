import pandas as pd,numpy as np
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   x=fn(s,days=5000)
   if x is not None:return x
  except Exception:pass
px=pd.DataFrame({s:get(s).set_index('date')['close'] for s in U}).sort_index(); r=px.pct_change()
# residual short-term reversal: remove contemporaneous cross-sectional market move from each asset's 3d return
r3=r.rolling(3).sum(); csmean=r3.mean(axis=1); f=-(r3.sub(csmean,axis=0)); f=f.sub(f.median(axis=1),axis=0)
out=[]
for h in [1,5,10]:
 fr=px.shift(-h)/px-1; ics=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1: ics.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
 a=np.array(ics); print('H',h,'dates',len(a),'avg_n',np.mean(ns),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0))
# regimes
fr=px.shift(-1)/px-1; vals=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1: vals.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman')))
for lab,a,b in [('2020-22','2020','2022-12-31'),('2023-24','2023','2024-12-31'),('2025-26','2025','2026-12-31'),('2027','2027','2027-02-25')]:
 q=[v for d,v in vals if str(d)>=a and str(d)<=b];print(lab,'dates',len(q),'IC',np.mean(q) if q else np.nan,'ICIR',np.mean(q)/np.std(q,ddof=1) if len(q)>1 else np.nan)
print('coverage',f.notna().sum(axis=1).mean()/15,'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('../persistent/factor_signals_miner_3_20270225_residual_reversal3.csv',index=False)
