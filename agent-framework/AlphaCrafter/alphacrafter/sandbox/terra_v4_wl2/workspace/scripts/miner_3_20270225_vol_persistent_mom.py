import pandas as pd,numpy as np
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,days=5000)
   if d is not None:return d
  except: pass
D={s:get(s) for s in U}; C=pd.DataFrame({s:d.set_index('date').close.astype(float) for s,d in D.items() if d is not None}).sort_index(); R=C.pct_change()
# Volatility-normalized medium momentum, with trend persistence and cross-sectional residualization
mom=C/C.shift(60)-1; vol=R.rolling(20,min_periods=15).std()*np.sqrt(252); persist=R.gt(0).rolling(40,min_periods=25).mean()
f=((mom/vol)*(persist-.5)).shift(1); out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('../persistent/factor_signals_miner_3_20270225_vol_persistent_mom.csv',index=False)
for h in [1,3,5,10]:
 y=C.shift(-h)/C-1; a=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
 a=np.array(a);print('H',h,'dates',len(a),'avgN',np.mean(ns),'IC',np.mean(a),'ICIR',np.mean(a)/np.std(a,ddof=1),'hit',np.mean(a>0))
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2027-02-25')]:
 y=C.shift(-1)/C-1;a=[]
 for dt in f.loc[lo:hi].index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 a=np.array(a);print('REG',lo,len(a),np.mean(a) if len(a) else np.nan,(np.mean(a)/np.std(a,ddof=1)) if len(a)>1 else np.nan)
print('coverage',f.notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
