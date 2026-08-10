import pandas as pd,numpy as np
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   x=fn(s,days=5000)
   if x is not None:return x
  except: pass
px=pd.DataFrame({s:get(s).set_index('date')['close'] for s in U}).sort_index();r=px.pct_change()
# Volatility carry: low realized volatility relative to its own long-run volatility, cross-sectionally ranked.
v20=r.rolling(20).std(); v120=r.rolling(120).std(); f=-(v20/v120)
f=f.sub(f.median(axis=1),axis=0)
for h in [1,5,10]:
 fr=px.shift(-h)/px-1;a=[];ns=[]
 for d in f.index:
  z=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
 a=np.array(a);print('H',h,'dates',len(a),'avg_n',np.mean(ns) if ns else 0,'IC',np.mean(a) if len(a) else np.nan,'ICIR',np.mean(a)/np.std(a,ddof=1) if len(a)>1 else np.nan,'hit',np.mean(a>0) if len(a) else np.nan)
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2027-02-25')]:
 fr=px.shift(-5)/px-1;a=[]
 for d in f.index:
  if str(d)>=lo and str(d)<=hi:
   z=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
   if len(z)>=8 and z.iloc[:,0].nunique()>1:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('REG',lo,len(a),np.mean(a) if a else np.nan)
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('../persistent/factor_signals_miner_1_20270225_vol_carry.csv',index=False)
print('coverage',f.notna().mean().mean(),'dates',len(f),'instruments',len(U))
