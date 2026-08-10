import pandas as pd, numpy as np
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   x=fn(s,days=5000)
   if x is not None:return x
  except Exception: pass
px=pd.DataFrame({s:get(s).set_index('date')['close'] for s in U}).sort_index(); r=px.pct_change()
# Trend consistency: medium return rewarded only when 5/20/60d directions agree, scaled by 20d volatility.
mom=px.pct_change(20); signs=(pd.concat([px.pct_change(5),px.pct_change(20),px.pct_change(60)],axis=1).apply(np.sign).sum(axis=1))
# per asset consistency is sign agreement score, not cross section aggregate
m5=px.pct_change(5);m20=px.pct_change(20);m60=px.pct_change(60)
cons=(np.sign(m5)+np.sign(m20)+np.sign(m60))/3
vol=r.rolling(20).std(); f=mom/vol*cons
f=f.sub(f.median(axis=1),axis=0)
for h in [1,5,10]:
 fr=px.shift(-h)/px-1; a=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
 a=np.array(a); print('H',h,'dates',len(a),'avg_n',np.mean(ns),'IC',np.mean(a),'ICIR',np.mean(a)/np.std(a,ddof=1),'hit',np.mean(a>0))
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2027-02-25')]:
 a=[]
 fr=px.shift(-5)/px-1
 for dt in f.index:
  if str(dt)>=lo and str(dt)<=hi:
   z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
   if len(z)>=8 and z.iloc[:,0].nunique()>1:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('REG',lo,len(a),np.mean(a) if a else np.nan,np.mean(a)/np.std(a,ddof=1) if len(a)>1 else np.nan)
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('../persistent/factor_signals_miner_1_20270225_trend_consistency.csv',index=False)
print('coverage',f.notna().mean().mean(),'dates',len(f),'instruments',len(U))
