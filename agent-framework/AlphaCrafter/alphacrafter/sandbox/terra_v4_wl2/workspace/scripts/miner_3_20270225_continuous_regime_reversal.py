import pandas as pd,numpy as np
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   x=fn(s,days=5000)
   if x is not None:return x
  except: pass
px=pd.DataFrame({s:get(s).set_index('date')['close'] for s in U}).sort_index(); r=px.pct_change()
v=get('VIX').set_index('date')['close'].reindex(px.index).ffill(); vx=v.pct_change()
disp=r.std(axis=1)
def pct(x): return x.rolling(120,min_periods=60).rank(pct=True)
# Continuous regime intensity, lagged to avoid lookahead; bounded and centered reversal
reg=(pct(vx).shift(1)*pct(disp).shift(1)).clip(0,1)
base=-r.rolling(3).sum(); f=base.mul(reg,axis=0); f=f.sub(f.median(axis=1),axis=0)
for h in [1,3,5,10]:
 fr=px.shift(-h)/px-1; a=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:
   a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z))
 a=np.array(a); print('H',h,'dates',len(a),'avg_n',np.mean(ns),'IC',np.mean(a),'ICIR',np.mean(a)/np.std(a,ddof=1),'hit',np.mean(a>0))
# active / regimes for 1d
fr=px.shift(-1)/px-1
for lab,lo,hi in [('2020-22','2020','2022-12-31'),('2023-24','2023','2024-12-31'),('2025-26','2025','2026-12-31'),('2027','2027','2027-02-25')]:
 a=[]
 for dt in f.index:
  if str(dt)<lo or str(dt)>hi: continue
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print(lab,len(a),np.mean(a) if a else np.nan)
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('../persistent/factor_signals_miner_3_20270225_continuous_regime_reversal.csv',index=False)
print('dates',len(f),'active',int((reg>0.5).sum()),'coverage',f.notna().sum(axis=1).mean())
