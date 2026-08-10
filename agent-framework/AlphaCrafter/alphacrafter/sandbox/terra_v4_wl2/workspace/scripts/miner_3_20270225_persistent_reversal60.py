import pandas as pd,numpy as np
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,days=5000)
   if d is not None:return d
  except Exception: pass
D={s:get(s) for s in U}; C=pd.DataFrame({s:d.set_index('date').close.astype(float) for s,d in D.items() if d is not None}).sort_index(); R=C.pct_change()
mom=C/C.shift(60)-1; vol=R.rolling(20,min_periods=15).std()*np.sqrt(252); persist=R.gt(0).rolling(40,min_periods=25).mean()
# Reversal of volatility-normalized 60d momentum, weighted by persistence distance from neutral.
f=(-(mom/vol)*(persist-.5)).shift(1)
y=C.shift(-5)/C-1; vals=[]; ns=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
a=np.array(vals); ic=float(np.mean(a)); icir=float(ic/np.std(a,ddof=1)); print('H5 dates',len(a),'avgN',float(np.mean(ns)),'IC',ic,'ICIR',icir,'hit',float(np.mean(a>0)))
for h in [1,3,10]:
 yy=C.shift(-h)/C-1; aa=[];nn=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1: aa.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));nn.append(len(z))
 aa=np.array(aa);print('H',h,'dates',len(aa),'avgN',float(np.mean(nn)),'IC',float(np.mean(aa)),'ICIR',float(np.mean(aa)/np.std(aa,ddof=1)))
print('coverage',float(f.notna().mean().mean()),'turnover',float(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean()))
for lo,hi in [('2025','2026-12-31'),('2027','2027-02-25')]:
 aa=[]
 for dt in f.loc[lo:hi].index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:aa.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 aa=np.array(aa);print('REG',lo,'dates',len(aa),'IC',float(np.mean(aa)),'ICIR',float(np.mean(aa)/np.std(aa,ddof=1)))
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('../persistent/factor_signals_miner_3_20270225_persistent_reversal60.csv',index=False)
