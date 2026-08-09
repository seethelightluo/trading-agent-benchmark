import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,days=5000)
   if d is not None:return d
  except Exception: pass
px=pd.DataFrame({s:get(s).set_index('date')['close'] for s in U}).sort_index(); r=px.pct_change()
vol=r.rolling(20).std(); disp=r.std(axis=1); high=(disp>disp.rolling(60,min_periods=30).quantile(.70)).shift(1)
base=-(r.rolling(3).sum().shift(1)/vol.shift(1)); sig=base.sub(base.mean(axis=1),axis=0).where(high,0.0)
for h in [1,5,10]:
 fr=px.shift(-h)/px-1; a=[]; ns=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
 a=np.array(a); print('H',h,'dates',len(a),'avg_n',np.mean(ns),'IC',np.mean(a),'ICIR',np.mean(a)/np.std(a,ddof=1),'hit',np.mean(a>0))
print('coverage',sig.notna().mean().mean(),'active',high.mean(),'assets',len(px.columns),'period',px.index.min(),px.index.max())
out=sig.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('../persistent/factor_signals_miner_2_20270225_cond_disp_volrev.csv',index=False)
