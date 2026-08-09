import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={a:get_stock_daily_data(a,days=4000) for a in U}
C=pd.DataFrame({a:d.set_index('date').close.astype(float) for a,d in D.items() if d is not None}).sort_index()
R=C.pct_change()
# Volatility term structure: compression (low recent risk relative to 60d risk)
short=R.rolling(10,min_periods=8).std(); long=R.rolling(60,min_periods=40).std()
f=-(short/(long+1e-12)-1.0)
# Winsorize cross section to avoid one noisy asset dominating
f=f.clip(f.quantile(.05,axis=1),f.quantile(.95,axis=1),axis=0)
print('assets',len(C.columns),'range',C.index.min(),C.index.max())
for h in [1,3,5,10]:
 y=C.shift(-h)/C-1; z=[]; ns=[]; dates=[]
 for dt in f.index:
  q=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(q)>=8:
   z.append(q.iloc[:,0].corr(q.iloc[:,1],method='spearman'));ns.append(len(q));dates.append(dt)
 s=pd.Series(z,index=dates).dropna(); print('H',h,'dates',len(s),'avgN',round(np.mean(ns),2),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6),'hit',round((s>0).mean(),4))
 if h==1:
  for j,part in enumerate(np.array_split(s,4)):
   print('Q',j+1,'IC',round(part.mean(),6),'IR',round(part.mean()/part.std(ddof=1),4),'n',len(part))
print('coverage',round(f.notna().mean().mean(),4),'turn',round(f.rank(pct=True,axis=1).diff().abs().mean(axis=1).mean(),4))
f.stack().rename('signal').to_csv('../persistent/factor_signals_miner_2_20270225_vol_term_structure.csv')
