import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; P={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)<200: d=get_index_daily_data(s,4000)
 if d is not None and len(d): P[s]=d.set_index('date').close.astype(float)
p=pd.DataFrame(P).sort_index(); r=np.log(p).diff()
# Volatility-expansion reversal: short-term shock reversal is activated when recent
# realized volatility is elevated relative to its own medium-term baseline.
ret5=np.log(p/p.shift(5)); v20=r.rolling(20).std()*np.sqrt(5); v60=r.rolling(60).std()*np.sqrt(5)
exp=(v20/v60).clip(0.5,2.5)
sig=(-ret5/v20*exp).replace([np.inf,-np.inf],np.nan)
sig=sig.sub(sig.median(axis=1),axis=0).shift(1)
for h in [1,3,5,10]:
 fwd=np.log(p.shift(-h)/p);ics=[];ns=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:
   ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 x=pd.Series(ics); print(f'h={h} dates={len(x)} avgN={np.mean(ns):.2f} IC={x.mean():.6f} ICIR={x.mean()/x.std(ddof=1):.6f} hit={np.mean(x>0):.4f}')
print(f'coverage={sig.notna().sum().sum()/(len(sig)*len(U)):.4f} turnover={sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean():.6f} instruments={len(P)} rows={len(p)}')
for a,b in [('2020','2022'),('2023','2025'),('2026','2027'),('2028','2029')]:
 q=[]
 for dt in sig.loc[a:b].index:
  z=pd.concat([sig.loc[dt],np.log(p.shift(-1)/p).loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print(f'regime={a}-{b} dates={len(q)} IC={np.mean(q) if q else np.nan:.6f}')
out=sig.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_2_20291115_vol_expansion_reversal_signal.csv',index=False)
