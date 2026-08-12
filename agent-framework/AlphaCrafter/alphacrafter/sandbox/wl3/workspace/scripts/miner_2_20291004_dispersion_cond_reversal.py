import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];P={}
for s in U:
 d=get_stock_daily_data(s,days=4000)
 if d is not None and len(d)>200:P[s]=d.set_index('date').close.astype(float)
p=pd.DataFrame(P).sort_index().ffill(); r=np.log(p).diff(); ret5=np.log(p/p.shift(5)); vol=r.rolling(60).std()*np.sqrt(60)
disp=r.rolling(5).std().mean(axis=1); threshold=disp.rolling(120).median(); active=(disp>threshold).astype(float)
sig=(-ret5.div(vol)*active.to_numpy()[:,None]).shift(1)
for h in [1,3,5,10]:
 fwd=np.log(p.shift(-h)/p);ics=[];ns=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8:ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 x=pd.Series(ics);print(f'h={h} dates={len(x)} avgN={np.mean(ns):.2f} IC={x.mean():.6f} ICIR={x.mean()/x.std(ddof=1):.6f} hit={np.mean(x>0):.4f}')
print(f'active_rate={active.mean():.4f} coverage={sig.notna().sum(axis=1).mean()/len(U):.4f} avg_valid={sig.notna().sum(axis=1).mean():.2f}')
print(f'turnover={sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean():.6f} instruments={len(U)} rows={len(p)}')
for a,b in [('2020','2022'),('2023','2025'),('2026','2027'),('2028','2029')]:
 q=[]
 for dt in sig.loc[a:b].index:
  z=pd.concat([sig.loc[dt],np.log(p.shift(-1)/p).loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print(f'regime={a}-{b} dates={len(q)} IC={np.mean(q) if q else np.nan:.6f}')
out=sig.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_2_20291004_dispersion_cond_reversal_signal.csv',index=False)
