import numpy as np, pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in U:
 d=get_stock_daily_data(s,days=4000)
 if d is not None and len(d)>200:
  P[s]=d.set_index('date').close.astype(float)
pd_=pd.DataFrame(P).sort_index().ffill(); r=np.log(pd_).diff()
# One interpretable idea: 60-session asset strength relative to contemporaneous universe median,
# normalized by 60-session volatility. Signal is lagged one completed session.
raw=np.log(pd_/pd_.shift(60)); rel=raw.sub(raw.median(axis=1),axis=0)
vol=r.rolling(60).std()*np.sqrt(60); sig=(rel/vol).shift(1)
rows=[]
for h in [1,3,5,10]:
 fwd=np.log(pd_.shift(-h)/pd_)
 ics=[]; ns=[]; dates=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z));dates.append(dt)
 x=pd.Series(ics); daily=x.mean(); icir=daily/x.std(ddof=1) if x.std(ddof=1)>0 else np.nan
 print(f'h={h} dates={len(x)} avgN={np.mean(ns):.2f} IC={daily:.6f} ICIR={icir:.6f} hit={np.mean(x>0):.4f}')
# coverage and rank turnover on valid cross sections
valid=sig.notna().sum(axis=1); print(f'coverage={valid.mean()/len(U):.4f} avg_valid={valid.mean():.2f}')
ranks=sig.rank(axis=1,pct=True); turn=ranks.diff().abs().mean(axis=1).dropna().mean(); print(f'turnover={turn:.6f} instruments={len(U)} rows={len(pd_)}')
# regime blocks
for a,b in [('2020','2022'),('2023','2025'),('2026','2027'),('2028','2029')]:
 q=[]
 for dt in sig.loc[a:b].index:
  z=pd.concat([sig.loc[dt],np.log(pd_.shift(-1)/pd_).loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print(f'regime={a}-{b} dates={len(q)} IC={np.mean(q) if q else np.nan:.6f}')
# signal artifact
out=sig.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_2_20291004_relative_strength60_signal.csv',index=False)
