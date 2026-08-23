import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)<100: d=get_index_daily_data(s,5000)
 return None if d is None else d.set_index(pd.to_datetime(d.date)).close.astype(float)
P=pd.concat({s:load(s) for s in U if load(s) is not None},axis=1).sort_index(); r=P.pct_change()
# Calm relative reversal: reverse 30-session risk-adjusted return, penalizing recent drawdown.
ret=P.pct_change(30); vol=r.rolling(30,min_periods=20).std(); peak=P.rolling(60,min_periods=30).max(); dd=P/peak-1
raw=-(ret/(vol+1e-8) + 0.25*dd)
f=raw.sub(raw.median(axis=1),axis=0).shift(1); f.to_csv('scripts/miner_1_20350705_calm_relative_reversal_signal.csv',index_label='date')
for h in [5,10,20,40]:
 fr=P.pct_change(h).shift(-h); vals=[]; ns=[]; turns=[]; prev=None
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(c):
    vals.append(c); ns.append(len(z)); rr=z.iloc[:,0].rank(pct=True)
    if prev is not None:
     common=rr.index.intersection(prev.index); turns.append(np.mean(abs(rr[common]-prev[common])))
    prev=rr
 q=np.array(vals); recent=q[-500:]
 print(f'{h}D dates={len(q)} avg_n={np.mean(ns):.3f} coverage={np.mean(ns)/15:.4f} IC={q.mean():.8f} ICIR={q.mean()/q.std(ddof=1)*np.sqrt(len(q)):.8f} hit={np.mean(q>0):.4f} turnover={np.mean(turns):.5f}')
 print(f'{h}D recent500 IC={recent.mean():.8f} ICIR={recent.mean()/recent.std(ddof=1)*np.sqrt(len(recent)):.8f} hit={np.mean(recent>0):.4f}')
print('range',P.index.min(),P.index.max(),'assets',P.shape[1])
