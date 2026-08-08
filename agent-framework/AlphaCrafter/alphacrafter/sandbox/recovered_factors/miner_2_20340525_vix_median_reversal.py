import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.concat([pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'].rename(a) for a in A],axis=1).sort_index()
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].reindex(P.index).ffill()
# High-VIX cross-sectional reversal: reverse recent 3d returns only when VIX is above its trailing 60d median; otherwise use mild continuation.
r=np.log(P).diff(); x=r.rolling(3,min_periods=3).sum(); high=(v>v.rolling(60,min_periods=40).median()).astype(float)
sig=(x*(1-2*high)).shift(1); sig=sig.sub(sig.mean(axis=1),axis=0)
print('END',P.index.max().date(),'assets',len(A),'dates',len(P),'coverage',sig.notna().mean().mean())
for h in [1,5,10,20]:
 f=P.pct_change(h).shift(-h); vals=[]; ns=[]
 for d in sig.index:
  z=pd.concat([sig.loc[d],f.loc[d]],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q): vals.append(q);ns.append(len(z))
 q=np.array(vals); print('H',h,'IC %.6f ICIR %.6f hit %.4f dates %d meanN %.2f'%(q.mean(),q.mean()/q.std(ddof=1),np.mean(q>0),len(q),np.mean(ns)))
rank=sig.rank(axis=1,pct=True); print('turn10',np.nanmean((rank-rank.shift(10)).abs().mean(axis=1)))
for lo,hi in [('2024','2027-12-31'),('2028','2030-12-31'),('2031','2034-12-31')]:
 vals=[]
 for d in sig.loc[lo:hi].index:
  z=pd.concat([sig.loc[d],P.pct_change(1).shift(-1).loc[d]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=np.array(vals); print('REG',lo,'dates',len(q),'IC',np.mean(q),'ICIR',np.mean(q)/np.std(q,ddof=1))
