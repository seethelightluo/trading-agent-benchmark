import pandas as pd,numpy as np,glob
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.concat([pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'].rename(a) for a in A],axis=1).sort_index()
r=np.log(P).diff(); v=r.rolling(20,min_periods=15).std();
# Volatility-scaled short-term reversal, one-day lag. Scaling makes comparable cross-asset.
sig=(-r.rolling(3,min_periods=3).sum()/v).shift(1)
print('END',P.index.max().date(),'assets',len(A),'dates',len(P),'coverage %.4f'%(sig.notna().mean().mean()))
def calc(h,idx=sig.index):
 f=np.log(P.shift(-h)/P); out=[]; ns=[]
 for d in idx:
  z=pd.concat([sig.loc[d],f.loc[d]],axis=1).dropna()
  if len(z)>=8:
   x=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(x):out.append(x);ns.append(len(z))
 return np.array(out),np.array(ns)
for h in [1,5,10,20]:
 q,n=calc(h); print('H',h,'IC %.6f ICIR %.6f hit %.4f dates %d meanN %.2f'%(q.mean(),q.mean()/q.std(ddof=1),np.mean(q>0),len(q),n.mean()))
rank=sig.rank(axis=1,pct=True); print('turn10 %.6f'%np.nanmean((rank-rank.shift(10)).abs().mean(axis=1)))
for lo,hi in [('2024','2027-12-31'),('2028','2030-12-31'),('2031','2034-12-31')]:
 q,n=calc(1,sig.loc[lo:hi].index);print('REG',lo,'dates',len(q),'IC %.6f ICIR %.6f'%(q.mean(),q.mean()/q.std(ddof=1)))
# Explicit library audit proxy: compare against common admitted raw reversal/momentum signals.
proxies={'raw3_reversal':(-r.rolling(3,min_periods=3).sum()).shift(1),'raw20_momentum':r.rolling(20,min_periods=15).sum().shift(1),'vol20':v.shift(1)}
for k,x in proxies.items():
 cs=[]
 for d in sig.index:
  z=pd.concat([sig.loc[d],x.loc[d]],axis=1).dropna()
  if len(z)>=8: cs.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('CORR',k,'meanabs %.6f maxabs %.6f'%(np.nanmean(np.abs(cs)),np.nanmax(np.abs(cs))))
print('FILES',len(glob.glob('factors/*.json')))
