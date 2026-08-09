import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.concat([pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'].rename(a) for a in A],axis=1).sort_index()
r=np.log(P).diff()
# Volatility compression / shock: inverse recent-to-long volatility ratio, lagged one day.
# Values above zero mean recent realized vol is below its long-run baseline.
v20=r.rolling(20,min_periods=15).std(); v60=r.rolling(60,min_periods=40).std()
sig=(-(v20/v60-1)).shift(1)
print('END',P.index.max().date(),'assets',len(A),'dates',len(P),'coverage %.4f'%sig.notna().mean().mean())
def calc(h,idx):
 f=np.log(P.shift(-h)/P); out=[]; ns=[]
 for d in idx:
  z=pd.concat([sig.loc[d],f.loc[d]],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q):out.append(q);ns.append(len(z))
 return np.array(out),np.array(ns)
for h in [1,5,10,20]:
 q,n=calc(h,sig.index); print('H',h,'IC %.6f ICIR %.6f hit %.4f dates %d meanN %.2f'%(q.mean(),q.mean()/q.std(ddof=1),np.mean(q>0),len(q),n.mean()))
rank=sig.rank(axis=1,pct=True); print('turn10 %.6f'%np.nanmean((rank-rank.shift(10)).abs().mean(axis=1)))
for lo,hi in [('2020','2023-12-31'),('2024','2027-12-31'),('2028','2030-12-31'),('2031','2034-12-31')]:
 q,n=calc(1,sig.loc[lo:hi].index);print('REG',lo,'dates',len(q),'IC %.6f ICIR %.6f hit %.4f'%(q.mean(),q.mean()/q.std(ddof=1),np.mean(q>0)))
# Split by recent vol shock cross-section; report conditional H1 IC.
for cond,name in [(v20.mean(axis=1)>v60.mean(axis=1),'high market vol'),(v20.mean(axis=1)<=v60.mean(axis=1),'low market vol')]:
 idx=sig.index[cond.reindex(sig.index).fillna(False)];q,n=calc(1,idx);print('COND',name,'dates',len(q),'IC %.6f ICIR %.6f'%(q.mean(),q.mean()/q.std(ddof=1)))
print('FILES',len(__import__('glob').glob('factors/*.json')))
