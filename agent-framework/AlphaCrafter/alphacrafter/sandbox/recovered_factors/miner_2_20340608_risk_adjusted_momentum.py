import pandas as pd,numpy as np,glob,json,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.concat([pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'].rename(a) for a in A],axis=1).sort_index()
r=np.log(P).diff(); vol=r.rolling(20,min_periods=15).std()*np.sqrt(252)
# Smooth risk-adjusted medium momentum; lag before forward-return test
sig=(r.rolling(20,min_periods=15).sum()/vol).shift(1)
print('END',P.index.max().date(),'assets',len(A),'dates',len(P),'coverage %.4f'%sig.notna().mean().mean())
def calc(h,idx=sig.index):
 f=P.pct_change(h).shift(-h); vals=[];ns=[]
 for d in idx:
  z=pd.concat([sig.loc[d],f.loc[d]],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q): vals.append(q);ns.append(len(z))
 q=np.array(vals); return q,np.array(ns)
for h in [1,5,10,20]:
 q,n=calc(h); print('H',h,'IC %.6f ICIR %.6f hit %.4f dates %d meanN %.2f'%(q.mean(),q.mean()/q.std(ddof=1),np.mean(q>0),len(q),n.mean()))
rank=sig.rank(axis=1,pct=True); print('turn10 %.6f'%np.nanmean((rank-rank.shift(10)).abs().mean(axis=1)))
for lo,hi in [('2024','2027-12-31'),('2028','2030-12-31'),('2031','2034-12-31')]:
 q,n=calc(1,sig.loc[lo:hi].index); print('REG',lo,'dates',len(q),'IC %.6f ICIR %.6f'%(q.mean(),q.mean()/q.std(ddof=1)))
# correlate with admitted factor expressions unavailable; compare signal vectors from json if expression recognizable
print('FILES',len(glob.glob('factors/*.json')))
# library correlation evidence: compute correlations against stored factor ids only if same expression metadata not computable
# report conservative proxy against common raw momentum and vol-normalized reversal
proxies={'raw20':r.rolling(20,min_periods=15).sum().shift(1),'raw5':r.rolling(5,min_periods=5).sum().shift(1),'vol20':vol.shift(1)}
for k,x in proxies.items():
 cs=[]
 for d in sig.index:
  z=pd.concat([sig.loc[d],x.loc[d]],axis=1).dropna()
  if len(z)>=8: cs.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('CORR',k,'maxabs',abs(np.nanmean(cs)))
