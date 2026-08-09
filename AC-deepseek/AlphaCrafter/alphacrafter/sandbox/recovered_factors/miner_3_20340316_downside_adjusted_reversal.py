import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def c(a): return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close.astype(float)
P=pd.DataFrame({a:c(a) for a in A}).sort_index(); lr=np.log(P).diff()
r20=lr.rolling(20,min_periods=15).sum(); dn=lr.where(lr<0); dvol=dn.rolling(40,min_periods=25).std()*np.sqrt(252)
# Penalize assets whose recent return is poor relative to downside risk; lagged contrarian score.
sig=(-r20/dvol).shift(1)
print('END',P.index.max(),'assets',len(A),'coverage %.4f'%sig.notna().mean().mean())
for h in [1,5,10,20]:
 f=P.shift(-h)/P-1; vals=[];ns=[]
 for d in sig.index:
  z=pd.concat([sig.loc[d],f.loc[d]],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q):vals.append(q);ns.append(len(z))
 x=np.array(vals); print('H',h,'IC %.6f ICIR %.6f hit %.4f dates %d meanN %.2f'%(x.mean(),x.mean()/x.std(ddof=1),np.mean(x>0),len(x),np.mean(ns)))
r=sig.rank(axis=1,pct=True); print('turn10 %.6f'%np.nanmean((r-r.shift(10)).abs().sum(axis=1)/r.notna().sum(axis=1)))
f=P.shift(-10)/P-1
for lo,hi in [('2020','2023-12-31'),('2024','2027-12-31'),('2028','2030-12-31'),('2031','2034-12-31')]:
 x=[]
 for d in sig.loc[lo:hi].index:
  z=pd.concat([sig.loc[d],f.loc[d]],axis=1).dropna()
  if len(z)>=8:x.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 x=np.array(x); print('REG10',lo,len(x),'IC %.6f ICIR %.6f hit %.4f'%(np.nanmean(x),np.nanmean(x)/np.nanstd(x,ddof=1),np.mean(x>0)))
