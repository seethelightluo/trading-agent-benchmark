import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; P={}
for a in A:
 p='../persistent/stock_data/'+a+'.csv'
 if os.path.exists(p): P[a]=pd.read_csv(p,parse_dates=['date']).set_index('date')['close'].astype(float)
P=pd.DataFrame(P).sort_index().ffill(); ret=P.pct_change(); r5=P/P.shift(5)-1
# residual reversal scaled by each asset's recent realized volatility
vol=ret.rolling(20).std()*np.sqrt(20); f=-(r5.sub(r5.mean(axis=1),axis=0))/vol
for h in [1,5,10,20]:
 vals=[]; ds=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],(P.shift(-h).loc[dt]/P.loc[dt]-1)],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q): vals.append(q);ds.append(dt)
 x=np.array(vals); print(h,len(x),round(x.mean(),6),round(x.mean()/x.std(ddof=1),6),round(np.mean(x>0),4))
print('coverage',round(f.notna().sum(axis=1).mean()/len(A),4),'avg names',f.notna().sum(axis=1).mean())
for label,cond in [('early',f.index<='2023-12-31'),('late',f.index>='2024-01-01'),('recent250',f.index>=f.index[-251])]:
 vals=[]
 for dt,q in zip(ds,vals if False else []): pass
 # recompute 10d subset
 allq=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],(P.shift(-10).loc[dt]/P.loc[dt]-1)],axis=1).dropna()
  if len(z)>=8: allq.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
 x=np.array([q for dt,q in allq if cond[f.index.get_loc(dt)]])
 print(label,len(x),round(x.mean(),6),round(x.mean()/x.std(ddof=1),6))
