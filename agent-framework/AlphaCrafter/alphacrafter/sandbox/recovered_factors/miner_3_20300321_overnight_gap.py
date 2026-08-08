import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; ds={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date') for a in A}; C=pd.DataFrame({a:ds[a].close for a in A}); O=pd.DataFrame({a:ds[a].open for a in A});
g=O/C.shift(1)-1; f=(-g.rolling(3,min_periods=2).mean()).shift(1)
for h in [1,5,10,20]:
 v=[]; n=[]
 for k,t in enumerate(C.index):
  if k+h>=len(C):continue
  z=pd.concat([f.loc[t],C.iloc[k+h]/C.loc[t]-1],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   v.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);n.append(len(z))
 v=np.array(v);print(h,len(v),round(np.mean(n),2),round(v.mean(),6),round(v.mean()/v.std(ddof=1),6),round(np.mean(v>0),4))
print('coverage',f.notna().sum().sum()/(f.shape[0]*15),'turnover',np.mean(f.rank(pct=True).diff().abs().sum(axis=1).dropna()/2))
