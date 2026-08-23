import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];b='../persistent/stock_data';D={}
for s in U:
 p=f'{b}/{s}.csv'
 if os.path.exists(p): D[s]=pd.read_csv(p,parse_dates=['date']).set_index('date')
C=pd.DataFrame({s:D[s].close for s in D}); R=C.pct_change(); V=pd.DataFrame({s:D[s].volume for s in D});
CL=pd.DataFrame({s:2*(D[s].close-D[s].low)/(D[s].high-D[s].low).replace(0,np.nan)-1 for s in D})
L=np.log1p(V); VS=L.sub(L.rolling(20,min_periods=10).mean()).div(L.rolling(20,min_periods=10).std()); F=-CL*VS
for h in [1,5,10]:
  z=[];ns=[]
  for dt in F.index:
   y=R.shift(-1).rolling(h).sum().shift(-(h-1)).loc[dt]; q=pd.concat([F.loc[dt],y],axis=1).dropna()
   if len(q)>=8:
    v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
    if np.isfinite(v):z.append(v);ns.append(len(q))
  z=np.array(z);print(h,'dates',len(z),'avg_n',np.mean(ns),'coverage',sum(ns)/(len(z)*15),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',np.mean(z>0))
