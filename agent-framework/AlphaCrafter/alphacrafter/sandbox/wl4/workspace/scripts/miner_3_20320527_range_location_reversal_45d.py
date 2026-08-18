import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 p='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(p): D[s]=pd.read_csv(p,parse_dates=['date']).set_index('date').sort_index()
c=pd.DataFrame({s:x.close for s,x in D.items()}); hi=pd.DataFrame({s:x.high for s,x in D.items()}); lo=pd.DataFrame({s:x.low for s,x in D.items()})
H=hi.rolling(45,min_periods=30).max(); L=lo.rolling(45,min_periods=30).min(); f=.5-(c-L)/(H-L).replace(0,np.nan)
print('universe',len(D),'dates',len(c))
for h in [5,10,20]:
 r=c.shift(-h)/c-1; z=[]; ns=[]
 for d in f.index:
  a=pd.concat([f.loc[d],r.loc[d]],axis=1).dropna()
  if len(a)>=8: z.append(spearmanr(a.iloc[:,0],a.iloc[:,1]).statistic); ns.append(len(a))
 z=pd.Series(z).dropna(); print('horizon',h,'dates',len(z),'avg_n',np.mean(ns),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',(z>0).mean())
q=f.rank(axis=1,pct=True); print('coverage',f.notna().sum().sum()/(f.shape[0]*f.shape[1]),'turnover',q.diff().abs().mean(axis=1).mean())
