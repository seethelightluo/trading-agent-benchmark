import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in U:
 p='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(p): D[s]=pd.read_csv(p,parse_dates=['date']).set_index('date').sort_index()
c=pd.DataFrame({s:x.close for s,x in D.items()}); hi=pd.DataFrame({s:x.high for s,x in D.items()}); lo=pd.DataFrame({s:x.low for s,x in D.items()})
H=hi.rolling(60,min_periods=40).max(); L=lo.rolling(60,min_periods=40).min(); f=(.5-(c-L)/(H-L).replace(0,np.nan)); r=c.shift(-10)/c-1
z=[]
for d in f.index:
 a=pd.concat([f.loc[d],r.loc[d]],axis=1).dropna()
 if len(a)>=8:z.append(spearmanr(a.iloc[:,0],a.iloc[:,1]).statistic)
z=pd.Series(z).dropna(); print('dates',len(z),'avg_n',len(U),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',(z>0).mean())
for h in [5,20]:
 rr=c.shift(-h)/c-1;q=[]
 for d in f.index:
  a=pd.concat([f.loc[d],rr.loc[d]],axis=1).dropna()
  if len(a)>=8:q.append(spearmanr(a.iloc[:,0],a.iloc[:,1]).statistic)
 q=pd.Series(q).dropna();print('decay',h,len(q),q.mean(),q.mean()/q.std(ddof=1))
print('coverage',f.notna().sum().sum()/(f.shape[0]*f.shape[1]),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
