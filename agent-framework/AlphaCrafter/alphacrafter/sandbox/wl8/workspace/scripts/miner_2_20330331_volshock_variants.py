import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in U:
 try:D[s]=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().close.astype(float).replace(0,np.nan)
 except:pass
p=pd.DataFrame(D).sort_index();r=np.log(p).diff(); rv5=r.rolling(5,min_periods=4).std();rv30=r.rolling(30,min_periods=20).std(); sh=(rv5/(rv30+1e-12)-1).clip(-2,3); ret=np.log(p/p.shift(5))
for a in [0.5,1.0,1.5,2.0]:
 f=(-ret*(1+a*sh)).rolling(2,min_periods=2).mean();q=np.log(p.shift(-10)/p);ii=[]
 for d in f.index:
  z=pd.concat([f.loc[d],q.loc[d]],axis=1).dropna()
  if len(z)>=8:ii.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 ii=pd.Series(ii).dropna();print('a',a,'dates',len(ii),'IC',ii.mean(),'ICIR',ii.mean()/ii.std(ddof=1),'hit',(ii>0).mean())
