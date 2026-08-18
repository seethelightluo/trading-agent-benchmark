import pandas as pd,numpy as np
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2032-03-31')
D={s:pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date')['close'].sort_index() for s in U}; p=pd.concat(D,axis=1).sort_index().loc[:cut]; lr=np.log(p).diff(); r10=lr.rolling(10).sum(); disp=r10.std(axis=1); vol=lr.rolling(20).std()*np.sqrt(20); raw=(-(r10-r10.mean(axis=1),)).__class__ if False else (-(r10.sub(r10.mean(axis=1),axis=0))).div(vol.clip(lower=.005,upper=1)).replace([np.inf,-np.inf],np.nan).clip(-5,5); f=raw.where(disp>disp.rolling(60).median()).shift(1); R=np.log(p.shift(-10)/p); A=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],R.loc[dt]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 if len(z)>=8:A.append(z.iloc[:,0].corr(z.iloc[:,1]))
a=pd.Series(A).dropna(); print('cutoff',cut.date(),'dates',len(a),'N',len(U),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',(a>0).mean()); print('coverage',f.notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for h in [5,10,20]:
 rr=np.log(p.shift(-h)/p); q=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],rr.loc[dt]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8:q.append(z.iloc[:,0].corr(z.iloc[:,1]))
 print('horizon',h,'IC',np.nanmean(q),'dates',len(q))
