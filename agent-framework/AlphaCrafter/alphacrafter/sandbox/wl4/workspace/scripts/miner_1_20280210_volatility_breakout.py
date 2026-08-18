import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2028-02-10'); p={}
for s in U:
 d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').sort_index();p[s]=d.close.loc[:end]
c=pd.DataFrame(p); r=c.pct_change(); vol=r.rolling(20).std(); f=c.pct_change(5)/(vol*np.sqrt(5)+1e-9); y=c.shift(-10)/c-1
A=[]; C=[]; T=[];prev=None
for dt in f.index:
 z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(q): A.append(q);C.append(len(z)/15)
  rr=f.loc[dt].rank(pct=True)
  if prev is not None:T.append((rr-prev).abs().mean())
  prev=rr
A=np.array(A);print('factor=volatility_breakout_5d dates=%d instruments=15 coverage=%.3f'%(len(A),np.mean(C)));print('IC %.5f ICIR %.5f hit %.3f turnover %.5f'%(A.mean(),A.mean()/A.std(ddof=1),np.mean(A>0),np.mean(T)))
for h in [1,5,10,20]:
 yy=c.shift(-h)/c-1;aa=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8:aa.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('decay',h,np.nanmean(aa))
