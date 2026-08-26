import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'
C=pd.concat({s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date')['close'] for s in U},axis=1).sort_index().loc[:'2035-05-27']
r=C.pct_change(); v=r.rolling(20,min_periods=15).std(); disp=r.rolling(20,min_periods=15).std().median(axis=1)
# conditional residualized short-term reversal: reverse 5d return, remove cross-sectional mean, activate above 60th percentile dispersion
x=-(C/C.shift(5)-1); x=x.sub(x.median(axis=1),axis=0); f=(x/v).where(disp>disp.rolling(252,min_periods=100).quantile(.6)).shift(1)
for h in [1,5,10,20]:
 y=C.shift(-h)/C-1; a=[]; ds=[]; ns=[]
 for d in f.index:
  z=pd.concat([f.loc[d],y.loc[d]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ds.append(d);ns.append(len(z))
 a=np.array(a); print('H',h,'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1)*np.sqrt(252),'dates',len(a),'N',np.mean(ns),'hit',np.mean(a>0))
 for st,en in [('2020','2026-12-31'),('2027','2030-12-31'),('2031','2034-12-31'),('2035','2035-05-27')]:
  q=a[(np.array(ds)>=pd.Timestamp(st))&(np.array(ds)<=pd.Timestamp(en))]; print(st,len(q),q.mean() if len(q) else np.nan)
print('coverage',f.notna().sum(axis=1).mean()/15)
