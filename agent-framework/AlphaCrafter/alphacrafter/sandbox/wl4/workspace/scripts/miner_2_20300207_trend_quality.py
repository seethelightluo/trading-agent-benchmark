import os,numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 p='../persistent/stock_data/'+s+'.csv'
 if not os.path.exists(p): p='../persistent/index_data/'+s+'.csv'
 d=pd.read_csv(p); d.columns=[str(c).lower() for c in d.columns]; date=next(c for c in d if c in ['date','datetime','trade_date']); close=next(c for c in d if c in ['close','收盘'])
 return pd.DataFrame({'c':pd.to_numeric(d[close],errors='coerce')},index=pd.to_datetime(d[date])).sort_index()
A=[]
for s in U:
 d=load(s); c=d.c; lr=np.log(c/c.shift(1)); f=(np.log(c/c.shift(60))/(lr.rolling(20).std()*np.sqrt(60))).shift(1); y=np.log(c.shift(-10)/c); A.append(pd.concat([f.rename('f'),y.rename('y')],axis=1).assign(s=s))
q=pd.concat(A); out=[]
for dt,g in q.groupby(level=0):
 g=g.dropna()
 if len(g)>=8: out.append(spearmanr(g.f,g.y).statistic)
a=np.array(out); print('dates',len(a),'avgN',q.dropna().groupby(level=0).size().mean(),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0),'recent250',a[-250:].mean()/a[-250:].std(ddof=1),'coverage',q.dropna().shape[0]/q.shape[0])
