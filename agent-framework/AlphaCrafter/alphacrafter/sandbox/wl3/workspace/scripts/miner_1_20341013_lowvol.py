import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];base='../persistent/stock_data';px={}
for s in U:
 f=f'{base}/{s}.csv'
 if os.path.exists(f):
  d=pd.read_csv(f);d.date=pd.to_datetime(d.date);px[s]=d.set_index('date').close
P=pd.DataFrame(px).sort_index();r=P.pct_change();F=(-r.rolling(30,min_periods=25).std()).shift(1)
print('assets',len(px),'dates',len(P),'coverage',F.notna().mean().mean())
for h in [1,3,5,10,20]:
 q=[];ns=[]
 for i in range(len(P)-h):
  z=pd.concat([F.iloc[i],P.iloc[i+h]/P.iloc[i]-1],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 q=np.array(q);print('horizon',h,'dates',len(q),'avg_n',np.mean(ns),'IC',np.mean(q),'ICIR',np.mean(q)/np.std(q,ddof=1)*np.sqrt(len(q)),'hit',np.mean(q>0))
 if h==10:
  for n in [120,252,756,1260]:
   x=q[-n:];print('recent10',n,'IC',np.mean(x),'ICIR',np.mean(x)/np.std(x,ddof=1)*np.sqrt(len(x)))
rank=F.rank(axis=1,pct=True);print('turnover',np.nanmean(np.abs(rank-rank.shift()).mean(axis=1)))
