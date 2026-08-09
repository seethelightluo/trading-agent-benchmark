import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={};H={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date');P[s]=d.close;H[s]=d.high-d.low
p=pd.DataFrame(P).sort_index(); ret=p.pct_change(); vol=ret.rolling(20).std(); rng=pd.DataFrame(H).sort_index().rolling(20).mean()/p
# volatility-adjusted short trend: 5d return / 20d vol, cross-sectional demean
f=(p/p.shift(5)-1)/vol
f=f.sub(f.median(axis=1),axis=0); fw=p.shift(-1)/p-1
ics=[];turn=[];prev=None;ns=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z));
  q=f.loc[dt].rank(pct=True)
  if prev is not None:turn.append(np.mean(abs(q-prev)))
  prev=q
x=np.array(ics);print('voladj5 trend',len(x),np.mean(ns),np.mean(ns)/15,np.mean(x),np.mean(x)/np.std(x,ddof=1),np.mean(x>0),np.mean(turn))
for h in [5,10,20]:
 fw=p.shift(-h)/p-1;xx=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8:xx.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print(h,len(xx),np.mean(xx),np.mean(xx)/np.std(xx,ddof=1))
f.stack().rename('signal').rename_axis(['date','symbol']).reset_index().to_csv('scripts/miner_1_20261217_voladj5_signal.csv',index=False)
