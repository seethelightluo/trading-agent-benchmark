import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; p={}
for a in A:
 f='../persistent/stock_data/'+a+'.csv'
 if os.path.exists(f): p[a]=pd.read_csv(f,parse_dates=['date']).set_index('date')['close'].replace(0,np.nan)
P=pd.DataFrame(p).sort_index(); R=P.pct_change(); market=R.mean(axis=1)
mu=market.rolling(60).mean(); vm=((market-mu)**2).rolling(60).mean()
beta=pd.DataFrame(index=R.index,columns=R.columns)
for a in R: beta[a]=((R[a]-R[a].rolling(60).mean())*(market-mu)).rolling(60).mean()/vm
beta=beta.shift(1); m40=market.rolling(40).sum().shift(1); ar40=R.rolling(40).sum().shift(1); res=ar40-beta.mul(m40,axis=0); rv=(R-beta.mul(market,axis=0)).rolling(30).std().shift(1); sig=(res/rv).rank(axis=1,pct=True)-.5
fwd=P.shift(-10)/P-1; q=[]
for d in sig.index:
 z=pd.concat([sig.loc[d],fwd.loc[d]],axis=1).dropna()
 if len(z)>=8:q.append((d,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
q=pd.DataFrame(q,columns=['d','ic','n']).set_index('d'); print('dates',len(q),'avgN',q.n.mean(),'coverage',q.n.sum()/(len(q)*15));print('IC',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std(ddof=1),'hit',(q.ic>0).mean(),'turn',sig.diff().abs().mean().mean())
for w in [365,750,1260]:
 z=q.tail(w);print('recent',w,z.ic.mean(),z.ic.mean()/z.ic.std(ddof=1))
for h in [1,5,10,20]:
 y=P.shift(-h)/P-1; zq=[]
 for d in sig.index:
  z=pd.concat([sig.loc[d],y.loc[d]],axis=1).dropna()
  if len(z)>=8:zq.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('decay',h,np.nanmean(zq))
sig.tail(500).to_csv('scripts/miner_3_20340928_beta_neutral_signal.csv')
