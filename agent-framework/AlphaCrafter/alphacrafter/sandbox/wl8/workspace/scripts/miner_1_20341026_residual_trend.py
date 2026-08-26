import os,numpy as np,pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for a in A:
 f='../persistent/stock_data/'+a+'.csv'
 if os.path.exists(f): px[a]=pd.read_csv(f,parse_dates=['date']).set_index('date')['close'].replace(0,np.nan)
P=pd.DataFrame(px).sort_index(); r=P.pct_change(); m=r.mean(axis=1)
beta=pd.DataFrame(index=P.index,columns=P.columns,dtype=float)
for a in P.columns: beta[a]=r[a].shift(1).rolling(60).cov(m.shift(1))/m.shift(1).rolling(60).var()
res=r.sub(beta.mul(m,axis=0),axis=0)
res20=res.shift(1).rolling(20).sum(); res60=res.shift(1).rolling(60).sum(); vol=res.shift(1).rolling(30).std()
# Reverse beta-neutral trend: negative residual trend is rewarded; all inputs lagged.
raw=((.6*res20+.4*res60)/vol).ewm(span=3,min_periods=1).mean()
sig=-(raw.rank(axis=1,pct=True)-.5)
fwd=P.shift(-10)/P-1; rows=[]
for d in sig.index:
 z=pd.concat([sig.loc[d],fwd.loc[d]],axis=1).dropna()
 if len(z)>=8: rows.append((d,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); print('dates',len(q),'avgN',q.n.mean(),'coverage',q.n.sum()/(len(q)*15)); print('IC',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std(ddof=1),'hit',(q.ic>0).mean(),'turnover',sig.diff().abs().mean().mean())
for w in [365,750,1260]:
 x=q.tail(w); print('recent',w,x.ic.mean(),x.ic.mean()/x.ic.std(ddof=1))
for h in [1,5,10,20]:
 y=P.shift(-h)/P-1; z=[]
 for d in sig.index:
  x=pd.concat([sig.loc[d],y.loc[d]],axis=1).dropna()
  if len(x)>=8:z.append(spearmanr(x.iloc[:,0],x.iloc[:,1]).statistic)
 print('decay',h,np.nanmean(z),len(z))
sig.tail(500).to_csv('scripts/miner_1_20341026_residual_trend_signal.csv'); q.to_csv('scripts/miner_1_20341026_residual_trend_ic.csv')
