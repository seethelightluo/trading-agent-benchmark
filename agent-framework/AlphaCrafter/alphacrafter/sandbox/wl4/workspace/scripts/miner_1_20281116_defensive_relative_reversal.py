import os,numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in U:
 p='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(p): P[s]=pd.read_csv(p,parse_dates=['date']).set_index('date')['close'].sort_index()
P=pd.DataFrame(P).loc[:'2028-11-16']; R=P.pct_change()
# short-horizon reversal scaled by idiosyncratic risk relative to defensive assets
market=R[['XAU','US10Y','CN10Y']].mean(axis=1)
res=R.sub(R.mean(axis=1),axis=0)
vol=res.rolling(20).std(); defensive=(R.rolling(10).mean().sub(market.rolling(10).mean(),axis=0))
F=(-R.rolling(3).sum().shift(1)/(vol*np.sqrt(20)+1e-12))*(1+defensive.clip(-.05,.05))
for h in [1,5,10]:
 y=P.pct_change(h).shift(-h); a=[]; nn=[]
 for dt in P.index:
  z=pd.concat([F.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8: a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); nn.append(len(z))
 a=np.array(a); print(h,'dates',len(a),'avgN',np.mean(nn),'IC',a.mean(),'ICIR',a.mean()/(a.std(ddof=1)+1e-12),'hit',np.mean(a>0))
 if h==1:
  print('recent250',a[-250:].mean(),a[-250:].mean()/(a[-250:].std(ddof=1)+1e-12),'turnover',np.nanmean(np.abs(F.rank(axis=1,pct=True).diff()).mean(axis=1)))
