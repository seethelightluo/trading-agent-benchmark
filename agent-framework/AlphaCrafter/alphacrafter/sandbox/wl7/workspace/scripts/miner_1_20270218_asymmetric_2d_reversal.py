import os,numpy as np,pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2027-02-18')
p={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'] for a in A}
P=pd.DataFrame(p).sort_index().loc[:end]; R=P.pct_change(); v=R.rolling(20,min_periods=15).std().shift(1)
# asymmetric: penalize downside volatility less? reversal strength based on prior 2d return / downside vol, lagged
r2=P.pct_change(2).shift(1); down=R.where(R<0,0).rolling(20,min_periods=15).std().shift(1)
f=-r2/(down+0.5*v+1e-8)
rows=[]
for h in [1,5,10]:
 y=P.shift(-h)/P-1; ic=[]; ns=[]
 for dt in P.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8: ic.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 s=pd.Series(ic); print(h,len(s),np.mean(ns),s.mean(),s.mean()/s.std(ddof=1)*np.sqrt(252),(s>0).mean())
print('rows',len(P),'coverage',f.notna().sum(axis=1).ge(8).mean(),'turnover',f.rank(pct=True).diff().abs().mean(axis=1).mean())
