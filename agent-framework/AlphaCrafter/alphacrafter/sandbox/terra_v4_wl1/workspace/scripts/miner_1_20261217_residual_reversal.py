import pandas as pd,numpy as np
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({s:pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index()
R=P.pct_change(); m=R.mean(axis=1)
# residual medium-term reversal: remove common cross-asset market move from each asset's cumulative return
beta=R.rolling(60,min_periods=30).cov(m).div(m.rolling(60,min_periods=30).var(),axis=0)
res=R.sub(beta.mul(m,axis=0),axis=0)
F=-res.rolling(10,min_periods=8).sum().shift(1)
F=F.sub(F.median(axis=1),axis=0)
for h in [1,5,10]:
 a=[]; ns=[]
 for i,d in enumerate(P.index[:-h]):
  z=pd.concat([F.iloc[i],P.iloc[i+h]/P.iloc[i]-1],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1: a.append(z.iloc[:,0].corr(z.iloc[:,1]));ns.append(len(z))
 a=np.array(a);print(h,'dates',len(a),'avg_n',round(np.mean(ns),2),'IC',round(a.mean(),8),'ICIR',round(a.mean()/a.std(ddof=1),8),'hit',round((a>0).mean(),4))
print('coverage',round(F.notna().sum().sum()/F.size,6),'turn',round(F.rank(pct=True).diff().abs().mean(axis=1).dropna().mean(),6))
