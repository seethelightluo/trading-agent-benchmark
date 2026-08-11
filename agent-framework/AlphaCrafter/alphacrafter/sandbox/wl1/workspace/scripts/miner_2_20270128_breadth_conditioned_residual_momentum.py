import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut='2027-01-27'
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index()
# Candidate: breadth-conditioned residual momentum. Cross-sectional residual return is
# strengthened in broad participation and attenuated/reversed in narrow markets.
R=P.pct_change(20); med=R.median(axis=1); resid=R.sub(med,axis=0)
breadth=(P.pct_change(5)>0).sum(axis=1)/P.notna().sum(axis=1)
# continuous, lagged state; broad positive breadth -> trend, narrow/negative -> reversal
state=(breadth.rolling(20,min_periods=10).mean()-0.5)*2
f=resid.mul(state,axis=0).shift(1)
print('candidate breadth-conditioned residual momentum')
for h in [5,10,20]:
 a=[]
 for dt in P.loc[:cut].index:
  fut=P.shift(-h).loc[dt]/P.loc[dt]-1
  q=pd.concat([f.loc[dt],fut],axis=1).dropna()
  if len(q)>=8:a.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
 a=np.asarray(a); print('h',h,'dates',len(a),'avg_n',15,'coverage_dates',len(a)/len(P.loc[:cut].index),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0))
valid=f.loc[:cut].notna().sum(axis=1); turnover=f.loc[:cut].rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(); print('coverage',valid.mean()/15,'avg_n',valid.mean(),'turnover',turnover)
# annual 10d and regime split breadth
out=[]
for dt in P.loc[:cut].index:
 q=pd.concat([f.loc[dt],(P.shift(-10).loc[dt]/P.loc[dt]-1)],axis=1).dropna()
 if len(q)>=8:out.append((dt,spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic,breadth.loc[dt]))
d=pd.DataFrame(out,columns=['date','ic','breadth']).set_index('date'); print('annual'); print(d.groupby(d.index.year).ic.agg(['count','mean']).to_string())
print('breadth_low',d[d.breadth<.4].ic.mean(),'breadth_high',d[d.breadth>.6].ic.mean())
print('period',P.index.min(),P.loc[:cut].index.max())
