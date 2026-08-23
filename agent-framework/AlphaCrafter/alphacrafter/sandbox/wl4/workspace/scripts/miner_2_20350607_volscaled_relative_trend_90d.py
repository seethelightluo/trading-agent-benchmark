import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close.astype(float) for a in A}
P=pd.DataFrame(P).sort_index().loc[:'2035-06-06']
r=P.pct_change(); mom=P.pct_change(90); vol=r.rolling(30,min_periods=20).std()*np.sqrt(252)
# volatility-scaled medium-term trend, cross-sectional centered to remove common market direction
F=(mom/vol).sub((mom/vol).mean(axis=1),axis=0).shift(1)
rows=[]
for dt in F.index:
 y=P.shift(-10).loc[dt]/P.loc[dt]-1; q=pd.concat([F.loc[dt],y],axis=1).dropna()
 if len(q)>=8: rows.append((dt,len(q),spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic))
a=pd.DataFrame(rows,columns=['date','n','ic'])
print('dates',len(a),'avgN',round(a.n.mean(),2),'coverage',round(a.n.sum()/(len(a)*15),4))
print('full IC ICIR hit',round(a.ic.mean(),6),round(a.ic.mean()/a.ic.std(ddof=1),6),round((a.ic>0).mean(),4))
for k in [120,260,520,780]:
 q=a.tail(k); print('recent',k,'IC ICIR hit',round(q.ic.mean(),6),round(q.ic.mean()/q.ic.std(ddof=1),6),round((q.ic>0).mean(),4))
# decay
for h in [1,5,10,20]:
 z=[]
 for dt in F.index:
  y=P.shift(-h).loc[dt]/P.loc[dt]-1; q=pd.concat([F.loc[dt],y],axis=1).dropna()
  if len(q)>=8:z.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
 z=pd.Series(z).dropna(); print('decay',h,round(z.mean(),6),round(z.mean()/z.std(ddof=1),6))
# signal turnover proxy
x=F.rank(axis=1,pct=True); print('turnover_proxy',round(x.diff().abs().mean(axis=1).mean(),6))
