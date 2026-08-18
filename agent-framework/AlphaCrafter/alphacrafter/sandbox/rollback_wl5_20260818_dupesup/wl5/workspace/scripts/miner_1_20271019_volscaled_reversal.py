import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; root='../persistent/stock_data'; end=pd.Timestamp('2027-10-18')
C={}
for s in U:
 d=pd.read_csv(os.path.join(root,s+'.csv')); d.date=pd.to_datetime(d.date); d=d[(d.date>='2020-01-01')&(d.date<=end)].sort_values('date'); C[s]=d.set_index('date').close.astype(float)
prices=pd.DataFrame(C).sort_index(); ret=prices.pct_change(); factor=-(prices/prices.shift(5)-1)/(ret.rolling(20,min_periods=15).std()*np.sqrt(5)); results={}; ys={}
for h in [1,5,10]:
 y=prices.shift(-h)/prices-1; ys[h]=y; ics=[]; ns=[]
 for dt in factor.index:
  z=pd.concat([factor.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8: ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 a=np.asarray(ics); results[h]=(len(a),float(a.mean()),float(a.mean()/a.std(ddof=1)),float(np.mean(a>0)),float(np.mean(ns)))
print('factor=5d reversal / 20d volatility'); print(results); print('dates',factor.index.min().date(),factor.index.max().date(),'assets',len(U),'coverage',float(factor.notna().sum(axis=1).mean()/15))
r=factor.rank(axis=1,pct=True); print('turnover_proxy',float((r-r.shift()).abs().mean(axis=1).mean()))
for a,b in [('2020-01-01','2022-02-28'),('2022-03-01','2024-04-30'),('2024-05-01','2027-10-18')]:
 q=[]
 for dt in factor.loc[a:b].index:
  z=pd.concat([factor.loc[dt],ys[1].loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=np.asarray(q);print('regime',a,b,'dates',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6))
