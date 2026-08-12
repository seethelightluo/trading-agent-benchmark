import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut='2027-01-13'; D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cut').set_index('date').sort_index()
 rng=(x.high-x.low).replace(0,np.nan); clv=(2*x.close-x.high-x.low)/rng
 rel=(rng/x.close)/((rng/x.close).rolling(20,min_periods=10).median())
 # inverse of tested pressure: weak closes / strong range expansion imply next-day reversal
 D[s]=pd.DataFrame({'f':(-clv*rel).rolling(5,min_periods=5).mean(),'r':x.close.pct_change()})
F=pd.concat({s:v.f for s,v in D.items()},axis=1)
def go(Y, dates=None):
 q=[]; ns=[]
 for d in (F.index if dates is None else dates):
  if d not in Y.index: continue
  z=pd.concat([F.loc[d],Y.loc[d]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 return np.asarray(q),ns
for h in [1,3,5,10]:
 Y=pd.concat({s:D[s].r.rolling(h).sum().shift(-h) for s in U},axis=1);q,n=go(Y); print('h',h,'dates',len(q),'avgN',round(np.mean(n),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
q,n=go(pd.concat({s:D[s].r.shift(-1) for s in U},axis=1)); print('coverage',round(F.notna().sum().sum()/F.size,4),'turnover',round(F.rank(pct=True).diff().abs().mean(axis=1).mean(),4))
for a,b in [('2020-01','2022-12'),('2023-01','2024-12'),('2025-01','2027-01')]:
 q,n=go(pd.concat({s:D[s].r.shift(-1) for s in U},axis=1),F.loc[a:b].index); print('regime',a,b,'dates',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
F.to_csv('scripts/miner_1_20270114_inverse_candle_pressure_signal.csv')
