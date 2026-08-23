import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'].astype(float) for s in U}).sort_index()
ret=P.pct_change(); vol=ret.rolling(20,min_periods=15).std()*np.sqrt(252)
F=(-(P/P.shift(5)-1)/vol).shift(1); R=P.shift(-10)/P-1
allv=[]; counts=[]; cov=[]; tr=[]; prev=None
for dt in F.index:
 x,y=F.loc[dt],R.loc[dt]; ok=x.notna()&y.notna()
 if ok.sum()>=8:
  allv.append(spearmanr(x[ok],y[ok]).statistic);counts.append(ok.sum());cov.append(ok.mean()); rr=x.rank(pct=True);tr.append(np.mean(np.abs(rr-(prev if prev is not None else rr))));prev=rr
z=np.array(allv); print('factor=lagged negative 5d return / 20d annualized vol; horizon=10d');print('dates',len(z),'avg_instruments',np.mean(counts),'coverage',np.mean(cov),'IC',z.mean(),'ICIR',z.mean()/(z.std(ddof=1)/np.sqrt(len(z))),'hit',np.mean(z>0),'turnover',np.mean(tr))
for n,a,b in [('2020-22','2020-01-01','2022-12-31'),('2023-24','2023-01-01','2024-12-31'),('2025-26','2025-01-01','2026-12-31'),('2027YTD','2027-01-01','2027-05-19')]:
 vals=[]
 for dt in F.index[(F.index>=a)&(F.index<=b)]:
  x,y=F.loc[dt],R.loc[dt];ok=x.notna()&y.notna()
  if ok.sum()>=8: vals.append(spearmanr(x[ok],y[ok]).statistic)
 q=np.array(vals); print(n,'dates',len(q),'IC',q.mean(),'ICIR',q.mean()/(q.std(ddof=1)/np.sqrt(len(q))))
pd.DataFrame(F).reset_index().to_csv('scripts/miner_2_20270520_short_reversal_volscaled_signal.csv',index=False)
