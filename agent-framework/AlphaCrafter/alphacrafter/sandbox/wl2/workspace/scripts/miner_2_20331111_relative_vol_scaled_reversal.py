import numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index()
# Volatility-scaled relative 5d reversal, lagged one day
r=P.pct_change(5); vol=P.pct_change().rolling(20).std(); f=(-(r.sub(r.median(axis=1),axis=0))/vol).shift(1)
ics=[]; ns=[]
for d in P.index:
 y=P.shift(-10)/P-1; ok=f.loc[d].notna()&y.loc[d].notna()
 if ok.sum()>=8: ics.append(spearmanr(f.loc[d][ok],y.loc[d][ok]).statistic); ns.append(ok.sum())
z=np.array(ics); print('candidate relative_vol_scaled_5d_reversal'); print('dates',len(z),'avg_n',np.mean(ns),'coverage',np.mean(ns)/15,'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',(z>0).mean())
for h in [5,10,20]:
 y=P.shift(-h)/P-1; q=[]
 for d in P.index:
  ok=f.loc[d].notna()&y.loc[d].notna()
  if ok.sum()>=8:q.append(spearmanr(f.loc[d][ok],y.loc[d][ok]).statistic)
 q=np.array(q); print('decay',h,'dates',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1))
print('turnover',f.rank(pct=True).diff().abs().mean(axis=1).mean())
f.to_csv('scripts/miner_2_20331111_relative_vol_scaled_reversal_signal.csv')
