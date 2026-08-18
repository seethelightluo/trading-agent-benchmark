import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close for a in A}).sort_index(); r=P.pct_change()
# long horizon trend, risk scaled and lagged
f=((P/P.shift(60)-1)/r.rolling(20).std()).shift(1); y=P.shift(-10)/P-1
ics=[]; ns=[]
for d in P.index:
 ok=f.loc[d].notna()&y.loc[d].notna()
 if ok.sum()>=8: ics.append(spearmanr(f.loc[d][ok],y.loc[d][ok]).statistic); ns.append(ok.sum())
z=np.array(ics);print('dates',len(z),'avg_n',np.mean(ns),'coverage',np.mean(ns)/15,'IC',z.mean(),'ICIR',z.mean()/z.std(),'hit',(z>0).mean())
for h in [1,3,5,10,20]:
 yy=P.shift(-h)/P-1;q=[]
 for d in P.index:
  ok=f.loc[d].notna()&yy.loc[d].notna()
  if ok.sum()>=8:q.append(spearmanr(f.loc[d][ok],yy.loc[d][ok]).statistic)
 q=np.array(q);print('decay',h,q.mean(),q.mean()/q.std())
print('turnover',f.rank(pct=True).diff().abs().mean(axis=1).mean());f.to_csv('scripts/miner_2_20331028_risk_trend60_signal.csv')
