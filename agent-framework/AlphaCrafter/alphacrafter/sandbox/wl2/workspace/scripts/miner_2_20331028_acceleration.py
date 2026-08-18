import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close for a in A}).sort_index()
# medium-term momentum acceleration: recent 10d return minus preceding 20d return, lagged
f=((P/P.shift(10)-1)-(P.shift(10)/P.shift(30)-1)).shift(1)
rows=[]
for d in P.index:
 y=P.shift(-10)/P-1; ok=f.loc[d].notna()&y.loc[d].notna()
 if ok.sum()>=8: rows.append(spearmanr(f.loc[d][ok],y.loc[d][ok]).statistic)
z=np.array(rows)
print('dates',len(z),'avg_n', 'see signal coverage')
print('IC',z.mean(),'ICIR',z.mean()/z.std(),'hit',(z>0).mean())
for h in [1,3,5,10,20]:
 y=P.shift(-h)/P-1; q=[]
 for d in P.index:
  ok=f.loc[d].notna()&y.loc[d].notna()
  if ok.sum()>=8:q.append(spearmanr(f.loc[d][ok],y.loc[d][ok]).statistic)
 q=np.array(q); print('decay',h,q.mean(),q.mean()/q.std())
print('coverage',f.notna().sum(axis=1).mean()/15,'turnover',f.rank(pct=True).diff().abs().mean(axis=1).mean())
f.to_csv('scripts/miner_2_20331028_acceleration_signal.csv')
