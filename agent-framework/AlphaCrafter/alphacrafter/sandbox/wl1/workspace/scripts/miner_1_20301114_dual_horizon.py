import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; b='../persistent/stock_data'
D=pd.concat([pd.read_csv(f'{b}/{a}.csv',parse_dates=['date']).set_index('date')['close'].rename(a) for a in A],axis=1).sort_index(); D=D.dropna(how='all')
r=D.pct_change(); v=r.rolling(20).std();
# interpretable dual-horizon signal: medium trend plus short reversal, risk scaled, lagged
s=(0.7*D.pct_change(20)/v - 0.3*r/v).shift(1)
for h in [1,5,10,20]:
 y=D.pct_change(h).shift(-h); q=[]; ns=[]
 for dt in s.index:
  ok=s.loc[dt].notna()&y.loc[dt].notna()
  if ok.sum()>=8:q.append(spearmanr(s.loc[dt,ok],y.loc[dt,ok]).statistic);ns.append(ok.sum())
 q=pd.Series(q).dropna();print(h,len(q),round(np.mean(ns),2),round(q.mean(),6),round(q.mean()/q.std(ddof=1)*np.sqrt(252),6))
print('coverage',s.notna().sum().sum()/s.size,'turnover',s.rank(pct=True).diff().abs().mean(axis=1).mean())
s.to_csv('scripts/miner_1_20301114_dual_horizon_signal.csv')
