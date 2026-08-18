import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index()
r=P.pct_change(); down=r.where(r<0).rolling(20).std(); f=(-r/down).shift(1)
for h in [5,10,20]:
 z=[]; ns=[]
 y=P.shift(-h)/P-1
 for d in P.index:
  ok=f.loc[d].notna()&y.loc[d].notna()
  if ok.sum()>=8:z.append(spearmanr(f.loc[d][ok],y.loc[d][ok]).statistic);ns.append(ok.sum())
 z=np.array(z);print('h',h,'dates',len(z),'avg_n',np.mean(ns),'coverage',np.mean(ns)/15,'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',(z>0).mean())
print('turnover',f.rank(pct=True).diff().abs().mean(axis=1).mean())
f.to_csv('scripts/miner_2_20331111_downside_vol_reversal_signal.csv')
