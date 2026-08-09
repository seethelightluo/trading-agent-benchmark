import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
S=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; rows=[]
for s in S:
 f='../persistent/stock_data/'+s+'.csv'
 if not os.path.exists(f): f='../persistent/index_data/'+s+'.csv'
 x=pd.read_csv(f,parse_dates=['date']).sort_values('date').set_index('date'); ret=x.close.pct_change()
 # volatility-scaled short reversal, distinct from raw reversal
 fac=-(ret.rolling(3,min_periods=3).sum()/(ret.rolling(20,min_periods=20).std()*np.sqrt(3)+1e-8))
 fr=x.close.shift(-1)/x.close-1
 rows.append(pd.DataFrame({'f':fac,'fr':fr}).dropna().assign(sym=s))
a=pd.concat(rows);p=a.pivot(columns='sym',values='f');r=a.pivot(columns='sym',values='fr')
def run(pp,rr):
 z=[];n=[]
 for d in pp.index:
  q=pd.concat([pp.loc[d],rr.loc[d]],axis=1).dropna()
  if len(q)>=8:z.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);n.append(len(q))
 z=np.array(z);return len(z),np.mean(n),z.mean(),z.mean()/z.std(ddof=1),np.mean(z>0),sum(n)/(len(z)*15),z
for a1,a2 in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2026-07-16','2027-02-10')]:print(a1,a2,run(p.loc[a1:a2],r.loc[a1:a2])[:-1])
print('FULL',run(p,r)[:-1]);p.stack().rename('signal').reset_index().to_csv('../persistent/factor_signals_miner_3_20270211_volscaled_rev3.csv',index=False)
