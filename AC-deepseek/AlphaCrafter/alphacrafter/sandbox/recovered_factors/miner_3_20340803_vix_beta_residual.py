import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index() for a in A}
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').sort_index().close
rows=[]
for dt in sorted(set.intersection(*[set(x.index) for x in D.values()]) & set(v.index)):
 vals=[]; f=[]
 for a in A:
  x=D[a]; i=x.index.get_loc(dt); j=v.index.get_loc(dt)
  if i<61 or j<21 or i+1>=len(x): continue
  ar=np.log(x.close.iloc[i-60:i+1].values[1:]/x.close.iloc[i-60:i].values)
  vr=np.log(v.iloc[j-20:j+1].values[1:]/v.iloc[j-20:j].values)
  beta=np.cov(ar[-20:],vr)[0,1]/(np.var(vr)+1e-9)
  # residual trend after removing contemporaneous VIX beta, lagged by daily close
  sig=np.log(x.close.iloc[i]/x.close.iloc[i-20])-beta*np.log(v.iloc[j]/v.iloc[j-20])
  vals.append(sig);f.append(np.log(x.close.iloc[i+1]/x.close.iloc[i]))
 if len(vals)>=8: rows.append(spearmanr(vals,f).statistic)
r=np.array(rows);print('candidate=vix_beta_residual_trend dates',len(r),'N=15; IC',r.mean(),'ICIR',r.mean()/r.std(),'hit',(r>0).mean())
for h in [5,10,20]:
 # skip decay detail, primary daily only
 pass
