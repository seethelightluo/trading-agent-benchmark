import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; qs=[]
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv');d.date=pd.to_datetime(d.date);qs.append(d.sort_values('date').set_index('date').close.astype(float))
z=pd.concat(qs,axis=1,keys=U).sort_index();r=z.pct_change(); out=[]; ns=[]
for i in range(60,len(z)-1):
 f=[]
 for s in U:
  q=r[s].iloc[i-20:i].dropna(); f.append(-q[q<0].std()/(q.std()+1e-9) if len(q)>=15 and (q<0).sum()>=3 else np.nan)
 y=r.iloc[i+1]; ok=np.isfinite(f)&np.isfinite(y.values)
 if ok.sum()>=8:out.append(spearmanr(np.array(f)[ok],y.values[ok]).statistic);ns.append(ok.sum())
a=np.array(out);print('dates',len(a),'mean_n',np.mean(ns),'coverage',np.mean(ns)/15,'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0))
