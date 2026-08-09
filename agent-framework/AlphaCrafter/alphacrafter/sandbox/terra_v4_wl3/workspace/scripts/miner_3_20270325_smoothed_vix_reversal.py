import pandas as pd,numpy as np
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24'); f=pd.read_csv('scripts/miner_3_20270325_vix_range_reversal_signal.csv',parse_dates=['date']).set_index('date'); f=f.rolling(5,min_periods=3).mean(); assets=list(f.columns)
p={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index(); p[a]=d.close
p=pd.DataFrame(p).loc[:cut]; out=[]
for h in [1,5,10]:
 fw=p.pct_change(h).shift(-h); q=[]; ns=[]
 for dt in f.index:
  if dt not in fw.index: continue
  z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8: q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 s=pd.Series(q); print(h,len(s),round(np.mean(ns),2),s.mean(),s.mean()/s.std(ddof=1),(s>0).mean())
print('coverage',f.notna().sum(axis=1).mean()/15,'turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
