import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24'); assets=[os.path.basename(x)[:-4] for x in glob.glob('../persistent/stock_data/*.csv')]
C={}; R={}; F={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=cut].set_index('date'); r=d.close.pct_change(); C[a]=d.close
 dn=r.clip(upper=0).rolling(30,min_periods=20).std(); F[a]=(d.close.pct_change(30)/dn).clip(-15,15); R[a]=d.close.pct_change(1).shift(-1)
fac=pd.DataFrame(F).sort_index(); returns={h:pd.DataFrame({a:C[a].pct_change(h).shift(-h) for a in assets}).reindex(fac.index) for h in [1,5,10]}; fwd=returns[1]; fac.to_csv('scripts/miner_3_20270325_tail_adjusted_trend_signal.csv')
def calc(y):
 vals=[]; ds=[]; ns=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ds.append(dt);ns.append(len(z))
 s=pd.Series(vals,index=ds); return s,ns
s,ns=calc(fwd); print('tail_adjusted_trend dates',len(s),'avgN',round(np.mean(ns),2),'IC',s.mean(),'ICIR',s.mean()/s.std(ddof=1),'hit',(s>0).mean(),'coverage',fac.notna().sum(axis=1).mean()/15,'turnover',fac.rank(axis=1,pct=True).diff().abs().mean().mean())
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]:
 q=s[(s.index>=lo)&(s.index<=hi)]; print(lo,'n',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1))
for h in [5,10]:
 q,_=calc(returns[h]); print(h,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1))
