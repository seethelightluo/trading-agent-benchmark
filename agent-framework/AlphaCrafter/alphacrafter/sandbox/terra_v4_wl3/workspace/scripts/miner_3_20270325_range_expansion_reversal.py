import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24'); A=[os.path.basename(x)[:-4] for x in glob.glob('../persistent/stock_data/*.csv')]; F={};P={}
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=cut].set_index('date'); r=d.close.pct_change(); P[a]=d.close
 # range expansion is a proxy for attention/stress; fade recent 3d move only when expansion is high
 rng=(d.high-d.low)/d.close; base=rng.rolling(30,min_periods=15).median(); exp=(rng/(base+1e-9)-1).clip(0,3)
 F[a]=(-r.rolling(3,min_periods=3).sum() * (1+0.6*exp.rolling(3,min_periods=2).mean())).clip(-.5,.5)
f=pd.DataFrame(F).sort_index(); p=pd.DataFrame(P).sort_index(); f.to_csv('scripts/miner_3_20270325_range_expansion_reversal_signal.csv')
print('assets',len(A),'rows',len(f),'period',f.index.min(),f.index.max())
for h in [1,5,10]:
 y=p.pct_change(h).shift(-h); vals=[];ns=[];ds=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z));ds.append(dt)
 s=pd.Series(vals,index=ds);print('H',h,'dates',len(s),'avgN',round(np.mean(ns),2),'IC %.6f ICIR %.6f hit %.4f'%(s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
 if h==1:
  for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]:
   q=s[(s.index>=lo)&(s.index<=hi)];print('regime',lo,'n',len(q),'IC %.6f ICIR %.6f'%(q.mean(),q.mean()/q.std(ddof=1)))
print('coverage',f.notna().sum(axis=1).mean()/len(A),'turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
