import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24'); assets=[os.path.basename(x)[:-4] for x in glob.glob('../persistent/stock_data/*.csv')]
P={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=cut].set_index('date'); P[a]=d.close
p=pd.DataFrame(P).sort_index(); ret=p.pct_change(); med=ret.median(axis=1)
F={}
for a in assets:
 r=ret[a]; beta=r.rolling(60,min_periods=30).cov(med)/med.rolling(60,min_periods=30).var()
 resid=r-beta*med
 vol=r.rolling(20,min_periods=15).std()
 # residual short reversal, damped by volatility and smoothed over 3 sessions
 F[a]=(-resid.rolling(3,min_periods=3).sum()/(vol*np.sqrt(3)+1e-8)).clip(-10,10)
fac=pd.DataFrame(F).sort_index(); fac.to_csv('scripts/miner_3_20270325_residual_volreversal_signal.csv')
print('assets',len(assets),'rows',len(fac))
for h in [1,5,10]:
 fwd=p.pct_change(h).shift(-h); vals=[]; ds=[]; ns=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ds.append(dt);ns.append(len(z))
 s=pd.Series(vals,index=ds); print('H',h,'dates',len(s),'avgN',round(np.mean(ns),2),'IC %.6f ICIR %.6f hit %.4f'%(s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
 if h==1:
  for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]: print('regime',lo,hi,s[(s.index>=lo)&(s.index<=hi)].mean())
print('coverage',fac.notna().sum(axis=1).mean()/len(assets),'turnover',fac.rank(axis=1,pct=True).diff().abs().mean().mean())
