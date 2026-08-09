import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24')
assets=sorted([os.path.basename(x)[:-4] for x in glob.glob('../persistent/stock_data/*.csv')])
F={}; fw={h:{} for h in (1,5,10)}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date')
 d=d[d.date<=cut].set_index('date'); c=d.close
 r=c.pct_change(); vol=r.rolling(10,min_periods=8).std()
 # Contrarian one-day return normalized by recent realized risk.
 F[a]=(-r/vol).replace([np.inf,-np.inf],np.nan)
 for k in fw: fw[k][a]=c.pct_change(k).shift(-k)
fac=pd.DataFrame(F).sort_index(); fac.to_csv('scripts/miner_1_20270325_risk_scaled_reversal_signal.csv')
print('assets',len(assets),'rows',len(fac),'period',fac.index.min(),fac.index.max())
for k in fw:
 fwd=pd.DataFrame(fw[k]).reindex(fac.index); vals=[]; ds=[]; ns=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ds.append(dt); ns.append(len(z))
 s=pd.Series(vals,index=ds)
 print('H',k,'dates',len(s),'avgN %.2f IC %.6f ICIR %.6f hit %.4f'%(np.mean(ns),s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
 if k==1:
  for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]:
   q=s[(s.index>=lo)&(s.index<=hi)]; print('regime',lo,hi,'IC %.6f n %d'%(q.mean(),len(q)))
print('coverage %.4f turnover %.4f'%(fac.notna().sum(axis=1).mean()/len(assets),fac.rank(axis=1,pct=True).diff().abs().mean().mean()))
