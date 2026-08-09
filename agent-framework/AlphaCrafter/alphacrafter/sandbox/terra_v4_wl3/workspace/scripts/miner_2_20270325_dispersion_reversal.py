import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24')
assets=[os.path.basename(x)[:-4] for x in glob.glob('../persistent/stock_data/*.csv')]
px={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date')
 px[a]=d[d.date<=cut].set_index('date').close
p=pd.DataFrame(px).sort_index(); r=p.pct_change()
# Cross-asset dispersion-conditioned reversal: reverse each asset's 3d return,
# and scale by current cross-sectional dispersion (high dispersion -> stronger).
disp=r.T.rolling(3,min_periods=2).std().T.mean(axis=1)
raw=-p.pct_change(3)
scale=(disp/disp.rolling(60,min_periods=20).median()).clip(0.5,2.0)
fac=raw.mul(scale,axis=0)
fac=fac.sub(fac.median(axis=1),axis=0)
fac=fac.clip(lower=fac.quantile(.05,axis=1),upper=fac.quantile(.95,axis=1),axis=0)
fac.to_csv('scripts/miner_2_20270325_dispersion_reversal_signal.csv')
print('assets',len(assets),'rows',len(fac),'period',fac.index.min(),fac.index.max())
for h in [1,5,10]:
 fwd=p.pct_change(h).shift(-h); vals=[]; ds=[]; ns=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ds.append(dt); ns.append(len(z))
 s=pd.Series(vals,index=ds)
 print('H',h,'dates',len(s),'avgN',round(np.mean(ns),2),'IC %.6f ICIR %.6f hit %.4f'%(s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
 if h==1:
  for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]:
   q=s[(s.index>=lo)&(s.index<=hi)]; print('regime',lo,hi,'IC %.6f n %d'%(q.mean(),len(q)))
print('coverage',fac.notna().sum(axis=1).mean()/len(assets),'turnover',fac.rank(axis=1,pct=True).diff().abs().mean().mean())
