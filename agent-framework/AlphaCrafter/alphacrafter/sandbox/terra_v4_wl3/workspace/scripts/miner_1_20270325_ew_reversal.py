import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24')
assets=[os.path.basename(x)[:-4] for x in glob.glob('../persistent/stock_data/*.csv')]
px={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date')
 px[a]=d[d.date<=cut].set_index('date').close
p=pd.DataFrame(px).sort_index(); r=p.pct_change()
# EW short reversal: negative exponentially weighted sum of recent 1d returns,
# with newest observation receiving highest weight, normalized by EW volatility.
mean=r.ewm(span=5,min_periods=5,adjust=False).mean()
vol=r.ewm(span=15,min_periods=10,adjust=False).std()
fac=(-mean/(vol+1e-8)).clip(-8,8)
fac=fac.sub(fac.median(axis=1),axis=0)
fac.to_csv('scripts/miner_1_20270325_ew_reversal_signal.csv')
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
