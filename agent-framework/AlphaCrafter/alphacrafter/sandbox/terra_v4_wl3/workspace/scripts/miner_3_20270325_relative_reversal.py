import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24'); paths=glob.glob('../persistent/stock_data/*.csv'); assets=[os.path.basename(x)[:-4] for x in paths]
cl={}
for p in paths:
 a=os.path.basename(p)[:-4]; d=pd.read_csv(p,parse_dates=['date']).sort_values('date').set_index('date'); cl[a]=d.close[d.index<=cut]
close=pd.DataFrame(cl).sort_index(); ret=close.pct_change()
# Relative reversal: fade each asset's 5d move relative to the contemporaneous cross-sectional median,
# scaled by its own 20d volatility. Cross-sectional demeaning avoids common market direction.
r5=close.pct_change(5); med=r5.median(axis=1); vol=ret.rolling(20,min_periods=10).std()
fac=(-(r5.sub(med,axis=0))/vol).replace([np.inf,-np.inf],np.nan)
fac.to_csv('scripts/miner_3_20270325_relative_reversal_signal.csv')
print('assets',len(assets),'rows',len(fac),'period',fac.index.min(),fac.index.max())
for h in [1,5,10]:
 y=close.pct_change(h).shift(-h); vals=[]; ds=[]; ns=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ds.append(dt); ns.append(len(z))
 s=pd.Series(vals,index=ds); print('H',h,'dates',len(s),'avgN',round(np.mean(ns),2),'IC %.6f ICIR %.6f hit %.4f'%(s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
 if h==1:
  for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]:
   q=s[(s.index>=lo)&(s.index<=hi)]; print('regime',lo,'dates',len(q),'IC %.6f ICIR %.6f'%(q.mean(),q.mean()/q.std(ddof=1)))
print('coverage',round(fac.notna().sum(axis=1).mean()/15,4),'turnover',round(fac.rank(axis=1,pct=True).diff().abs().mean().mean(),4))
