import pandas as pd,numpy as np,glob
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-04-07'); base='../persistent/stock_data/'
assets=[x.split('/')[-1][:-4] for x in glob.glob(base+'*.csv')]
px={}
for a in assets:
 d=pd.read_csv(base+a+'.csv',parse_dates=['date']); px[a]=d[d.date<=cut].set_index('date').close
px=pd.DataFrame(px).sort_index(); r=px.pct_change()
# Multi-horizon residual reversal: average cross-sectional demeaned 3d and 5d reversal,
# scaled by 20d volatility. Designed to retain breadth while reducing one-day noise.
r3=r.rolling(3,min_periods=3).sum(); r5=r.rolling(5,min_periods=5).sum()
raw=-(0.5*r3+0.5*r5)
factor=raw.sub(raw.median(axis=1),axis=0)/(r.rolling(20,min_periods=15).std()+1e-8)
factor.to_csv('scripts/miner_2_20270408_multihorizon_reversal_signal.csv')
print('assets',len(assets),'rows',len(factor),'period',factor.index.min(),factor.index.max())
for h in [1,5,10]:
 fwd=px.pct_change(h).shift(-h); vals=[];ds=[];ns=[]
 for dt in factor.index:
  z=pd.concat([factor.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ds.append(dt);ns.append(len(z))
 s=pd.Series(vals,index=ds); print('H',h,'dates',len(s),'avgN %.2f IC %.7f ICIR %.7f hit %.4f'%(np.mean(ns),s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
 if h==1:
  for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-04-07')]:
   q=s[(s.index>=lo)&(s.index<=hi)]; print('regime',lo,hi,'IC %.7f ICIR %.7f n %d'%(q.mean(),q.mean()/q.std(ddof=1),len(q)))
print('coverage %.6f turnover %.6f'%(factor.notna().sum(axis=1).mean()/len(assets),factor.rank(axis=1,pct=True).diff().abs().mean().mean()))
