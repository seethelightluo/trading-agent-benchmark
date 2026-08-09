import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24'); root='../persistent/stock_data'
assets=[os.path.basename(x)[:-4] for x in glob.glob(root+'/*.csv')]
C={}
for a in assets:
 d=pd.read_csv(f'{root}/{a}.csv',parse_dates=['date']).sort_values('date').set_index('date'); d=d[d.index<=cut]; C[a]=d.close
close=pd.DataFrame(C).sort_index(); r=close.pct_change()
# Orthogonal cross-asset residual reversal: fade each asset's 5d return relative to
# the contemporaneous cross-sectional median, scaled by its 20d volatility.
r5=close.pct_change(5); med=r5.median(axis=1); residual=r5.sub(med,axis=0)
rv=r.rolling(20,min_periods=12).std(); fac=(-residual/(rv*np.sqrt(5))).replace([np.inf,-np.inf],np.nan).clip(-10,10)
fac.to_csv('scripts/miner_3_20270325_residual_reversal_signal.csv')
print('assets',len(assets),'rows',len(fac),'period',fac.index.min(),fac.index.max())
for h in [1,5,10]:
 y=close.pct_change(h).shift(-h); vals=[]; ns=[]; ds=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z)); ds.append(dt)
 s=pd.Series(vals,index=ds); print('H',h,'dates',len(s),'avgN',round(np.mean(ns),2),'IC %.7f ICIR %.7f hit %.4f'%(s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
 if h==1:
  for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]:
   q=s[(s.index>=lo)&(s.index<=hi)]; print('regime',lo,hi,'dates',len(q),'IC %.7f ICIR %.7f'%(q.mean(),q.mean()/q.std(ddof=1)))
print('coverage',fac.notna().sum(axis=1).mean()/len(assets),'turnover',fac.rank(axis=1,pct=True).diff().abs().mean().mean())
