import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24'); files=glob.glob('../persistent/stock_data/*.csv'); assets=[os.path.basename(x)[:-4] for x in files]
C={}; H={}
for p in files:
 a=os.path.basename(p)[:-4]; d=pd.read_csv(p,parse_dates=['date']).sort_values('date').set_index('date'); d=d[d.index<=cut]; C[a]=d.close; H[a]=d.high
close=pd.DataFrame(C).sort_index(); high=pd.DataFrame(H).reindex(close.index)
# Trend persistence: signed 20d return, but penalize assets whose recent path is excessively volatile.
r=close.pct_change(); r20=close.pct_change(20); rv=r.rolling(20,min_periods=15).std();
# Use cross-sectional relative trend and risk adjustment, with a mild 5d confirmation.
rel=r20.sub(r20.median(axis=1),axis=0); confirm=r.rolling(5,min_periods=4).mean(); crel=confirm.sub(confirm.median(axis=1),axis=0)
fac=(rel/rv)*(1+0.5*np.sign(rel)*crel.abs()).replace([np.inf,-np.inf],np.nan)
fac.to_csv('scripts/miner_3_20270325_trend_persistence_signal.csv')
def evalh(h):
 y=close.pct_change(h).shift(-h); vals=[]; ns=[]; ds=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z));ds.append(dt)
 s=pd.Series(vals,index=ds); return s,ns
print('assets',len(assets),'rows',len(fac),'period',fac.index.min(),fac.index.max())
for h in [1,5,10]:
 s,ns=evalh(h); print('H',h,'dates',len(s),'avgN',round(np.mean(ns),2),'IC %.7f ICIR %.7f hit %.4f'%(s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
 if h==1:
  for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]:
   q=s[(s.index>=lo)&(s.index<=hi)]; print('regime',lo,'to',hi,'dates',len(q),'IC %.7f ICIR %.7f'%(q.mean(),q.mean()/q.std(ddof=1)))
print('coverage',fac.notna().sum(axis=1).mean()/len(assets),'turnover',fac.rank(axis=1,pct=True).diff().abs().mean().mean())
