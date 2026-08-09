import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24'); files=glob.glob('../persistent/stock_data/*.csv'); assets=[os.path.basename(x)[:-4] for x in files]
C={}
for p in files:
 d=pd.read_csv(p,parse_dates=['date']).sort_values('date').set_index('date'); C[os.path.basename(p)[:-4]]=d.close.where(d.close>0)
close=pd.DataFrame(C).sort_index(); close=close.loc[close.index<=cut]
r=close.pct_change(); rv=r.rolling(20,min_periods=15).std()
# Multi-horizon contrarian signal: blend 1d and 5d returns, scaled by trailing risk.
# Cross-sectional centering avoids common market direction and preserves relative ranking.
r1=r.sub(r.median(axis=1),axis=0); r5=close.pct_change(5).sub(close.pct_change(5).median(axis=1),axis=0)
fac=(-(0.5*r1+0.5*r5)/rv).replace([np.inf,-np.inf],np.nan)
fac.to_csv('scripts/miner_3_20270325_blended_reversal_signal.csv')
def ev(h):
 y=close.pct_change(h).shift(-h); vals=[]; ds=[]; ns=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ds.append(dt);ns.append(len(z))
 s=pd.Series(vals,index=ds); return s,ns
print('assets',len(assets),'rows',len(fac),'period',fac.index.min(),fac.index.max())
for h in [1,5,10]:
 s,ns=ev(h); print('H',h,'dates',len(s),'avgN',round(np.mean(ns),2),'IC %.7f ICIR %.7f hit %.4f'%(s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
 if h==1:
  for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]:
   q=s[(s.index>=lo)&(s.index<=hi)];print('regime',lo,hi,'dates',len(q),'IC %.7f ICIR %.7f'%(q.mean(),q.mean()/q.std(ddof=1)))
print('coverage',fac.notna().sum(axis=1).mean()/len(assets),'turnover',fac.rank(axis=1,pct=True).diff().abs().mean().mean())
