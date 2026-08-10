import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24'); files=glob.glob('../persistent/stock_data/*.csv'); names=[os.path.basename(p)[:-4] for p in files]
C={}
for p in files:
 n=os.path.basename(p)[:-4]; d=pd.read_csv(p,parse_dates=['date']).sort_values('date').set_index('date'); C[n]=d.close[d.index<=cut]
close=pd.DataFrame(C).sort_index(); r=close.pct_change(); r3=close.pct_change(3)
# Cross-asset stress rebound: reverse relative 3d performance only when broad 5d downside breadth is high.
rel=r3.sub(r3.median(axis=1),axis=0); breadth=(close.pct_change(5)<0).mean(axis=1)
stress=((breadth-0.5)/0.25).clip(0,1)
z=rel.sub(rel.mean(axis=1),axis=0).div(rel.std(axis=1).replace(0,np.nan),axis=0)
fac=(-z).mul(stress,axis=0); fac.to_csv('scripts/miner_2_20270325_stress_rebound_signal.csv')
def ev(h):
 y=close.pct_change(h).shift(-h);v=[];ds=[];ns=[]
 for dt in fac.index:
  x=pd.concat([fac.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(x)>=8 and x.iloc[:,0].nunique()>1 and x.iloc[:,1].nunique()>1: v.append(spearmanr(x.iloc[:,0],x.iloc[:,1]).statistic);ds.append(dt);ns.append(len(x))
 return pd.Series(v,index=ds),ns
print('assets',len(names),'rows',len(fac),'period',fac.index.min().date(),fac.index.max().date())
for h in [1,5,10]:
 s,n=ev(h); print('H',h,'dates',len(s),'avgN',round(np.mean(n),2),'IC %.7f ICIR %.7f hit %.4f'%(s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
 if h==1:
  for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]:
   q=s[(s.index>=lo)&(s.index<=hi)]; print('regime',lo,hi,len(q),'IC %.7f ICIR %.7f'%(q.mean(),q.mean()/q.std(ddof=1)))
print('coverage',fac.notna().sum(axis=1).mean()/len(names),'turnover',fac.rank(axis=1,pct=True).diff().abs().mean().mean(),'stress_obs',int((stress>0).sum()))
