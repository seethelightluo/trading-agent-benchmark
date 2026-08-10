import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24')
files=glob.glob('../persistent/stock_data/*.csv')
C={}
for p in files:
 a=os.path.basename(p)[:-4]; d=pd.read_csv(p,parse_dates=['date']).sort_values('date').set_index('date'); C[a]=d.close[d.index<=cut]
close=pd.DataFrame(C).sort_index(); r=close.pct_change()
# Breadth-conditioned short-horizon rebound with a stricter, persistent stress regime.
r3=close.pct_change(3); rel=r3.sub(r3.median(axis=1),axis=0)
breadth=(close.pct_change(5)<0).mean(axis=1)
# Require >=60% downside breadth and scale activation to 90%.
stress=((breadth-0.60)/0.30).clip(0,1)
z=rel.sub(rel.mean(axis=1),axis=0).div(rel.std(axis=1).replace(0,np.nan),axis=0)
fac=(-z).mul(stress,axis=0)
fac.to_csv('scripts/miner_3_20270325_strict_stress_signal.csv')
def ev(h):
 y=close.pct_change(h).shift(-h); vals=[]; ns=[]; ds=[]
 for dt in fac.index:
  x=pd.concat([fac.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(x)>=8 and x.iloc[:,0].nunique()>1 and x.iloc[:,1].nunique()>1:
   vals.append(spearmanr(x.iloc[:,0],x.iloc[:,1]).statistic); ns.append(len(x)); ds.append(dt)
 return pd.Series(vals,index=ds),ns
print('assets',len(C),'rows',len(fac),'period',fac.index.min().date(),fac.index.max().date())
for h in [1,5,10]:
 s,n=ev(h); print('H',h,'dates',len(s),'avgN',round(np.mean(n),2),'IC %.7f ICIR %.7f hit %.4f'%(s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
 if h==1:
  for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]:
   q=s[(s.index>=lo)&(s.index<=hi)]; print('regime',lo,hi,len(q),'IC %.7f'%(q.mean() if len(q) else np.nan))
print('coverage',fac.notna().sum(axis=1).mean()/len(C),'turnover',fac.rank(axis=1,pct=True).diff().abs().mean().mean())
print('active dates',int((stress>0).sum()),'of',len(stress))
