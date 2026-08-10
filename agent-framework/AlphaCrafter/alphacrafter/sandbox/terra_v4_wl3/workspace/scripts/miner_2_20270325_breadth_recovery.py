import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24')
files=glob.glob('../persistent/stock_data/*.csv'); C={}
for p in files:
 a=os.path.basename(p)[:-4]; d=pd.read_csv(p,parse_dates=['date']).sort_values('date').set_index('date'); C[a]=d.close[d.index<=cut]
close=pd.DataFrame(C).sort_index();
# Breadth-recovery continuation: relative 5d strength, activated as market breadth improves
r5=close.pct_change(5); rel=r5.sub(r5.median(axis=1),axis=0)
b5=(r5<0).mean(axis=1); b20=(close.pct_change(20)<0).mean(axis=1)
recovery=((b20-b5)/0.25).clip(0,1)
z=rel.sub(rel.mean(axis=1),axis=0).div(rel.std(axis=1).replace(0,np.nan),axis=0)
fac=z.mul(recovery,axis=0); fac.to_csv('scripts/miner_2_20270325_breadth_recovery_signal.csv')
def ev(h):
 y=close.pct_change(h).shift(-h); vals=[]; ds=[]; ns=[]
 for dt in fac.index:
  x=pd.concat([fac.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(x)>=8 and x.iloc[:,0].nunique()>1 and x.iloc[:,1].nunique()>1:
   vals.append(spearmanr(x.iloc[:,0],x.iloc[:,1]).statistic); ds.append(dt); ns.append(len(x))
 return pd.Series(vals,index=ds),ns
print('assets',len(C),'rows',len(fac))
for h in [1,5,10]:
 s,n=ev(h); print('H',h,'dates',len(s),'avgN',round(np.mean(n),2) if n else 0,'IC',round(s.mean(),7) if len(s) else np.nan,'ICIR',round(s.mean()/s.std(ddof=1),7) if len(s)>1 else np.nan,'hit',round((s>0).mean(),4) if len(s) else np.nan)
 if h==1:
  for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]:
   q=s[(s.index>=lo)&(s.index<=hi)]; print('regime',lo,len(q),round(q.mean(),7) if len(q) else np.nan)
print('coverage',fac.notna().sum(axis=1).mean()/len(C),'turnover',fac.rank(axis=1,pct=True).diff().abs().mean().mean(),'active',int((recovery>0).sum()))
