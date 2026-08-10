import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24')
files=glob.glob('../persistent/stock_data/*.csv'); names=[os.path.basename(p)[:-4] for p in files]
C={}
for p in files:
 n=os.path.basename(p)[:-4]; d=pd.read_csv(p,parse_dates=['date']).sort_values('date').set_index('date'); C[n]=d.close[d.index<=cut]
close=pd.DataFrame(C).sort_index(); r=close.pct_change()
# Fade 3-session cross-sectional relative moves only while observation-only VIX level is elevated.
# The 60-session rolling median is lagged one session, avoiding use of today's macro close.
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).sort_values('date').set_index('date').close
v=v.reindex(close.index).ffill(); threshold=v.rolling(60,min_periods=30).median().shift(1)
stress=(v.shift(1)>threshold).astype(float)
base=r.rolling(3).sum().shift(0); rel=base.sub(base.median(axis=1),axis=0)
fac=-rel.mul(stress,axis=0)
fac.to_csv('scripts/miner_1_20270325_vix_level_reversal_signal.csv')
def ev(h):
 y=close.pct_change(h).shift(-h); vals=[]; ds=[]; ns=[]
 for dt in fac.index:
  x=pd.concat([fac.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(x)>=8 and x.iloc[:,0].nunique()>1 and x.iloc[:,1].nunique()>1:
   vals.append(spearmanr(x.iloc[:,0],x.iloc[:,1]).statistic);ds.append(dt);ns.append(len(x))
 s=pd.Series(vals,index=ds); return s,ns
print('assets',len(names),'rows',len(fac),'active',int(stress.sum()))
for h in [1,5,10]:
 s,n=ev(h); print('H',h,'dates',len(s),'avgN',round(np.mean(n),2),'IC',round(s.mean(),7),'ICIR',round(s.mean()/s.std(ddof=1),7),'hit',round((s>0).mean(),4))
 if h==1:
  for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]:
   q=s[(s.index>=lo)&(s.index<=hi)]; print('regime',lo,len(q),round(q.mean(),7),round(q.mean()/q.std(ddof=1),7) if len(q)>1 else np.nan)
print('coverage',fac.notna().sum(axis=1).mean()/len(names),'turnover',fac.rank(axis=1,pct=True).diff().abs().mean().mean())
