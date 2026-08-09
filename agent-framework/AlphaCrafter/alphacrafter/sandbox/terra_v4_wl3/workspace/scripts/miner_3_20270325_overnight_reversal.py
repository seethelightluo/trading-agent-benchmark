import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24'); files=glob.glob('../persistent/stock_data/*.csv'); A=[os.path.basename(p)[:-4] for p in files]; C={}
for p in files:
 d=pd.read_csv(p,parse_dates=['date']).sort_values('date').set_index('date'); C[os.path.basename(p)[:-4]]=d[d.index<=cut].close
c=pd.DataFrame(C).sort_index(); r=c.pct_change(); rv=r.rolling(20,min_periods=10).std()
# Short-horizon contrarian signal, centered cross-section and risk scaled; nonlinear winsorization avoids crypto outliers.
x=r.sub(r.median(axis=1),axis=0); fac=(-x/rv).clip(-4,4); fac.to_csv('scripts/miner_3_20270325_overnight_reversal_signal.csv')
for h in [1,5,10]:
 y=c.pct_change(h).shift(-h); vals=[]; ns=[]; ds=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z));ds.append(dt)
 s=pd.Series(vals,index=ds); print('H',h,'dates',len(s),'avgN',np.mean(ns),'IC',s.mean(),'ICIR',s.mean()/s.std(ddof=1),'hit',(s>0).mean())
 if h==1:
  for lo,hi in [('2026-07-28','2026-12-31'),('2027-01-01','2027-03-24')]:
   q=s[(s.index>=lo)&(s.index<=hi)]; print('regime',lo,hi,len(q),q.mean(),q.mean()/q.std(ddof=1))
print('coverage',fac.notna().sum(axis=1).mean()/len(A),'turnover',fac.rank(axis=1,pct=True).diff().abs().mean().mean(),'assets',len(A))
