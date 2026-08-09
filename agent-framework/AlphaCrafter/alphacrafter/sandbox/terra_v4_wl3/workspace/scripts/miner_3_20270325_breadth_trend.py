import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24'); assets=[os.path.basename(x)[:-4] for x in glob.glob('../persistent/stock_data/*.csv')]
R={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date').set_index('date')
 d=d[d.index<=cut]; R[a]=d.close.pct_change(20)
r=pd.DataFrame(R).sort_index()
# Cross-asset trend agreement: medium-term asset momentum weighted by contemporaneous breadth,
# with breadth lagged one day to avoid using same-day close in the signal.
breadth=(r>0).mean(axis=1).shift(1)
fac=r.mul((breadth-0.5).clip(-0.5,0.5),axis=0)
fac.to_csv('scripts/miner_3_20270325_breadth_trend_signal.csv')
print('assets',len(assets),'rows',len(fac),'period',fac.index.min(),fac.index.max(),'coverage',fac.notna().sum(axis=1).mean()/len(assets))
for h in [1,5,10]:
 fwd=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close.pct_change(h).shift(-h) for a in assets}).reindex(fac.index)
 vals=[]; ds=[]; ns=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ds.append(dt);ns.append(len(z))
 s=pd.Series(vals,index=ds); print('H',h,'dates',len(s),'avgN',round(np.mean(ns),2),'IC %.6f ICIR %.6f hit %.4f'%(s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
 if h==1:
  for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]:
   q=s[(s.index>=lo)&(s.index<=hi)];print('regime',lo,hi,'IC %.6f ICIR %.6f n %d'%(q.mean(),q.mean()/q.std(ddof=1),len(q)))
print('turnover',fac.rank(axis=1,pct=True).diff().abs().mean().mean())
