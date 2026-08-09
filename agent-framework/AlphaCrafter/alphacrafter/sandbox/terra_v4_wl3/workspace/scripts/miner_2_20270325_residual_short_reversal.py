import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24'); assets=[os.path.basename(x)[:-4] for x in glob.glob('../persistent/stock_data/*.csv')]; F={}; fw={h:{} for h in [1,5,10]}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=cut].set_index('date'); r=d.close.pct_change(); vol=r.rolling(20,min_periods=15).std()
 rev=-r.rolling(3,min_periods=3).sum()/(vol+1e-8); mom=r.rolling(20,min_periods=15).sum()
 # Cross-sectional residual: short reversal stripped of same-date 20d momentum exposure.
 z=pd.DataFrame({'rev':rev,'mom':mom}); out=[]
 # temporarily collect then residualize cross-section below
 F[a]=z
 for h in fw: fw[h][a]=d.close.pct_change(h).shift(-h)
raw=pd.concat({a:F[a]['rev'] for a in assets},axis=1); mm=pd.concat({a:F[a]['mom'] for a in assets},axis=1)
res=pd.DataFrame(index=raw.index,columns=raw.columns,dtype=float)
for dt in raw.index:
 q=pd.concat([raw.loc[dt],mm.loc[dt]],axis=1).dropna()
 if len(q)>=8:
  x=q.iloc[:,1].values; y=q.iloc[:,0].values; b=np.polyfit(x,y,1) if np.ptp(x)>0 else [0,y.mean()]
  res.loc[dt,q.index]=q.iloc[:,0]-np.polyval(b,x)
res.to_csv('scripts/miner_2_20270325_residual_short_reversal_signal.csv')
print('assets',len(assets),'rows',len(res),'period',res.index.min(),res.index.max())
for h in fw:
 fwd=pd.DataFrame(fw[h]).reindex(res.index); vals=[];ds=[];ns=[]
 for dt in res.index:
  q=pd.concat([res.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1: vals.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);ds.append(dt);ns.append(len(q))
 s=pd.Series(vals,index=ds); print('H',h,'dates',len(s),'avgN',round(np.mean(ns),2),'IC %.6f ICIR %.6f hit %.4f cov %.4f'%(s.mean(),s.mean()/s.std(ddof=1),(s>0).mean(),res.notna().sum(axis=1).mean()/len(assets)))
 if h==1:
  for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]:
   q=s[(s.index>=lo)&(s.index<=hi)];print('regime',lo,hi,'IC %.6f n %d'%(q.mean(),len(q)))
print('turnover',res.rank(axis=1,pct=True).diff().abs().mean().mean())
