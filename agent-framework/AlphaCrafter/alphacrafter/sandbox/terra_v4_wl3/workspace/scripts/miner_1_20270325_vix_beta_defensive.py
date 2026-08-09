import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24'); assets=[os.path.basename(x)[:-4] for x in glob.glob('../persistent/stock_data/*.csv')]
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).sort_values('date').set_index('date'); v=v[v.index<=cut]; vr=v.close.pct_change()
# Defensive beta: assets with low beta to VIX shocks should outperform on next day when volatility regime is elevated.
vz=(v.close-v.close.rolling(60,min_periods=40).mean())/(v.close.rolling(60,min_periods=40).std()+1e-12)
allf={}; allfw={h:{} for h in [1,5,10]}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=cut].set_index('date'); r=d.close.pct_change()
 cov=r.rolling(40,min_periods=25).cov(vr); vv=vr.rolling(40,min_periods=25).var(); beta=cov/(vv+1e-12)
 # activate only elevated VIX; continuous score remains comparable cross-section
 allf[a]=(-beta*vz).where(vz>0)
 for h in allfw: allfw[h][a]=d.close.pct_change(h).shift(-h)
fac=pd.DataFrame(allf).sort_index(); fac.to_csv('scripts/miner_1_20270325_vix_beta_defensive_signal.csv')
print('assets',len(assets),'rows',len(fac),'period',fac.index.min(),fac.index.max())
for h in allfw:
 fwd=pd.DataFrame(allfw[h]).reindex(fac.index); vals=[];ds=[];ns=[]
 for dt in fac.index:
  q=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1: vals.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);ds.append(dt);ns.append(len(q))
 s=pd.Series(vals,index=ds); print('H',h,'dates',len(s),'avgN',round(np.mean(ns),2),'IC %.6f ICIR %.6f hit %.4f cov %.4f'%(s.mean(),s.mean()/s.std(ddof=1),(s>0).mean(),fac.notna().sum(axis=1).mean()/len(assets)))
 if h==1:
  for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]:
   q=s[(s.index>=lo)&(s.index<=hi)];print('regime',lo,hi,'IC %.6f n %d'%(q.mean(),len(q)))
print('turnover',fac.rank(axis=1,pct=True).diff().abs().mean().mean())
