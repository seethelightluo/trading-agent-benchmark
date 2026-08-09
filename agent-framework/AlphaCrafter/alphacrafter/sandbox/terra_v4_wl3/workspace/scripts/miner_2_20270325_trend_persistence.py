import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24'); assets=[os.path.basename(x)[:-4] for x in glob.glob('../persistent/stock_data/*.csv')]; px={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date'); px[a]=d[d.date<=cut].set_index('date').close
idx=sorted(set().union(*[set(x.index) for x in px.values()])); idx=pd.DatetimeIndex(idx)
F={}; returns={h:{} for h in [1,5,10]}
for a,p in px.items():
 r=p.pct_change(); F[a]=(p.pct_change(20)*r.gt(0).rolling(20,min_periods=15).mean()).clip(-10,10)
 for h in returns: returns[h][a]=p.pct_change(h).shift(-h)
fac=pd.DataFrame(F).reindex(idx).sort_index(); fac.to_csv('scripts/miner_2_20270325_trend_persistence_signal.csv')
for h in [1,5,10]:
 fwd=pd.DataFrame(returns[h]).reindex(fac.index); vals=[];dates=[];ns=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);dates.append(dt);ns.append(len(z))
 s=pd.Series(vals,index=pd.DatetimeIndex(dates)); print('horizon',h,'dates',len(s),'avgN',round(np.mean(ns),2),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6),'hit',round((s>0).mean(),4))
 if h==1:
  print('coverage',round(fac.notna().sum(axis=1).mean()/len(assets),4),'turnover',round(fac.rank(axis=1,pct=True).diff().abs().mean().mean(),4))
  for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]:
   q=s[(s.index>=lo)&(s.index<=hi)]; print('regime',lo,'dates',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6))
