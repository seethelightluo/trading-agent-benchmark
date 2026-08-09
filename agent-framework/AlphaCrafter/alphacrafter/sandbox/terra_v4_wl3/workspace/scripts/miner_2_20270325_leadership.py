import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24')
assets=sorted([os.path.basename(x)[:-4] for x in glob.glob('../persistent/stock_data/*.csv')])
F={}; closes={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=cut].set_index('date'); c=d.close.astype(float); closes[a]=c
 F[a]=c.pct_change(20)-0.5*c.pct_change(5)
fac=pd.DataFrame(F).sort_index(); fac.to_csv('scripts/miner_2_20270325_leadership_signal.csv')
for h in [1,5,10]:
 fwd=pd.DataFrame({a:c.pct_change(h).shift(-h) for a,c in closes.items()}).reindex(fac.index)
 vals=[]; dates=[]; ns=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); dates.append(dt); ns.append(len(z))
 s=pd.Series(vals,index=pd.to_datetime(dates)); print('horizon',h,'dates',len(s),'avgN',round(np.mean(ns),3),'IC',round(float(s.mean()),6),'ICIR',round(float(s.mean()/s.std(ddof=1)),6),'hit',round(float((s>0).mean()),6))
 if h==1:
  print('coverage',round(float(fac.notna().sum(axis=1).mean()/len(assets)),6),'turnover',round(float(fac.rank(axis=1,pct=True).diff().abs().mean().mean()),6))
  for lo,hi in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2027-03-24')]:
   q=s[(s.index>=lo)&(s.index<=hi)]; print('regime',lo,hi,'dates',len(q),'IC',round(float(q.mean()),6),'ICIR',round(float(q.mean()/q.std(ddof=1)),6))
print('assets',len(assets))
