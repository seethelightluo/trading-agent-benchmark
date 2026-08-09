import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24')
assets=[os.path.basename(x)[:-4] for x in glob.glob('../persistent/stock_data/*.csv')]
F={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date')
 d=d[d.date<=cut].set_index('date')
 rng=(d.high-d.low).replace(0,np.nan)
 pressure=((d.close-d.open)/rng).clip(-1,1)
 # Three-session exponentially weighted fade; completed sessions only.
 F[a]=-pressure.ewm(span=3,adjust=False,min_periods=3).mean()
fac=pd.DataFrame(F).sort_index(); fac.to_csv('scripts/miner_3_20270325_ewm_pressure_signal.csv')
fwd={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=cut].set_index('date')
 fwd[a]=d.close.pct_change().shift(-1)
fwd=pd.DataFrame(fwd).reindex(fac.index)
vals=[]; ds=[]; ns=[]
for dt in fac.index:
 z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
  vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ds.append(dt); ns.append(len(z))
s=pd.Series(vals,index=ds)
print('assets',len(assets),'dates',len(s),'avgN',round(np.mean(ns),2),'IC %.6f ICIR %.6f hit %.4f coverage %.4f turnover %.6f'%(s.mean(),s.mean()/s.std(ddof=1),(s>0).mean(),fac.notna().sum(axis=1).mean()/len(assets),fac.rank(axis=1,pct=True).diff().abs().mean().mean()))
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]:
 q=s[(s.index>=lo)&(s.index<=hi)]; print('regime',lo,hi,'IC %.6f dates %d'%(q.mean(),len(q)))
for h in [5,10]:
 ff={}
 for a in assets:
  d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=cut].set_index('date'); ff[a]=d.close.pct_change(h).shift(-h)
 fw=pd.DataFrame(ff).reindex(fac.index); vv=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: vv.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 ss=pd.Series(vv); print('horizon',h,'IC %.6f ICIR %.6f dates %d'%(ss.mean(),ss.mean()/ss.std(ddof=1),len(ss)))
