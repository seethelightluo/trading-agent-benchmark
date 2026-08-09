import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24'); assets=[os.path.basename(x)[:-4] for x in glob.glob('../persistent/stock_data/*.csv')]
for kind in ['volscaled20','volscaled60','risktrend']:
 F={}; fw={1:{}}
 for a in assets:
  d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=cut].set_index('date'); r=d.close.pct_change()
  if kind=='volscaled20': x=d.close.pct_change(20)/r.rolling(20,min_periods=15).std()
  elif kind=='volscaled60': x=d.close.pct_change(60)/r.rolling(60,min_periods=40).std()
  else: x=(d.close.pct_change(20)+0.5*d.close.pct_change(60))/(r.rolling(20,min_periods=15).std())
  F[a]=x.clip(-10,10); fw[1][a]=d.close.pct_change(1).shift(-1)
 fac=pd.DataFrame(F).sort_index(); fac.to_csv('scripts/miner_3_20270325_'+kind+'_signal.csv'); fwd=pd.DataFrame(fw[1]).reindex(fac.index)
 vals=[];ds=[];ns=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ds.append(dt);ns.append(len(z))
 s=pd.Series(vals,index=ds); print(kind,'dates',len(s),'avgN',np.mean(ns),'IC',s.mean(),'ICIR',s.mean()/s.std(ddof=1),'hit',(s>0).mean(),'cov',fac.notna().sum(axis=1).mean()/15,'turn',fac.rank(axis=1,pct=True).diff().abs().mean().mean())
 for p in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]: print(p,s[(s.index>=p[0])&(s.index<=p[1])].mean())
