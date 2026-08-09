import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-12-31'); D={}
for s in U:
 p='../persistent/stock_data/'+s+'.csv'; x=pd.read_csv(p,parse_dates=['date']).sort_values('date').set_index('date'); D[s]=x.close.loc[:cut]
p=pd.concat(D,axis=1).sort_index(); r=p.pct_change(); m=r.mean(axis=1)
for w in [20,40,60,120]:
 f=pd.DataFrame(index=r.index,columns=U,dtype=float)
 for i in range(w,len(r)):
  z=m.iloc[i-w:i]; vz=z.var()
  if vz>0:
   for s in U: f.loc[f.index[i],s]=-(r[s].iloc[i-w:i].cov(z)/vz)
 vals=[]; dates=[]; ns=[]
 for i in range(len(r)-1):
  q=pd.concat([f.iloc[i],r.iloc[i+1].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1:
   vals.append(spearmanr(q.iloc[:,0],q.y).statistic); dates.append(r.index[i]); ns.append(len(q))
 a=np.array(vals); di=pd.DatetimeIndex(dates)
 print('W',w,'dates',len(a),'N',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
 for lo,hi in [(2020,2022),(2023,2024),(2025,2026)]:
  z=a[(di.year>=lo)&(di.year<=hi)]; print(' reg',lo,hi,'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'n',len(z))
 print('turn',round(np.nanmean(np.abs(f.rank(axis=1,pct=True).diff()).mean(axis=1)),4))
 if w==60:
  rows=[(dt,s,f.loc[dt,s]) for dt in f.index for s in U if pd.notna(f.loc[dt,s])]
  pd.DataFrame(rows,columns=['date','symbol','signal']).to_csv('../persistent/factor_signals_miner_2_20261231_low_beta.csv',index=False)
