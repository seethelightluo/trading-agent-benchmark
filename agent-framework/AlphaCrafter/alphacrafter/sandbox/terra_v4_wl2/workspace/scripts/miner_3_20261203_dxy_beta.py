import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-12-03')
D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date').set_index('date')
 D[s]=x.close.loc[:cut]
p=pd.concat(D,axis=1).sort_index(); r=p.pct_change()
dxy=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).set_index('date').sort_index().close.loc[:cut]
dr=dxy.pct_change().reindex(r.index)
for w in [20,40,60,120]:
 f=pd.DataFrame(index=r.index,columns=U,dtype=float)
 for i in range(w,len(r)):
  x=r.iloc[i-w:i]; z=dr.iloc[i-w:i]; vz=z.var()
  if vz>0: f.iloc[i]=-(x.apply(lambda q:q.cov(z)/vz))
 vals=[]; dates=[]; ns=[]
 for i in range(len(r)-1):
  q=pd.concat([f.iloc[i],r.iloc[i+1].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1: vals.append(spearmanr(q.iloc[:,0],q.y).statistic);dates.append(r.index[i]);ns.append(len(q))
 a=np.array(vals); print('W',w,'dates',len(a),'N',round(np.mean(ns),2),'cov',round(np.mean(ns)/15,3),'IC',round(a.mean(),5),'ICIR',round(a.mean()/a.std(ddof=1),5),'hit',round(np.mean(a>0),4))
 for lo,hi in [(2020,2022),(2023,2024),(2025,2026)]:
  z=a[(pd.DatetimeIndex(dates).year>=lo)&(pd.DatetimeIndex(dates).year<=hi)];print(' reg',lo,hi,'ICIR',round(z.mean()/z.std(ddof=1),4),'n',len(z))
 ranks=f.rank(axis=1,pct=True);print('turn',round(np.nanmean(np.abs(ranks.diff()).mean(axis=1)),4))
 if w==60:
  rows=[]
  for dt in f.index:
   for s in U:
    if pd.notna(f.loc[dt,s]):rows.append((dt,s,f.loc[dt,s]))
  pd.DataFrame(rows,columns=['date','symbol','signal']).to_csv('../persistent/factor_signals_miner_3_20261203_dxy_beta.csv',index=False)
