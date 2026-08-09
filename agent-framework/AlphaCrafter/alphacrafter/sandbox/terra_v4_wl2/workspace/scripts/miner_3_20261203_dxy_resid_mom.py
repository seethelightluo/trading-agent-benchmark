import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];cut=pd.Timestamp('2026-12-03')
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().close.loc[:cut] for s in U}
p=pd.concat(D,axis=1).sort_index();r=p.pct_change(); d=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).set_index('date').sort_index().close.loc[:cut].pct_change().reindex(r.index)
for w in [20,60,120]:
 f=pd.DataFrame(index=r.index,columns=U,dtype=float)
 for i in range(w,len(r)):
  x=r.iloc[i-w:i]; z=d.iloc[i-w:i]; vz=z.var()
  if vz>0:
   b=x.apply(lambda q:q.cov(z)/vz)
   f.iloc[i]=x.sum()-b*z.sum()
 vals=[];dates=[];ns=[]
 for i in range(len(r)-1):
  q=pd.concat([f.iloc[i],r.iloc[i+1].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1: vals.append(spearmanr(q.iloc[:,0],q.y).statistic);dates.append(r.index[i]);ns.append(len(q))
 a=np.array(vals);print(w,len(a),round(np.mean(ns),2),round(a.mean(),5),round(a.mean()/a.std(ddof=1),5),round(np.mean(a>0),4),round(np.nanmean(np.abs(f.rank(pct=True).diff()).mean(axis=1)),4))
 if w==60:
  rows=[(dt,s,f.loc[dt,s]) for dt in f.index for s in U if pd.notna(f.loc[dt,s])]
  pd.DataFrame(rows,columns=['date','symbol','signal']).to_csv('../persistent/factor_signals_miner_3_20261203_dxy_resid_mom.csv',index=False)
