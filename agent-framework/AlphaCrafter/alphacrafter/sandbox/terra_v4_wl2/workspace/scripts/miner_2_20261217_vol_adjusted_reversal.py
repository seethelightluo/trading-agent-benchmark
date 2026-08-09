import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-12-16')
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().close.loc[:cut] for s in U}
p=pd.concat(D,axis=1).sort_index(); r=p.pct_change()
for h,vw in [(3,20),(5,20),(5,60),(10,60)]:
 f=-p.pct_change(h)/r.rolling(vw).std()
 vals=[];ns=[];ds=[]
 for i in range(len(r)-1):
  q=pd.concat([f.iloc[i],r.iloc[i+1].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.y.nunique()>1:
   vals.append(spearmanr(q.iloc[:,0],q.y).statistic);ns.append(len(q));ds.append(r.index[i])
 a=np.array(vals); rr=f.rank(axis=1,pct=True).loc[ds]
 print(h,vw,len(a),round(np.mean(ns),2),round(a.mean(),6),round(a.mean()/a.std(ddof=1),6),round(np.mean(a>0),4),round(np.nanmean(np.abs(rr.diff()).mean(axis=1)),4),str(ds[0].date()),str(ds[-1].date()))
 print('regimes',[(y,len(a[[d.year==y for d in ds]]),round(a[[d.year==y for d in ds]].mean(),5)) for y in range(2020,2027) if any(d.year==y for d in ds)])
