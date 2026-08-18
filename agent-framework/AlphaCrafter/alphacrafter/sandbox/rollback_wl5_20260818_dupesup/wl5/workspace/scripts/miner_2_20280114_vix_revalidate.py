import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv'); d.date=pd.to_datetime(d.date); P[s]=d.sort_values('date').set_index('date').close.astype(float)
P=pd.DataFrame(P).sort_index(); v=pd.read_csv('../persistent/index_data/VIX.csv'); v.date=pd.to_datetime(v.date); v=v.sort_values('date').set_index('date').close.astype(float).reindex(P.index).ffill(); R=P.pct_change(); vu=v.diff()>0
f=pd.DataFrame(index=P.index,columns=P.columns,dtype=float)
for dt in P.index:
 ix=P.index.get_loc(dt); lo=max(0,ix-60); rr=R.iloc[lo:ix]; q=vu.iloc[lo:ix];
 if q.sum()>=5: f.loc[dt]=-(rr[q].mean()-rr.mean())
fwd=P.shift(-10)/P-1
for name,mask in [('all',slice('2020','2027')),('recent',slice('2026','2028')),('latest',slice('2027','2028'))]:
 rows=[]
 for dt in f.loc[mask].index:
  z=pd.concat([f.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8: rows.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 a=np.array(rows); print(name,'dates',len(a),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',(a>0).mean())
