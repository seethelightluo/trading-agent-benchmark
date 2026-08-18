import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv'); d['date']=pd.to_datetime(d['date']); return d.sort_values('date').set_index('date')['close'].astype(float)
p={s:load(s) for s in U}; x=pd.concat(p,axis=1).sort_index(); ret=x.pct_change()
# residual momentum: 20d asset return residual to equal-weight benchmark, standardized by idiosyncratic vol
bench=ret[U].mean(axis=1); out=[];  dates=x.index
for i in range(60,len(dates)-1):
 d=dates[i]; nxt=dates[i+1]; vals={}
 for s in U:
  rr=ret[s].loc[:d].tail(60); bb=bench.loc[rr.index]
  z=pd.concat([rr,bb],axis=1).dropna()
  if len(z)<40: continue
  beta=np.cov(z.iloc[:,0],z.iloc[:,1],ddof=1)[0,1]/(np.var(z.iloc[:,1],ddof=1)+1e-12)
  r20= x[s].loc[d]/x[s].shift(20).loc[d]-1 if pd.notna(x[s].shift(20).loc[d]) else np.nan
  br20=bench.loc[:d].tail(20).add(1).prod()-1
  resid=r20-beta*br20
  idio=z.iloc[:,0]-beta*z.iloc[:,1]
  vals[s]=resid/(idio.std()*np.sqrt(252)+1e-8)
 f=pd.Series(vals); fr=ret.loc[nxt]
 z=pd.concat([f,fr],axis=1).dropna()
 if len(z)>=8:
  ic=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  out.append((d,ic,len(z)))
a=np.array([v[1] for v in out]); print('dates',len(out),'mean_n',np.mean([v[2] for v in out]),'IC',np.nanmean(a),'ICIR',np.nanmean(a)/(np.nanstd(a,ddof=1)+1e-12),'hit',np.mean(a>0),'coverage',np.mean([v[2] for v in out])/15)
for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026')]:
 q=np.array([v[1] for v in out if lo<=str(v[0].year)<=hi]); print(lo,hi,len(q),np.mean(q) if len(q) else np.nan,np.mean(q)/(np.std(q,ddof=1)+1e-12) if len(q)>1 else np.nan)
for h in [5,10]:
 oo=[]
 for i in range(60,len(dates)-h):
  d=dates[i]; end=dates[i+h]; vals={}
  for s in U:
   rr=ret[s].loc[:d].tail(60); bb=bench.loc[rr.index]; z=pd.concat([rr,bb],axis=1).dropna()
   if len(z)<40:continue
   beta=np.cov(z.iloc[:,0],z.iloc[:,1],ddof=1)[0,1]/(np.var(z.iloc[:,1],ddof=1)+1e-12); r20=x[s].loc[d]/x[s].shift(20).loc[d]-1; br20=bench.loc[:d].tail(20).add(1).prod()-1; vals[s]=(r20-beta*br20)/(z.iloc[:,0].sub(beta*z.iloc[:,1]).std()*np.sqrt(252)+1e-8)
  z=pd.concat([pd.Series(vals),x.pct_change(h).loc[end]],axis=1).dropna()
  if len(z)>=8:oo.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('h',h,'IC',np.mean(oo),'ICIR',np.mean(oo)/(np.std(oo,ddof=1)+1e-12),'N',len(oo))
