import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-07-15')
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().close.loc[:cut] for s in U}
idx=sorted(set().union(*[set(v.index) for v in P.values()])); p=pd.DataFrame({s:v.reindex(idx) for s,v in P.items()}); r=p.ffill().pct_change()
# Cross-sectional relative-strength persistence: trailing asset return relative to same-day universe median,
# multiplied by sign-consistency of relative daily returns. This removes common market direction.
rel=r.sub(r.median(axis=1),axis=0); f=pd.DataFrame(index=idx,columns=U,dtype=float)
for s in U:
 f[s]=rel[s].rolling(20,min_periods=15).sum() * (rel[s].gt(0).rolling(20,min_periods=15).mean()-rel[s].lt(0).rolling(20,min_periods=15).mean())
for h in [5,10,20]:
 I=[];N=[];D=[]
 for i in range(len(p)-h):
  q=pd.concat([f.iloc[i].rename('f'),(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:I.append(spearmanr(q.f,q.y).statistic);N.append(len(q));D.append(p.index[i])
 a=np.array(I); print('h',h,'dates',len(a),'avgN',round(np.mean(N),2),'coverage',round(np.mean(N)/15,4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
 if h==10: print('annual10',{y:round(a[[d.year==y for d in D]].mean(),6) for y in sorted(set(d.year for d in D))})
z=[];prev=None
for _,row in f.iterrows():
 q=row.dropna()
 if len(q)>=8:
  rr=q.rank(pct=True)
  if prev is not None:
   ix=rr.index.intersection(prev.index);z.append(np.abs(rr[ix]-prev[ix]).mean())
  prev=rr
print('turnover',round(np.mean(z),6),'factor_dates',int(f.notna().sum(axis=1).ge(8).sum()),'instruments',len(U))
print('signal_corr_mom',f.stack().corr(r.rolling(20).sum().stack()))
