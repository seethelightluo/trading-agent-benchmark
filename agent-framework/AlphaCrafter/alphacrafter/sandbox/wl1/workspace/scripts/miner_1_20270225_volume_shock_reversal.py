import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
idx=sorted(set().union(*[set(x.index) for x in P.values()]))
idx=[d for d in idx if d<=pd.Timestamp('2027-02-25')]
c=pd.DataFrame({s:x.close.reindex(idx) for s,x in P.items()}).ffill(); v=pd.DataFrame({s:x.volume.reindex(idx) for s,x in P.items()}).replace(0,np.nan).ffill()
r=c.pct_change(); shock=np.log(v.rolling(5).mean()/v.rolling(60).mean()); f=-r.rolling(5).sum()*shock
for h in [5,10,20]:
 I=[];N=[];D=[]
 for i in range(len(c)-h):
  q=pd.concat([f.iloc[i].rename('f'),(c.iloc[i+h]/c.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:I.append(spearmanr(q.f,q.y).statistic);N.append(len(q));D.append(idx[i])
 a=np.array(I);print('h',h,'dates',len(a),'N',round(np.mean(N),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
 if h==10:print('annual',{y:round(a[[d.year==y for d in D]].mean(),6) for y in sorted(set(d.year for d in D))})
print('coverage',round(np.mean(f.notna().sum(axis=1))/15,4),'turnover',round(np.mean([np.abs(x.rank(pct=True)-y.rank(pct=True)).mean() for x,y in zip(f.dropna().iloc[:-1].values,f.dropna().iloc[1:].values)]),6))
