import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2026-07-15')
def load(s):
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv');d.date=pd.to_datetime(d.date);return d[d.date<=END].sort_values('date').drop_duplicates('date').set_index('date')
D={s:load(s) for s in U}; sig={}; ret={}
for s,d in D.items():
 r=d.close.pct_change(); ret[s]=r
 tr=(d.high-d.low)/d.close.shift(1); sig[s]=-(r*tr/tr.rolling(20,min_periods=10).median())
rows=[]
for s in U:
 for dt in sig[s].index:
  f=sig[s].get(dt); y=ret[s].shift(-1).get(dt)
  if np.isfinite(f) and np.isfinite(y): rows.append((dt,s,f,y))
q=pd.DataFrame(rows,columns=['date','s','f','y']);ics=[]
for dt,g in q.groupby('date'):
 if len(g)>=8: ics.append((dt,spearmanr(g.f,g.y).statistic))
a=np.array([x[1] for x in ics]);print('candidate L20 dates',len(a),'avgN',q.groupby('date').size().mean(),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',(a>0).mean(),'coverage',len(a)/len(set(q.date)))
for lo,hi in [(2020,2022),(2023,2024),(2025,2026)]:
 x=np.array([v for d,v in ics if lo<=d.year<=hi]);print('regime',lo,hi,'n',len(x),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1))
# decay
for h in [1,5,10,20]:
 rr=[]
 for s in U:
  y=D[s].close.pct_change(h).shift(-h); z=pd.concat([sig[s],y],axis=1).dropna();
  for dt,v in z.iterrows(): rr.append((dt,s,v.iloc[0],v.iloc[1]))
 z=pd.DataFrame(rr,columns=['date','s','f','y']); x=[]
 for dt,g in z.groupby('date'):
  if len(g)>=8:x.append(spearmanr(g.f,g.y).statistic)
 x=np.array(x);print('decay',h,'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1))
# pooled rank correlations with simple production analogues
allc=[];clv=[];rev=[];mom=[]
for s,d in D.items():
 r=d.close.pct_change(); tr=(d.high-d.low)/d.close.shift(1)
 for dt in sig[s].index:
  vals=[sig[s].get(dt), -(2*(d.close-d.low)/(d.high-d.low)-1).get(dt), -r.rolling(5).sum().get(dt), (r.rolling(20).sum()/r.rolling(20).std()).get(dt)]
  if all(np.isfinite(vals)):allc.append(vals)
c=np.array(allc);print('pooled Spearman corr candidate vs CLV/rev/mom',*[spearmanr(c[:,0],c[:,i]).statistic for i in [1,2,3]])
