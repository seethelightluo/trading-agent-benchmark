import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; rows=[];D={}
for s in U:
 p='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(p):
  x=pd.read_csv(p,parse_dates=['date']).sort_values('date').set_index('date');D[s]=x
for s,x in D.items():
 r=x.close.pct_change();f=-r/r.rolling(20).std();y=r.shift(-1);rows.append(pd.DataFrame({'date':x.index,'f':f,'y':y,'s':s}).dropna())
a=pd.concat(rows,ignore_index=True); q=[]
for d,g in a.groupby('date'):
 if len(g)>=8:q.append((d,spearmanr(g.f,g.y).statistic,len(g)))
ic=pd.Series({d:v for d,v,n in q});print('dates',len(ic),'avg_n',np.mean([n for d,v,n in q]),'coverage',len(a)/sum(len(x) for x in D.values()),'IC',ic.mean(),'ICIR',ic.mean()/ic.std(),'hit',(ic>0).mean())
for lo,hi in [('2020','2025'),('2026','2029'),('2030','2033')]:
 z=ic[(ic.index>=lo)&(ic.index<=hi+'-12-31')];print('regime',lo,hi,len(z),z.mean(),z.mean()/z.std())
for h in [1,3,5,10]:
 z=[]
 for s,x in D.items():
  r=x.close.pct_change();f=-r/r.rolling(20).std();y=r.rolling(h).sum().shift(-h);z.append(pd.DataFrame({'date':x.index,'f':f,'y':y}))
 b=pd.concat(z,ignore_index=True);v=[]
 for d,g in b.groupby('date'):
  g=g.dropna()
  if len(g)>=8:v.append(spearmanr(g.f,g.y).statistic)
 print('horizon',h,'IC',np.mean(v),'n',len(v))
