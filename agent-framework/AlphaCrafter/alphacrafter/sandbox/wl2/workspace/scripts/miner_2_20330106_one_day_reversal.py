import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in U:
 p='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(p): D[s]=pd.read_csv(p,parse_dates=['date']).sort_values('date').set_index('date')
# Candidate: lagged one-day reversal normalized by 20d volatility, smoothed 3 observations
R={s:x.close.pct_change() for s,x in D.items()}; rows=[]
for s,x in D.items():
 f=(-(R[s]/R[s].rolling(20).std())).rolling(3).mean(); y=R[s].shift(-1)
 rows.append(pd.DataFrame({'date':x.index,'f':f.values,'y':y.values,'s':s}).dropna())
a=pd.concat(rows,ignore_index=True); q=[]; ranks=[]
for d,g in a.groupby('date'):
 if len(g)>=8:
  q.append((d,spearmanr(g.f,g.y).statistic,len(g))); ranks.append((d,g.set_index('s').f.rank(pct=True).to_dict()))
ic=pd.Series({d:v for d,v,n in q}); print('dates',len(ic),'avg_n',np.mean([n for d,v,n in q]),'coverage',len(a)/sum(len(x) for x in D.values()),'IC',ic.mean(),'ICIR',ic.mean()/ic.std(),'hit',(ic>0).mean())
prev=None;ts=[]
for d,z in ranks:
 if prev: ts.append(np.mean([abs(z[s]-prev[s]) for s in set(z)&set(prev)]))
 prev=z
print('turnover',np.mean(ts))
for lo,hi in [('2020','2025'),('2026','2029'),('2030','2033')]:
 z=ic[(ic.index>=lo)&(ic.index<=hi+'-12-31')];print('regime',lo,hi,len(z),z.mean(),z.mean()/z.std())
for h in [1,3,5,10]:
 z=[]
 for s,x in D.items():
  f=(-(R[s]/R[s].rolling(20).std())).rolling(3).mean();y=R[s].rolling(h).sum().shift(-h);z.append(pd.DataFrame({'date':x.index,'f':f,'y':y}))
 b=pd.concat(z,ignore_index=True);v=[]
 for d,g in b.groupby('date'):
  g=g.dropna()
  if len(g)>=8:v.append(spearmanr(g.f,g.y).statistic)
 print('horizon',h,'IC',np.mean(v),'n',len(v))
out=[]
for s,x in D.items():
 f=(-(R[s]/R[s].rolling(20).std())).rolling(3).mean();out.append(pd.DataFrame({'date':x.index,'symbol':s,'signal':f}))
pd.concat(out).to_csv('scripts/miner_2_20330106_one_day_reversal_signal.csv',index=False)
