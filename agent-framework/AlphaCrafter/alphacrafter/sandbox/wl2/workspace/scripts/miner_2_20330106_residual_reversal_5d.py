import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 p=f'../persistent/stock_data/{s}.csv'
 if os.path.exists(p):
  x=pd.read_csv(p,parse_dates=['date']).sort_values('date').set_index('date')
  D[s]=x
# candidate: medium-horizon residual reversal, volatility scaled, with 5d smoothing
rows=[]
for s,x in D.items():
 c=x.close.astype(float); r=c.pct_change()
 f=-(c.pct_change(5)/r.rolling(20).std()).rolling(3).mean()
 y=c.pct_change().shift(-1)
 z=pd.DataFrame({'f':f,'y':y,'s':s}).dropna(); rows.append(z)
a=pd.concat(rows)
ics=[]; turnovers=[]
for dt,g in a.groupby(level=0):
 if len(g)>=8:
  ic=spearmanr(g.f,g.y).statistic
  ics.append((dt,ic,len(g)))
  turnovers.append((dt,g.set_index('s').f.rank(pct=True).to_dict()))
ic=pd.Series({d:v for d,v,n in ics})
print('dates',len(ic),'avg_n',np.mean([n for d,v,n in ics]),'coverage',len(a)/sum(len(x) for x in D.values()),'IC',ic.mean(),'ICIR',ic.mean()/ic.std(ddof=1),'hit', (ic>0).mean())
# rank turnover using adjacent common names
prev=None; ts=[]
for d,rr in turnovers:
 if prev:
  names=set(rr)&set(prev); ts.append(np.mean([abs(rr[s]-prev[s]) for s in names]))
 prev=rr
print('turnover',np.mean(ts))
for h in [1,3,5,10]:
 vals=[]
 for s,x in D.items():
  c=x.close.astype(float); r=c.pct_change(); f=-(c.pct_change(5)/r.rolling(20).std()).rolling(3).mean(); y=c.pct_change(h).shift(-h)
  vals.append(pd.DataFrame({'f':f,'y':y,'date':x.index}).dropna())
 b=pd.concat(vals).reset_index(drop=True)
 q=[]
 for dt,g in b.groupby('date'):
  if len(g)>=8:q.append(spearmanr(g.f,g.y).statistic)
 print('horizon',h,'IC',np.nanmean(q),'n',len(q))
for lo,hi in [('2020','2025'),('2026','2029'),('2030','2033')]:
 q=ic[(ic.index>=lo)&(ic.index<=hi+'-12-31')]
 print('regime',lo,hi,len(q),q.mean(),q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
# artifact for reproducibility
out=[]
for s,x in D.items():
 c=x.close.astype(float); r=c.pct_change(); f=-(c.pct_change(5)/r.rolling(20).std()).rolling(3).mean()
 out.append(pd.DataFrame({'date':x.index,'symbol':s,'signal':f}))
pd.concat(out).to_csv('scripts/miner_2_20330106_residual_reversal_5d_signal.csv',index=False)
