import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date').set_index('date')
 D[s]=x
# Candidate: shock-reversal signal. Large true-range expansion relative to trailing range,
# signed by the day's return; fading the shock should predict next-day return.
rows=[]
for s,x in D.items():
 r=x.close.pct_change(); tr=(x.high-x.low)/x.close.shift(1)
 shock=tr/(tr.rolling(20,min_periods=15).median()+1e-12)
 f=-np.sign(r)*shock
 for i,dt in enumerate(x.index[:-1]):
  if pd.notna(f.iloc[i]): rows.append((dt,s,float(f.iloc[i]),float(x.close.iloc[i+1]/x.close.iloc[i]-1)))
a=pd.DataFrame(rows,columns=['date','s','f','y'])
for h in [1,5,10]:
 rows=[]
 for s,x in D.items():
  r=x.close.pct_change(); tr=(x.high-x.low)/x.close.shift(1); f=-np.sign(r)*tr/(tr.rolling(20,min_periods=15).median()+1e-12)
  for i,dt in enumerate(x.index):
   if pd.notna(f.iloc[i]) and i+h<len(x): rows.append((dt,s,float(f.iloc[i]),float(x.close.iloc[i+h]/x.close.iloc[i]-1)))
 z=pd.DataFrame(rows,columns=['date','s','f','y']); ic=[]; ns=[]
 for dt,g in z.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1:
   ic.append(spearmanr(g.f,g.y).statistic); ns.append(len(g))
 q=np.array(ic); print('h',h,'dates',len(q),'avgN',np.mean(ns),'IC',np.mean(q),'ICIR',np.mean(q)/(np.std(q,ddof=1)+1e-12),'hit',np.mean(q>0))
# rank turnover and coverage
v=a.sort_values(['date','s']); ranks=v.groupby('date').f.rank(pct=True); print('coverage',a.s.nunique()/15,'turnover',ranks.groupby(a.date).mean().diff().abs().mean())
for yr,g in a.groupby(a.date.dt.year):
 q=[]
 for dt,k in g.groupby('date'):
  if len(k)>=8 and k.f.nunique()>1:q.append(spearmanr(k.f,k.y).statistic)
 print('regime',yr,np.mean(q),len(q))
