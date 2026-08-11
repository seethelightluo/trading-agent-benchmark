import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-07-15'); D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cut').sort_values('date').set_index('date')
 D[s]=x
# Shock exhaustion: unusually large completed daily true range, signed by close-to-close direction.
# Negative sign says large up shocks mean weaker next return and large down shocks mean stronger next return.
rec=[]
for s,x in D.items():
 c=x.close; r=c.pct_change(); tr=(x.high-x.low)/c
 baseline=tr.rolling(20,min_periods=15).median()
 f=-(r)*(tr/(baseline+1e-12))
 for i,dt in enumerate(x.index):
  if pd.notna(f.iloc[i]) and i+1<len(x): rec.append((dt,s,float(f.iloc[i]),float(c.iloc[i+1]/c.iloc[i]-1)))
a=pd.DataFrame(rec,columns=['date','s','f','y'])
for h in [1,5,10]:
 # rebuild h-th observation
 rec=[]
 for s,x in D.items():
  c=x.close;r=c.pct_change();tr=(x.high-x.low)/c;base=tr.rolling(20,min_periods=15).median();f=-(r)*(tr/(base+1e-12))
  for i,dt in enumerate(x.index):
   if pd.notna(f.iloc[i]) and i+h<len(x):rec.append((dt,s,float(f.iloc[i]),float(c.iloc[i+h]/c.iloc[i]-1)))
 z=pd.DataFrame(rec,columns=['date','s','f','y']);ics=[];ns=[];rk=[]
 for dt,g in z.groupby('date'):
  g=g.dropna()
  if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1:
   ics.append(spearmanr(g.f,g.y).statistic);ns.append(len(g));rk.append(g.f.rank(pct=True))
 q=np.asarray(ics); turn=np.mean([np.abs(rk[i]-rk[i-1]).mean() for i in range(1,len(rk))])
 print('horizon',h,'dates',len(q),'avgN',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round(np.mean(q>0),4),'turn',round(turn,4))
if len(a):
 vals=[]
 for dt,g in a.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1: vals.append((dt,spearmanr(g.f,g.y).statistic))
 q=pd.Series(dict(vals));print('regime', {int(y):round(q[q.index.year==y].mean(),5) for y in sorted(q.index.year.unique())})
print('assets',len(D),'raw dates',a.date.nunique())
