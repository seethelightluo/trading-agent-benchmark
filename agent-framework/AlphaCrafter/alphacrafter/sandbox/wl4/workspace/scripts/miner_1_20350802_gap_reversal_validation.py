import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
CUT=pd.Timestamp('2035-08-01'); U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 p=f'../persistent/stock_data/{s}.csv'
 if not os.path.exists(p): p=f'../persistent/index_data/{s}.csv'
 if os.path.exists(p):
  x=pd.read_csv(p); x.columns=[c.lower() for c in x.columns]; x['date']=pd.to_datetime(x['date']); D[s]=x[x.date<=CUT].sort_values('date').drop_duplicates('date').set_index('date')
rows=[]
for s,x in D.items():
 q=x.copy(); q['factor']=-q['open']/q['close'].shift(1)+1; q['fwd']=q['close'].shift(-10)/q['close']-1
 for dt,r in q[['factor','fwd']].dropna().iterrows(): rows.append((dt,s,r.factor,r.fwd))
z=pd.DataFrame(rows,columns=['date','sym','factor','fwd']); ics=[]; ns=[]
for _,g in z.groupby('date'):
 if len(g)>=8 and g.factor.nunique()>1 and g.fwd.nunique()>1: ics.append(spearmanr(g.factor,g.fwd).statistic); ns.append(len(g))
a=np.array(ics); print({'dates':len(a),'avg_instruments':round(np.mean(ns),2),'coverage':round(len(z)/sum(len(x) for x in D.values()),4),'ic':round(a.mean(),6),'icir':round(a.mean()/a.std(ddof=1)*np.sqrt(len(a)),6),'hit':round(np.mean(a>0),4),'period':f'{z.date.min().date()} to {z.date.max().date()}'})
for h in [1,5,10,20]:
 rr=[]
 for s,x in D.items():
  q=x.copy(); q['f']=-q.open/q.close.shift(1)+1; q['r']=q.close.shift(-h)/q.close-1
  for dt,r in q[['f','r']].dropna().iterrows(): rr.append((dt,r.f,r.r))
 q=pd.DataFrame(rr,columns=['date','f','r']); ii=[spearmanr(g.f,g.r).statistic for _,g in q.groupby('date') if len(g)>=8 and g.f.nunique()>1 and g.r.nunique()>1]
 print('h',h,'ic',round(np.mean(ii),6),'n',len(ii))
