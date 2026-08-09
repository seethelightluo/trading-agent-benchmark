import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index().loc[:'2026-07-15']
r5=px.pct_change(5); rows=[]
for dt in r5.index:
 x=r5.loc[dt]
 if x.notna().sum()<8: continue
 f=pd.Series({s:x.drop(labels=s).median() for s in U})
 fut=px.shift(-1).loc[dt]/px.loc[dt]-1
 z=pd.concat([f,fut],axis=1).dropna()
 if len(z)>=8: rows.append((dt,z.iloc[:,0],z.iloc[:,1],len(z)))
ics=np.array([spearmanr(a,b).statistic for _,a,b,_ in rows])
print('candidate dates',len(rows),'meanIC',ics.mean(),'ICIR',ics.mean()/ics.std(ddof=1),'hit',(ics>0).mean(),'meanN',np.mean([n for *_,n in rows]))
# pooled rank correlation versus existing expressions
for name,fun in [('rev5',lambda x:-x),('rev3',lambda x:-x),('mom20',lambda x:x)]:
 vals=[]; cand=[]
 for dt,f,y,n in rows:
  x=r5.loc[dt] if name!='rev3' else px.pct_change(3).loc[dt]
  if name=='mom20': x=px.pct_change(20).loc[dt]
  q=pd.concat([f.rename('c'),fun(x).rename('l')],axis=1).dropna()
  vals.extend(q.c.tolist());cand.extend(q.l.tolist())
 print(name,'pooled_spearman',spearmanr(vals,cand).statistic)
for h in [5,10]:
 q=[]
 for dt,f,_,n in rows:
  fut=px.shift(-h).loc[dt]/px.loc[dt]-1
  z=pd.concat([f,fut],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('decay',h,'dates',len(q),'IC',np.mean(q),'ICIR',np.mean(q)/np.std(q,ddof=1))
for a,b in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2026-07-15')]:
 q=[v for dt,_,v,_ in rows if a<=str(dt)[:10]<=b]
 # no
 print('regime',a,'dates',len(q))
