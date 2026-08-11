import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date') for s in U}
rows=[]
for s,d in D.items():
 d=d.loc[:'2027-07-28']; p=d.shift(1)
 fac=-(p.close/p.open-1)/(((p.high-p.low)/p.close)+.002)
 r=d.close.shift(-1)/d.close-1
 rows.append(pd.DataFrame({'f':fac,'r':r,'s':s},index=d.index))
X=pd.concat(rows).sort_index(); ics=[]; ns=[]
for dt,g in X.dropna().groupby(level=0):
 if len(g)>=8: ics.append(spearmanr(g.f,g.r).statistic); ns.append(len(g))
ics=np.array(ics)
print('dates',len(ics),'avgN',np.mean(ns),'IC %.6f ICIR %.6f hit %.4f coverage %.4f'%(np.mean(ics),np.mean(ics)/np.std(ics,ddof=1),np.mean(ics>0),len(X.dropna())/(len(ics)*15)))
for a,b in [('2020','2021'),('2022','2023'),('2024','2025'),('2026','2027')]:
 q=X.loc[a:b]; z=[]
 for _,g in q.dropna().groupby(level=0):
  if len(g)>=8:z.append(spearmanr(g.f,g.r).statistic)
 z=np.array(z); print(a,b,len(z),np.mean(z),np.mean(z)/np.std(z,ddof=1))
for h in [1,3,5,10]:
 vals=[]
 for s,d in D.items():
  d=d.loc[:'2027-07-28']; p=d.shift(1)
  fac=-(p.close/p.open-1)/(((p.high-p.low)/p.close)+.002)
  r=d.close.shift(-h)/d.close-1; vals.append(pd.DataFrame({'f':fac,'r':r},index=d.index))
 q=pd.concat(vals).sort_index(); z=[]
 for _,g in q.dropna().groupby(level=0):
  if len(g)>=8:z.append(spearmanr(g.f,g.r).statistic)
 z=np.array(z); print('h',h,'IC',np.mean(z),'ICIR',np.mean(z)/np.std(z,ddof=1))
