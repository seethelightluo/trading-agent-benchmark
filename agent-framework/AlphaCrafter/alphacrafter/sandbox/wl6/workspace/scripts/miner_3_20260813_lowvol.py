import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; b='../persistent/stock_data'
P={}
for s in U:
 d=pd.read_csv(f'{b}/{s}.csv'); d.date=pd.to_datetime(d.date); P[s]=d.set_index('date').close.sort_index()
P=pd.DataFrame(P).sort_index(); R=P.pct_change(); rows=[]
for s in U:
 f=-R[s].rolling(20,min_periods=15).std(); y=R[s].shift(-1)
 rows.append(pd.DataFrame({'date':P.index,'f':f.values,'y':y.values,'asset':s}))
A=pd.concat(rows,ignore_index=True).dropna(); vals=[]
for dt,g in A.groupby('date'):
 if len(g)>=8: vals.append((dt,spearmanr(g.f,g.y).statistic))
i=pd.Series(dict(vals)); print('dates',len(i),'assets',A.groupby('date').size().loc[i.index].mean(),'IC',round(i.mean(),5),'ICIR',round(i.mean()/i.std(ddof=1),5),'hit',round((i>0).mean(),4))
for h in [5,10]:
 rows=[]
 for s in U:
  rows.append(pd.DataFrame({'date':P.index,'f':(-R[s].rolling(20,min_periods=15).std()).values,'y':(P[s].shift(-h)/P[s]-1).values,'asset':s}))
 B=pd.concat(rows,ignore_index=True).dropna(); q=[]
 for dt,g in B.groupby('date'):
  if len(g)>=8:q.append(spearmanr(g.f,g.y).statistic)
 q=pd.Series(q); print('h',h,'n',len(q),'IC',round(q.mean(),5),'ICIR',round(q.mean()/q.std(ddof=1),5))
for yr in range(2020,2027):
 q=i[i.index.year==yr]
 if len(q): print(yr,len(q),round(q.mean(),4),round(q.mean()/q.std(ddof=1),4))
print('coverage',round(len(A)/(len(P)*15),4))
