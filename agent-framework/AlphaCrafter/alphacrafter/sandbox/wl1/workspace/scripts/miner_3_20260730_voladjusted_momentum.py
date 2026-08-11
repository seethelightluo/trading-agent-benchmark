import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-07-15')
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close.loc[:cut] for s in U}; p=pd.concat(D,axis=1).sort_index()
r=pd.DataFrame({s:p[s].pct_change(fill_method=None) for s in U})
for w in [10,20,40]:
 f=pd.DataFrame({s:p[s].pct_change(w)/(r[s].rolling(w,min_periods=w).std()*np.sqrt(w)) for s in U})
 for h in [5,10]:
  rows=[]
  for dt in p.index:
   if dt not in f.index: continue
   q=pd.concat([f.loc[dt].rename('f'),(p.shift(-h).loc[dt]/p.loc[dt]-1).rename('y')],axis=1).dropna()
   if len(q)>=8 and q.f.nunique()>1: rows.append((dt,spearmanr(q.f,q.y).statistic,len(q)))
  a=pd.DataFrame(rows,columns=['date','ic','n']); x=a.ic.to_numpy(); sd=x.std(ddof=1) if len(x)>1 else np.nan
  print('w',w,'h',h,'dates',len(x),'avgN',round(a.n.mean(),2),'coverage',round(a.n.mean()/15,4),'IC',round(x.mean(),6),'ICIR',round(x.mean()/sd,6),'hit',round((x>0).mean(),4))
  print('regime', {int(y):round(a.loc[a.date.dt.year==y,'ic'].mean(),5) for y in sorted(a.date.dt.year.unique())})
