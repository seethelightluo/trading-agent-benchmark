import numpy as np, pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2026-12-17')
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@CUT').set_index('date').sort_index() for s in U}
def fac(d):
 c=d.close.astype(float); lo=c.rolling(60,min_periods=40).min(); hi=c.rolling(60,min_periods=40).max(); return ((c-lo)/(hi-lo).replace(0,np.nan)).ewm(span=5,min_periods=5).mean()
def run(h):
 rows=[]
 for s,d in D.items():
  f=fac(d); r=d.close.shift(-h)/d.close-1
  rows += [(dt,s,f.loc[dt],r.loc[dt]) for dt in d.index]
 x=pd.DataFrame(rows,columns=['date','sym','f','r']).dropna(); q=[]; ns=[]
 for dt,g in x.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.r.nunique()>1: q.append(g.f.corr(g.r,method='spearman')); ns.append(len(g))
 q=pd.Series(q).dropna(); print('H',h,'dates',len(q),'avgN',np.mean(ns),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',np.mean(q>0),'coverage',len(x)/sum(len(d) for d in D.values()))
 for a,b in [(2020,2022),(2023,2024),(2025,2026)]:
  z=x[(x.date.dt.year>=a)&(x.date.dt.year<=b)]; y=[]
  for _,g in z.groupby('date'):
   if len(g)>=8 and g.f.nunique()>1 and g.r.nunique()>1:y.append(g.f.corr(g.r,method='spearman'))
  y=pd.Series(y).dropna(); print('regime',a,b,len(y),y.mean(),y.mean()/y.std(ddof=1))
 return x
x=run(10)
fmat=pd.DataFrame({s:fac(d) for s,d in D.items()}); fmat.index.name='date'; fmat.to_csv('scripts/miner_2_20261217_recovery60_signal.csv')
# rank turnover
print('turnover',fmat.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
