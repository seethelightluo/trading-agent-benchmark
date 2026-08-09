import numpy as np, pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
def fac(d):
 c=d.close.astype(float); lo=c.rolling(60,min_periods=40).min(); hi=c.rolling(60,min_periods=40).max(); return ((c-lo)/(hi-lo).replace(0,np.nan)).ewm(span=5,min_periods=5).mean()
rows=[]
for s,d in frames.items():
 f=fac(d); r=d.close.shift(-1)/d.close-1
 for dt in d.index: rows.append((dt,s,f.loc[dt],r.loc[dt]))
x=pd.DataFrame(rows,columns=['date','sym','f','r']).dropna(); ics=[]; counts=[]
for dt,g in x.groupby('date'):
 if len(g)>=8 and g.f.nunique()>1 and g.r.nunique()>1: ics.append(g.f.corr(g.r,method='spearman')); counts.append(len(g))
ics=pd.Series(ics).dropna(); print('dates',len(ics),'avg_n',np.mean(counts),'IC',ics.mean(),'ICIR',ics.mean()/ics.std(ddof=1),'hit',np.mean(ics>0),'coverage',len(x)/sum(len(d) for d in frames.values()))
for h in [5,10]:
 rows=[]
 for s,d in frames.items():
  f=fac(d); r=d.close.shift(-h)/d.close-1
  rows += [(dt,f.loc[dt],r.loc[dt]) for dt in d.index]
 z=pd.DataFrame(rows,columns=['date','f','r']).dropna(); q=[]
 for dt,g in z.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1:q.append(g.f.corr(g.r,method='spearman'))
 q=pd.Series(q).dropna(); print(h,'dates',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1))
x['year']=x.date.dt.year
for y,g in x.groupby('year'):
 q=[a.f.corr(a.r,method='spearman') for _,a in g.groupby('date') if len(a)>=8 and a.f.nunique()>1]; q=pd.Series(q).dropna(); print(y,len(q),q.mean(),q.mean()/q.std(ddof=1))
x.pivot(index='date',columns='sym',values='f').to_csv('scripts/miner_2_20261217_recovery60_signal.csv')
