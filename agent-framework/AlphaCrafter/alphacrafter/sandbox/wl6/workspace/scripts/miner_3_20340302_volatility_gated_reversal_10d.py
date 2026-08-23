import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2034-03-01')
px={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cut').drop_duplicates('date').sort_values('date').set_index('date') for s in U}
rets=pd.concat({s:d.close.pct_change() for s,d in px.items()},axis=1)
rows=[]
for s,d in px.items():
 r=d.close.pct_change(); rv=r.rolling(20).std(); cs=rets.rolling(20).std().median(axis=1).reindex(d.index)
 # contrarian 10d return, strengthened when own vol is below cross-sectional median
 gate=(1+(cs-rv).clip(-cs,cs)/(cs+1e-12))
 f=-d.close.pct_change(10)*gate
 y=d.close.shift(-10)/d.close-1
 for dt in d.index:
  if np.isfinite(f.loc[dt]) and np.isfinite(y.loc[dt]): rows.append((dt,s,f.loc[dt],y.loc[dt]))
x=pd.DataFrame(rows,columns=['date','symbol','factor','fwd'])
q=[]
for _,g in x.groupby('date'):
 if len(g)>=8:
  v=g.factor.corr(g.fwd,method='spearman')
  if np.isfinite(v):q.append(v)
q=pd.Series(q)
def st(z):
 a=[]
 for _,g in z.groupby('date'):
  if len(g)>=8:
   v=g.factor.corr(g.fwd,method='spearman')
   if np.isfinite(v):a.append(v)
 a=pd.Series(a);return len(a),round(a.mean(),6),round(a.mean()/a.std(ddof=1),6),round((a>0).mean(),4)
print('volatility-gated 10d reversal; dates',x.date.min().date(),x.date.max().date(),'rows',len(x),'assets',x.symbol.nunique(),'avg_n',round(x.groupby('date').size().mean(),2))
print('overall',st(x))
for a,b in [('2020','2022'),('2023','2024'),('2025','2026'),('2027','2029'),('2030','2034')]:print('regime',a,b,st(x[(x.date>=a)&(x.date<=b)]))
r=x.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True);print('coverage',round(x.symbol.nunique()/15,4),'turnover',round(r.diff().abs().mean(axis=1).mean(),6))
