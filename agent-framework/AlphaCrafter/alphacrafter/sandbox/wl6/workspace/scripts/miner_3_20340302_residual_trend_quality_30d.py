import numpy as np, pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2034-03-01')
px={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cut').drop_duplicates('date').sort_values('date').set_index('date') for s in U}
# Residual trend: 30d return relative to cross-sectional median, scaled by 30d vol;
# quality multiplier is fraction of positive daily returns in last 30d.
rets=pd.concat({s:d.close.pct_change() for s,d in px.items()},axis=1)
rows=[]
for s,d in px.items():
 r=d.close.pct_change(); common=rets.reindex(d.index)
 rel30=d.close.pct_change(30)-common.median(axis=1).reindex(d.index)
 vol=r.rolling(30).std()*np.sqrt(30)+1e-12
 breadth=(r.gt(0).astype(float).rolling(30).mean())
 f=rel30/vol*(0.5+0.5*breadth)
 fwd=d.close.shift(-10)/d.close-1
 for dt in d.index:
  if np.isfinite(f.loc[dt]) and np.isfinite(fwd.loc[dt]): rows.append((dt,s,f.loc[dt],fwd.loc[dt]))
x=pd.DataFrame(rows,columns=['date','symbol','factor','fwd'])
def stat(z):
 q=[]
 for _,g in z.groupby('date'):
  if len(g)>=8:
   v=g.factor.corr(g.fwd,method='spearman')
   if np.isfinite(v):q.append(v)
 q=pd.Series(q); return len(q),round(z.groupby('date').size().mean(),2),round(q.mean(),6),round(q.mean()/q.std(ddof=1),6),round((q>0).mean(),4),z.symbol.nunique()
print('residual 30d momentum / vol * positive breadth quality')
print('range',x.date.min().date(),x.date.max().date(),'rows',len(x),'assets',x.symbol.nunique())
print('overall',stat(x))
for a,b in [('2020','2022'),('2023','2024'),('2025','2026'),('2027','2029'),('2030','2034')]: print('regime',a,b,stat(x[(x.date>=a)&(x.date<=b)]))
r=x.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True)
print('turnover',round(r.diff().abs().mean(axis=1).mean(),6),'coverage',round(x.symbol.nunique()/15,4),'avg_n',round(x.groupby('date').size().mean(),2))
