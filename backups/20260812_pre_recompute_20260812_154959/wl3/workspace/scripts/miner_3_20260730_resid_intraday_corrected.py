import pandas as pd, numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut='2026-07-15'
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date <= @cut').set_index('date').sort_index() for s in U}
rows=[]
for s,x in D.items():
 rng=(x.high-x.low).replace(0,np.nan); body=-(x.close-x.open)/x.open; clv=2*(x.close-x.low)/rng-1
 rows.append(pd.DataFrame({'date':x.index,'body':body,'clv':clv,'symbol':s,'r1':x.close.shift(-1)/x.close-1,'r5':x.close.shift(-5)/x.close-1}))
a=pd.concat(rows,ignore_index=True)
def calc(g):
 q=g.dropna(subset=['body','clv']); out=pd.Series(np.nan,index=g.index)
 if len(q)>=8 and q.clv.var()>0: out.loc[q.index]=q.body-q.clv.cov(q.body)/q.clv.var()*q.clv
 return out
a['f']=a.groupby('date',group_keys=False).apply(calc).reindex(a.index)
def evaluate(col):
 z=[]
 for dt,g in a.dropna(subset=['f',col]).groupby('date'):
  if len(g)>=8:
   c=g.f.corr(g[col],method='spearman')
   if pd.notna(c): z.append(c)
 z=np.array(z); return len(z),z.mean(),z.mean()/z.std(ddof=1),(z>0).mean()
print('dates total',a.date.nunique(),'instruments',len(U))
for col in ['r1','r5']: print(col,evaluate(col))
rank=a.pivot(index='date',columns='symbol',values='f').rank(axis=1,pct=True)
print('coverage',a.f.notna().mean(),'turnover',rank.diff().abs().mean(axis=1).mean())
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-07-15')]:
 q=a.dropna(subset=['f','r1']).groupby('date').apply(lambda g:g.f.corr(g.r1,method='spearman') if len(g)>=8 else np.nan).dropna().loc[lo:hi]; print('regime',lo,len(q),q.mean(),q.mean()/q.std(ddof=1))
for h in [1,5,10]:
 col='r1' if h==1 else 'r5'; n,ic,ir,hit=evaluate(col); print('decay',h,n,ic,ir)
print('provenance','body_residualized_on_clv; daily cross-sectional Spearman; cutoff',cut)
