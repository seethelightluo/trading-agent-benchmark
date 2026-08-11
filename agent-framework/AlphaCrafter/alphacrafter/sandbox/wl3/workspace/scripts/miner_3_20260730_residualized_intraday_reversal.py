import pandas as pd, numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut='2026-07-29'
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date <= @cut').set_index('date').sort_index() for s in U}
rows=[]
for s,x in D.items():
 rng=(x.high-x.low).replace(0,np.nan)
 # Intraday body reversal, cross-sectionally residualized against close location.
 body=-(x.close-x.open)/x.open
 clv=2*(x.close-x.low)/rng-1
 tmp=pd.DataFrame({'date':x.index,'body':body,'clv':clv,'r':x.close.shift(-1)/x.close-1,'symbol':s})
 rows.append(tmp.reset_index(drop=True))
a=pd.concat(rows,ignore_index=True)
def resid(g):
 q=g.dropna(subset=['body','clv'])
 if len(q)<8 or q.clv.std()==0: return pd.Series(np.nan,index=g.index)
 return pd.Series(q.body-q.clv.cov(q.body)/q.clv.var()*q.clv, index=q.index)
a['f']=a.groupby('date',group_keys=False).apply(resid).reindex(a.index)
a=a.dropna(subset=['f','r'])
out=[]
for dt,g in a.groupby('date'):
 if len(g)>=8:
  c=g.f.corr(g.r,method='spearman')
  if pd.notna(c): out.append((dt,c,len(g)))
z=pd.DataFrame(out,columns=['date','ic','n']).set_index('date')
rank=a.assign(rank=a.groupby('date').f.rank(pct=True)).pivot(index='date',columns='symbol',values='rank').sort_index()
print('factor residualized_intraday_reversal; cutoff',cut,'dates',len(z),'avg_n',z.n.mean(),'coverage',len(a)/sum(len(x) for x in D.values()),'IC',z.ic.mean(),'ICIR',z.ic.mean()/z.ic.std(ddof=1),'hit',(z.ic>0).mean(),'turnover',rank.diff().abs().mean(axis=1).mean())
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-07-29')]:
 q=z.loc[lo:hi].ic; print('regime',lo,hi,len(q),q.mean(),q.mean()/q.std(ddof=1))
for h in [1,5,10]:
 vals=[]
 for s,x in D.items():
  rng=(x.high-x.low).replace(0,np.nan); body=-(x.close-x.open)/x.open; clv=2*(x.close-x.low)/rng-1
  t=pd.DataFrame({'date':x.index,'body':body,'clv':clv,'r':x.close.shift(-h)/x.close-1})
  vals.append(t)
 b=pd.concat(vals,ignore_index=True).dropna()
 # residual by date using same cross-sectional regression
 b['f']=b.groupby('date',group_keys=False).apply(resid).reindex(b.index)
 vv=[]
 for dt,g in b.dropna().groupby('date'):
  if len(g)>=8:
   c=g.f.corr(g.r,method='spearman')
   if pd.notna(c): vv.append(c)
 print('decay',h,len(vv),np.mean(vv))
print('pooled correlations',a[['f','clv']].corr().iloc[0,1])
