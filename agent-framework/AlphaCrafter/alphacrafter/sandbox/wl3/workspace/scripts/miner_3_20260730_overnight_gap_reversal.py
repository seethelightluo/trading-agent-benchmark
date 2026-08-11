import pandas as pd, numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut='2026-07-29'
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cut').set_index('date').sort_index() for s in U}
rows=[]
for s,x in D.items():
    gap=x.open/x.close.shift(1)-1; vol=x.close.pct_change().rolling(20,min_periods=10).std()
    rows.append(pd.DataFrame({'date':x.index,'f':-gap/vol.replace(0,np.nan),'r':x.close.shift(-1)/x.close-1,'s':s}))
a=pd.concat(rows,ignore_index=True); res=[]
for dt,g in a.dropna().groupby('date'):
    if len(g)>=8:
        c=g.f.corr(g.r,method='spearman')
        if pd.notna(c): res.append((dt,c,len(g)))
z=pd.DataFrame(res,columns=['date','ic','n']).set_index('date')
rnk=a.assign(rank=a.groupby('date').f.rank(pct=True)).pivot(index='date',columns='s',values='rank')
print('candidate overnight_gap_reversal; cutoff',cut,'dates',len(z),'avg_n',z.n.mean(),'coverage',len(a.dropna())/len(a),'IC',z.ic.mean(),'ICIR',z.ic.mean()/z.ic.std(ddof=1),'hit',(z.ic>0).mean(),'turnover',rnk.diff().abs().mean(axis=1).mean())
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-07-29')]:
 q=z.loc[lo:hi].ic; print('regime',lo,hi,len(q),q.mean(),q.mean()/q.std(ddof=1))
for h in [3,5,10]:
 vals=[]
 for s,x in D.items():
  gap=x.open/x.close.shift(1)-1; vol=x.close.pct_change().rolling(20,min_periods=10).std()
  vals.append(pd.DataFrame({'date':x.index,'f':-gap/vol.replace(0,np.nan),'r':x.close.shift(-h)/x.close-1}))
 b=pd.concat(vals,ignore_index=True).dropna(); vv=[]
 for dt,g in b.groupby('date'):
  if len(g)>=8:
   c=g.f.corr(g.r,method='spearman')
   if pd.notna(c): vv.append(c)
 print('decay',h,len(vv),np.mean(vv))
print('unique instruments',a.s.nunique())
