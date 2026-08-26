import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    x=get_stock_daily_data(s, days=3600)
    if x is not None and len(x):
        x=x.copy(); x['date']=pd.to_datetime(x['date']); x=x.drop_duplicates('date').set_index('date').sort_index()
        D[s]=x['close'].astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change()
# Defensive low-volatility factor: lower trailing realized volatility ranks higher.
f=-(r.rolling(20).std()*np.sqrt(20))
rows=[]
for d in f.index:
    j=p.index.get_loc(d); j2=j+10
    if j2>=len(p): continue
    z=pd.concat([f.loc[d].rename('f'),(p.iloc[j2]/p.iloc[j]-1).rename('y')],axis=1).dropna()
    if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1: rows.append((d,z.f.corr(z.y),len(z)))
a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(a),'mean_n',a.n.mean(),'coverage',a.n.sum()/(len(a)*15))
print('IC %.8f ICIR %.8f hit %.4f'%(a.ic.mean(),a.ic.mean()/a.ic.std(ddof=1),np.mean(a.ic>0)))
for label,q in [('online',a[a.index>=pd.Timestamp('2026-07-16')]),('recent252',a.tail(252)),('2029',a[a.index.year==2029]),('2028',a[a.index.year==2028]),('2027',a[a.index.year==2027])]: print(label,len(q),'IC %.8f ICIR %.8f hit %.4f'%(q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1),np.mean(q.ic>0)))
ranks=f.rank(axis=1,pct=True); turn=(ranks.diff(10).abs().sum(axis=1)/ranks.notna().sum(axis=1)).dropna(); print('turnover10',turn.mean())
# decay diagnostics
for h in [5,20,40]:
 rows=[]
 for d in f.index:
  j=p.index.get_loc(d); j2=j+h
  if j2>=len(p): continue
  z=pd.concat([f.loc[d].rename('f'),(p.iloc[j2]/p.iloc[j]-1).rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1: rows.append(z.f.corr(z.y))
 print('decay',h,'dates',len(rows),'IC',np.mean(rows),'ICIR',np.mean(rows)/np.std(rows,ddof=1))
