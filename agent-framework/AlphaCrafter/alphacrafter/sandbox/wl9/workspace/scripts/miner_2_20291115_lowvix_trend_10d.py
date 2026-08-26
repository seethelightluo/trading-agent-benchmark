import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=get_stock_daily_data(s,days=3600)
 if x is not None and len(x):
  x=x.copy();x.date=pd.to_datetime(x.date);D[s]=x.drop_duplicates('date').set_index('date').sort_index().close.astype(float)
p=pd.DataFrame(D).sort_index().ffill();r=p.pct_change()
v=pd.read_csv('../persistent/index_data/VIX.csv');v.date=pd.to_datetime(v.date);v=v.drop_duplicates('date').set_index('date').close.astype(float).reindex(p.index).ffill()
q=v.rolling(252,min_periods=60).quantile(.25); gate=(v<q).astype(float)
f=(p.pct_change(40)/(r.rolling(20).std()*np.sqrt(20)+1e-8))*gate.values[:,None]
rows=[]
for d in f.index:
 j=p.index.get_loc(d);k=j+10
 if k>=len(p):continue
 z=pd.concat([f.loc[d].rename('f'),(p.iloc[k]/p.iloc[j]-1).rename('y')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1:rows.append((d,z.f.corr(z.y),len(z)))
a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(a),'mean_n',a.n.mean(),'coverage',a.n.sum()/(len(a)*15));print('IC %.8f ICIR %.8f hit %.4f'%(a.ic.mean(),a.ic.mean()/a.ic.std(ddof=1),np.mean(a.ic>0)))
for label,x in [('online',a[a.index>=pd.Timestamp('2026-07-16')]),('recent252',a.tail(252)),('2029',a[a.index.year==2029]),('2028',a[a.index.year==2028])]:
 print(label,len(x),'IC %.8f ICIR %.8f hit %.4f'%(x.ic.mean(),x.ic.mean()/x.ic.std(ddof=1),np.mean(x.ic>0)))
rr=f.rank(axis=1,pct=True);print('turnover10',((rr.diff(10).abs().sum(axis=1)/rr.notna().sum(axis=1)).dropna()).mean())
