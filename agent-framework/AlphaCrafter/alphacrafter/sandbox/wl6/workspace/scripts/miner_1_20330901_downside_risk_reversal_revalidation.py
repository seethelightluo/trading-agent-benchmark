import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)==0:d=get_index_daily_data(s,4000)
 if d is None:return None
 return d.assign(date=pd.to_datetime(d.date)).drop_duplicates('date').set_index('date').sort_index().close.astype(float)
p=pd.DataFrame({s:get(s) for s in U}).sort_index(); r=p.pct_change()
f=-(p.pct_change(20)).div(r.where(r<0,0).rolling(40).std().replace(0,np.nan),axis=0)
fu=p.shift(-10)/p-1
rows=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fu.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
o=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); mu=o.ic.mean(); sd=o.ic.std(ddof=1)
print('data_end',p.index.max().date(),'dates',len(o),'assets',len(U),'avg_n',round(o.n.mean(),2),'coverage',round(f.notna().sum(axis=1).mean()/len(U),4))
print('IC %.8f ICIR %.8f hit %.4f turnover %.4f'%(mu,mu/sd*np.sqrt(252),(o.ic>0).mean(),f.rank(axis=1,pct=True).diff().abs().mean().mean()))
for a,b in [('2020','2024'),('2025','2026'),('2027','2029'),('2030','2033')]:
 q=o.loc[a:b,'ic']; print('regime',a,b,len(q),round(q.mean(),8),round(q.mean()/q.std(ddof=1)*np.sqrt(252),6))
for h in [5,10,20,40]:
 x=p.shift(-h)/p-1; q=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],x.loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,len(q),round(float(np.nanmean(q)),8))
