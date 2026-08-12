import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in U:
 x=get_stock_daily_data(s,days=3000)
 if x is None or len(x)<100:x=get_index_daily_data(s,days=3000)
 if x is not None:D[s]=x.set_index('date')['close'].astype(float)
p=pd.DataFrame(D).sort_index().ffill();r=p.pct_change();b=r.mean(axis=1); breadth=(r<0).mean(axis=1); bv=b.rolling(20,min_periods=15).std()
stress=((breadth.rolling(5,min_periods=5).mean()-0.50).clip(lower=0)/0.50)*((bv/bv.rolling(120,min_periods=60).median()).clip(lower=0.8)-0.8).clip(lower=0)
m3=r.rolling(3,min_periods=3).sum(); win=60; eb=b.rolling(win,min_periods=30).mean(); er=r.rolling(win,min_periods=30).mean(); erb=r.mul(b,axis=0).rolling(win,min_periods=30).mean(); beta=erb.sub(er.mul(eb,axis=0)).div(b.rolling(win,min_periods=30).var(),axis=0)
resid=m3.sub(beta.mul(b.rolling(3,min_periods=3).sum(),axis=0),axis=0);vol=r.rolling(20,min_periods=15).std();f=(-resid/vol).mul(stress,axis=0)
rows=[]
for i in range(len(p)-1):
 z=pd.concat([f.iloc[i].rename('f'),r.iloc[i+1].rename('y')],axis=1).dropna()
 if len(z)>=8 and z.f.nunique()>1:rows.append((p.index[i],len(z),z.f.corr(z.y)))
a=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date');x=a.ic
print('dates',len(x),'avgN',a.n.mean(),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'hit',(x>0).mean(),'coverage',f.notna().mean().mean(),'active',(stress>0).mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for nm,mask in [('2020-22',a.index<'2023-01-01'),('2023-25',(a.index>='2023-01-01')&(a.index<'2026-01-01')),('2026-30',a.index>='2026-01-01')]:
 q=a.loc[mask].ic;print(nm,len(q),q.mean(),q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
for h in [3,5,10]:
 y=p.pct_change(h).shift(-h+1);q=[]
 for i in range(len(p)-h):
  z=pd.concat([f.iloc[i],y.iloc[i]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:q.append(z.iloc[:,0].corr(z.iloc[:,1]))
 print('h',h,'IC',np.nanmean(q),'n',len(q))
f.to_csv('scripts/miner_1_20300808_stress_recovery_signal.csv')
