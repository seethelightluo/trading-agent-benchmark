import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=get_stock_daily_data(s,4000)
 if x is None: x=get_index_daily_data(s,4000)
 if x is not None:
  x=x.copy(); x.date=pd.to_datetime(x.date); D[s]=x.set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change();
# Relative short-term acceleration, scaled by own recent risk; all inputs lagged one session.
m20=p.pct_change(20); m60=p.pct_change(60); v20=r.rolling(20).std()*np.sqrt(252)
raw=(m20-m60).sub((m20-m60).median(axis=1),axis=0)
sig=(raw/(v20+1e-8)).shift(1)
rows=[]
for d in sig.index:
 z=pd.concat([sig.loc[d],p.shift(-10).loc[d]/p.loc[d]-1],axis=1).dropna()
 if len(z)>=8: rows.append((d,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('candidate relative risk-scaled acceleration 20/60')
print('dates',len(q),'avgN',round(q.n.mean(),2),'coverage',round(q.n.mean()/15,4))
print('IC10',round(q.ic.mean(),6),'ICIRdaily',round(q.ic.mean()/q.ic.std(),6),'hit',round((q.ic>0).mean(),4))
print('turnover',round(sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
for h in [5,10,20,40]:
 vals=[]
 for d in sig.index:
  z=pd.concat([sig.loc[d],p.shift(-h).loc[d]/p.loc[d]-1],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1]))
 print('decay',h,round(np.nanmean(vals),6), 'n',len(vals))
for a,b in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032'),('2033','2035')]:
 z=q.loc[a:b,'ic']; print('regime',a,b,'n',len(z),'ic',round(z.mean(),6),'icir',round(z.mean()/z.std(),6))
sig.to_csv('scripts/miner_2_20350119_rel_risk_accel_signal.csv',index_label='date');q.to_csv('scripts/miner_2_20350119_rel_risk_accel_ic.csv')
