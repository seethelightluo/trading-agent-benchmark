import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=get_stock_daily_data(s,days=4500)
 if d is None or len(d)<200: d=get_index_daily_data(s,days=4500)
 if d is not None: D[s]=d.set_index('date')['close'].astype(float)
C=pd.concat(D,axis=1).sort_index().ffill(); R=C.pct_change()
# Path-efficiency trend: net 20d move divided by total absolute daily movement.
# This rewards persistent directional movement and penalizes noisy/choppy trends.
net=C/C.shift(20)-1
path=R.abs().rolling(20,min_periods=15).sum()
vol=R.rolling(20,min_periods=15).std()
f=(net/path.replace(0,np.nan))/vol.replace(0,np.nan)
f=f.sub(f.mean(axis=1),axis=0)
f=f.replace([np.inf,-np.inf],np.nan)

def eval_h(h):
 rows=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],(C.shift(-h)/C-1).loc[dt]],axis=1).dropna()
  if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
 q=pd.Series({d:v for d,v,n in rows})
 return q, np.mean([n for d,v,n in rows])
for h in [1,3,5,10,20]:
 q,n=eval_h(h); print('h',h,'dates',len(q),'avg_n',round(n,3),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
 if h==10:
  for name,sel in [('2020-25',q.index<'2026-01-01'),('2026-29',(q.index>='2026-01-01')&(q.index<'2030-01-01')),('2030+',q.index>='2030-01-01'),('last365',q.index>=q.index.max()-pd.Timedelta(days=365))]:
   x=q[sel]; print(name,len(x),round(x.mean(),6),round(x.mean()/x.std(ddof=1),6),round((x>0).mean(),4))
print('date_coverage',len(eval_h(10)[0])/len(f),'asset_coverage',round(C.notna().mean().mean(),5),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),5))
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20330930_path_efficiency_trend_signal.csv',index=False)
