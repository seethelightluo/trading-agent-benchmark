import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 for f in (get_stock_daily_data,get_index_daily_data):
  try:
   x=f(symbol=s,days=5000)
   if x is not None and len(x)>300:return x[['date','close']]
  except:pass
 return None
p={s:get(s) for s in U};p={s:x for s,x in p.items() if x is not None}
c=pd.concat([x.set_index('date').close.rename(s) for s,x in p.items()],axis=1).sort_index().ffill(); lr=np.log(c).diff()
# Shock-recovery: recent selloff is more likely to mean-revert when short vol is unusually elevated.
vol5=lr.rolling(5).std(); vol60=lr.rolling(60).std(); shock=(vol5/vol60).clip(0.5,3)
r5=c/c.shift(5)-1
sig=(-r5*shock).replace([np.inf,-np.inf],np.nan)
rows=[]
for h in [5,10,20,40,60]:
 f=c.shift(-h)/c-1; a=[]
 for d in sig.index:
  z=pd.concat([sig.loc[d],f.loc[d]],axis=1).dropna()
  if len(z)>=8:a.append((d,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
 q=pd.DataFrame(a,columns=['d','ic','n']).set_index('d'); x=q.ic
 rows.append((h,len(x),q.n.mean(),x.mean(),x.mean()/x.std(ddof=1)*np.sqrt(252), (x>0).mean()))
print('assets',len(p),'dates',len(c),'range',c.index.min(),c.index.max())
print('h dates n IC ICIR hit');[print('%d %d %.2f %+.6f %+.6f %.4f'%r) for r in rows]
h=20;f=c.shift(-h)/c-1;a=[]
for d in sig.index:
 z=pd.concat([sig.loc[d],f.loc[d]],axis=1).dropna()
 if len(z)>=8:a.append((d,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
q=pd.DataFrame(a,columns=['d','ic','n']).set_index('d')
for nm,a,b in [('2024-26','2024-01-01','2026-12-31'),('2027-29','2027-01-01','2029-12-31'),('2030','2030-01-01','2030-12-31'),('2031','2031-01-01','2031-10-15')]:
 x=q.loc[a:b].ic;print('regime',nm,len(x),x.mean(),x.mean()/x.std(ddof=1)*np.sqrt(252) if x.std()>0 else np.nan)
print('coverage',sig.notna().mean().mean(),'turnover',sig.rank(pct=True).diff().abs().mean(axis=1).dropna().mean())
sig.to_csv('scripts/miner_3_20311016_volshock_reversal_signal.csv',index_label='date');print('artifact scripts/miner_3_20311016_volshock_reversal_signal.csv')
