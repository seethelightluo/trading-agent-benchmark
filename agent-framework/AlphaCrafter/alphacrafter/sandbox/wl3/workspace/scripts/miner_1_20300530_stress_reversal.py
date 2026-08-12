import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)<300: d=get_index_daily_data(s,4000)
 if d is not None and len(d): D[s]=d.set_index('date').close.astype(float)
px=pd.DataFrame(D).sort_index().ffill(); lr=np.log(px).diff()
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').close.astype(float).reindex(px.index).ffill()
# Continuous stress-conditioned residual reversal. Fade 5d relative losers, with stronger weight when VIX is above its 60d mean.
r5=lr.rolling(5).sum(); resid=r5.sub(r5.median(axis=1),axis=0)
vol=lr.rolling(20).std()*np.sqrt(5)+1e-9
base=(-resid/vol).clip(-3,3)
vz=(vix-vix.rolling(60).mean())/(vix.rolling(60).std()+1e-9)
stress=(1+0.75*vz.clip(lower=0,upper=2))
f=base.mul(stress,axis=0).shift(1)
f=f.sub(f.median(axis=1),axis=0)
rows_by={}
for h in [1,3,5,10]:
 rows=[]
 for i,dt in enumerate(px.index[:-h]):
  z=pd.concat([f.loc[dt],np.log(px.iloc[i+h]/px.iloc[i])],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1: rows.append((dt,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
 q=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date'); rows_by[h]=q
 ic=q.ic.mean(); ir=ic/(q.ic.std(ddof=1)+1e-12)
 print('H',h,'obs',len(q),'avgN %.2f'%q.n.mean(),'IC %.6f ICIR %.6f hit %.4f'%(ic,ir,(q.ic>0).mean()))
 for a,b in [('2020','2022'),('2023','2025'),('2026','2027'),('2028','2030')]:
  y=q.loc[a:b].ic
  if len(y): print(' ',a+'-'+b,'n',len(y),'IC %.6f ICIR %.6f'%(y.mean(),y.mean()/(y.std(ddof=1)+1e-12)))
rank=f.rank(axis=1,pct=True)
print('dates',len(px),'instruments',len(D),'coverage %.4f'%(f.notna().mean().mean()),'turnover %.4f'%(rank.diff().abs().mean(axis=1).mean()))
# Persist artifact for the selected horizon only after inspection; save all signals for audit.
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20300530_stress_reversal_signal.csv',index=False)
