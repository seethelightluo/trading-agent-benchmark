import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)<300:d=get_index_daily_data(s,4000)
 if d is not None and len(d):D[s]=d.set_index('date').close.astype(float)
px=pd.DataFrame(D).sort_index().ffill(); r=np.log(px).diff()
# Stress-conditioned short reversal: recent losers are favored only when cross-asset breadth
# is weak, isolating rebound behavior from ordinary trend. Signal lagged one day.
stress=(-r.rolling(15,min_periods=10).sum()).rank(axis=1,pct=True).mean(axis=1)
f=(-r.rolling(3,min_periods=3).sum()).mul((stress-0.5).clip(lower=0),axis=0).shift(1)
for h in [1,3,5,10]:
 q=[]
 for i,dt in enumerate(px.index[:-h]):
  z=pd.concat([f.loc[dt],r.iloc[i+1:i+h+1].sum()],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:q.append((dt,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
 q=pd.DataFrame(q,columns=['date','n','ic']).set_index('date');print('H',h,'obs',len(q),'avgN %.2f IC %.6f ICIR %.6f hit %.4f'%(q.n.mean(),q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1),(q.ic>0).mean()))
 for a,b in [('2020','2022'),('2023','2025'),('2026','2027'),('2028','2030')]:
  y=q.loc[a:b].ic
  if len(y):print(a+'-'+b,'n',len(y),'IC %.6f ICIR %.6f'%(len(y) and y.mean(),len(y) and y.mean()/y.std(ddof=1)))
print('dates',len(px),'instruments',len(D),'coverage %.4f'%f.notna().mean().mean())
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_2_20300711_stress_rebound_signal.csv',index=False)
