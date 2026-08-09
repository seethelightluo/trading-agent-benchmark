import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   x=fn(s,days=2300)
   if x is not None and len(x)>100:return x[['date','close']]
  except: pass
x={s:fetch(s) for s in U};x={s:v for s,v in x.items() if v is not None}
p=pd.concat([v.assign(symbol=s) for s,v in x.items()]).pivot(index='date',columns='symbol',values='close').sort_index().ffill(); r=p.pct_change()
# Breadth-conditioned reversal: contrarian 3d return, scaled by prior 20d breadth distance from neutral.
b=((r.rolling(20).sum()>0).mean(axis=1)).shift(1)
f=(-p.pct_change(3).shift(1)).mul((1+2*(b-.5).abs()),axis=0)
print('instruments',len(p.columns),'dates',p.index.min(),p.index.max())
for h in [1,5,10]:
 y=p.shift(-h)/p-1;a=[];ns=[]
 for d in f.index:
  z=pd.concat([f.loc[d],y.loc[d]],axis=1).dropna()
  if len(z)>=8:
   q=z.iloc[:,0].rank().corr(z.iloc[:,1].rank());
   if np.isfinite(q):a.append(q);ns.append(len(z))
 a=np.array(a);print('H',h,'dates',len(a),'avgN',np.mean(ns),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0))
print('coverage',f.notna().sum(axis=1).div(len(U)).mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
f.stack().rename('factor').to_csv('scripts/miner_2_20261217_breadth_distance_reversal_signal.csv')
