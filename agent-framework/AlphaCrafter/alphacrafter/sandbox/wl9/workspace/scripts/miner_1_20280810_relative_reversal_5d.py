import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=get_stock_daily_data(s,days=4000)
 if d is not None and len(d): D[s]=d.set_index('date')['close']
close=pd.DataFrame(D).sort_index().ffill(); r=close.pct_change()
# Relative short-horizon reversal: sell recent winners and buy losers,
# subtracting contemporaneous peer median to remove common market moves.
rel5=close/close.shift(5)-1
f=-(rel5-rel5.median(axis=1).values[:,None])
# damp extreme cross-sectional observations, retain interpretable reversal
f=f.clip(f.quantile(.02,axis=1),f.quantile(.98,axis=1),axis=0)
res={h:[] for h in [1,5,10,20]}; counts={h:[] for h in res}
for i in range(len(close)-20):
 for h in res:
  if i+h>=len(close): continue
  z=pd.concat([f.iloc[i].rename('x'),(close.iloc[i+h]/close.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8:
   q=z.x.corr(z.y,method='spearman')
   if np.isfinite(q): res[h].append(q); counts[h].append(len(z))
for h,a0 in res.items():
 a=np.asarray(a0,float); print(h,'dates',len(a),'mean_names',round(np.mean(counts[h]),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1)*np.sqrt(len(a)),4),'hit',round((a>0).mean(),4))
 for n,label in [(252,'recent252'),(504,'recent504')]:
  b=a[-n:] if len(a)>n else a; print(label,'n',len(b),'IC',round(b.mean(),6),'ICIR',round(b.mean()/b.std(ddof=1)*np.sqrt(len(b)),4))
print('coverage',round(f.notna().sum(axis=1).mean()/15,4),'turnover5',round(f.rank(axis=1,pct=True).diff(5).abs().mean(axis=1).dropna().mean(),4),'assets',len(D),'period',close.index.min(),close.index.max())
