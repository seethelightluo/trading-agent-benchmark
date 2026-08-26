import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=get_stock_daily_data(s,days=4000)
 if d is not None and len(d): D[s]=d.set_index('date')['close']
close=pd.DataFrame(D).sort_index().ffill()
ret20=close/close.shift(20)-1
lo=close.rolling(60,min_periods=40).min(); hi=close.rolling(60,min_periods=40).max()
pos=(close-lo)/(hi-lo).replace(0,np.nan)
f=ret20*(0.5+pos)
res={h:[] for h in [1,5,10,20]}; dates={h:[] for h in res}; counts={h:[] for h in res}
for i in range(len(close)-20):
 dt=close.index[i]; x=f.iloc[i]
 for h in res:
  if i+h>=len(close): continue
  y=close.iloc[i+h]/close.iloc[i]-1; z=pd.concat([x.rename('x'),y.rename('y')],axis=1).dropna()
  if len(z)>=8:
   res[h].append(z.x.corr(z.y,method='spearman')); dates[h].append(dt); counts[h].append(len(z))
for h in res:
 a=np.array(res[h],float); a=a[np.isfinite(a)]
 print(h,'dates',len(a),'mean_names',round(float(np.mean(counts[h])),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1)*np.sqrt(len(a)),4),'hit',round(float((a>0).mean()),4))
 for n,label in [(252,'recent252'),(504,'recent504')]:
  b=a[-n:] if len(a)>n else a
  print(label,'n',len(b),'IC',round(b.mean(),6),'ICIR',round(b.mean()/b.std(ddof=1)*np.sqrt(len(b)),4))
r=f.rank(axis=1,pct=True); turn=(r-r.shift(5)).abs().mean(axis=1).dropna()
print('coverage',round(f.notna().sum(axis=1).mean()/15,4),'turnover5',round(turn.mean(),4),'period',close.index.min(),close.index.max(),'assets',len(D))
