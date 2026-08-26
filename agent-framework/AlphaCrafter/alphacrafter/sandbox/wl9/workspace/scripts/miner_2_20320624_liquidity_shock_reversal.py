import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
raw={s:get_stock_daily_data(s,days=3200) for s in U}
# Construct from close and use traded-value proxy when volume is absent/zero.
px=pd.DataFrame({s:(d.set_index('date')['close'] if d is not None else pd.Series(dtype=float)) for s,d in raw.items()}).sort_index()
vr={}
for s,d in raw.items():
 if d is None: vr[s]=pd.Series(dtype=float)
 else:
  v=pd.to_numeric(d.set_index('date')['volume'],errors='coerce')
  vr[s]=v.where(v>0,1.0)
vol=pd.DataFrame(vr).reindex(px.index)
r=px.pct_change(); lv=np.log(vol)
rv=r.rolling(5).sum(); vz=(lv-lv.rolling(40).mean()).div(lv.rolling(40).std())
sig=(-rv*vz.clip(-3,3)).shift(1)
for h in [5,10,20,40,60]:
 ic=[]; n=[]; turnovers=[]
 for i in range(1,len(px)-h):
  z=pd.concat([sig.iloc[i],px.iloc[i+h]/px.iloc[i]-1],axis=1).dropna()
  if len(z)>=8:
   ic.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); n.append(len(z))
   if i>0: turnovers.append((sig.iloc[i-1].rank(pct=True)-sig.iloc[i].rank(pct=True)).abs().mean())
 x=pd.Series(ic).dropna(); print(f'h={h} dates={len(x)} avgN={np.mean(n):.2f} IC={x.mean():.6f} ICIR={x.mean()/x.std(ddof=1):.6f} hit={(x>0).mean():.4f} turnover={np.mean(turnovers):.6f}')
print('date_range',px.index.min(),px.index.max(),'coverage',sig.notna().sum(axis=1).mean()/15)
