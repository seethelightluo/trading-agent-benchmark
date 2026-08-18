import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   x=fn(s,days=5000)
   if x is not None and len(x): return x[['date','close']].copy()
  except Exception: pass
px={s:load(s) for s in U}; px={s:x for s,x in px.items() if x is not None}
wide=pd.concat([x.rename(columns={'close':s}).set_index('date')[[s]] for s,x in px.items()],axis=1).sort_index()
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].rename('VIX')
wide=wide.join(vix,how='left').ffill(); ret=wide[U].pct_change()
def stress(a):
 a=np.asarray(a); return float(a[-1]>np.nanpercentile(a,70))
vp=wide.VIX.rolling(252,min_periods=60).apply(stress,raw=True)
r10=wide[U].pct_change(10); resid=r10.sub(r10.median(axis=1),axis=0)
f=(-resid).rolling(3,min_periods=3).mean().shift(1).mul(1+0.75*vp.shift(1),axis=0)
def calc(h):
 fr=wide[U].shift(-h)/wide[U]-1; vals=[]; dates=[]; ns=[]
 for d in f.index:
  a=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
  if len(a)>=8: vals.append(a.iloc[:,0].corr(a.iloc[:,1],method='spearman')); dates.append(d); ns.append(len(a))
 z=pd.Series(vals,index=dates).dropna(); return z,ns
for h in [5,10,20]:
 z,ns=calc(h); print(f'H{h} dates={len(z)} avgN={np.mean(ns):.2f} IC={z.mean():.6f} ICIR={z.mean()/z.std(ddof=1)*np.sqrt(252):.6f} hit={(z>0).mean():.4f}')
z,_=calc(10)
for n in [365,730,1095]:
 q=z.tail(n); print(f'recent{n} IC={q.mean():.6f} ICIR={q.mean()/q.std(ddof=1)*np.sqrt(252):.6f}')
print('coverage',f[U].notna().mean().mean(),'dates',len(z),'instruments',len(U))
