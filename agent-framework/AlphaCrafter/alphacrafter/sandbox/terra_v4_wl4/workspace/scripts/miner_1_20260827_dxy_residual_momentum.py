import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for a in A:
 d=get_stock_daily_data(a,days=2400)
 if d is not None and len(d)>150: px[a]=d.set_index('date').close.astype(float)
p=pd.concat(px,axis=1).sort_index().ffill(); r=p.pct_change()
d=get_index_daily_data('DXY',days=2400)
if d is None: raise RuntimeError('no DXY')
x=d.set_index('date').close.astype(float).reindex(p.index).ffill().pct_change()
# Residual momentum: 20d asset return after removing rolling 60d DXY beta exposure.
# All beta/returns use data through factor date only.
co=r.rolling(60,min_periods=40).cov(x)
va=x.rolling(60,min_periods=40).var()
beta=co.div(va,axis=0)
f=r.rolling(20,min_periods=15).sum()-beta.mul(x.rolling(20,min_periods=15).sum(),axis=0)
# evaluate same-date factor vs next h-session cumulative return
for h in [1,5,10]:
 y=p.shift(-h).div(p)-1; vals=[]; ns=[]; turns=[]
 for i in range(len(p)-h):
  q=pd.concat([f.iloc[i].rename('f'),y.iloc[i].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1: vals.append(q.f.corr(q.y));ns.append(len(q))
  if i>0:
   z=f.iloc[i].rank(pct=True); zp=f.iloc[i-1].rank(pct=True); turns.append(np.nanmean(abs(z-zp)))
 z=np.array(vals); z=z[np.isfinite(z)]
 print('h',h,'dates',len(z),'meanN',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC',round(z.mean(),5),'ICIR',round(z.mean()/z.std(ddof=1),5),'hit',round(np.mean(z>0),4),'turn',round(np.nanmean(turns),4))
# regimes
for label,mask in [('2020-22',p.index<'2023-01-01'),('2023-24',(p.index>='2023-01-01')&(p.index<'2025-01-01')),('2025-26',p.index>='2025-01-01')]:
 z=[]
 for i in range(len(p)-1):
  if not mask[i]: continue
  q=pd.concat([f.iloc[i].rename('f'),(p.shift(-1).iloc[i].div(p.iloc[i])-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1:z.append(q.f.corr(q.y))
 z=np.array(z);print(label,'dates',len(z),'IC',round(np.nanmean(z),5),'ICIR',round(np.nanmean(z)/np.nanstd(z,ddof=1),5))
# independent diagnostics
base=-p.pct_change(5)
q=pd.concat([f.stack(),base.stack()],axis=1).dropna();print('corr_simple_reversal5',round(q.iloc[:,0].corr(q.iloc[:,1]),5))
print('last_factor_date',p.index.max().date())
