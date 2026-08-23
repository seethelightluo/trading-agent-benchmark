import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
    for f in (get_index_daily_data,get_stock_daily_data):
        try:
            d=f(s,days=4000)
            if d is not None and len(d): return d
        except Exception: pass
P={}
for s in U:
 d=fetch(s)
 if d is None: continue
 d=d.copy(); d.date=pd.to_datetime(d.date); d=d.set_index('date').sort_index()
 c=pd.to_numeric(d.close,errors='coerce'); P[s]=c.pct_change()
R=pd.concat(P,axis=1).sort_index(); m=R.mean(axis=1)
# beta-neutral residual momentum: trailing covariance beta to equal-weight benchmark,
# residual cumulative return divided by residual volatility, with trailing-only inputs.
beta=R.rolling(60,min_periods=40).cov(m).div(m.rolling(60,min_periods=40).var(),axis=0)
res=R.sub(beta.mul(m,axis=0),axis=0)
sig=res.rolling(40,min_periods=30).sum()/(res.rolling(20,min_periods=15).std()*np.sqrt(20)+0.02)
F={h:R.shift(-h).rolling(h).sum().shift(-(h-1)) for h in [1,5,10,20]}
# use forward compounded returns directly from prices equivalent enough
rows=[]
for dt in R.index:
 q=pd.DataFrame({'sig':sig.loc[dt] if dt in sig.index else np.nan})
 for h in [1,5,10,20]:
  q['f']=R.shift(-1).rolling(h).sum().shift(-(h-1)).loc[dt] if dt in R.index else np.nan
  q=q.dropna()
  if len(q)>=8:
   rows.append((dt,h,q.sig.rank().corr(q.f.rank()),len(q)))
q=pd.DataFrame(rows,columns=['date','h','ic','n'])
for h in [1,5,10,20]:
 x=q[q.h==h]; m=x.ic.mean(); sd=x.ic.std(ddof=1)
 print(f'{h}d dates={len(x)} avg_n={x.n.mean():.2f} coverage={x.n.mean()/15:.4f} IC={m:.6f} ICIR={m/sd*np.sqrt(252):.4f} hit={(x.ic>0).mean():.4f} turnover_proxy={x.ic.diff().abs().mean():.6f}')
 for a,b in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2028-12-31'),('2029','2029-12-31')]:
  y=x[(x.date.astype(str)>=a)&(x.date.astype(str)<=b)].ic; print('regime',a,len(y),f'{y.mean():.6f}')
