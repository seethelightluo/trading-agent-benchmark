import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
    for fn in (get_index_daily_data,get_stock_daily_data):
        try:
            x=fn(s,days=5000)
            if x is not None and len(x)>100:return x[['date','close']].copy()
        except Exception:pass
    return None
parts=[]
for s in U:
    x=get(s)
    if x is not None:
        x['r']=x.close.pct_change();x['r5']=x.close.pct_change(5);x['v20']=x.r.rolling(20).std()
        parts.append(x.set_index('date').rename(columns={'r':s+'_r','r5':s+'_r5','v20':s+'_v'}))
d=pd.concat(parts,axis=1).sort_index()
rets=d[[s+'_r' for s in U]].rename(columns=lambda z:z[:-2])
r5=d[[s+'_r5' for s in U]].rename(columns=lambda z:z[:-3])
vol=d[[s+'_v' for s in U]].rename(columns=lambda z:z[:-2])
# Reversal of 5-day cross-sectional residual return, strengthened in high-dispersion regimes.
resid=r5.sub(r5.median(axis=1),axis=0)
disp=r5.std(axis=1).rolling(20,min_periods=10).mean()
base=-resid/(vol*np.sqrt(5)+1e-8)
mult=(disp/disp.rolling(120,min_periods=40).median()).clip(0.5,2.0)
f=base.mul(mult,axis=0).rolling(3,min_periods=3).mean().shift(1)
for h in [1,5,10,20]:
 fr=(1+rets).rolling(h).apply(np.prod,raw=True).shift(-h+1)-1; vals=[];ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 q=pd.Series(vals); rr=q.tail(250)
 print(f'H{h}: dates={len(q)} avg_n={np.mean(ns):.2f} min_n={min(ns) if ns else 0} IC={q.mean():.6f} ICIR={q.mean()/q.std(ddof=1):.6f} hit={(q>0).mean():.4f} recent250={rr.mean():.6f}/{rr.mean()/rr.std(ddof=1):.6f}')
print(f'coverage={f.notna().sum(axis=1).mean()/15:.6f} dates={len(f)} assets={len(parts)}')
