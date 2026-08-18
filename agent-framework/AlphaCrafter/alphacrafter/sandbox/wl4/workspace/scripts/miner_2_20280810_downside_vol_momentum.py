import numpy as np, pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
    for fn in (get_index_daily_data,get_stock_daily_data):
        try:
            x=fn(s,days=5000)
            if x is not None and len(x)>100:return x[['date','close']]
        except Exception: pass
parts=[]
for s in U:
    x=get(s)
    if x is not None:
        x=x.set_index('date').sort_index(); r=x.close.pct_change()
        parts.append(pd.DataFrame({s+'_r':r,s+'_r20':x.close.pct_change(20),s+'_down':r.clip(upper=0).rolling(40).std(),s+'_v':r.rolling(20).std()}))
d=pd.concat(parts,axis=1).sort_index()
r20=d[[s+'_r20' for s in U]].rename(columns=lambda z:z[:-4])
down=d[[s+'_down' for s in U]].rename(columns=lambda z:z[:-5])
vol=d[[s+'_v' for s in U]].rename(columns=lambda z:z[:-2])
# Momentum rewarded when downside risk is low; cross-sectional rank-like scaling and breadth gate.
breadth=(r20>0).mean(axis=1)
raw=r20/(down*np.sqrt(20)+1e-8)
gate=(breadth-0.5).abs()+0.5
f=raw.mul(gate,axis=0).rolling(3,min_periods=3).mean().shift(1)
rets=d[[s+'_r' for s in U]].rename(columns=lambda z:z[:-2])
print('assets',len(parts),'dates',len(f),'coverage',f.notna().sum(axis=1).mean()/15)
for h in [1,5,10,20]:
 fr=(1+rets).rolling(h).apply(np.prod,raw=True).shift(-h+1)-1; vals=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 q=pd.Series(vals); rr=q.tail(250)
 print(f'H{h}: dates={len(q)} avg_n={np.mean(ns):.2f} min_n={min(ns)} IC={q.mean():.6f} ICIR={q.mean()/q.std(ddof=1):.6f} hit={(q>0).mean():.4f} recent250={rr.mean():.6f}/{rr.mean()/rr.std(ddof=1):.6f}')
