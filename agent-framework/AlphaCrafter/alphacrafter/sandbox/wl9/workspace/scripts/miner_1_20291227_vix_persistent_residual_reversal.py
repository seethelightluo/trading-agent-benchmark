import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,3000)
 if d is None or len(d)==0: d=get_index_daily_data(s,3000)
 if d is not None and len(d): px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index().ffill(); ret=P.pct_change()
v=get_index_daily_data('VIX',3000)
if v is None: v=get_stock_daily_data('VIX',3000)
V=v.set_index('date').close.astype(float).reindex(P.index).ffill()
rows=[]
for i in range(80,len(P)-40):
 # signal uses only through date i-1; persistent elevated VIX on both preceding sessions
 hist=V.iloc[i-61:i]; med=hist.median()
 if len(hist)<60 or V.iloc[i-1] <= med or V.iloc[i-2] <= med: continue
 r10=ret.iloc[i-10:i].sum(); resid=r10-r10.median(); down=ret.iloc[i-20:i].clip(upper=0).std()
 sig=-(resid/down.replace(0,np.nan)); z=sig.rank(pct=True)
 row={'date':P.index[i]}
 for s in U: row['s_'+s]=sig.get(s,np.nan)
 for h in (5,10,20):
  f=ret.iloc[i:i+h].sum(); q=z.dropna().index.intersection(f.dropna().index)
  row['ic'+str(h)]=z[q].corr(f[q]) if len(q)>=8 else np.nan
 row['n']=z.notna().sum(); rows.append(row)
R=pd.DataFrame(rows).set_index('date')
print('dates',len(R),'avg_n',R.n.mean() if len(R) else 0,'active_obs',len(R)/max(1,len(P)))
for h in (5,10,20):
 a=R['ic'+str(h)].dropna(); print('h',h,'dates',len(a),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',(a>0).mean())
rank=R[['s_'+s for s in U]].rank(axis=1,pct=True)
print('turnover',rank.diff().abs().mean().mean(),'coverage',R.n.mean()/15)
R.to_csv('scripts/miner_1_20291227_vix_persistent_residual_reversal_signal.csv')
