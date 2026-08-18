import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=get_stock_daily_data(s,days=4000)
 if d is None or len(d)<150: d=get_index_daily_data(s,days=4000)
 if d is not None: D[s]=d.set_index('date')[['open','close']].astype(float)
O=pd.concat({s:d['open'] for s,d in D.items()},axis=1).sort_index().ffill(); C=pd.concat({s:d['close'] for s,d in D.items()},axis=1).sort_index().ffill()
# One-day intraday shock reversal, with cross-sectional market-relative demeaning.
gap=C/O-1
m=gap.mean(axis=1)
f=-(gap.sub(m,axis=0)).shift(0)
f=f.replace([np.inf,-np.inf],np.nan)
fwd=C.shift(-10)/C-1
rows=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
ic=pd.Series({d:v for d,v,n in rows}); ns=[n for d,v,n in rows]
print('dates',len(ic),'avg_n',np.mean(ns),'coverage',len(ic)/len(f))
print('IC',ic.mean(),'ICIR',ic.mean()/ic.std(ddof=1),'hit',(ic>0).mean())
for h in [3,5,10,20]:
 fw=C.shift(-h)/C-1; rr=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8: rr.append(z.iloc[:,0].corr(z.iloc[:,1]))
 q=pd.Series(rr).dropna(); print('h',h,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'n',len(q))
for name,sel in [('2020-25',ic.index<'2026-01-01'),('2026-29',(ic.index>='2026-01-01')&(ic.index<'2030-01-01')),('2030+',ic.index>='2030-01-01'),('last365',ic.index>=ic.index.max()-pd.Timedelta(days=365))]:
 q=ic[sel]; print(name,len(q),q.mean(),q.mean()/q.std(ddof=1) if len(q)>2 else np.nan,(q>0).mean())
rank=f.rank(axis=1,pct=True); print('turnover',rank.diff().abs().mean(axis=1).mean(),'valid_assets',C.notna().mean().mean())
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20330819_intraday_gap_reversal_signal.csv',index=False)
