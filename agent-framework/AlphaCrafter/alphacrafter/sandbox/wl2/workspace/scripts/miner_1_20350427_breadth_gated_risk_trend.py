import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)<150: d=get_index_daily_data(s,5000)
 if d is not None: px[s]=d.set_index('date')['close']
C=pd.DataFrame(px).sort_index().ffill(); R=C.pct_change()
# Candidate: 20d risk-adjusted trend, only when lagged cross-asset breadth is positive.
# Signal at t uses observations through t-1; rank-like raw score is lagged 20d return / 40d vol.
vol=R.rolling(40,min_periods=25).std(); raw=R.rolling(20,min_periods=20).sum()/vol
breadth=(R.rolling(20,min_periods=15).sum()>0).mean(axis=1)
gate=(breadth.shift(1)>0.5)
S=raw.shift(1).where(gate,0.0)
rows=[]
for h in [5,10,20,40]:
 f=C.shift(-h)/C-1
 vals=[]
 for dt in S.index:
  a=S.loc[dt]; b=f.loc[dt]; z=pd.concat([a,b],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].std()>0 and z.iloc[:,1].std()>0: vals.append(z.iloc[:,0].corr(z.iloc[:,1]))
 q=pd.Series(vals).dropna(); rows.append((h,len(q),q.mean(),q.mean()/q.std(ddof=1), (q>0).mean()))
print('rows',len(C),'assets',len(px),'dates',len(S))
for x in rows: print('h%d dates=%d IC=%.8f ICIR=%.8f hit=%.4f'%x)
print('coverage',S.notna().sum(axis=1).mean()/len(U),'activation',gate.mean())
# save artifact for audit
out=S.copy(); out.index.name='date'; out.to_csv('../persistent/miner_1_20350427_breadth_gated_risk_trend_signal.csv')
