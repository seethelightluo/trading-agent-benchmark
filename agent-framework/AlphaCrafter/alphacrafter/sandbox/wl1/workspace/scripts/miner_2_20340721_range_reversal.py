import pandas as pd, numpy as np
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)==0: d=get_index_daily_data(s,5000)
 if d is not None and len(d): px[s]=d.set_index(pd.to_datetime(d.date)).close.astype(float)
P=pd.DataFrame(px).sort_index().ffill()
# Range-location reversal: fade recent 5d move more strongly at extreme 60d range positions.
r5=P.pct_change(5); lo=P.rolling(60).min(); hi=P.rolling(60).max()
loc=(P-lo)/(hi-lo).replace(0,np.nan)
# extreme multiplier is symmetric, with sign fading the recent move
extreme=1+2*(loc-0.5).abs()
sig=(-r5/P.pct_change().rolling(20).std().replace(0,np.nan))*extreme
sig=sig.shift(1)
rows=[]
for h in [5,10,20,40]:
 fwd=P.shift(-h)/P-1
 vals=[]
 for dt in sig.index:
  a=sig.loc[dt]; b=fwd.loc[dt]; z=pd.concat([a,b],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1]))
 x=pd.Series(vals).dropna(); print('H',h,'dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(),6),'hit',round((x>0).mean(),4))
# coverage and signal turnover
valid=sig.notna().sum(axis=1); print('overall dates',len(sig),'avgN',round(valid.mean(),2),'coverage',round(valid.mean()/len(U),4))
ranks=sig.rank(axis=1,pct=True); turn=(ranks.diff().abs().mean(axis=1)>0.15).mean(); print('turnover_proxy',round(turn,4))
for a,b in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032'),('2033','2034')]:
 q=[]
 for dt in sig.loc[a:b].index:
  z=pd.concat([sig.loc[dt],(P.shift(-10)/P-1).loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append(z.iloc[:,0].corr(z.iloc[:,1]))
 q=pd.Series(q).dropna(); print(a,b,len(q),round(q.mean(),6),round(q.mean()/q.std(),6) if len(q)>1 else np.nan)
# artifact for audit
out=sig.copy(); out.index.name='date'; out.to_csv('scripts/miner_2_20340721_range_reversal_signal.csv')
