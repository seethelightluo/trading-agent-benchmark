import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,4000) for s in U}
close=pd.DataFrame({s:d.set_index('date')['close'] for s,d in D.items() if d is not None}).sort_index().ffill()
r=close.pct_change()
# Range-efficiency trend: directional 20d move divided by path length;
# use sign of 60d trend as a causal trend regime, and risk-scale by 20d volatility.
m20=close/close.shift(20)-1; m60=close/close.shift(60)-1
path=r.abs().rolling(20).sum(); vol=r.rolling(20).std()
eff=m20/path
f=(eff/(vol*np.sqrt(20))).where(m60>0, -eff/(vol*np.sqrt(20)))
rows=[]; turnover=[]
for i in range(80,len(close)-10):
 z=pd.concat([f.iloc[i],close.iloc[i+10]/close.iloc[i]-1],axis=1).dropna()
 if len(z)>=8: rows.append((z.iloc[:,0].corr(z.iloc[:,1]),len(z),close.index[i]))
 if i>80:
  a,b=f.iloc[i].rank(pct=True),f.iloc[i-1].rank(pct=True)
  turnover.append(np.nanmean(abs(a-b)))
ics=np.array([x[0] for x in rows]); print('candidate=range_efficiency_conditioned_trend'); print('dates',len(rows),'instruments_mean',np.mean([x[1] for x in rows]),'coverage',len(rows)/len(close)); print('IC',np.nanmean(ics),'ICIR',np.nanmean(ics)/np.nanstd(ics,ddof=1),'hit',np.mean(ics>0),'turnover',np.nanmean(turnover))
for a,b in [('2020','2024-12-31'),('2025','2026-12-31'),('2027','2028-12-31'),('2029','2029-12-31')]:
 q=[v for v,_,dt in rows if pd.Timestamp(a)<=dt<=pd.Timestamp(b)]
 print(a,'n',len(q),'ic',np.mean(q) if q else np.nan,'icir',np.mean(q)/np.std(q,ddof=1) if len(q)>1 else np.nan)
for h in [5,20]:
 rr=[]
 for i in range(80,len(close)-h):
  z=pd.concat([f.iloc[i],close.iloc[i+h]/close.iloc[i]-1],axis=1).dropna()
  if len(z)>=8: rr.append(z.iloc[:,0].corr(z.iloc[:,1]))
 print('decay',h,'ic',np.mean(rr),'dates',len(rr))
f.to_csv('scripts/miner_2_20291213_range_efficiency_signal.csv')
