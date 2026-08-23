import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,4000) for s in U}
close=pd.DataFrame({s:d.set_index('date')['close'] for s,d in D.items() if d is not None}).sort_index().ffill()
r=close.pct_change(); m20=close/close.shift(20)-1; m60=close/close.shift(60)-1
# downside-risk-adjusted intermediate momentum, with 5d confirmation
neg=r.where(r<0,0).rolling(40).std(); f=(m20/(neg*np.sqrt(20))).where(m60>0, -m20/(neg*np.sqrt(20)))
rows=[]; sig=[]
for i in range(60,len(close)-10):
 z=pd.concat([f.iloc[i],close.iloc[i+10]/close.iloc[i]-1],axis=1).dropna()
 if len(z)>=8: rows.append((z.iloc[:,0].corr(z.iloc[:,1]),len(z),close.index[i]))
 if i>60: sig.append(np.nanmean(abs(f.iloc[i].rank(pct=True)-f.iloc[i-1].rank(pct=True))))
r=np.array([x[0] for x in rows]); print('candidate=trend_conditioned_downside_momentum'); print('dates',len(rows),'instruments_mean',np.mean([x[1] for x in rows]),'coverage',len(rows)/len(close)); print('IC',np.nanmean(r),'ICIR',np.nanmean(r)/np.nanstd(r,ddof=1),'hit',np.mean(r>0),'turnover',np.nanmean(sig))
for a,b in [('2020','2024-12-31'),('2025','2026-12-31'),('2027','2028-12-31'),('2029','2029-12-31')]:
 q=[v for v,_,dt in rows if pd.Timestamp(a)<=dt<=pd.Timestamp(b)]; print(a,'n',len(q),'ic',np.mean(q) if q else np.nan,'icir',np.mean(q)/np.std(q,ddof=1) if len(q)>1 else np.nan)
for h in [5,20]:
 rr=[]
 for i in range(60,len(close)-h):
  z=pd.concat([f.iloc[i],close.iloc[i+h]/close.iloc[i]-1],axis=1).dropna()
  if len(z)>=8: rr.append(z.iloc[:,0].corr(z.iloc[:,1]))
 print('decay',h,np.mean(rr),len(rr))
f.to_csv('scripts/miner_2_20291129_trend_conditioned_downside_signal.csv')
