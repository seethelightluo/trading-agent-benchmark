import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,4000) for s in U}
close=pd.DataFrame({s:d.set_index('date')['close'] for s,d in D.items() if d is not None}).sort_index().ffill()
ret=close.pct_change(); net=close/close.shift(30)-1; path=ret.abs().rolling(30).sum(); eff=(net.abs()/path).clip(0,1); vol=ret.rolling(20).std(); f=(net*eff/(vol*np.sqrt(20))).replace([np.inf,-np.inf],np.nan)
rows=[]; turnover=[]
for i in range(30,len(close)-10):
 z=pd.concat([f.iloc[i],close.iloc[i+10]/close.iloc[i]-1],axis=1).dropna()
 if len(z)>=8: rows.append((z.iloc[:,0].corr(z.iloc[:,1]),len(z),close.index[i]))
 if i>30: turnover.append(np.nanmean(abs(f.iloc[i].rank(pct=True)-f.iloc[i-1].rank(pct=True))))
r=np.array([x[0] for x in rows]); print('candidate=efficiency_weighted_30d_trend'); print('dates',len(rows),'instruments_mean',np.mean([x[1] for x in rows]),'coverage',len(rows)/len(close)); print('IC',np.nanmean(r),'ICIR',np.nanmean(r)/np.nanstd(r,ddof=1),'hit',np.mean(r>0),'turnover',np.nanmean(turnover))
for a,b in [('2020','2024-12-31'),('2025','2026-12-31'),('2027','2028-12-31'),('2029','2029-12-31')]:
 q=[v for v,_,dt in rows if pd.Timestamp(a)<=dt<=pd.Timestamp(b)]; print(a,'n',len(q),'ic',np.mean(q) if q else np.nan,'icir',np.mean(q)/np.std(q,ddof=1) if len(q)>1 else np.nan)
for h in [5,20]:
 rr=[]
 for i in range(30,len(close)-h):
  z=pd.concat([f.iloc[i],close.iloc[i+h]/close.iloc[i]-1],axis=1).dropna()
  if len(z)>=8: rr.append(z.iloc[:,0].corr(z.iloc[:,1]))
 print('decay',h,np.mean(rr),len(rr))
f.to_csv('scripts/miner_2_20291115_efficiency_weighted_30d_signal.csv')
