import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,4000) for s in U}
close=pd.DataFrame({s:d.set_index('date')['close'] for s,d in D.items() if d is not None}).sort_index().ffill()
vol=pd.DataFrame({s:d.set_index('date')['volume'] for s,d in D.items() if d is not None}).sort_index().reindex(close.index).ffill()
ret=close.pct_change(); fwd=close.shift(-10)/close-1
r5=close.pct_change(5); v20=ret.rolling(20).std(); volshock=(vol/vol.rolling(60).median()).clip(0.5,4)
f=((-r5/(v20*np.sqrt(5))).clip(-6,6)*(volshock**0.25)).replace([np.inf,-np.inf],np.nan)
ics=[]; dates=[]; ns=[]; ranks=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if np.isfinite(c):ics.append(c);dates.append(dt);ns.append(len(z));ranks.append(f.loc[dt].rank(pct=True))
a=np.array(ics); print('dates',len(a),'start',dates[0].date(),'end',dates[-1].date(),'meanN',np.mean(ns),'coverage',np.mean(ns)/len(U),'IC10',np.mean(a),'ICIR_daily',np.mean(a)/np.std(a,ddof=1),'hit',np.mean(a>0),'turnover',np.mean(pd.DataFrame(ranks).diff().abs().mean(axis=1)))
for h in [5,20]:
 yy=close.shift(-h)/close-1;q=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,np.nanmean(q))
for x,y in [('2020-07-01','2024-12-31'),('2025-01-01','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2032-12-31'),('2033-01-01','2034-08-17')]:
 q=[c for c,d in zip(a,dates) if pd.Timestamp(x)<=d<=pd.Timestamp(y)]
 if len(q)>1:print('regime',x,len(q),np.mean(q),np.mean(q)/np.std(q,ddof=1))
out=[(dt,s,float(f.loc[dt,s])) for dt in f.index for s in f.columns if pd.notna(f.loc[dt,s])]
pd.DataFrame(out,columns=['date','symbol','factor_value']).to_csv('scripts/miner_2_20340817_liquidity_shock_reversal_signal.csv',index=False)
