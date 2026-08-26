import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=get_stock_daily_data(s,days=6000)
 if d is not None and len(d):
  d=d.copy(); d.date=pd.to_datetime(d.date); D[s]=d.set_index('date')
px=pd.DataFrame({s:x.close.astype(float) for s,x in D.items()}).sort_index()
hi=pd.DataFrame({s:x.high.astype(float) for s,x in D.items()}).reindex(px.index)
lo=pd.DataFrame({s:x.low.astype(float) for s,x in D.items()}).reindex(px.index)
ret=px.pct_change(); risk=ret.rolling(20,min_periods=15).std()
a1=(hi-lo)/px; a2=(hi-px.shift(1)).abs()/px.shift(1); a3=(lo-px.shift(1)).abs()/px.shift(1)
tr=pd.DataFrame(np.maximum.reduce([a1.values,a2.values,a3.values]),index=px.index,columns=px.columns)
atr20=tr.rolling(20,min_periods=15).mean(); atr60=tr.rolling(60,min_periods=40).mean()
compression=(atr20/(atr60+1e-12)).clip(.5,1.5)
acc=px.pct_change(20)-px.pct_change(60)
f=(acc/(risk+1e-12))*(1.25-compression).clip(.25,1.75)
f=f.sub(f.mean(axis=1),axis=0).div(f.std(axis=1),axis=0)
fr=px.shift(-10)/px-1; rows=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
a=np.array([x[1] for x in rows]); dates=[x[0] for x in rows]
print('factor range_compressed_trend_acceleration'); print('dates',len(a),'avgN',np.mean([x[2] for x in rows]),'start',min(dates),'end',max(dates))
print('IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0),'coverage',np.mean([x[2] for x in rows])/15)
for n in [365,750,1260]:
 q=a[-n:]; print('recent',n,q.mean(),q.mean()/q.std(ddof=1),len(q))
for h in [1,5,20]:
 yy=px.shift(-h)/px-1; q=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('decay',h,np.mean(q),len(q))
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
f.to_csv('scripts/miner_2_20340427_range_compressed_trend_signal.csv')
pd.DataFrame(rows,columns=['date','ic','n']).to_csv('scripts/miner_2_20340427_range_compressed_trend_ic.csv',index=False)
