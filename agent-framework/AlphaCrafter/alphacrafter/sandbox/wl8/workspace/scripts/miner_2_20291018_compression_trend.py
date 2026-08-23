import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
symbols=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2029-10-18')
D={}
for s in symbols:
 f='../persistent/stock_data/'+s+'.csv'
 if not os.path.exists(f): f='../persistent/index_data/'+s+'.csv'
 d=pd.read_csv(f,parse_dates=['date']).sort_values('date').set_index('date')
 D[s]=d['close'].astype(float)
p=pd.DataFrame(D).loc[:end]
# compression-adjusted medium trend: lagged 20d return, amplified when recent range/vol is compressed
ret=p.pct_change()
vol=ret.rolling(20,min_periods=15).std()
range20=(p.rolling(20,min_periods=15).max()-p.rolling(20,min_periods=15).min())/p.rolling(20,min_periods=15).mean()
compression=(range20 / (vol*np.sqrt(20))).replace([np.inf,-np.inf],np.nan)
# lower ratio = compressed; use inverse clipped, neutralized cross-sectionally
comp=(1/compression).clip(0.25,4)
sig=(p.pct_change(20).shift(1)/vol.shift(1)*comp.shift(1))
# forward returns, aligned dates
fwd=p.shift(-10)/p-1
ics=[]; turnovers=[]; nins=[]
prev=None
for dt in sig.index:
 x=sig.loc[dt]; y=fwd.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
 if len(z)>=8:
  ic=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(ic):
   ics.append(ic); nins.append(len(z))
   ranks=x.rank(pct=True)
   if prev is not None: turnovers.append(np.mean(abs(ranks-prev)))
   prev=ranks
ics=np.array(ics)
print('dates',len(ics),'period',sig.index.min().date(),sig.index.max().date(),'avg_instruments',np.mean(nins),'coverage',np.mean(nins)/15)
print('IC',ics.mean(),'ICIR',ics.mean()/ics.std(ddof=1),'hit',np.mean(ics>0),'turnover',np.mean(turnovers))
for name,mask in [('2020-22',(np.array([d.year for d in sig.index if d in sig.index])>=2020)),]: pass
for lo,hi in [('2020-01-01','2022-12-31'),('2023-01-01','2025-12-31'),('2026-01-01','2027-12-31'),('2028-01-01','2029-10-18')]:
 vals=[]
 for dt in sig.loc[lo:hi].index:
  z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 a=np.array(vals); print(lo, 'n',len(a),'IC',a.mean() if len(a) else np.nan,'ICIR',a.mean()/a.std(ddof=1) if len(a)>1 else np.nan)
# decay
for h in [1,5,10,20]:
 ff=p.shift(-h)/p-1; aa=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],ff.loc[dt]],axis=1).dropna()
  if len(z)>=8: aa.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 a=np.array(aa); print('decay',h,a.mean(),a.mean()/a.std(ddof=1),len(a))
# signal artifact
out=sig.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20291018_compression_trend_signal.csv',index=False)
