import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
files=glob.glob('../persistent/stock_data/*.csv')
data={os.path.basename(f)[:-4]:pd.read_csv(f,parse_dates=['date']).set_index('date') for f in files}
p=pd.DataFrame({s:d['close'] for s,d in data.items()}).sort_index(); r=p.pct_change()
v5=r.rolling(5).std(); v20=r.rolling(20).std(); v60=r.rolling(60).std()
# Volatility acceleration, rather than volatility level, weights short reversal.
acc=(v5/(v20+1e-12)).clip(0.5,2.5)
sig=(-(r.rolling(3).sum())/(v5*np.sqrt(3)+1e-12)*acc).shift(1); sig.index.name='date'
ics=[]; dates=[]; ns=[]; cov=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],r.shift(-1).loc[dt]],axis=1).dropna()
 if len(z)>=8:
  q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(q):ics.append(q); dates.append(dt); ns.append(len(z)); cov.append(len(z)/15)
a=np.array(ics); print('dates',len(a),'avg_instruments',np.mean(ns),'universe',15); print('daily_ic %.9f daily_icir %.9f hit %.4f coverage %.4f'%(a.mean(),a.mean()/a.std(ddof=1)*np.sqrt(252),(a>0).mean(),np.mean(cov)))
for lo,hi in [('2020-01-01','2022-12-31'),('2023-01-01','2025-12-31'),('2026-01-01','2031-12-31'),('2032-01-01','2032-12-31')]:
 b=a[[lo<=str(d.date())<=hi for d in dates]]; print('regime',lo,hi,len(b),np.nanmean(b),np.nanmean(b)/np.nanstd(b,ddof=1)*np.sqrt(252) if len(b)>1 else np.nan)
for h in [1,3,5,10]:
 yy=p.pct_change(h).shift(-h);q=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('decay',h,np.nanmean(q),len(q))
sig.to_csv('scripts/miner_2_20320415_vol_accel_reversal_signal.csv')
