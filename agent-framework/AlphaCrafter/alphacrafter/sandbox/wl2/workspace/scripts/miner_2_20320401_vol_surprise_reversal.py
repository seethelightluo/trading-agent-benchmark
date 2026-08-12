import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
files=glob.glob('../persistent/stock_data/*.csv'); data={os.path.basename(f)[:-4]:pd.read_csv(f,parse_dates=['date']).set_index('date') for f in files}
p=pd.DataFrame({s:d['close'] for s,d in data.items()}).sort_index(); r=p.pct_change()
v5=r.rolling(5).std(); v60=r.rolling(60).std(); surprise=(v5/(v60+1e-12)).clip(0.5,3)
# volatility-surprise weighted short-horizon reversal; continuous coverage and interpretable
sig=(-(r.rolling(3).sum())/(v5*np.sqrt(3)+1e-12)*surprise).shift(1); y=r.shift(-1)
ics=[]; dates=[]; ns=[]; cov=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],y.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(q): ics.append(q);dates.append(dt);ns.append(len(z));cov.append(len(z)/15)
a=np.array(ics); m=a.mean(); sd=a.std(ddof=1); print('dates',len(a),'avg_instruments',np.mean(ns),'universe',15); print('daily_ic %.9f daily_icir %.9f hit %.4f coverage %.4f'%(m,m/sd*np.sqrt(252),(a>0).mean(),np.mean(cov)))
for lo,hi in [('2020-01-01','2022-12-31'),('2023-01-01','2025-12-31'),('2026-01-01','2031-12-31'),('2032-01-01','2032-12-31')]:
 b=a[[lo<=str(d.date())<=hi for d in dates]]; print(lo[:4]+'-'+hi[:4],len(b),b.mean() if len(b) else np.nan,(b.mean()/b.std(ddof=1)*np.sqrt(252)) if len(b)>1 and b.std(ddof=1)>0 else np.nan)
for h in [1,3,5,10]:
 yy=p.pct_change(h).shift(-h); q=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('decay',h,np.nanmean(q),len(q))
sig.index.name='date';sig.to_csv('scripts/miner_2_20320401_vol_surprise_reversal_signal.csv')
