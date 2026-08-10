import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-12-17')
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:cut] for s in U}
P=pd.DataFrame({s:d.close for s,d in D.items()}).sort_index().ffill(); R=P.pct_change(); V=pd.DataFrame({s:d.volume for s,d in D.items()}).reindex(P.index).ffill()
# Novel: fade the 2-day move, but suppress moves confirmed by unusually high volume.
r2=R.rolling(2,min_periods=2).sum(); vr=V/V.rolling(30,min_periods=20).median(); F=-r2*(2-vr.clip(0,2)); F=F.sub(F.median(axis=1),axis=0)
F.to_csv('scripts/miner_1_20261217_volume_confirmed_reversal_signal.csv',index_label='date')
for h in [1,5,10]:
 Y=P.pct_change(h).shift(-h); vals=[]; ns=[]; ds=[]
 for dt in F.index:
  z=pd.concat([F.loc[dt].rename('f'),Y.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1:
   q=spearmanr(z.f,z.y).statistic
   if np.isfinite(q): vals.append(q);ns.append(len(z));ds.append(dt)
 a=np.array(vals); print('h',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4),'coverage',round(len(a)/len(P),4))
 for lo,hi in [(2020,2022),(2023,2024),(2025,2026)]:
  z=a[(pd.DatetimeIndex(ds).year>=lo)&(pd.DatetimeIndex(ds).year<=hi)]
  print('regime',lo,hi,'n',len(z),'ic',round(z.mean(),6) if len(z) else None)
r=F.rank(pct=True,axis=1); print('turnover',round(np.nanmean(np.abs(r.diff()).mean(axis=1)),6),'factor_dates',len(F),'universe',len(U))
