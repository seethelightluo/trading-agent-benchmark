import pandas as pd, numpy as np
from scipy.stats import spearmanr
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2027-11-02')
px={}
for s in U:
 d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date')['close'].sort_index(); px[s]=d[d.index<=cut]
p=pd.DataFrame(px).ffill(); r=p.pct_change(); neg2=(r.clip(upper=0)**2); downside=np.sqrt(neg2.rolling(20,min_periods=10).mean())
f=(p/p.shift(20)-1)/(downside*np.sqrt(20)+1e-8)
for h in [1,5,10]:
 vals=[]; dates=[]; nobs=[]; fr=p.shift(-h)/p-1
 for dt in f.index:
  z=pd.concat([f.loc[dt].rename('f'),fr.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8:
   vals.append(spearmanr(z.f,z.y).statistic); dates.append(dt); nobs.append(len(z))
 a=np.array(vals); ds=pd.DatetimeIndex(dates)
 print('horizon',h,'dates',len(a),'meanIC %.5f ICIR %.5f hit %.3f meanN %.1f'%(np.nanmean(a),np.nanmean(a)/(np.nanstd(a,ddof=1)+1e-12),np.mean(a>0),np.mean(nobs)))
 for lo,hi in [('2020','2022-12-31'),('2023','2025-12-31'),('2026','2027-11-02')]:
  q=a[(ds>=pd.Timestamp(lo))&(ds<=pd.Timestamp(hi))]; print(lo,hi,len(q),'IC %.5f ICIR %.5f'%(np.nanmean(q),np.nanmean(q)/(np.nanstd(q,ddof=1)+1e-12)))
rank=f.rank(axis=1,pct=True); print('coverage %.3f turnover %.4f instruments %d'%(f.notna().mean(axis=1).mean(),(rank-rank.shift(1)).abs().mean(axis=1).mean(),len(U)))
print('max_abs_library_correlation unavailable: no signal artifact comparison')
