import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; px={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None: d=get_index_daily_data(s,4000)
 if d is not None: px[s]=d.sort_values('date').set_index('date')['close'].astype(float)
P=pd.DataFrame(px).sort_index().ffill(); R=P.pct_change(); r20=P.pct_change(20); r60=P.pct_change(60)
neg2=(R.clip(upper=0)**2).rolling(40,min_periods=20).mean(); down=np.sqrt(neg2)
raw=(.65*r20+.35*r60)/(down*np.sqrt(252)+1e-8); sig=raw.rank(axis=1,pct=True).shift(1)
vals={h:[] for h in [1,5,10,20]}; ns={h:[] for h in vals}
for dt in sig.index:
 for h in vals:
  y=P.shift(-h).loc[dt]/P.loc[dt]-1; z=pd.concat([sig.loc[dt],y],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: vals[h].append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns[h].append(len(z))
for h,a0 in vals.items():
 a=np.array(a0); print('%dd dates=%d avgN=%.2f IC=%.6f ICIR=%.6f hit=%.4f'%(h,len(a),np.mean(ns[h]),np.nanmean(a),np.nanmean(a)/np.nanstd(a,ddof=1),np.mean(a>0)))
print('coverage %.4f turnover %.6f period %s %s'%(np.isfinite(sig).mean().mean(),sig.diff().abs().mean(axis=1).dropna().mean(),P.index.min(),P.index.max()))
out=sig.copy(); out.index.name='date'; out.to_csv('scripts/miner_1_20300822_recovery_quality_signal.csv')
