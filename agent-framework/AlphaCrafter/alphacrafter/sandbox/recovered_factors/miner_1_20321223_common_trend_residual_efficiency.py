"""Explore one idea: common-trend-conditioned residual directional efficiency.
At each date, score an asset's 20-bar idiosyncratic directional efficiency only
when the cross-asset median has a positive 20-bar trend.  The construction asks
whether orderly asset-specific advances persist in a supportive common regime.
Completed bars through 2032-12-22; no future data are used in signal formation.
"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
START=pd.Timestamp('2026-07-16'); CUT=pd.Timestamp('2032-12-22')
d={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index() for a in A}
idx=pd.DatetimeIndex(sorted(set.intersection(*[set(x.loc[START:CUT].index) for x in d.values()])))
c=pd.DataFrame({a:d[a].reindex(idx)['close'] for a in A}); r=c.pct_change()
# 60-bar rolling residual relative to contemporaneous cross-sectional median return
m=r.median(axis=1); beta=r.rolling(60,min_periods=45).cov(m).div(m.rolling(60,min_periods=45).var(),axis=0); e=r-beta.mul(m,axis=0)
# One factor only: residual directional efficiency, gated by positive common 20d return
raw=e.rolling(20,min_periods=18).sum()/e.abs().rolling(20,min_periods=18).sum()
common20=m.rolling(20,min_periods=18).sum()
s=raw.where(common20>0)
print('FACTOR common_trend_conditioned_residual_directional_efficiency_20_60obs')
print('cutoff',CUT.date(),'assets',len(A),'calendar_dates',len(idx),'coverage',int(s.notna().sum().sum()),'/',s.size,round(s.notna().mean().mean(),6))
for q in (1,5,10,20):
 f=c.shift(-q).div(c).sub(1); vals=[]; ns=[]; ds=[]
 for t in idx:
  z=pd.concat([s.loc[t],f.loc[t]],axis=1).dropna()
  if len(z)>=8:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z));ds.append(t)
 vals=np.asarray(vals);ds=pd.DatetimeIndex(ds); ns=np.asarray(ns); ic=vals.mean();ir=ic/vals.std(ddof=1)
 print('H',q,'dates',len(vals),'IC',round(ic,6),'ICIR',round(ir,6),'hit',round((vals>0).mean(),6),'mean_n',round(ns.mean(),3),'min_n',ns.min(),'PASS',abs(ic)>=.007 and abs(ir)>=.084)
 if q==20:
  for n,lo in [('2026_2029','2026-07-16'),('2030_2031','2030-01-01'),('recent_12m','2031-12-23')]:
   hi='2029-12-31' if n=='2026_2029' else ('2031-12-31' if n=='2030_2031' else CUT)
   x=vals[(ds>=lo)&(ds<=hi)];print('REGIME',n,'dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),6))
ranks=s.rank(axis=1,pct=True); print('turnover',round((ranks-ranks.shift()).abs().stack().mean(),6),'comparisons',len(idx)-1,'median_iqr',round((s.quantile(.75,axis=1)-s.quantile(.25,axis=1)).median(),6))
