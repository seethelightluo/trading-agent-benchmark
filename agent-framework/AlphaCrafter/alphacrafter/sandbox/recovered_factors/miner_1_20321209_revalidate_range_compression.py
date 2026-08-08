"""Scheduled enhanced revalidation of range-compression persistence; completed bars through 2032-12-08."""
import pandas as pd, numpy as np
from scipy.stats import spearmanr
AS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2032-12-08'); START=pd.Timestamp('2026-07-16')
D={a:pd.read_csv(f'../persistent/stock_data/{a}.csv',parse_dates=['date']).set_index('date').sort_index() for a in AS}
idx=pd.DatetimeIndex(sorted(set.intersection(*[set(x.loc[(x.index>=START)&(x.index<=CUT)].index) for x in D.values()])))
c=pd.DataFrame({a:D[a].reindex(idx).close for a in AS}); h=pd.DataFrame({a:D[a].reindex(idx).high for a in AS}); l=pd.DataFrame({a:D[a].reindex(idx).low for a in AS})
r=c.pct_change(); m=r.median(axis=1); beta=r.rolling(60,min_periods=45).cov(m).div(m.rolling(60,min_periods=45).var(),axis=0); res=r-beta.mul(m,axis=0); z=res.div(res.rolling(60,min_periods=45).std())
# A downside event is residual return <= -0.75 rolling residual standard deviations. Range is compressed if its intraday range is low relative to its 60d asset baseline.
rng=((h-l)/c).replace([np.inf,-np.inf],np.nan); relrng=rng.div(rng.rolling(60,min_periods=45).median())
event=(z<=-0.75); compressed=(1-relrng).clip(-3,3).where(event)
# Persistence: 20-observation event-conditioned compression minus 60-observation baseline.
s=compressed.rolling(20,min_periods=8).mean()-compressed.rolling(60,min_periods=18).mean()
print('FACTOR residual_downside_range_compression_persistence_20_60obs cutoff',CUT.date(),'assets',len(AS),'calendar_dates',len(idx)); print('coverage',int(s.notna().sum().sum()),'/',s.size,round(s.notna().mean().mean(),6))
for q in [1,5,10,20]:
 f=c.shift(-q)/c-1; vals=[]; ns=[]; dates=[]
 for t in idx:
  x=pd.concat([s.loc[t],f.loc[t]],axis=1).dropna()
  if len(x)>=8: vals.append(spearmanr(x.iloc[:,0],x.iloc[:,1]).statistic); ns.append(len(x)); dates.append(t)
 vals=np.array(vals); ds=pd.DatetimeIndex(dates); ns=np.array(ns); ic=vals.mean(); ir=ic/vals.std(ddof=1)
 print('H',q,'dates',len(vals),'IC',round(ic,6),'ICIR',round(ir,6),'hit',round((vals>0).mean(),6),'mean_n',round(ns.mean(),3),'min_n',int(ns.min()),'PASS',abs(ic)>=.007 and abs(ir)>=.084)
 if q==20:
  for name,lo,hi in [('2026_2029','2026-07-16','2029-12-31'),('2030_to_cutoff','2030-01-01',CUT),('recent_12m','2031-12-09',CUT)]:
   v=vals[(ds>=lo)&(ds<=hi)]; print('REGIME',name,'dates',len(v),'IC',round(v.mean(),6),'ICIR',round(v.mean()/v.std(ddof=1),6),'hit',round((v>0).mean(),6))
ranks=s.rank(axis=1,pct=True); print('turnover',round((ranks-ranks.shift()).abs().stack().mean(),6),'median_iqr',round((s.quantile(.75,axis=1)-s.quantile(.25,axis=1)).median(),6))
