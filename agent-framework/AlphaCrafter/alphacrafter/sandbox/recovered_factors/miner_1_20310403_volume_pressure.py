import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
px={}; vol={}
for a in assets:
 d=pd.read_csv(f'{base}/{a}.csv'); d.date=pd.to_datetime(d.date); d=d.set_index('date')
 px[a]=d.close; vol[a]=d.volume.replace(0,np.nan)
px=pd.DataFrame(px).sort_index(); vol=pd.DataFrame(vol).reindex(px.index)
r=px.pct_change()
# volume-weighted directional pressure: signed return weighted by relative volume, smoothed 5d
rv=vol/vol.rolling(20,min_periods=10).mean()
sig=(r*rv).rolling(5,min_periods=4).sum()
# cross-sectional standardized signal
fwd={h:px.shift(-h)/px-1 for h in [1,5,10,20]}
allics={}
for h,y in fwd.items():
 rows=[]
 for dt in px.index:
  z=pd.concat([sig.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8: rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
 s=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); all ics if False else None
 all ics
 print('H%d dates=%d meanN=%.2f IC=%+.6f ICIR=%+.6f hit=%.3f'%(h,len(s),s.n.mean(),s.ic.mean(),s.ic.mean()/s.ic.std(ddof=1),(s.ic>0).mean()))
 if h==10:
  for name,a,b in [('2020-23','2020','2023-12-31'),('2024-27','2024','2027-12-31'),('2028-30','2028','2030-12-31'),('latest120',None,None)]:
   q=s if name=='latest120' else s.loc[a:b]
   if name=='latest120': q=s.tail(120)
   print(name,len(q),q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1),(q.ic>0).mean())
print('coverage',sig.notna().mean().mean(),'active_dates',sig.notna().any(axis=1).sum())
rr=sig.rank(axis=1,pct=True).iloc[::10]
print('turnover10',rr.diff().abs().mean().mean())
print('decay shown above')
print('LIBRARY_AUDIT max_abs_library_correlation=UNAVAILABLE')
