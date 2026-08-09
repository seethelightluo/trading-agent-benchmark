import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(a):
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv'); d.date=pd.to_datetime(d.date); return d.set_index('date')
raw={a:load(a) for a in assets}
px=pd.concat([raw[a].close.rename(a) for a in assets],axis=1).sort_index().ffill()
ret=px.pct_change()
# Volatility-adjusted acceleration: recent 5d return relative to the average daily pace of 20d return,
# scaled by trailing 20d volatility. This seeks inflection/continuation without raw momentum level.
r5=ret.rolling(5,min_periods=4).sum(); r20=ret.rolling(20,min_periods=15).sum(); vol=ret.rolling(20,min_periods=15).std()
sig=(r5-r20/4).div(vol.replace(0,np.nan)).clip(-10,10)
# library proxies using exact broad constructions
trend=r20.div(vol.replace(0,np.nan))
rv=pd.concat([(raw[a].volume/ raw[a].volume.rolling(20,min_periods=10).mean()).rename(a) for a in assets],axis=1).reindex(px.index).ffill().rolling(20,min_periods=10).mean()
rev=-ret
lowvol=-vol
signals={'risk_adjusted_trend':trend,'relative_volume':rv,'short_reversal':rev,'inverse_vol':lowvol}
print('period',px.index.min().date(),px.index.max().date(),'dates',len(px),'assets',len(assets))
for h in [1,5,10,20]:
 fwd=px.pct_change(h).shift(-h); vals=[]; ns=[]; dates=[]
 for dt in sig.index:
  ok=sig.loc[dt].notna()&fwd.loc[dt].notna()
  if ok.sum()>=8:
   x=sig.loc[dt,ok]; y=fwd.loc[dt,ok]; q=spearmanr(x,y).statistic
   if np.isfinite(q): vals.append(q); ns.append(ok.sum()); dates.append(dt)
 z=pd.Series(vals,index=dates); print('H',h,'dates',len(z),'meanN',round(np.mean(ns),2),'IC %.6f ICIR %.6f hit %.4f'%(z.mean(),z.mean()/z.std(ddof=1),(z>0).mean()))
 if h==10:
  for yr,g in z.groupby(z.index.year): print(' year',yr,'IC %.6f n %d'%(g.mean(),len(g)))
print('coverage %.4f turnover %.4f'%(sig.notna().sum(axis=1).mean()/15,sig.rank(axis=1,pct=True).diff().abs().mean().mean()))
# pooled signal correlation evidence against current library proxies
for name,s in signals.items():
 x=[];y=[]
 for dt in sig.index:
  ok=sig.loc[dt].notna()&s.loc[dt].notna()
  x.extend(sig.loc[dt,ok].tolist()); y.extend(s.loc[dt,ok].tolist())
 print('corr',name,'rho %.6f cells %d'%(spearmanr(x,y).statistic,len(x)))
