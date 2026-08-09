import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'] for a in assets}
d=pd.concat(px,axis=1).sort_index(); r=d.pct_change()
# lagged 20d path efficiency: cumulative return divided by total absolute daily movement, shifted one day
ret20=d.pct_change(20); path=r.abs().rolling(20,min_periods=15).sum(); sig=(ret20/path).shift(1)
# evaluate horizons
for h in [1,5,10,20]:
  fwd=d.shift(-h)/d-1; vals=[]; ns=[]; dates=[]
  for dt in d.index:
    x=sig.loc[dt]; y=fwd.loc[dt]; ok=x.notna()&y.notna()
    if ok.sum()>=8:
      vals.append(spearmanr(x[ok],y[ok]).statistic); ns.append(ok.sum()); dates.append(dt)
  z=np.array(vals); print('H',h,'dates',len(z),'meanN',np.mean(ns),'IC %.6f ICIR %.6f hit %.4f'%(np.nanmean(z),np.nanmean(z)/np.nanstd(z,ddof=1),np.mean(z>0)))
  if h==10:
   for lo,hi in [('2020','2023'),('2024','2027'),('2028','2030'),('2031','2033')]:
    q=z[(np.array(dates)>=lo)&(np.array(dates)<=hi)]; print(lo,len(q),round(np.nanmean(q),6),round(np.nanmean(q)/np.nanstd(q,ddof=1),6))
# turnover rank changes every 10 dates
rank=sig.rank(axis=1,pct=True); print('coverage',sig.notna().mean().mean(),'turnover',rank.diff(10).abs().mean().mean())
# save signal reconstruction inputs not needed
