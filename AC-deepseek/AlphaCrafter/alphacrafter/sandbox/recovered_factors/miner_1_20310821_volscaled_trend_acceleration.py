import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
px={a:pd.read_csv(f'{base}/{a}.csv',parse_dates=['date']).set_index('date')['close'] for a in assets}
p=pd.DataFrame(px).sort_index(); r=p.pct_change()
# Candidate: volatility-scaled medium-term trend acceleration, designed to be less raw momentum
# signal = (ret20/vol20) - (ret60/vol60), using only t close; forward returns tested.
f=(p.pct_change(20)/r.rolling(20).std())-(p.pct_change(60)/r.rolling(60).std())
# robust cross-sectional winsorization/rank not needed for Spearman
print('dates',len(p),'assets',len(assets),'range',p.index.min(),p.index.max())
for h in [1,5,10,20]:
  vals=[]; ns=[]; dates=[]
  fr=p.shift(-h)/p-1
  for d in p.index:
    x=f.loc[d]; y=fr.loc[d]; ok=x.notna()&y.notna()
    if ok.sum()>=8:
      vals.append(spearmanr(x[ok],y[ok]).statistic); ns.append(ok.sum()); dates.append(d)
  z=np.asarray(vals); print('H',h,'n_dates',len(z),'meanN',np.mean(ns),'IC %.6f ICIR %.6f hit %.3f'%(np.nanmean(z),np.nanmean(z)/(np.nanstd(z,ddof=1)+1e-12)*np.sqrt(len(z)),np.mean(z>0)))
  for lo,hi in [('2020','2023'),('2024','2027'),('2028','2030'),('2031','2031')]:
   q=z[(np.array(dates)>=lo)&(np.array(dates)<=hi)]; print(lo, 'n',len(q),'ic %.5f'%(np.nanmean(q) if len(q) else np.nan))
# turnover rank changes every 10 observations
rank=f.rank(axis=1,pct=True); print('coverage',f.notna().mean().mean(),'turn10',np.nanmean((rank-rank.shift(10)).abs().mean(axis=1)))
