import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
px={}
for a in assets:
 d=pd.read_csv(os.path.join(base,a+'.csv'),parse_dates=['date']).sort_values('date').set_index('date')
 px[a]=d['close'].replace(0,np.nan)
prices=pd.DataFrame(px).sort_index(); r=np.log(prices).diff()
# Trend persistence: 30d compounded log momentum, weighted by fraction of positive sessions, normalized by realized vol.
ret30=r.rolling(30,min_periods=25).sum(); pos30=(r>0).rolling(30,min_periods=25).mean(); vol30=r.rolling(30,min_periods=25).std()
f=(ret30*(0.5+pos30)/vol30).shift(1)
fr=np.log(prices).shift(-10)-np.log(prices)
rows=[]; dates=[]
for dt in f.index:
 x=f.loc[dt]; y=fr.loc[dt]; ok=x.notna()&y.notna()
 if ok.sum()>=8:
  rows.append(spearmanr(x[ok],y[ok]).statistic); dates.append(dt)
ic=pd.Series(rows,index=dates).dropna(); print('dates',len(ic),'avgN',np.mean([((f.loc[d].notna())&(fr.loc[d].notna())).sum() for d in dates]),'coverage',f.loc[dates].notna().mean().mean())
print('IC %.8f ICIR %.8f hit %.4f turnover %.5f'%(ic.mean(),ic.mean()/ic.std(ddof=1), (ic>0).mean(), f.rank(axis=1,pct=True).diff().abs().mean().mean()))
for n in [60,120,252,756]:
 z=ic.tail(n); print('recent',n,'IC %.6f ICIR %.6f'%(z.mean(),z.mean()/z.std(ddof=1)))
for lo,hi in [('2020','2022'),('2023','2025'),('2026','2028'),('2029','2030'),('2031','2032')]:
 z=ic.loc[lo:hi]; print('regime',lo,hi,len(z),'%.6f %.6f'%(z.mean(),z.mean()/z.std(ddof=1)))
f.to_csv('scripts/miner_2_20320415_trend_persistence_signal.csv')
