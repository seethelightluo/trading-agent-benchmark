import pandas as pd, numpy as np, glob, json
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'] for a in assets}).sort_index()
r=p.pct_change()
# Downside-risk adjusted medium horizon trend: denominator uses only negative daily returns.
down=r.where(r<0).rolling(20,min_periods=15).std()
sig=p.pct_change(20)/down
fwd=p.shift(-1)/p-1
ics=[]; dates=[]; counts=[]
for dt in p.index:
 x=sig.loc[dt]; y=fwd.loc[dt]; m=x.notna()&y.notna()
 if m.sum()>=8:
  ics.append(spearmanr(x[m],y[m]).statistic); dates.append(dt); counts.append(m.sum())
ics=np.asarray(ics)
print('idea=downside_risk_adjusted_trend_20d')
print('dates',len(ics),'mean_instruments',np.mean(counts),'coverage',np.mean(counts)/15)
print('daily IC %.9f ICIR %.9f hit %.4f std %.9f'%(np.nanmean(ics),np.nanmean(ics)/np.nanstd(ics,ddof=1),np.mean(ics>0),np.nanstd(ics,ddof=1)))
for h in [5,10,20]:
 yy=p.shift(-h)/p-1; z=[]; n=[]
 for dt in p.index:
  m=sig.loc[dt].notna()&yy.loc[dt].notna()
  if m.sum()>=8: z.append(spearmanr(sig.loc[dt][m],yy.loc[dt][m]).statistic); n.append(m.sum())
 z=np.asarray(z); print('horizon',h,'dates',len(z),'meanIC',np.nanmean(z),'ICIR',np.nanmean(z)/np.nanstd(z,ddof=1),'hit',np.mean(z>0),'n',np.mean(n))
for name,expr in [('trend','p.pct_change(20)/r.rolling(20).std()'),('rav','p.pct_change(20)/r.rolling(20).std()'),('rev','-p.pct_change(5)/r.rolling(5).std()'),('vol','r.rolling(20).std()')]:
 if name=='trend': other=p.pct_change(20)/r.rolling(20).std()
 elif name=='rev': other=-p.pct_change(5)/r.rolling(5).std()
 elif name=='vol': other=r.rolling(20).std()
 else: continue
 m=sig.stack().notna()&other.stack().notna(); print('corr',name, spearmanr(sig.stack()[m],other.stack()[m]).statistic)
# regimes
for label, a,b in [('2020', '2020-01-01','2020-12-31'),('2021_22','2021-01-01','2022-12-31'),('2023_24','2023-01-01','2024-12-31'),('2025_26','2025-01-01','2026-12-31')]:
 z=[v for d,v in zip(dates,ics) if str(d)>=a and str(d)<=b]; z=np.array(z); print('regime',label,'dates',len(z),'IC',np.nanmean(z),'ICIR',np.nanmean(z)/np.nanstd(z,ddof=1) if len(z)>1 else np.nan)
print('turnover_rank',np.mean((sig.rank(axis=1,pct=True)-sig.shift(1).rank(axis=1,pct=True)).abs().mean(axis=1).dropna()))
print('latest',p.index[-1])
