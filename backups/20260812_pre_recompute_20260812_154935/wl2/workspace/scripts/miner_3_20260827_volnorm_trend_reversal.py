import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
dates=D['SPX'].index[(D['SPX'].index>='2020-01-01')&(D['SPX'].index<='2026-07-15')]
C=pd.DataFrame({s:D[s].close.reindex(dates) for s in U}); r=C.pct_change()
# Volatility-normalized trend with a short-horizon reversal overlay.
# The signal rewards persistent 12-day moves per unit of recent risk and
# subtracts the most recent 3-day move, all lagged before the forecast day.
vol=r.rolling(20,min_periods=15).std()
trend=r.rolling(12,min_periods=10).sum()/vol
rev=r.rolling(3,min_periods=3).sum()/vol
F=(trend.rank(axis=1,pct=True)-0.35*rev.rank(axis=1,pct=True)).rank(axis=1,pct=True).shift(1)
Y=C.pct_change().shift(-1)
q=[];ns=[]
for dt in dates:
 z=pd.DataFrame({'f':F.loc[dt],'y':Y.loc[dt]}).dropna()
 if len(z)>=8:q.append(spearmanr(z.f,z.y).statistic);ns.append(len(z))
q=np.asarray(q)
print('dates',len(q),'avgN',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4),'coverage',round(F.notna().sum().sum()/F.size,4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
for k in [63,126,252,504]:
 x=q[-k:]; print('recent',k,'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
for h in [3,5]:
 yy=C.pct_change(h).shift(-h);qq=[]
 for dt in dates:
  z=pd.DataFrame({'f':F.loc[dt],'y':yy.loc[dt]}).dropna()
  if len(z)>=8:qq.append(spearmanr(z.f,z.y).statistic)
 qq=np.asarray(qq);print('horizon',h,'IC',round(qq.mean(),6),'ICIR',round(qq.mean()/qq.std(ddof=1),6),'n',len(qq))
# regime slices provide robustness context
for a,b in [('2020-01-01','2021-12-31'),('2022-01-01','2023-12-31'),('2024-01-01','2026-07-15')]:
 x=[]
 for dt,ic in zip(dates[-len(q):],q):
  if a<=str(dt.date())<=b:x.append(ic)
 if len(x):print('regime',a,b,'n',len(x),'IC',round(np.mean(x),6),'ICIR',round(np.mean(x)/np.std(x,ddof=1),6))
