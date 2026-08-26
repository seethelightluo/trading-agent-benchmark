import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for a in assets:
 f='../persistent/stock_data/'+a+'.csv'
 if os.path.exists(f):
  d=pd.read_csv(f,parse_dates=['date']).set_index('date')
  px[a]=d['close'].replace(0,np.nan)
P=pd.DataFrame(px).sort_index(); r=P.pct_change()
# Every input is lagged: smoothed multi-horizon trend, normalized by lagged volatility.
r20=P.shift(1).pct_change(20); r60=P.shift(1).pct_change(60)
v30=r.shift(1).rolling(30).std()
# persistence rewards agreement, while retaining signed trend direction
agreement=((np.sign(r20)+np.sign(r60))/2).clip(-1,1)
raw=((0.55*r20+0.45*r60)/v30)*(1+0.35*agreement)
# 3-day EWMA reduces signal churn; cross-sectional rank is robust to scale
sig=raw.ewm(span=3,min_periods=1).mean().rank(axis=1,pct=True)-.5
rows=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],(P.shift(-10)/P-1).loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(q),'avgN',q.n.mean(),'coverage',q.n.sum()/(len(q)*15))
print('IC',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std(ddof=1),'hit',(q.ic>0).mean(),'turnover',sig.diff().abs().mean().mean())
for w in [365,750,1260]:
 z=q.tail(w); print('recent',w,'IC',z.ic.mean(),'ICIR',z.ic.mean()/z.ic.std(ddof=1))
for h in [1,5,10,20]:
 yy=P.shift(-h)/P-1; rr=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8: rr.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('decay',h,np.nanmean(rr),'n',len(rr))
sig.tail(500).to_csv('scripts/miner_1_20341026_smoothed_trend_consistency_signal.csv')
q.to_csv('scripts/miner_1_20341026_smoothed_trend_consistency_ic.csv')
