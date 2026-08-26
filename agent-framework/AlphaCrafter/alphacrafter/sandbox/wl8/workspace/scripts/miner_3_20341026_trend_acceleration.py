import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for a in assets:
 f='../persistent/stock_data/'+a+'.csv'
 if os.path.exists(f):
  d=pd.read_csv(f,parse_dates=['date']).set_index('date')
  px[a]=d['close'].replace(0,np.nan)
P=pd.DataFrame(px).sort_index(); ret=P.pct_change()
# All predictor inputs are lagged through the prior completed session.
lag=P.shift(1)
r10=lag.pct_change(10); r40=lag.pct_change(40); v20=ret.shift(1).rolling(20).std()
# Acceleration: recent trend relative to its slower baseline, risk-normalized.
raw=(r10-r40/4.0)/v20
sig=raw.rank(axis=1,pct=True)-0.5
fwd=P.shift(-10)/P-1
rows=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(q),'avgN',round(q.n.mean(),3),'coverage',round(q.n.sum()/(len(q)*15),4))
for direction in [1,-1]:
 x=q.ic*direction
 print('direction',direction,'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
 for w in [365,750,1260]:
  z=x.tail(w); print('recent',w,round(z.mean(),6),round(z.mean()/z.std(ddof=1),6))
for h in [1,5,10,20]:
 yy=P.shift(-h)/P-1; rr=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8: rr.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('decay',h,round(np.nanmean(rr),6), 'n',len(rr))
print('turnover',round(sig.diff().abs().mean().mean(),6))
sig.tail(500).to_csv('scripts/miner_3_20341026_trend_acceleration_signal.csv')
q.to_csv('scripts/miner_3_20341026_trend_acceleration_ic.csv')
