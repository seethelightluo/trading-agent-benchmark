import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2033-03-17')
xs={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index()
 xs[s]=d.loc[:end,'close'].astype(float)
p=pd.DataFrame(xs).sort_index(); r=np.log(p).diff()
# candidate: short-term reversal residualized against medium trend, volatility normalized
# residualize 5d return by 20d return, using cross-sectional beta per date is not predictive; use per asset rolling regression lag-safe
f=pd.DataFrame(index=p.index)
for s in U:
 rr=r[s]
 # short reversal, remove persistent trend component and scale by lagged vol
 f[s]=-(rr.rolling(5).sum() - 0.25*rr.rolling(20).sum()) / rr.rolling(20).std().shift(1)
# rank cross section at each date; forward 10 trading day return
fr=p.shift(-10)/p-1
rows=[]
for dt in f.index:
 z=f.loc[dt]; y=fr.loc[dt]; q=pd.concat([z,y],axis=1).dropna()
 if len(q)>=8:
  rows.append((dt,len(q),spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic))
a=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date')
mu=a.ic.mean(); sd=a.ic.std(ddof=1); icir=mu/sd*np.sqrt(252) if sd else np.nan
# turnover of rank-normalized signal, averaged adjacent valid dates
rank=f.rank(axis=1,pct=True); turnover=rank.diff().abs().mean(axis=1).mean()
print('dates',len(a),'avgN',a.n.mean(),'coverage',a.n.sum()/(len(a)*len(U)),'IC',mu,'ICIR',icir,'hit', (a.ic>0).mean(),'turnover',turnover)
for h in [1,5,10,20]:
 yy=p.shift(-h)/p-1; vals=[]
 for dt in f.index:
  q=pd.concat([f.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(q)>=8: vals.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
 print('decay',h,np.nanmean(vals),len(vals))
# artifacts
sig=f.copy(); sig.index=sig.index.strftime('%Y-%m-%d'); sig.to_csv('scripts/miner_1_20330317_short_reversal_residual_signal.csv')
a.to_csv('scripts/miner_1_20330317_short_reversal_residual_ic.csv')
