import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
px={}
for s in U:
 d=pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date')['close'].sort_index()
 px[s]=d
p=pd.DataFrame(px).sort_index()
# signal available at t: 30d lagged return divided by lagged 20d realized vol
ret=p.pct_change()
sig=p.shift(1).pct_change(30) / (ret.shift(1).rolling(20,min_periods=15).std()*np.sqrt(20))
# forward return from t close to t+10 close
fwd=p.shift(-10)/p-1
rows=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  rows.append((dt,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
r=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date')
for end in ['2033-10-12','2032-10-12','2031-10-12']:
 x=r.loc[:end]
 if len(x): print(end,'dates',len(x),'avgN',x.n.mean(),'IC',x.ic.mean(),'ICIR',x.ic.mean()/x.ic.std(ddof=1),'hit',(x.ic>0).mean())
print('coverage',sig.loc[:'2033-10-12'].notna().sum(axis=1).mean()/15)
# rank turnover
q=sig.rank(axis=1,pct=True); common=q.index.intersection(q.index.to_series().shift(1).dropna().index)
print('turnover',((q.loc[common]-q.shift(1).loc[common]).abs().mean(axis=1)).mean())
print('horizons')
for h in [1,5,10,20]:
 ff=p.shift(-h)/p-1; rr=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],ff.loc[dt]],axis=1).dropna()
  if len(z)>=8: rr.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 rr=np.array(rr); print(h,len(rr),np.nanmean(rr),np.nanmean(rr)/np.nanstd(rr,ddof=1))
