import numpy as np, pandas as pd
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
C=pd.concat({s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date')['close'] for s in assets},axis=1).sort_index().loc[:'2035-11-07']
r=C.pct_change(); vol=r.rolling(20).std()*np.sqrt(20)
# Cross-asset dispersion is an observable regime variable. Trend in calm markets,
# reversal after unusually dispersed 10d moves; volatility-normalized for comparability.
mom=C.pct_change(20)/vol.replace(0,np.nan)
disp=C.pct_change(10).std(axis=1)
stress=(disp>disp.rolling(60,min_periods=30).median()).astype(float)
sig=mom.mul(1-2*stress,axis=0).shift(1)
rows=[]
for i in range(1,len(C)-20):
 x=sig.iloc[i].values; ok=np.isfinite(x)
 if ok.sum()<8: continue
 for h in [1,5,10,20]:
  y=C.iloc[i+h].values/C.iloc[i].values-1; q=ok&np.isfinite(y)
  if q.sum()>=8: rows.append((C.index[i],h,np.corrcoef(x[q],y[q])[0,1],q.sum()))
df=pd.DataFrame(rows,columns=['date','h','ic','n'])
print('rows',len(df),'dates',df.date.nunique(),'assets',C.shape[1],'period',C.index.min().date(),C.index.max().date())
for h in [1,5,10,20]:
 z=df[df.h==h].dropna(); print('H%d IC %.8f ICIR %.8f hit %.4f nobs %d avgN %.2f'%(h,z.ic.mean(),z.ic.mean()/z.ic.std(ddof=1)*np.sqrt(252), (z.ic>0).mean(),len(z),z.n.mean()))
z=df[df.h==10].dropna()
for lo,hi in [('2020-01-01','2024-12-31'),('2025-01-01','2029-12-31'),('2030-01-01','2034-12-31'),('2035-01-01','2035-11-07'),('2034-11-01','2035-11-07')]:
 a=z[(z.date>=lo)&(z.date<=hi)]; print('regime',lo,hi,'n',len(a),'IC',a.ic.mean(),'ICIR',a.ic.mean()/a.ic.std(ddof=1)*np.sqrt(252) if len(a)>2 else np.nan)
print('coverage',np.isfinite(sig).sum().sum()/sig.size,'turnover',np.nanmean(np.abs(np.diff(np.nan_to_num(sig.values,nan=0),axis=0))))
sig.index.name='date'; sig.to_csv('scripts/miner_3_20351112_dispersion_switch_signal.csv')
