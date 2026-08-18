import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def ld(s):
 d=get_stock_daily_data(s,days=5000)
 if d is None:return None
 d=d.copy();d.date=pd.to_datetime(d.date);return d.drop_duplicates('date').set_index('date').sort_index()
ds={s:ld(s) for s in U}; ds={s:d for s,d in ds.items() if d is not None}
P=pd.DataFrame({s:d.close.astype(float) for s,d in ds.items()}).sort_index(); R=P.pct_change()
# Multi-horizon volatility-normalized residual reversal, lagged one day.
csmed=R.median(axis=1)
res=R.sub(csmed,axis=0)
vol=R.rolling(20).std()*np.sqrt(20)
z=res/vol
raw=-(0.50*z + 0.30*z.rolling(3).sum()/np.sqrt(3) + 0.20*z.rolling(5).sum()/np.sqrt(5))
f=raw.shift(1); fr=f.rank(axis=1,pct=True)
rows=[]; dates=[]; cov=[]; turn=[]
for i in range(len(P)-10):
 x=fr.iloc[i]
 if x.notna().sum()>=8:
  dates.append(P.index[i]);cov.append(x.notna().mean())
  if i: turn.append((x-fr.iloc[i-1]).abs().mean())
  zc=pd.concat([x,P.iloc[i+1].div(P.iloc[i])-1],axis=1).dropna()
  if len(zc)>=8: rows.append(zc.iloc[:,0].corr(zc.iloc[:,1]))
print('assets',len(P.columns),'dates',len(P),'valid',len(dates),'coverage',round(np.mean(cov),4),'turnover',round(np.nanmean(turn),4))
a=np.array(rows); print('horizon 1 n',len(a),'IC',round(np.nanmean(a),6),'ICIR',round(np.nanmean(a)/np.nanstd(a,ddof=1),6),'hit',round(np.mean(a>0),4))
for lo,hi in [(0,len(a)//2),(len(a)//2,len(a))]: print('subperiod',lo,hi,'n',hi-lo,'IC',round(float(np.nanmean(a[lo:hi])),6),'ICIR',round(float(np.nanmean(a[lo:hi])/np.nanstd(a[lo:hi],ddof=1)),6))
fr.index.name='date';fr.to_csv('scripts/miner_3_20330217_multihorizon_reversal_signal.csv')
