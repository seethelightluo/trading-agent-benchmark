import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 d=get_stock_daily_data(s,days=5000)
 if d is None or len(d)<120:return None
 d=d.copy(); d.date=pd.to_datetime(d.date); return d.drop_duplicates('date').set_index('date').sort_index()
ds={s:load(s) for s in U}; ds={s:d for s,d in ds.items() if d is not None}
P=pd.DataFrame({s:d.close.astype(float) for s,d in ds.items()}).sort_index(); R=P.pct_change()
# Range-efficiency: directional displacement relative to path length, lagged one day.
path=R.abs().rolling(20).sum(); disp=P.pct_change(20); f=(disp/path).shift(1)
# cross-sectional rank is the portfolio-useable representation
fr=f.rank(axis=1,pct=True)
rows={h:[] for h in [1,3,5,10]}; dates=[]; cov=[]; turns=[]
for i in range(len(P)-10):
 x=fr.iloc[i]
 if x.notna().sum()>=8:
  dates.append(P.index[i]); cov.append(x.notna().mean())
  if i: turns.append((x-fr.rank(axis=1,pct=True).iloc[i-1]).abs().mean())
 for h in rows:
  z=pd.concat([x,P.iloc[i+h].div(P.iloc[i])-1],axis=1).dropna()
  if len(z)>=8: rows[h].append(z.iloc[:,0].corr(z.iloc[:,1]))
print('assets',len(P.columns),'total_dates',len(P),'valid_dates',len(dates),'coverage',round(float(np.mean(cov)),4),'rank_turnover',round(float(np.nanmean(turns)),4))
for h,a in rows.items():
 a=np.asarray(a); print('horizon',h,'n',len(a),'IC',round(float(np.nanmean(a)),6),'ICIR',round(float(np.nanmean(a)/np.nanstd(a,ddof=1)),6),'hit',round(float(np.mean(a>0)),4))
for lo,hi in [(0,len(dates)//2),(len(dates)//2,len(dates))]:
 a=[]
 for dt in dates[lo:hi]:
  i=P.index.get_loc(dt); z=pd.concat([fr.loc[dt],P.iloc[i+1].div(P.iloc[i])-1],axis=1).dropna(); a.append(z.iloc[:,0].corr(z.iloc[:,1]))
 print('subperiod',lo,hi,'n',len(a),'IC',round(float(np.nanmean(a)),6))
fr.index.name='date'; fr.to_csv('scripts/miner_3_20330203_range_efficiency_signal.csv')
