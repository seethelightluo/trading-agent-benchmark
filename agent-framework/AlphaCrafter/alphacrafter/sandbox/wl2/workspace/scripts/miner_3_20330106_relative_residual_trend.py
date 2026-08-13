import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 d=get_stock_daily_data(s, days=5000)
 if d is None or len(d)<120:return None
 d=d.copy(); d['date']=pd.to_datetime(d['date']); return d.drop_duplicates('date').set_index('date').sort_index()['close'].astype(float)
px={s:load(s) for s in U}; px={s:x for s,x in px.items() if x is not None}; P=pd.DataFrame(px).sort_index(); R=P.pct_change()
ret20=P.pct_change(20); vol20=R.rolling(20).std(); med=ret20.median(axis=1); f=(ret20.sub(med,axis=0)/vol20).shift(1)
rows={h:[] for h in [1,3,5,10]}; dates=[]; cov=[]; turnover=[]
for i in range(len(P)-10):
 dt=P.index[i]; x=f.iloc[i]
 for h in rows:
  z=pd.concat([x,P.iloc[i+h].div(P.iloc[i])-1],axis=1).dropna()
  if len(z)>=8: rows[h].append(z.iloc[:,0].corr(z.iloc[:,1]))
 if x.notna().sum()>=8:
  dates.append(dt); cov.append(x.notna().mean())
  if i: turnover.append((x.rank(pct=True)-f.iloc[i-1].rank(pct=True)).abs().mean())
print('dates',len(dates),'assets',len(P.columns),'coverage',np.mean(cov),'turnover',np.mean(turnover))
for h,v in rows.items():
 a=np.array(v); print(h,'n',len(a),'IC',np.nanmean(a),'ICIR',np.nanmean(a)/np.nanstd(a,ddof=1),'hit',np.mean(a>0))
for lo,hi in [(0,len(dates)//2),(len(dates)//2,len(dates))]:
 vals=[]
 for dt in dates[lo:hi]:
  i=P.index.get_loc(dt); z=pd.concat([f.loc[dt],P.iloc[i+1].div(P.iloc[i])-1],axis=1).dropna(); vals.append(z.iloc[:,0].corr(z.iloc[:,1]))
 print('sub',lo,hi,np.mean(vals),len(vals))
f.index.name='date'; f.to_csv('scripts/miner_3_20330106_relative_residual_trend_signal.csv')
