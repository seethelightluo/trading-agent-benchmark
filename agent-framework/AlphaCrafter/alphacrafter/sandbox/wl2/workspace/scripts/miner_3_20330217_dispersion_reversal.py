import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def ld(s):
 d=get_stock_daily_data(s,days=5000)
 if d is None:return None
 d=d.copy();d.date=pd.to_datetime(d.date);return d.drop_duplicates('date').set_index('date').sort_index()
ds={s:ld(s) for s in U};ds={s:d for s,d in ds.items() if d is not None}
P=pd.DataFrame({s:d.close.astype(float) for s,d in ds.items()}).sort_index();R=P.pct_change()
v=R.rolling(20).std(); disp=R.std(axis=1).rolling(20).rank(pct=True)
# Dispersion-conditioned short-term reversal, lagged: signal only in top quartile cross-asset dispersion.
raw=-R.rolling(3).sum()/v
active=(disp>0.75).astype(float); f=(raw.mul(active,axis=0)).shift(1);fr=f.rank(axis=1,pct=True)
rows={h:[] for h in [1,3,5,10]};dates=[];cov=[];turn=[]
for i in range(len(P)-10):
 x=fr.iloc[i]
 if x.notna().sum()>=8:
  dates.append(P.index[i]);cov.append(x.notna().mean())
  if i:turn.append((x-fr.iloc[i-1]).abs().mean())
  for h in rows:
   z=pd.concat([x,P.iloc[i+h].div(P.iloc[i])-1],axis=1).dropna()
   if len(z)>=8:rows[h].append(z.iloc[:,0].corr(z.iloc[:,1]))
print('assets',len(P.columns),'dates',len(P),'valid',len(dates),'coverage',round(np.mean(cov),4),'turnover',round(np.nanmean(turn),4))
for h,a in rows.items():
 a=np.array(a);print('horizon',h,'n',len(a),'IC',round(np.nanmean(a),6),'ICIR',round(np.nanmean(a)/np.nanstd(a,ddof=1),6),'hit',round(np.mean(a>0),4))
for lo,hi in [(0,len(dates)//2),(len(dates)//2,len(dates))]:
 a=[]
 for dt in dates[lo:hi]:
  i=P.index.get_loc(dt);z=pd.concat([fr.loc[dt],P.iloc[i+1].div(P.iloc[i])-1],axis=1).dropna();a.append(z.iloc[:,0].corr(z.iloc[:,1]))
 print('subperiod',lo,hi,'n',len(a),'IC',round(np.nanmean(a),6))
fr.index.name='date';fr.to_csv('scripts/miner_3_20330217_dispersion_reversal_signal.csv')
