import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,5000)
   if d is not None and len(d)>100:
    d=d.copy();d.date=pd.to_datetime(d.date);return d.drop_duplicates('date').set_index('date').sort_index().close.astype(float)
  except: pass
P=pd.DataFrame({s:load(s) for s in U}).sort_index();R=P.pct_change(); breadth=(R<0).mean(axis=1)
# Recovery breadth impulse: lagged improvement from widespread weakness, multiplied by asset residual momentum.
shock=(breadth.rolling(5).mean().shift(1)-breadth.rolling(20).mean().shift(1))
impulse=((shock<-.15)&(breadth.shift(1)<.55)).astype(float)
res=R.sub(R.mean(axis=1),axis=0).rolling(10).sum().shift(1)
f=res.mul(impulse,axis=0)
rows={h:[] for h in [1,3,5,10]}; ds=[];cov=[];turn=[]
for i in range(len(P)-10):
 x=f.iloc[i]
 if x.notna().sum()>=8 and x.nunique()>1:
  ds.append(P.index[i]);cov.append(x.notna().mean())
  if i:turn.append((x.rank(pct=True)-f.iloc[i-1].rank(pct=True)).abs().mean())
 for h in rows:
  y=P.iloc[i+h].div(P.iloc[i])-1;z=pd.concat([x,y],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:rows[h].append(z.iloc[:,0].corr(z.iloc[:,1]))
print('assets',len(P.columns),'total_dates',len(P),'valid_dates',len(ds),'active',int(impulse.sum()),'coverage',round(float(np.mean(cov)),4),'turnover',round(float(np.nanmean(turn)),4))
for h,a in rows.items():
 a=np.asarray(a);print('horizon',h,'n',len(a),'IC',round(float(np.nanmean(a)),6),'ICIR',round(float(np.nanmean(a)/np.nanstd(a,ddof=1)),6),'hit',round(float(np.mean(a>0)),4))
f.to_csv('scripts/miner_3_20330203_recovery_breadth_signal.csv',index_label='date')
