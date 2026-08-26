import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d):
  x=d[['date','close']].drop_duplicates('date'); x.date=pd.to_datetime(x.date); px[s]=x.set_index('date').close
px=pd.DataFrame(px).sort_index().ffill()
# Cross-sectional residual trend: 40d return relative to daily universe median, scaled by own 40d volatility.
r=np.log(px/px.shift(40)); med=r.median(axis=1)
f=((r.sub(med,axis=0))/np.log(px/px.shift(40)).rolling(40).std()).shift(1)
# Require broad, valid values; no future information
f=f.replace([np.inf,-np.inf],np.nan)
rows=[]
for h in [1,5,10,20]:
 vals=[]
 for dt in f.index:
  y=np.log(px.shift(-h).loc[dt]/px.loc[dt])
  z=pd.concat([f.loc[dt],y],axis=1).dropna()
  if len(z)>=8:
   q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(q): vals.append((dt,q,len(z)))
 a=pd.Series([x[1] for x in vals])
 print('horizon',h,'dates',len(a),'avgN',np.mean([x[2] for x in vals]),'coverage',np.mean([x[2]/15 for x in vals]),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',(a>0).mean())
 for nm,sub in [('first',a.iloc[:len(a)//3]),('middle',a.iloc[len(a)//3:2*len(a)//3]),('recent',a.iloc[2*len(a)//3:])]:
  print(' ',nm,len(sub),sub.mean(),sub.mean()/sub.std(ddof=1) if len(sub)>1 else np.nan)
# rank turnover
rr=f.rank(axis=1,pct=True); t=[]
for i in range(1,len(rr)):
 z=pd.concat([rr.iloc[i-1],rr.iloc[i]],axis=1).dropna()
 if len(z)>=8:t.append((z.iloc[:,0]-z.iloc[:,1]).abs().mean())
print('turnover',np.mean(t),'calendar_dates',len(f.index),'assets',len(px.columns))
out=f.copy();out.index.name='date';out.reset_index().to_csv('scripts/miner_1_20320809_residual_trend40_signal.csv',index=False)
