import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
fs={}
for s in U:
 d=get_stock_daily_data(s,days=6000)
 if d is not None and len(d)>100:
  d=d.copy(); d.date=pd.to_datetime(d.date); fs[s]=d.set_index('date').close.astype(float)
px=pd.DataFrame(fs).sort_index(); r=px.pct_change()
sig=(-(px/px.shift(60)-1)/(r.rolling(60,min_periods=40).std()*np.sqrt(252)+1e-6)).shift(1)
y=px.shift(-10)/px-1
ics=[]; rows=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],y.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if np.isfinite(q): ics.append(q); rows.append((dt,q,len(z)))
a=np.array(ics); print('cutoff',px.index.max().date(),'dates',len(a),'assets',len(fs),'avg_n',np.mean([x[2] for x in rows])); print('IC %.8f ICIR %.8f hit %.4f'%(a.mean(),a.mean()/(a.std(ddof=1)+1e-12),np.mean(a>0)))
rk=sig.rank(axis=1,pct=True); tt=[]
for x,z in zip(rk.index[:-1],rk.index[1:]):
 q=pd.concat([rk.loc[x],rk.loc[z]],axis=1).dropna()
 if len(q)>=8: tt.append(np.mean(abs(q.iloc[:,0]-q.iloc[:,1])))
print('turnover',np.mean(tt),'coverage',np.mean([x[2] for x in rows])/len(fs))
for h in [1,5,10,20]:
 yy=px.shift(-h)/px-1; aa=[]
 for dt in sig.index:
  q=pd.concat([sig.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(q)>=8: aa.append(q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
 print('decay',h,np.nanmean(aa))
for label,lo,hi in [('2020-24','2020','2024-12-31'),('2025-30','2025','2030-12-31'),('2031-35','2031','2035-12-31'),('recent120',str(px.index.max()-pd.Timedelta(days=180)),str(px.index.max()))]:
 v=[q for d,q,n in rows if str(d.date())>=lo and str(d.date())<=hi]; print(label,len(v),np.mean(v) if v else np.nan,(np.mean(v)/(np.std(v,ddof=1)+1e-12)) if len(v)>1 else np.nan)
sig.to_csv('scripts/miner_2_20351109_risk_adjusted_momentum60_signal.csv')
