import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is not None and len(d)>100:
  x=d[['date','close']].copy(); x.date=pd.to_datetime(x.date); px[s]=x.set_index('date').close
P=pd.DataFrame(px).sort_index(); R=P.pct_change(); r5=P.pct_change(5); vol=R.rolling(20).std()*np.sqrt(252)
# Cross-sectional residual short-term reversal, activated when cross-sectional dispersion is elevated.
resid=r5.sub(r5.median(axis=1),axis=0); disp=r5.std(axis=1); threshold=disp.rolling(60,min_periods=30).median()
f=(-resid/vol).where(disp.ge(threshold),0.0)
fr=P.shift(-10)/P-1
ics=[]; ns=[]; ds=[]
for i in range(1,len(P)-11):
 z=pd.concat([f.iloc[i-1],fr.iloc[i]],axis=1).dropna()
 if len(z)>=8: ics.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z)); ds.append(P.index[i])
ser=pd.Series(ics,index=pd.to_datetime(ds)).dropna(); print('rows',len(P),'instruments',len(px),'ic_dates',len(ser),'avg_n',round(np.mean(ns),2)); print('IC %.6f ICIR %.6f hit %.4f coverage %.4f'%(ser.mean(),ser.mean()/ser.std(ddof=1),(ser>0).mean(),np.mean(ns)/len(U)))
rank=f.rank(axis=1,pct=True); print('turnover_proxy %.5f'%((rank-rank.shift(10)).abs().mean(axis=1).dropna().mean()))
for a,b in [('2020','2024-12-31'),('2025','2027-12-31'),('2028','2031-12-31'),('2032','2035-11-23')]:
 q=ser.loc[a:b]; print(a,b,'n',len(q),'IC %.6f ICIR %.6f'%(q.mean(),q.mean()/q.std(ddof=1) if len(q)>1 else np.nan))
for h in [1,5,10,20]:
 ff=P.shift(-h)/P-1; ii=[]
 for i in range(1,len(P)-h):
  z=pd.concat([f.iloc[i-1],ff.iloc[i]],axis=1).dropna()
  if len(z)>=8: ii.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('h',h,'IC',np.nanmean(ii),'n',len(ii))
f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna().to_csv('../persistent/miner_1_20351123_dispersion_residual_reversal_signal.csv',index=False); ser.rename('ic').to_csv('../persistent/miner_1_20351123_dispersion_residual_reversal_ic.csv')
