import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}; vv={}
for s in U:
 d=get_stock_daily_data(s,2500)
 if d is not None and len(d)>=140:
  x=d.set_index('date'); px[s]=x.close.astype(float); vv[s]=x.volume.astype(float)
P=pd.DataFrame(px).sort_index(); V=pd.DataFrame(vv).reindex(P.index)
r20=P/P.shift(20)-1
rv=V.rolling(20,min_periods=15).mean()/(V.rolling(60,min_periods=40).mean()+1e-12)
sig=(-r20*rv.pow(.35)).clip(-8,8)
fw=P.shift(-10)/P-1; ic=[]; dates=[]; ns=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],fw.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if np.isfinite(c): ic.append(c); dates.append(dt); ns.append(len(z))
a=np.asarray(ic); dates=pd.DatetimeIndex(dates)
rows=[(dt,s,float(sig.loc[dt,s])) for dt in sig.index for s in sig.columns if pd.notna(sig.loc[dt,s])]
pd.DataFrame(rows,columns=['date','symbol','factor_value']).to_csv('scripts/miner_3_20340831_volume_confirmed_reversal_signal.csv',index=False)
print('dates',len(a),'start',dates[0].date(),'end',dates[-1].date(),'mean_n',round(np.mean(ns),3),'coverage',round(np.mean(ns)/15,6),'IC',round(a.mean(),6),'ICIR_daily',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),6))
for x,y in [('2026-08-31','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2032-12-31'),('2033-01-01','2034-08-21')]:
 z=a[(dates>=pd.Timestamp(x))&(dates<=pd.Timestamp(y))]
 if len(z)>1: print('regime',x,len(z),round(z.mean(),6),round(z.mean()/z.std(ddof=1),6))
S=pd.DataFrame([sig.loc[d].rank(pct=True) for d in dates],index=dates); print('turnover',round(S.diff().abs().mean().mean(),6))
for h in [5,20]:
 q=P.shift(-h)/P-1; aa=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],q.loc[dt]],axis=1).dropna()
  if len(z)>=8: aa.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 aa=np.asarray([x for x in aa if np.isfinite(x)]); print('decay',h,round(aa.mean(),6))
