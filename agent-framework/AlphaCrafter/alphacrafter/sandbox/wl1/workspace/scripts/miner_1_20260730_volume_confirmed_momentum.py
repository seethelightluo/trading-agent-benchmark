import os
import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
fac={}; fwds={}
for s in U:
 d=pd.read_csv(os.path.join('../persistent/stock_data',s+'.csv')); d['date']=pd.to_datetime(d.date); d=d.sort_values('date').set_index('date')
 p=np.log(pd.to_numeric(d.close,errors='coerce')); v=pd.to_numeric(d.volume,errors='coerce')
 vr=(v.rolling(20,min_periods=15).mean()/v.rolling(60,min_periods=40).mean()).clip(.5,2)
 fac[s]=(p-p.shift(20))*vr
 fwds[s]=p.shift(-10)-p
factor=pd.DataFrame(fac).sort_index(); fwd=pd.DataFrame(fwds).reindex(factor.index)
ics=[]; dates=[]; ns=[]; turnovers=[]; prev=None
for dt in factor.index:
 if not (pd.Timestamp('2020-01-02')<=dt<=pd.Timestamp('2026-07-15')): continue
 z=pd.concat([factor.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(q): ics.append(q);dates.append(dt);ns.append(len(z))
 sig=factor.loc[dt].rank(pct=True)
 if prev is not None: turnovers.append(np.nanmean(np.abs(sig-prev)))
 prev=sig
ics=np.asarray(ics); print('dates',len(ics),'avg_instruments',np.mean(ns),'coverage',np.mean(ns)/15)
print('IC %.8f ICIR %.8f hit %.5f turnover %.8f'%(ics.mean(),ics.mean()/ics.std(ddof=1),np.mean(ics>0),np.mean(turnovers)))
for h in [1,5,10,20]:
 aa=[]
 for s in U:
  d=pd.read_csv(os.path.join('../persistent/stock_data',s+'.csv'));d['date']=pd.to_datetime(d.date);d=d.sort_values('date').set_index('date');p=np.log(d.close);fwds[s]=p.shift(-h)-p
 ff=pd.DataFrame(fwds).sort_index()
 for dt in factor.index:
  if pd.Timestamp('2020-01-02')<=dt<=pd.Timestamp('2026-07-15'):
   z=pd.concat([factor.loc[dt],ff.loc[dt]],axis=1).dropna()
   if len(z)>=8: aa.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('decay',h,'%.8f'%np.mean(aa))
for yr in range(2020,2027):
 a=[v for d,v in zip(dates,ics) if d.year==yr]
 if a:print('annual',yr,'%.8f'%np.mean(a),len(a))
print('last_date',dates[-1])
