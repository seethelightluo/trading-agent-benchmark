import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is not None and len(d)>=180: px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index(); R=P.pct_change()
# Short-horizon shock reversal normalized by slower risk baseline: rewards recent losses
ret10=R.rolling(10,min_periods=8).sum(); risk60=R.rolling(60,min_periods=40).std()
f=(-ret10/risk60.replace(0,np.nan)).replace([np.inf,-np.inf],np.nan)
fr=R.shift(-10).rolling(10,min_periods=10).sum()
ics=[]; dates=[]; ns=[]; ranks=[]; rows=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if np.isfinite(c):
   ics.append(c); dates.append(dt); ns.append(len(z)); ranks.append(f.loc[dt].rank(pct=True))
   for s,v in f.loc[dt].dropna().items(): rows.append((dt,s,float(v)))
a=np.array(ics); S=pd.DataFrame(ranks,index=dates)
pd.DataFrame(rows,columns=['date','symbol','factor_value']).to_csv('scripts/miner_1_20331013_shock_reversal_10d_signal.csv',index=False)
ir=float(a.mean()/a.std(ddof=1)*np.sqrt(252)) if len(a)>1 and a.std(ddof=1)>0 else None
print({'dates':len(a),'start':str(dates[0].date()),'end':str(dates[-1].date()),'mean_n':round(float(np.mean(ns)),3),'coverage':round(float(np.mean(ns)/15),6),'IC':round(float(a.mean()),6),'ICIR':round(ir,6),'hit':round(float(np.mean(a>0)),6),'turnover':round(float(S.diff().abs().mean().mean()),6)})
for x,y in [('2020-01-01','2024-12-31'),('2025-01-01','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2033-10-01')]:
 z=a[(np.array(dates)>=pd.Timestamp(x))&(np.array(dates)<=pd.Timestamp(y))]
 q=float(z.mean()/z.std(ddof=1)*np.sqrt(252)) if len(z)>1 and z.std(ddof=1)>0 else None
 print(x,len(z),round(float(z.mean()),6) if len(z) else None,round(q,6) if q is not None else None)
