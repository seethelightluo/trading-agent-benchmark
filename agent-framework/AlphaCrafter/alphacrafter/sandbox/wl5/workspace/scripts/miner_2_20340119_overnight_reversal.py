import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
F={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is not None and len(d)>=80:
  q=d.set_index('date').sort_index().astype(float); F[s]=q
C=pd.DataFrame({s:q.close for s,q in F.items()}).sort_index(); O=pd.DataFrame({s:q.open for s,q in F.items()}).reindex(C.index)
# Overnight gap relative to prior close; contrarian mean of last 5 sessions.
gap=(O/C.shift(1)-1).replace([np.inf,-np.inf],np.nan)
f=(-gap.rolling(5,min_periods=4).mean()).clip(-.5,.5)
ics=[]; dates=[]; ns=[]; ranks=[]; rows=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],(C.shift(-10)/C-1).loc[dt]],axis=1).dropna()
 if len(z)>=8:
  ic=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if np.isfinite(ic): ics.append(ic);dates.append(dt);ns.append(len(z));ranks.append(f.loc[dt].rank(pct=True))
 for s in f.columns:
  if pd.notna(f.loc[dt,s]): rows.append((dt,s,float(f.loc[dt,s])))
pd.DataFrame(rows,columns=['date','symbol','factor_value']).to_csv('scripts/miner_2_20340119_overnight_reversal_signal.csv',index=False)
a=np.array(ics); S=pd.DataFrame(ranks,index=dates)
print({'dates':len(a),'start':str(dates[0].date()),'end':str(dates[-1].date()),'mean_n':round(np.mean(ns),3),'coverage':round(np.mean(ns)/15,6),'IC':round(a.mean(),6),'ICIR':round(a.mean()/a.std(ddof=1)*np.sqrt(252),6),'hit':round(np.mean(a>0),6),'turnover':round(S.diff().abs().mean().mean(),6)})
for x,y in [('2020-01-01','2025-12-31'),('2026-07-16','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2034-01-18')]:
 z=a[(np.array(dates)>=pd.Timestamp(x))&(np.array(dates)<=pd.Timestamp(y))]
 if len(z)>1: print('regime',x,len(z),round(z.mean(),6),round(z.mean()/z.std(ddof=1)*np.sqrt(252),6))
for h in [5,10,20]:
 aa=[]; ff=C.shift(-h)/C-1
 for dt in f.index:
  z=pd.concat([f.loc[dt],ff.loc[dt]],axis=1).dropna()
  if len(z)>=8: aa.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('horizon',h,'n',len(aa),'IC',round(np.nanmean(aa),6))
