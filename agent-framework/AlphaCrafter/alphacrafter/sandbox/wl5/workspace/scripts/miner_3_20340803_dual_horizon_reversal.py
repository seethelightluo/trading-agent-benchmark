import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,2500)
 if d is not None and len(d)>=140: px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index(); R=P.pct_change(); v30=R.rolling(30,min_periods=20).std(); r10=P/P.shift(10)-1; r60=P/P.shift(60)-1
csmed=r60.median(axis=1); cshort=r10.median(axis=1)
f=(((r60.sub(csmed,axis=0))-(r10.sub(cshort,axis=0)))/(v30*np.sqrt(10)+1e-8)).clip(-8,8)
fwds={h:P.shift(-h)/P-1 for h in [5,10,20]}; ics={h:[] for h in fwds}; dates={h:[] for h in fwds}; ns={h:[] for h in fwds}
for dt in f.index:
 for h in fwds:
  z=pd.concat([f.loc[dt],fwds[h].loc[dt]],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(c): ics[h].append(c); dates[h].append(dt); ns[h].append(len(z))
rows=[(dt,s,float(f.loc[dt,s])) for dt in f.index for s in f.columns if pd.notna(f.loc[dt,s])]
pd.DataFrame(rows,columns=['date','symbol','factor_value']).to_csv('scripts/miner_3_20340803_dual_horizon_reversal_signal.csv',index=False)
for h in fwds:
 a=np.array(ics[h]); ds=pd.DatetimeIndex(dates[h]); sd=a.std(ddof=1); print('horizon',h,'dates',len(a),'start',ds[0].date(),'end',ds[-1].date(),'mean_n',round(np.mean(ns[h]),3),'coverage',round(np.mean(ns[h])/15,6),'IC',round(a.mean(),6),'ICIR_daily',round(a.mean()/sd,6),'ICIR_ann',round(a.mean()/sd*np.sqrt(252),6),'hit',round(np.mean(a>0),6),flush=True)
 for x,y in [('2024-07-01','2024-12-31'),('2025-01-01','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2032-12-31'),('2033-01-01','2034-08-02')]:
  z=a[(ds>=pd.Timestamp(x))&(ds<=pd.Timestamp(y))]
  if len(z)>1: print('regime',x,len(z),round(z.mean(),6),round(z.mean()/z.std(ddof=1),6))
S=pd.DataFrame([f.loc[d].rank(pct=True) for d in dates[10]],index=dates[10]); print('turnover',round(S.diff().abs().mean().mean(),6),flush=True)
