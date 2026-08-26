import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is not None and len(d)>=160: px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index(); P.index=pd.to_datetime(P.index).normalize(); R=P.pct_change(); m=R.mean(axis=1); W=60
b=R.rolling(W,min_periods=35).cov(m).div(m.rolling(W,min_periods=35).var()+1e-10,axis=0).shift(1)
e=R-b.mul(m,axis=0); cum=e.rolling(W,min_periods=45).sum().shift(1); v=e.rolling(W,min_periods=45).std().shift(1)*np.sqrt(W)
f=(-cum/(v+1e-8)).replace([np.inf,-np.inf],np.nan).clip(-10,10)
rows=[]; all_ic=[]; dates=[]; ns=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],(P.shift(-10)/P-1).loc[dt]],axis=1).dropna()
 if len(z)>=8:
  c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if np.isfinite(c): all_ic.append(c);dates.append(dt);ns.append(len(z))
a=np.array(all_ic); ds=pd.DatetimeIndex(dates)
print('dates',len(a),'start',ds[0].date(),'end',ds[-1].date(),'mean_n',round(np.mean(ns),3),'coverage',round(np.mean(ns)/15,6),'IC',round(a.mean(),6),'daily_ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),6))
for x,y in [('2026-07-01','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2032-12-31'),('2033-01-01','2034-07-20')]:
 z=a[(ds>=x)&(ds<=y)]
 if len(z)>1: print('regime',x,y,'dates',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6))
S=pd.DataFrame([f.loc[d].rank(pct=True) for d in ds],index=ds); print('turnover',round(S.diff().abs().mean().mean(),6))
for h in [5,20]:
 aa=[]; F=P.shift(-h)/P-1
 for dt in f.index:
  z=pd.concat([f.loc[dt],F.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(c): aa.append(c)
 print('decay',h,'IC',round(np.mean(aa),6),'n',len(aa))
for dt in f.index:
 for s in f.columns:
  if pd.notna(f.loc[dt,s]): rows.append((dt,s,float(f.loc[dt,s])))
pd.DataFrame(rows,columns=['date','symbol','factor_value']).to_csv('scripts/miner_1_20340720_residual_momentum60_signal.csv',index=False)
