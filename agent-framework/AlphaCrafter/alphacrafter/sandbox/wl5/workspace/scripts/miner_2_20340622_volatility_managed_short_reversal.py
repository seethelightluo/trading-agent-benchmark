import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is not None and len(d)>=80: px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index(); P.index=pd.to_datetime(P.index).normalize(); R=P.pct_change()
# Short-horizon reversal, volatility-managed and conditioned on the lagged cross-sectional
# dispersion: dampen the signal during unusually high dispersion, when reversal is less stable.
ret5=P/P.shift(5)-1
vol20=R.rolling(20,min_periods=15).std()*np.sqrt(20)
disp=ret5.mad(axis=1) if hasattr(ret5,'mad') else ret5.sub(ret5.mean(axis=1),axis=0).abs().mean(axis=1)
base=-(ret5/(vol20+1e-8))
threshold=disp.rolling(120,min_periods=60).mean()+0.5*disp.rolling(120,min_periods=60).std()
# high dispersion continuation, normal regime reversal; all state inputs lagged one session
f=base.where(disp.shift(1)<=threshold.shift(1),-base).shift(1).clip(-10,10)
fw={h:P.shift(-h)/P-1 for h in [5,10,20]}
all_ic={}
for h,F in fw.items():
 a=[];ds=[];ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],F.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(c):a.append(c);ds.append(dt);ns.append(len(z))
 a=np.array(a); ds=pd.DatetimeIndex(ds); all_ic[h]=(a,ds,ns)
 print('horizon',h,'dates',len(a),'start',ds[0].date(),'end',ds[-1].date(),'mean_n',round(np.mean(ns),3),'coverage',round(np.mean(ns)/15,6),'IC',round(a.mean(),6),'daily_ICIR',round(a.mean()/a.std(ddof=1),6),'annualized_ICIR',round(a.mean()/a.std(ddof=1)*np.sqrt(252),6),'hit',round(np.mean(a>0),6))
 if h==10:
  for x,y in [('2026-07-01','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2032-12-31'),('2033-01-01','2034-06-22')]:
   z=a[(ds>=pd.Timestamp(x))&(ds<=pd.Timestamp(y))]
   if len(z)>1: print('regime',x,y,'dates',len(z),'IC',round(z.mean(),6))
  S=pd.DataFrame([f.loc[d].rank(pct=True) for d in ds],index=ds)
  print('turnover',round(S.diff().abs().mean().mean(),6))
  rows=[(dt,s,float(f.loc[dt,s])) for dt in f.index for s in f.columns if pd.notna(f.loc[dt,s])]
  pd.DataFrame(rows,columns=['date','symbol','factor_value']).to_csv('scripts/miner_2_20340622_volatility_managed_short_reversal_signal.csv',index=False)
