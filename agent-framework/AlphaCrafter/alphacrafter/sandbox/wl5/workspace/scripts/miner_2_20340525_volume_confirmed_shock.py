import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}; vv={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is not None and len(d)>=120:
  q=d.set_index('date'); px[s]=q.close.astype(float); vv[s]=q.volume.astype(float)
P=pd.DataFrame(px).sort_index(); V=pd.DataFrame(vv).reindex(P.index)
R=P.pct_change(); r5=P/P.shift(5)-1
# Candidate: volume-confirmed idiosyncratic shock reversal. Large unusual turnover
# shocks are expected to mean-revert; cross-sectional median removes common move.
res=r5.sub(r5.median(axis=1),axis=0)
volshock=(V/(V.rolling(30,min_periods=15).median()+1e-12)).clip(0.25,4.0)
rv=R.rolling(30,min_periods=20).std()*np.sqrt(5)
f=(-res*volshock/(rv+1e-8)).clip(-10,10)
fw=P.shift(-10)/P-1
ics=[]; dates=[]; ns=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if np.isfinite(c): ics.append(c); dates.append(dt); ns.append(len(z))
a=np.asarray(ics); dates=pd.DatetimeIndex(dates)
print('dates',len(a),'start',dates[0].date(),'end',dates[-1].date(),'mean_n',round(np.mean(ns),3),'coverage',round(np.mean(ns)/15,6),'IC',round(a.mean(),6),'daily_ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),6))
for x,y in [('2026-07-01','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2032-12-31'),('2033-01-01','2034-05-24')]:
 z=a[(dates>=pd.Timestamp(x))&(dates<=pd.Timestamp(y))]
 if len(z)>1: print('regime',x,y,'dates',len(z),'IC',round(z.mean(),6),'hit',round(np.mean(z>0),4))
S=pd.DataFrame([f.loc[d].rank(pct=True) for d in dates],index=dates)
print('turnover',round(S.diff().abs().mean().mean(),6))
for h in [5,10,20]:
 ff=P.shift(-h)/P-1; aa=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],ff.loc[dt]],axis=1).dropna()
  if len(z)>=8: aa.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,round(np.nanmean(aa),6),'n',len(aa))
rows=[(dt,s,float(f.loc[dt,s])) for dt in f.index for s in f.columns if pd.notna(f.loc[dt,s])]
pd.DataFrame(rows,columns=['date','symbol','factor_value']).to_csv('scripts/miner_2_20340525_volume_confirmed_shock_signal.csv',index=False)
