import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is not None and len(d)>=180: px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index(); R=P.pct_change()
# candidate: residual of volatility-scaled 20d reversal after removing cross-sectional dispersion-conditioned reversal component
r20=R.rolling(20,min_periods=15).sum(); v20=R.rolling(20,min_periods=15).std()
base=(-r20/v20.replace(0,np.nan))
csdisp=R.rolling(20,min_periods=15).std().mean(axis=1)
# dispersion conditioned reversal: reversal attenuated in low dispersion, as prior factor proxy
cond=(-r20).div(csdisp.replace(0,np.nan),axis=0)
F=pd.DataFrame(index=P.index,columns=P.columns,dtype=float)
for dt in P.index:
 z=pd.concat([base.loc[dt],cond.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  x=z.iloc[:,1].values; y=z.iloc[:,0].values
  x=x-x.mean(); y=y-y.mean(); den=np.dot(x,x)
  F.loc[dt,z.index]=y-(np.dot(x,y)/den*x if den>1e-12 else 0)
fr=R.shift(-10).rolling(10,min_periods=10).sum()
a=[]; dates=[]; ns=[]; ranks=[]; rows=[]
for dt in F.index:
 z=pd.concat([F.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if np.isfinite(c): a.append(c); dates.append(dt); ns.append(len(z)); ranks.append(F.loc[dt].rank(pct=True))
 for s in F.columns:
  if pd.notna(F.loc[dt,s]): rows.append((dt,s,float(F.loc[dt,s])))
a=np.array(a); S=pd.DataFrame(ranks,index=dates)
pd.DataFrame(rows,columns=['date','symbol','factor_value']).to_csv('scripts/miner_3_20330929_residual_volscaled_reversal_20d_signal.csv',index=False)
print({'dates':len(a),'start':str(dates[0].date()),'end':str(dates[-1].date()),'mean_n':round(float(np.mean(ns)),3),'coverage':round(float(np.mean(ns)/15),6),'IC':round(float(a.mean()),6),'ICIR':round(float(a.mean()/a.std(ddof=1)*np.sqrt(252)),6),'hit':round(float(np.mean(a>0)),6),'turnover':round(float(S.diff().abs().mean().mean()),6)})
for x,y in [('2025-01-01','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2033-09-28')]:
 z=a[(np.array(dates)>=pd.Timestamp(x))&(np.array(dates)<=pd.Timestamp(y))]
 print(x,len(z),round(float(z.mean()),6) if len(z) else None,round(float(z.mean()/z.std(ddof=1)*np.sqrt(252)),6) if len(z)>1 else None)
