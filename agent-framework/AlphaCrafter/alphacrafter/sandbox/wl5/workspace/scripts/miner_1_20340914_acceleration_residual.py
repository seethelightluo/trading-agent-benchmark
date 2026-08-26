import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,3000)
 if d is not None and len(d)>=180: px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index(); R=P.pct_change()
# lagged cross-asset beta and residual returns; acceleration compares recent residual trend with medium residual trend
M=R.mean(axis=1); beta=R.rolling(60,min_periods=45).cov(M)/M.rolling(60,min_periods=45).var()
res=R-beta.mul(M,axis=0)
short=res.rolling(20,min_periods=15).sum().shift(1)
medium=res.rolling(90,min_periods=70).sum().shift(1)
rv=res.rolling(60,min_periods=45).std().shift(1)
f=((short-medium/4)/(rv*np.sqrt(20)+1e-8)).clip(-8,8)
fw=P.shift(-10)/P-1
vals=[];ds=[];ns=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if np.isfinite(c): vals.append(c);ds.append(dt);ns.append(len(z))
a=np.array(vals); dsi=pd.DatetimeIndex(ds)
print('dates',len(a),'start',dsi[0].date() if len(dsi) else 'NA','end',dsi[-1].date() if len(dsi) else 'NA','mean_n',round(np.mean(ns),3),'coverage',round(np.mean(ns)/15,6),'IC',round(a.mean(),6),'ICIR_daily',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),6),flush=True)
for x,y in [('2020-01-01','2024-12-31'),('2025-01-01','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2032-12-31'),('2033-01-01','2034-09-01')]:
 z=a[(dsi>=pd.Timestamp(x))&(dsi<=pd.Timestamp(y))]
 if len(z)>1: print('regime',x,len(z),round(z.mean(),6),round(z.mean()/z.std(ddof=1),6),flush=True)
S=pd.DataFrame([f.loc[d].rank(pct=True) for d in ds],index=ds)
print('turnover',round(S.diff().abs().mean().mean(),6),flush=True)
rows=[(dt,s,float(f.loc[dt,s])) for dt in f.index for s in f.columns if pd.notna(f.loc[dt,s])]
pd.DataFrame(rows,columns=['date','symbol','factor_value']).to_csv('scripts/miner_1_20340914_acceleration_residual_signal.csv',index=False)
