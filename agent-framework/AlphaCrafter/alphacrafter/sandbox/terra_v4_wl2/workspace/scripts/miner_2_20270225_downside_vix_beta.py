import pandas as pd, numpy as np
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
    for fn in (get_index_daily_data,get_stock_daily_data):
        try:
            x=fn(s,days=5000)
            if x is not None:return x
        except Exception: pass
v=get('VIX').set_index('date')['close']; px=pd.DataFrame({s:get(s).set_index('date')['close'] for s in U}).sort_index()
r=px.pct_change(); vx=v.reindex(px.index).ffill().pct_change()
# Lagged downside beta: covariance to positive VIX-return shocks only, robustly ranked cross-sectionally.
pos=vx.where(vx>0,0.0); w=60
mu_r=r.rolling(w,min_periods=40).mean(); mu_p=pos.rolling(w,min_periods=40).mean()
cov=((r-mu_r).mul(pos-mu_p,axis=0)).rolling(w,min_periods=40).mean()
var=((pos-mu_p)**2).rolling(w,min_periods=40).mean()
beta=cov.div(var,axis=0).replace([np.inf,-np.inf],np.nan).shift(1)
# Favor low downside/VIX beta; neutralize daily cross-sectional level.
f=-beta
f=f.sub(f.median(axis=1),axis=0)
fr=px.shift(-1)/px-1
ics=[]; ns=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1:
  ics.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z))
a=np.array(ics)
print('factor=downside_vix_beta_resilience dates',len(a),'avg_n',np.mean(ns),'coverage',np.mean(ns)/15)
print('IC',np.mean(a),'ICIR',np.mean(a)/np.std(a,ddof=1),'hit',np.mean(a>0))
for lab,lo,hi in [('2020-22','2020','2022-12-31'),('2023-24','2023','2024-12-31'),('2025-26','2025','2026-12-31'),('2027','2027','2027-02-25')]:
 q=[x for d,x in zip(f.index[0:len(ics)],ics) if str(d)>=lo and str(d)<=hi]
 print(lab,len(q),np.mean(q) if q else np.nan)
for h in [5,10]:
 zics=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],(px.shift(-h)/px-1).loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:zics.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 q=np.array(zics); print('H',h,'dates',len(q),'IC',np.mean(q),'ICIR',np.mean(q)/np.std(q,ddof=1))
print('turnover',np.mean((f.rank(axis=1,pct=True).diff().abs().sum(axis=1)>0)))
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('../persistent/factor_signals_miner_2_20270225_downside_vix_beta.csv',index=False)
