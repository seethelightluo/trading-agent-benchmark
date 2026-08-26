import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 p=pd.read_csv('../persistent/stock_data/'+s+'.csv'); p.date=pd.to_datetime(p.date); D[s]=p.set_index('date').close.astype(float)
px=pd.DataFrame(D).sort_index().ffill().loc[:'2033-11-27']; r=px.pct_change()
# Continuous severity: prior-day broad negative return, scaled by its own 60d
# volatility, and cross-sectional dispersion. Signal is lagged 5d reversal
# multiplied by bounded shock severity; all inputs are known at decision date.
mu=r.mean(axis=1); disp=r.std(axis=1)
muvol=mu.rolling(60,min_periods=30).std(); dv=disp.rolling(60,min_periods=30).median()
severity=(-mu.shift(1)/muvol.shift(1)).clip(lower=0,upper=3) * (disp.shift(1)/dv.shift(1)).clip(lower=0.5,upper=2)
active=(mu.shift(1)<0)&severity.notna()
raw=-px.shift(1).pct_change(5)
f=raw.mul(severity,axis=0).where(active,np.nan).rank(axis=1,pct=True)
print('cutoff','2033-11-27','dates',len(px),'instruments',len(U),'coverage',round(f.notna().mean().mean(),4),'activation',round(active.mean(),4),'active_dates',int(active.sum()))
for h in [1,5,10,20]:
 vals=[]; ns=[]
 for i in range(len(px)-h):
  z=pd.concat([f.iloc[i].rename('x'),(px.iloc[i+h]/px.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8 and z.x.nunique()>2: vals.append(z.x.corr(z.y)); ns.append(len(z))
 q=pd.Series(vals).dropna(); print('H',h,'dates',len(q),'avgN',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
print('turnover',round(f.diff().abs().mean().mean(),6))
f.to_csv('scripts/miner_2_20331128_continuous_shock_recovery_signal.csv')
