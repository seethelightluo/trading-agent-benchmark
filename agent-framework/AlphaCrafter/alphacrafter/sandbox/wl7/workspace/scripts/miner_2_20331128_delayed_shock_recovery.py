import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 p=pd.read_csv('../persistent/stock_data/'+s+'.csv');p.date=pd.to_datetime(p.date);D[s]=p.set_index('date').close.astype(float)
px=pd.DataFrame(D).sort_index().ffill().loc[:'2033-11-27']; r=px.pct_change(); mu=r.mean(1); disp=r.std(1)
# delayed post-shock reversal: shock observed 3 sessions ago, enter only after
# immediate rebound/noise has passed; factor uses return ending yesterday.
shock=(mu.shift(3)<0)&(disp.shift(3)>disp.shift(3).rolling(60,min_periods=30).median())
raw=-px.shift(1).pct_change(5)
f=raw.where(shock,np.nan).rank(axis=1,pct=True)
print('cutoff 2033-11-27 dates',len(px),'instruments',len(U),'coverage',round(f.notna().mean().mean(),4),'activation',round(shock.mean(),4),'active_dates',int(shock.sum()))
for h in [1,5,10,20]:
 v=[];n=[]
 for i in range(len(px)-h):
  z=pd.concat([f.iloc[i].rename('x'),(px.iloc[i+h]/px.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8 and z.x.nunique()>2:v.append(z.x.corr(z.y));n.append(len(z))
 q=pd.Series(v).dropna();print('H',h,'dates',len(q),'avgN',round(np.mean(n),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
print('turnover',round(f.diff().abs().mean().mean(),6));f.to_csv('scripts/miner_2_20331128_delayed_shock_recovery_signal.csv')
