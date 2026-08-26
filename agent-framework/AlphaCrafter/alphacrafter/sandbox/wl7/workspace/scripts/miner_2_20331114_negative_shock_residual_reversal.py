import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 p=pd.read_csv('../persistent/stock_data/'+s+'.csv'); p.date=pd.to_datetime(p.date); D[s]=p.set_index('date').close.astype(float)
px=pd.DataFrame(D).sort_index().ffill().loc[:'2033-11-13']; r=px.pct_change(); m=r.mean(axis=1)
disp=r.std(axis=1); shock=(m.shift(1)<0)&(disp.shift(1)>disp.shift(1).rolling(60,min_periods=30).median())
# Lagged 5d asset return residualized to the common cross-asset 5d move.
r5=px.pct_change(5); m5=m.rolling(5).sum(); beta=r.rolling(60,min_periods=30).cov(m).div(m.rolling(60,min_periods=30).var(),axis=0)
resid=r5-beta.mul(m5,axis=0); raw=-resid.shift(1); f=raw.where(shock,np.nan).rank(axis=1,pct=True)
print('cutoff','2033-11-13','dates',len(px),'instruments',len(U),'coverage',round(f.notna().mean().mean(),4),'activation',round(shock.mean(),4))
for h in [1,5,10,20]:
 vals=[]; ns=[]
 for i in range(len(px)-h):
  z=pd.concat([f.iloc[i].rename('x'),(px.iloc[i+h]/px.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8 and z.x.nunique()>2: vals.append(z.x.corr(z.y)); ns.append(len(z))
 q=pd.Series(vals).dropna(); print('H',h,'dates',len(q),'avgN',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
print('turnover',round(f.diff().abs().mean().mean(),6)); f.to_csv('scripts/miner_2_20331114_negative_shock_residual_reversal_signal.csv')
