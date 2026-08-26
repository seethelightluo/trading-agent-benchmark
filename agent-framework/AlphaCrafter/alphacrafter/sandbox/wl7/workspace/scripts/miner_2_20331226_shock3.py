import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 p=pd.read_csv('../persistent/stock_data/'+s+'.csv'); p.date=pd.to_datetime(p.date); D[s]=p.set_index('date').close.astype(float)
px=pd.DataFrame(D).sort_index().ffill().loc[:'2033-12-25']; r=px.pct_change(); breadth=(r<0).mean(axis=1)
# Conditional short-horizon residual rebound: following broad weakness, favor
# assets with the most negative 3-session residual return. Inputs lagged one day.
cs=r.mean(axis=1); vol=cs.rolling(60,min_periods=30).std(); stress=(-cs.shift(1)/vol.shift(1)).clip(0,3)
active=(breadth.shift(1)>=.60)&stress.notna(); rr=px.shift(1).pct_change(3)
raw=-rr.sub(rr.mean(axis=1),axis=0); f=raw.mul(stress,axis=0).where(active,np.nan).rank(axis=1,pct=True)
print('cutoff 2033-12-25 dates',len(px),'instruments',len(U),'coverage',round(f.notna().mean().mean(),4),'active_dates',int(active.sum()))
for h in [1,5,10,20]:
 v=[]; ns=[]
 for i in range(len(px)-h):
  z=pd.concat([f.iloc[i].rename('x'),(px.iloc[i+h]/px.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8 and z.x.nunique()>2: v.append(z.x.corr(z.y)); ns.append(len(z))
 q=pd.Series(v).dropna(); print('H',h,'dates',len(q),'avgN',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
print('turnover',round(f.diff().abs().mean().mean(),6)); f.to_csv('scripts/miner_2_20331226_shock3_signal.csv')
