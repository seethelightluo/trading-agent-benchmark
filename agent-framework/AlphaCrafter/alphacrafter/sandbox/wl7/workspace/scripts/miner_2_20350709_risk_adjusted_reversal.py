import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
u=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; P={}
for s in u:
 d=None
 try:d=get_index_daily_data(s,4200)
 except:pass
 if d is None:
  try:d=get_stock_daily_data(s,4200)
  except:pass
 if d is not None and len(d): P[s]=d.set_index(pd.to_datetime(d.date)).close
p=pd.DataFrame(P).sort_index(); r=p.pct_change(); mom=p.pct_change(20); v=r.rolling(30, min_periods=20).std(); dn=r.where(r<0,0).rolling(30,min_periods=20).std(); s=-(mom/(v+1e-12))*(v/(dn+v+1e-12)); s=s.div((1+r.rolling(20,min_periods=15).std().median(axis=1)).replace(0,np.nan),axis=0).shift(1)
f=p.shift(-20).div(p)-1; vals=[]; ns=[]
for dt in s.index:
 z=pd.concat([s.loc[dt],f.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  c=z.iloc[:,0].corr(z.iloc[:,1]);
  if np.isfinite(c): vals.append(c);ns.append(len(z))
a=np.array(vals); print('dates',len(a),'avgN',np.mean(ns),'IC',np.mean(a),'ICIR',np.mean(a)/np.std(a,ddof=1),'hit',np.mean(a>0))
for n in [252,756,1260]:
 q=a[-n:];print('recent',n,np.mean(q),np.mean(q)/np.std(q,ddof=1))
# top-quintile membership turnover across valid dates
ranks=s.rank(axis=1,pct=True); top=(ranks>=.8); changes=[]; cov=[]
for i in range(1,len(top)):
 a1=top.iloc[i-1];a2=top.iloc[i]; z=a1.notna()&a2.notna();
 if z.sum(): changes.append((a1[z]!=a2[z]).mean());cov.append(s.iloc[i].notna().mean())
print('turnover',np.mean(changes),'coverage',np.mean(cov))
s.to_csv('scripts/miner_2_20350709_risk_adjusted_reversal_signal.csv',index_label='date')
